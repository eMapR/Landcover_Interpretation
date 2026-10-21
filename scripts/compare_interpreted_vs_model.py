#!/usr/bin/env python3
"""Compare interpreted RF land cover cells against the AlphaEarth model map.

For every interpreted Sentinel-2 grid cell (``rf_class_*.tif``), this script:
  1. stitches the model-map GeoTIFF tiles (a GEE export split into tiles) and
     clips them to the cell's exact frame (both are 10 m, EPSG:5070, so the clip
     is a nearest-neighbour lattice match, no resampling distortion);
  2. translates both maps to a common 10-class scheme via
     ``data/reference/class_crosswalk.csv``;
  3. computes agreement statistics (overall accuracy, per-class precision/recall/
     F1/IoU, macro-F1, mean IoU, Cohen's kappa) from a confusion matrix; and
  4. saves a side-by-side figure: Interpreted | Model | Agreement.

Cells that fall in the model's background (no classified data) are reported as
"no overlap" and excluded from the statistics.

Outputs (under ``outputs/comparison_<version>/``):
  - ``<cell>.png``                 per-cell side-by-side figure
  - ``per_cell_metrics.csv``       one row per overlapping cell
  - ``global_confusion_matrix.csv``/``.png``
  - ``global_metrics.txt``         pooled metrics across all cells

Model versions v2-v5 are spatially-smooth classifiers; v6 is the speckly
dot-product classifier. All are compared; the per-version "neighbor-change"
value reports how spatially smooth vs per-pixel each one is.

Usage:
    python scripts/compare_interpreted_vs_model.py                 # all cells, v2-v6
    python scripts/compare_interpreted_vs_model.py --versions v2 v6
    python scripts/compare_interpreted_vs_model.py --limit 6       # quick preview
    python scripts/compare_interpreted_vs_model.py --no-figures    # metrics only

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import glob
import os
import random
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

warnings.filterwarnings("ignore")

RF_DIR = SC.RF_DIR
MODEL_DIR = "data/raw/model_maps"
CROSSWALK = "data/reference/class_crosswalk.csv"
MODEL_LEGEND = "data/reference/model_maps_10class_legend.csv"
OUT_ROOT = SC.out_path("outputs")


def load_mappings():
    """Return (rf2common, common_names, common_colors) keyed by common code 1..10.

    The common code is the model-map code (1..10); each maps to one RF code.
    """
    cw = pd.read_csv(CROSSWALK)
    rf2common = {}
    for r in cw.itertuples():
        if pd.notna(r.model_code) and pd.notna(r.rf_code) and int(r.model_code) > 0:
            rf2common[int(r.rf_code)] = int(r.model_code)
    leg = pd.read_csv(MODEL_LEGEND)
    names, colors = {}, {}
    for r in leg.itertuples():
        if int(r.code) > 0:
            names[int(r.code)] = r.display_name
            colors[int(r.code)] = r.color
    return rf2common, names, colors


def model_tiles(version):
    folder = os.path.join(MODEL_DIR, f"classified_maps_10class_{version}")
    tiles = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not tiles:
        raise SystemExit(f"no model tiles found in {folder} (run fetch_model_maps.py)")
    return tiles


def bounds_intersect(a, b):
    return not (a.right <= b.left or a.left >= b.right or
                a.top <= b.bottom or a.bottom >= b.top)


def noise_level(tile_src, sample=2000):
    """Fraction of horizontally-adjacent pixels that differ, over a sample window.

    A spatially-smooth classified map is well under ~0.15; a speckly per-pixel
    method (e.g. the v6 dot-product classifier) runs much higher (~0.8). Reported
    per version as context for the agreement metrics, not as a pass/fail.
    """
    w = min(sample, tile_src.width); h = min(sample, tile_src.height)
    r0 = tile_src.height // 2 - h // 2
    a = tile_src.read(1, window=((r0, r0 + h), (0, w)))
    a = a[(a > 0).any(axis=1)] if (a > 0).any() else a
    if a.size < 2 or a.shape[1] < 2:
        return 0.0
    return float((a[:, 1:] != a[:, :-1]).mean())


def stitch_model_to_cell(cell_ds, tile_srcs):
    """Reproject/stitch the model tiles into the cell's exact grid. 0 = background.

    NB: rasterio.reproject re-initializes the whole destination to 0 on every
    call, so each tile must be warped into its own temp array and then merged
    (keeping non-background pixels); accumulating into one shared array would let
    a non-overlapping tile erase a previous tile's data for boundary cells.
    """
    dst = np.zeros((cell_ds.height, cell_ds.width), dtype=np.uint8)
    for src in tile_srcs:
        if not bounds_intersect(cell_ds.bounds, src.bounds):
            continue
        tmp = np.zeros_like(dst)
        reproject(
            source=rasterio.band(src, 1), destination=tmp,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=cell_ds.transform, dst_crs=cell_ds.crs,
            resampling=Resampling.nearest,
        )
        dst = np.where(tmp > 0, tmp, dst)
    return dst


def to_common_rf(rf_arr, rf2common):
    out = np.zeros_like(rf_arr, dtype=np.uint8)
    for rf_code, common in rf2common.items():
        out[rf_arr == rf_code] = common
    return out


def confusion(interp, model, n=10):
    """Confusion matrix over common classes 1..n. Rows=interpreted, cols=model.

    Only pixels valid in BOTH maps (class in 1..n) are counted.
    """
    valid = (interp >= 1) & (interp <= n) & (model >= 1) & (model <= n)
    i = interp[valid].astype(int) - 1
    m = model[valid].astype(int) - 1
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (i, m), 1)
    return cm, int(valid.sum())


def metrics_from_cm(cm):
    """Per-class + summary metrics. Rows=truth(interpreted), cols=pred(model)."""
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float)   # interpreted support (truth)
    col = cm.sum(0).astype(float)   # model predicted
    total = cm.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(row > 0, tp / row, np.nan)
        precision = np.where(col > 0, tp / col, np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
        iou = np.where((row + col - tp) > 0, tp / (row + col - tp), np.nan)
    present = row > 0  # classes actually present in the interpreted reference
    oa = tp.sum() / total if total else np.nan
    # Cohen's kappa
    pe = (row * col).sum() / (total * total) if total else np.nan
    kappa = (oa - pe) / (1 - pe) if total and (1 - pe) != 0 else np.nan
    return dict(
        overall_accuracy=oa,
        macro_f1=np.nanmean(f1[present]) if present.any() else np.nan,
        mean_iou=np.nanmean(iou[present]) if present.any() else np.nan,
        kappa=kappa,
        precision=precision, recall=recall, f1=f1, iou=iou,
        support=row, n_valid=int(total),
    )


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def plot_cell(name, rf_common, model_common, colors, names, m, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    codes = sorted(colors)
    cmap = ListedColormap(["#ffffff"] + [colors[c] for c in codes])  # index 0 = white (bg)
    norm = BoundaryNorm(np.arange(-0.5, len(codes) + 1.5), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    axes[0].imshow(rf_common, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Interpreted (RF)", fontsize=10)
    axes[1].imshow(model_common, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Model (AlphaEarth)", fontsize=10)

    valid = (rf_common >= 1) & (model_common >= 1)
    agree = np.full(rf_common.shape, 0, dtype=np.uint8)   # 0 = n/a
    agree[valid & (rf_common == model_common)] = 1        # match
    agree[valid & (rf_common != model_common)] = 2        # mismatch
    acmap = ListedColormap(["#eeeeee", "#2ca02c", "#d62728"])
    axes[2].imshow(agree, cmap=acmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                   interpolation="nearest")
    axes[2].set_title("Agreement", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    class_handles = [Patch(facecolor=colors[c], edgecolor="k", label=f"{c} {names[c]}")
                     for c in codes]
    agree_handles = [Patch(facecolor="#2ca02c", edgecolor="k", label="match"),
                     Patch(facecolor="#d62728", edgecolor="k", label="mismatch"),
                     Patch(facecolor="#eeeeee", edgecolor="k", label="no model data")]
    axes[2].legend(handles=agree_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.legend(handles=class_handles, loc="lower center", ncol=min(10, len(codes)), fontsize=8,
               bbox_to_anchor=(0.5, 0.16))

    fig.suptitle(f"{name}\nvalid={100*m['n_valid']/rf_common.size:.0f}%  "
                 f"OA={m['overall_accuracy']:.2f}  macroF1={m['macro_f1']:.2f}  "
                 f"mIoU={m['mean_iou']:.2f}  kappa={m['kappa']:.2f}", fontsize=9)
    _caption(fig, "Side-by-side maps for one interpreted grid cell: the interpreted RF land cover "
             "(left), the AlphaEarth model map (center), and their per-pixel agreement (right), all "
             "in the shared 10-class scheme. In the agreement panel green marks matching pixels, red "
             "marks mismatches, and light grey marks pixels with no model data, which are excluded "
             "from the statistics. The header reports the valid overlap fraction along with overall "
             "accuracy, macro-F1, mean IoU, and kappa for this cell.", top=0.93)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, names, out_path):
    import matplotlib.pyplot as plt
    codes = sorted(names)
    labels = [names[c] for c in codes]
    with np.errstate(invalid="ignore"):
        norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Model (AlphaEarth)"); ax.set_ylabel("Interpreted (RF)")
    ax.set_title("Row-normalized confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if norm[i, j] < 0.6 else "white")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def target_year(path):
    m = re.search(r"target_(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def location_key(path):
    """Identify a physical cell/observation: grid id + sample + target year.

    Rasters sharing this key are the same location+time labeled by different
    reviewers (repeats); the reviewer name is intentionally excluded.
    """
    m = re.search(rf"grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)",
                  os.path.basename(path))
    return (m.group(1), m.group(2), m.group(3)) if m else (os.path.basename(path),)


def dedupe_cells(cells, seed):
    """Keep one randomly-chosen interpretation per location. Returns (kept, n_dup_locations)."""
    groups = defaultdict(list)
    for c in cells:
        groups[location_key(c)].append(c)
    rng = random.Random(seed)
    kept, n_dup = [], 0
    for k in sorted(groups):
        v = sorted(groups[k])
        if len(v) > 1:
            n_dup += 1
            kept.append(rng.choice(v))
        else:
            kept.append(v[0])
    return sorted(kept), n_dup


def run_version(version, cells, rf2common, names, colors, limit, no_figures, min_valid, suffix=""):
    """Compare every overlapping cell against one model version. Returns a summary dict."""
    tiles = model_tiles(version)
    tile_srcs = [rasterio.open(t) for t in tiles]
    out_dir = os.path.join(OUT_ROOT, f"comparison_{version}{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    nl = noise_level(tile_srcs[0])
    speckle = "  (speckly/per-pixel, e.g. dot-product)" if nl > 0.5 else ""
    print(f"\n=== model {version} ({len(tiles)} tiles)  neighbor-change={nl:.2f}{speckle} ===")

    global_cm = np.zeros((10, 10), dtype=np.int64)
    rows = []
    n_overlap = n_processed = 0

    for cell in cells:
        name = os.path.splitext(os.path.basename(cell))[0]
        with rasterio.open(cell) as ds:
            rf_arr = ds.read(1)
            model_arr = stitch_model_to_cell(ds, tile_srcs)
            cell_size = ds.height * ds.width

        if (model_arr > 0).mean() < min_valid:
            continue  # cell sits in model background -> no overlap
        n_overlap += 1
        if limit is not None and n_processed >= limit:
            continue

        rf_common = to_common_rf(rf_arr, rf2common)
        cm, n_valid = confusion(rf_common, model_arr)
        if n_valid == 0:
            continue
        global_cm += cm
        m = metrics_from_cm(cm)
        rows.append(dict(cell=name, n_valid=n_valid,
                         valid_frac=round(n_valid / cell_size, 3),
                         overall_accuracy=round(m["overall_accuracy"], 4),
                         macro_f1=round(m["macro_f1"], 4),
                         mean_iou=round(m["mean_iou"], 4),
                         kappa=round(m["kappa"], 4)))
        if not no_figures:
            plot_cell(name, rf_common, model_arr, colors, names, m,
                      os.path.join(out_dir, name + ".png"))
        n_processed += 1
        if n_processed % 25 == 0:
            print(f"  processed {n_processed} overlapping cells ...")

    for s in tile_srcs:
        s.close()

    if not rows:
        print(f"  no overlapping cells for {version}")
        return None

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_cell_metrics.csv"), index=False)

    codes = sorted(names)
    cm_df = pd.DataFrame(global_cm, index=[f"{c} {names[c]}" for c in codes],
                         columns=[f"{c} {names[c]}" for c in codes])
    cm_df.to_csv(os.path.join(out_dir, "global_confusion_matrix.csv"))
    if not no_figures:
        plot_confusion(global_cm, names, os.path.join(out_dir, "global_confusion_matrix.png"))

    gm = metrics_from_cm(global_cm)
    with open(os.path.join(out_dir, "global_metrics.txt"), "w") as fh:
        fh.write(f"model version: {version}   neighbor-change: {nl:.3f}\n")
        fh.write(f"overlapping cells: {n_overlap}   scored cells: {len(rows)}\n")
        fh.write(f"pooled valid pixels: {gm['n_valid']:,}\n\n")
        fh.write(f"overall accuracy: {gm['overall_accuracy']:.4f}\n")
        fh.write(f"macro F1:         {gm['macro_f1']:.4f}\n")
        fh.write(f"mean IoU:         {gm['mean_iou']:.4f}\n")
        fh.write(f"Cohen's kappa:    {gm['kappa']:.4f}\n\n")
        fh.write(f"{'class':<18}{'precision':>10}{'recall':>10}{'f1':>10}{'iou':>10}{'support':>12}\n")
        for k, c in enumerate(codes):
            fh.write(f"{names[c]:<18}{gm['precision'][k]:>10.3f}{gm['recall'][k]:>10.3f}"
                     f"{gm['f1'][k]:>10.3f}{gm['iou'][k]:>10.3f}{int(gm['support'][k]):>12,}\n")

    print(f"  overlapping={n_overlap}  scored={len(rows)}  OA={gm['overall_accuracy']:.3f}  "
          f"macroF1={gm['macro_f1']:.3f}  mIoU={gm['mean_iou']:.3f}  kappa={gm['kappa']:.3f}  -> {out_dir}/")
    return dict(version=version, neighbor_change=round(nl, 3), overlapping=n_overlap,
                scored=len(rows), overall_accuracy=round(gm["overall_accuracy"], 4),
                macro_f1=round(gm["macro_f1"], 4), mean_iou=round(gm["mean_iou"], 4),
                kappa=round(gm["kappa"], 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5", "v6"],
                    help="model versions to compare (default: v2 v3 v4 v5 v6)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N overlapping cells per version")
    ap.add_argument("--no-figures", action="store_true", help="compute metrics only, skip per-cell PNGs")
    ap.add_argument("--min-valid", type=float, default=0.01,
                    help="min fraction of cell with model data to count as overlap (default: 0.01)")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="only compare interpreted cells with these target years "
                         "(e.g. 2019 to match the model's 2018-2020 window)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="score every raster, incl. locations labeled by multiple reviewers "
                         "(default: keep one random interpretation per location)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for picking one interpretation per location (default: 0)")
    args = ap.parse_args()

    rf2common, names, colors = load_mappings()
    cells = sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True))

    suffix = ""
    if args.targets:
        keep = set(args.targets)
        cells = [c for c in cells if target_year(c) in keep]
        suffix = "_target" + "-".join(args.targets)

    if not args.keep_duplicates:
        cells, n_dup = dedupe_cells(cells, args.seed)
        print(f"de-duplicated: {n_dup} location(s) had multiple interpretations; "
              f"kept one each (seed={args.seed})")

    print(f"interpreted cells: {len(cells)}   versions: {' '.join(args.versions)}   "
          f"targets: {' '.join(args.targets) if args.targets else 'all'}")

    summary = []
    for v in args.versions:
        r = run_version(v, cells, rf2common, names, colors,
                        args.limit, args.no_figures, args.min_valid, suffix=suffix)
        if r:
            summary.append(r)

    if summary:
        os.makedirs(OUT_ROOT, exist_ok=True)
        sdf = pd.DataFrame(summary)
        sdf.to_csv(os.path.join(OUT_ROOT, f"comparison_summary_by_version{suffix}.csv"), index=False)
        print("\n" + "=" * 70)
        print("cross-version summary (interpreted vs model):")
        print(sdf.to_string(index=False))
        print(f"\nwrote {OUT_ROOT}/comparison_summary_by_version{suffix}.csv")


if __name__ == "__main__":
    main()
