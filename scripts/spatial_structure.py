#!/usr/bin/env python3
"""Spatial-structure diagnostics for the model maps and the interpreted cells.

`neighbor_change` cleanly separates the speckly v6 but barely distinguishes the
smooth variants (v2/v3/v5). These richer diagnostics capture spatial grain:

  - mean patch size per class -- connected-component labeling (8-connectivity) on
    each class mask; a patch is a contiguous run of one class.
  - Moran's I on the class raster -- global spatial autocorrelation (queen/8-neighbor
    contiguity). High = smooth, near 0 = per-pixel noise.

Both are computed on the interpreted cells too, so the interpretations act as the
reference scale (not an absolute target). To keep the comparison fair, the model is
measured **within the same interpreted-cell footprints** (same 10 m window size), and
both maps are put on the common 10-class scheme via the crosswalk.

Note on Moran's I: class codes are nominal, so treat the value as a smoothness
diagnostic (structure vs. speckle), not a quantitative autocorrelation of a
meaningful variable.

Outputs (reports/spatial_structure/):
  - spatial_structure_summary.csv        per source: patches, mean/median patch, Moran's I
  - patch_size_by_class.csv              per source x class
  - patch_size_ecdf.png                  patch-size ECDF, all sources vs interpreted
  - patch_size_hist_smallmultiples.png   per-variant distribution vs interpreted
  - mean_patch_size_by_class.png         per-class mean patch size by source
  - morans_i_by_source.png               Moran's I by source (interpreted = reference)

Usage:
    python scripts/spatial_structure.py
    python scripts/spatial_structure.py --targets 2019
    python scripts/spatial_structure.py --versions v2 v6

Requires: rasterio, numpy, pandas, matplotlib, scipy
"""

import argparse
import glob
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreted_vs_model as C  # reuse stitch/crosswalk/dedup helpers
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

warnings.filterwarnings("ignore")

OUT = "reports/spatial_structure"
PIX_HA = None               # ha per pixel, set in main() from the reference rasters' transforms
STRUCT = np.ones((3, 3), int)  # 8-connectivity
PRED_BAND = {"v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5}   # band order in pred_<bracket>_cell*.tif


def _cell_bracket_gid(path):
    m = re.search(r"grid_(\d+)_.*opt_(\d{4}_\d{4})", os.path.basename(path))
    return str(int(m.group(1))).zfill(5), m.group(2)


def select_by_truth(cells, truth_csv):
    """Keep one raster per location using the adjudicated reviewer from the truth CSV."""
    df = pd.read_csv(truth_csv, dtype=str, keep_default_na=False)
    truth = {str(int(g)).zfill(5): rev.strip().lower() for g, rev in zip(df.grid_id, df.reviewer)}
    rx = re.compile(r"reviewer_([A-Za-z]+)_grid_(\d+)_", re.I)
    by_grid = defaultdict(list)
    for c in cells:
        m = rx.search(os.path.basename(c))
        if m:
            by_grid[str(int(m.group(2))).zfill(5)].append((m.group(1).lower(), c))
    kept, missing, mismatch = [], [], []
    for gid, revpaths in by_grid.items():
        want = truth.get(gid)
        if want is None:
            missing.append(gid); continue
        match = [c for r, c in revpaths if r == want]
        if not match:
            mismatch.append((gid, want)); continue
        kept.append(match[0])
    return sorted(kept), missing, mismatch


def patch_sizes(arr_common, classes):
    """Return {class: np.array of patch sizes (px)} via 8-connected labeling."""
    out = {}
    for c in classes:
        mask = arr_common == c
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=STRUCT)
        if n:
            out[c] = np.bincount(lab.ravel())[1:]  # drop background label 0
    return out


def morans_i(x, valid):
    """Global Moran's I, queen (8-neighbour) contiguity, over valid pixels.

    Counts each adjacency once (E, S, SE, SW); edges handled by slicing (no wrap).
    """
    x = x.astype(np.float64)
    N = int(valid.sum())
    if N < 3:
        return np.nan
    m = x[valid].mean()
    d = np.where(valid, x - m, 0.0)
    denom = float(((x[valid] - m) ** 2).sum())
    if denom == 0:
        return np.nan
    pairs = [((slice(None), slice(0, -1)), (slice(None), slice(1, None))),   # E
             ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),   # S
             ((slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None))),   # SE
             ((slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1)))]   # SW
    num = 0.0
    W = 0
    for sa, sb in pairs:
        both = valid[sa] & valid[sb]
        num += float((d[sa][both] * d[sb][both]).sum())
        W += int(both.sum())
    if W == 0:
        return np.nan
    return (N / W) * (num / denom)


def collect_source(name, arrays_iter, classes):
    """Aggregate patch sizes (per class + pooled) and per-cell Moran's I for a source."""
    by_class = {c: [] for c in classes}
    pooled = []
    morans = []
    for arr_common in arrays_iter:
        ps = patch_sizes(arr_common, classes)
        for c, sizes in ps.items():
            by_class[c].append(sizes)
            pooled.append(sizes)
        morans.append(morans_i(arr_common, arr_common > 0))
    pooled = np.concatenate(pooled) if pooled else np.array([])
    by_class = {c: (np.concatenate(v) if v else np.array([])) for c, v in by_class.items()}
    morans = np.array([x for x in morans if not np.isnan(x)])
    return dict(name=name, pooled=pooled, by_class=by_class, morans=morans)


def area_ecdf(sizes_px):
    """Cumulative fraction of total area (a patch's area is its pixel count) in patches of size <= x.

    Sorting by size and taking cumsum(size)/sum(size) weights each patch by its area, not by a count
    of one, so the single-pixel floor collapses and the grain of the landscape shows.
    """
    s = np.sort(np.asarray(sizes_px, dtype=float))
    if s.size == 0:
        return s, s
    return s, np.cumsum(s) / s.sum()


def median_by_area(sizes_px):
    """Patch size (ha) where the area-ECDF crosses 0.5: half the class area sits in smaller patches."""
    s, cf = area_ecdf(sizes_px)
    if s.size == 0:
        return np.nan
    return float(s[np.searchsorted(cf, 0.5)]) * PIX_HA


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5", "v6"])
    ap.add_argument("--targets", nargs="+", default=None, help="restrict to these target years")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="use every raster (default: one interpretation per location)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth", default=None,
                    help="use the adjudicated reviewer per location from this CSV (overrides dedup)")
    ap.add_argument("--preds", default=None,
                    help="use the temporally-matched per-bracket prediction rasters in this dir as "
                         "the model field instead of the static mosaics")
    ap.add_argument("--spec", default=None,
                    help="also include spec_all from this dir of single-band pred_specall rasters")
    ap.add_argument("--out", default=None, help="output folder (default reports/spatial_structure)")
    args = ap.parse_args()

    global OUT, PIX_HA
    if args.out:
        OUT = args.out

    rf2common, names, colors = C.load_mappings()
    classes = sorted(names)  # common codes 1..10

    cells = sorted(glob.glob(os.path.join(C.RF_DIR, "**", SC.RF_GLOB), recursive=True))
    if args.targets:
        keep = set(args.targets)
        cells = [c for c in cells if C.target_year(c) in keep]
    if args.truth:
        cells, missing, mismatch = select_by_truth(cells, args.truth)
        if mismatch:
            print("STOP: truth reviewer with no matching raster:", mismatch); return
        ref_src = f"adjudicated reviewer per location ({os.path.basename(args.truth)})"
        print(f"adjudicated reference: {len(cells)} cells from {args.truth}")
    elif not args.keep_duplicates:
        cells, n_dup = C.dedupe_cells(cells, args.seed)
        ref_src = f"one interpretation per location, random dedup seed {args.seed}"
        print(f"de-duplicated: {n_dup} location(s); {len(cells)} unique cells")
    else:
        ref_src = "every raster (duplicates kept)"
    map_src = (f"temporally-matched per-bracket predictions ({args.preds})" if args.preds
               else "static v2-v6 mosaics (single 2018/2020 classification)")
    print(f"cells: {len(cells)}   sources: interpreted + {' '.join(args.versions)}")
    # model fields are measured on the cell footprints, so the cells' pixel size applies to all sources
    PIX_HA = SC.uniform_pixel_area_m2(cells) / SC.M2_PER_HA

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reference.txt"), "w") as fh:
        fh.write(f"reference points: {ref_src}\nmodel field: {map_src}\n")

    # interpreted (remap RF codes -> common)
    def interp_arrays():
        for f in cells:
            with rasterio.open(f) as ds:
                yield C.to_common_rf(ds.read(1), rf2common)
    print("  interpreted ...", flush=True)
    sources = [collect_source("interpreted", interp_arrays(), classes)]

    # each model version, measured on the same cell footprints; codes already common. with --preds
    # the field is the cell's temporally-matched per-bracket prediction band, else the static mosaic
    for v in args.versions:
        tiles = None if args.preds else [rasterio.open(t) for t in C.model_tiles(v)]

        def model_arrays(v=v, tiles=tiles):
            band = PRED_BAND[v]
            for f in cells:
                if args.preds:
                    gid, bracket = _cell_bracket_gid(f)
                    pp = os.path.join(args.preds, bracket, f"pred_{bracket}_cell{gid}.tif")
                    with rasterio.open(pp) as ds:
                        yield ds.read(band)
                else:
                    with rasterio.open(f) as ds:
                        yield C.stitch_model_to_cell(ds, tiles)
        print(f"  model {v} ...", flush=True)
        sources.append(collect_source(v, model_arrays(), classes))
        if tiles:
            for t in tiles:
                t.close()

    # optional spec_all source: single-band per-bracket predictions on the same cell footprints
    model_names = list(args.versions)
    if args.spec:
        model_names.append("spec_all")
        blank = []

        def spec_arrays():
            for f in cells:
                gid, bracket = _cell_bracket_gid(f)
                pp = os.path.join(args.spec, bracket, f"pred_specall_{bracket}_cell{gid}.tif")
                with rasterio.open(pp) as ds:
                    a = ds.read(1)
                if not (a >= 1).any():                    # entirely nodata: contributes no patches
                    blank.append((bracket, gid))
                yield a
        print("  model spec_all ...", flush=True)
        sources.append(collect_source("spec_all", spec_arrays(), classes))
        print(f"    spec_all: {len(blank)} blank cells contributed no patches")
        with open(os.path.join(OUT, "reference.txt"), "a") as fh:
            fh.write(f"spec_all: {args.spec} (single-band pred_specall per bracket); "
                     f"{len(blank)} of {len(cells)} cells are entirely nodata and contribute no "
                     f"patches\n")

    # ---- summary table ----
    rows = []
    for s in sources:
        p = s["pooled"]
        rows.append(dict(source=s["name"], n_patches=int(p.size),
                         mean_patch_px=round(float(p.mean()), 2) if p.size else np.nan,
                         mean_patch_ha=round(float(p.mean()) * PIX_HA, 4) if p.size else np.nan,
                         median_patch_px=float(np.median(p)) if p.size else np.nan,
                         median_by_area_ha=round(median_by_area(p), 4) if p.size else np.nan,
                         morans_i_mean=round(float(s["morans"].mean()), 4) if s["morans"].size else np.nan,
                         morans_i_std=round(float(s["morans"].std()), 4) if s["morans"].size else np.nan))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "spatial_structure_summary.csv"), index=False)

    # ---- per-class table ----
    crows = []
    for s in sources:
        for c in classes:
            v = s["by_class"][c]
            if v.size:
                crows.append(dict(source=s["name"], code=c, cls=names[c], n_patches=int(v.size),
                                  mean_patch_px=round(float(v.mean()), 2),
                                  median_patch_px=float(np.median(v))))
    pd.DataFrame(crows).to_csv(os.path.join(OUT, "patch_size_by_class.csv"), index=False)

    make_plots(sources, classes, names, colors, model_names)

    print("\n" + "=" * 60)
    print(summ.to_string(index=False))
    print("\nmedian-by-area patch size (ha), smallest to largest grain "
          "(half of a source's class area sits in patches smaller than this):")
    for r in sorted(rows, key=lambda d: (d["median_by_area_ha"] if d["median_by_area_ha"] == d["median_by_area_ha"] else 1e9)):
        print(f"  {r['source']:12} {r['median_by_area_ha']}")
    print(f"\noutputs -> {OUT}/")


def _caption(fig, text, top=1.0, width=125, fontsize=8, color="0.35"):
    """Add a wrapped descriptive caption below the figure, reserving space for it (and leaving room
    above for a suptitle when top < 1)."""
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=fontsize, color=color)


def make_plots(sources, classes, names, colors, versions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {s["name"]: s for s in sources}
    order = ["interpreted"] + list(versions)
    palette = {"interpreted": "black", "v2": "#1f77b4", "v3": "#2ca02c",
               "v4": "#9467bd", "v5": "#ff7f0e", "v6": "#d62728", "spec_all": "#8c564b"}

    def ecdf(a):
        a = np.sort(a)
        return a, np.arange(1, a.size + 1) / a.size

    # 1) ECDF overlay (patch size in ha, log-x)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for nm in order:
        p = by_name[nm]["pooled"] * PIX_HA
        if p.size:
            xs, ys = ecdf(p)
            ax.plot(xs, ys, label=nm, color=palette.get(nm), lw=2 if nm == "interpreted" else 1.3,
                    ls="-" if nm == "interpreted" else "--")
    ax.set_xscale("log")
    ax.set_xlabel("patch size (ha, log scale)")
    ax.set_ylabel("cumulative fraction of patches")
    ax.set_title("Patch-size distribution: model variants vs. interpreted (reference)")
    ax.legend(); ax.grid(alpha=0.3)
    xlim = ax.get_xlim()                               # reuse the same log x-range for the area ECDF
    _caption(fig, "Cumulative fraction of patches at or below a given size (hectares, log scale), "
                  "pooled over all cells and classes, one line per source. Each patch is one "
                  "8-connected same-class region and counts once regardless of its area, so a curve "
                  "shifted left means more small patches, a count-based fragmentation ranking. The "
                  "interpreted reference is the solid black line.")
    fig.savefig(os.path.join(OUT, "patch_size_ecdf.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 1b) AREA-WEIGHTED ECDF: cumulative fraction of class AREA in patches of size <= x. same pooling,
    # colors, and log x-range as the count ECDF above; single-pixel specks now collapse to the left
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for nm in order:
        p = by_name[nm]["pooled"]                     # patch sizes in px (area = pixel count)
        if not p.size:
            continue
        s, cf = area_ecdf(p)
        xs = s * PIX_HA
        ref = nm == "interpreted"
        ax.plot(xs, cf, label=nm, color=palette.get(nm), lw=2.5 if ref else 1.3,
                ls="-" if ref else "--", zorder=3 if ref else 2)
        # mark the median-by-area (x where the curve crosses 0.5) as an ordering anchor
        mx = xs[np.searchsorted(cf, 0.5)]
        ax.plot([mx], [0.5], marker="o", ms=7 if ref else 5, color=palette.get(nm),
                mec="white", mew=0.8, zorder=4)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=1)   # 0.5 reference for the median-by-area
    ax.set_xscale("log"); ax.set_xlim(xlim)
    ax.set_xlabel("Patch Size (ha, Log Scale)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction of Class Area", fontsize=12)
    ax.set_title("Area-Weighted Patch-Size Distribution: Model Variants vs. Interpreted Reference",
                 fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(False)
    _caption(fig, "Cumulative fraction of class area held in patches at or below a given size "
                  "(hectares, log scale), the same patches as the count ECDF but weighted by pixel "
                  "area, so single-pixel specks collapse to the left and the grain of the landscape "
                  "shows. The dot marks each source's median-by-area patch size, where half of its "
                  "class area sits in smaller patches. Sources further right hold their area in larger, "
                  "more contiguous patches; the interpreted reference (solid black) is the coarsest.",
             width=115, fontsize=9, color="0.3")
    fig.savefig(os.path.join(OUT, "patch_size_ecdf_area_weighted.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) small multiples: each variant's histogram vs interpreted
    ref = by_name["interpreted"]["pooled"] * PIX_HA
    bins = np.logspace(np.log10(max(ref.min(), PIX_HA)), np.log10(ref.max() + 1), 40) if ref.size else 40
    fig, axes = plt.subplots(1, len(versions), figsize=(3.2 * len(versions), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, versions):
        p = by_name[v]["pooled"] * PIX_HA
        ax.hist(ref, bins=bins, density=True, color="black", histtype="step", lw=2, label="interpreted")
        if p.size:
            ax.hist(p, bins=bins, density=True, color=palette.get(v), alpha=0.5, label=v)
        ax.set_xscale("log"); ax.set_title(v, fontsize=10); ax.set_xlabel("patch ha")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.suptitle("Per-variant patch-size distribution vs. interpreted", fontsize=11)
    _caption(fig, "Patch-size histograms (density, hectares on a log axis) for each classified map "
                  "(filled) overlaid on the interpreted reference (black outline), one panel per "
                  "source. Mass shifted toward smaller patches, relative to the reference outline, "
                  "indicates a more fragmented map.", top=0.93)
    fig.savefig(os.path.join(OUT, "patch_size_hist_smallmultiples.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3) mean patch size per class, grouped by source
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [names[c] for c in classes]
    x = np.arange(len(classes)); w = 0.8 / len(order)
    for i, nm in enumerate(order):
        vals = [by_name[nm]["by_class"][c].mean() * PIX_HA if by_name[nm]["by_class"][c].size else 0
                for c in classes]
        ax.bar(x + i * w, vals, w, label=nm, color=palette.get(nm))
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean patch size (ha)")
    ax.set_title("Mean patch size per class, by source")
    ax.legend(ncol=len(order), fontsize=8)
    _caption(fig, "Mean patch size (hectares) per class, grouped bars by source. A larger bar means "
                  "more contiguous patches of that class. The change classes and the speckly v6 have "
                  "very small mean patches, while the interpreted reference and the smoother variants "
                  "reach larger sizes on the abundant classes.")
    fig.savefig(os.path.join(OUT, "mean_patch_size_by_class.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 4) Moran's I by source
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = [by_name[nm]["morans"].mean() if by_name[nm]["morans"].size else np.nan for nm in order]
    errs = [by_name[nm]["morans"].std() if by_name[nm]["morans"].size else 0 for nm in order]
    ax.bar(order, means, yerr=errs, capsize=4,
           color=[palette.get(nm) for nm in order])
    ref_i = by_name["interpreted"]["morans"].mean() if by_name["interpreted"]["morans"].size else np.nan
    ax.axhline(ref_i, ls="--", color="black", lw=1, label=f"interpreted ref ({ref_i:.2f})")
    ax.set_ylabel("Moran's I (Mean per Cell)", fontsize=12)
    ax.set_title("Spatial Autocorrelation of the Class Raster (Moran's I by Source)",
                 fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(False)
    _caption(fig, "Mean per-cell Moran's I (queen contiguity over the class raster) for each source, "
                  "with error bars for the between-cell standard deviation and the interpreted "
                  "reference value marked (dashed). Class codes are nominal, so read Moran's I as a "
                  "spatial-smoothness diagnostic (structure vs. speckle), not a quantitative "
                  "autocorrelation of a meaningful variable. Higher Moran's I means more spatial "
                  "autocorrelation (smoother maps); v6 is near zero (salt-and-pepper speckle), while "
                  "the smoother classifiers sit close to the interpreted reference.",
             width=115, fontsize=9, color="0.3")
    fig.savefig(os.path.join(OUT, "morans_i_by_source.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
