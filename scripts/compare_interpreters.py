#!/usr/bin/env python3
"""Inter-interpreter agreement: same grid cell labeled by two reviewers.

Some grid cells were independently interpreted by more than one reviewer (matched
on grid id + sample + target year). For each such pair this script:
  1. loads both reviewers' classified rasters (identical footprint, so a direct
     pixel-for-pixel comparison, no reprojection);
  2. builds a confusion matrix over the RF land cover classes and computes
     agreement stats (overall agreement, per-class F1/IoU, macro-F1, mean IoU,
     Cohen's kappa); and
  3. renders a side-by-side figure: Reviewer A | Reviewer B | Agreement.

Metrics reported are direction-independent (F1, IoU, overall agreement, kappa are
symmetric between the two reviewers).

Outputs (under ``outputs/interpreter_agreement/``):
  - ``<grid>_<revA>_vs_<revB>.png``   per-pair side-by-side figure
  - ``per_pair_metrics.csv``          one row per reviewer pair
  - ``by_reviewer_pair.csv``          mean agreement grouped by reviewer pairing
  - ``global_confusion_matrix.csv``/``.png``  pooled over all pairs
  - ``global_metrics.txt``            pooled per-class stats

Usage:
    python scripts/compare_interpreters.py
    python scripts/compare_interpreters.py --limit 6      # quick preview
    python scripts/compare_interpreters.py --no-figures   # metrics only

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import glob
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

warnings.filterwarnings("ignore")

RF_DIR = SC.RF_DIR
RF_LEGEND = "data/reference/label_lookup.csv"
OUT_DIR = SC.out_path("outputs/interpreter_agreement")
NAME_RE = re.compile(rf"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)", re.I)


def load_legend():
    df = pd.read_csv(RF_LEGEND)
    codes = [int(c) for c in df.code]
    names = {int(r.code): r.display_name for r in df.itertuples()}
    colors = {int(r.code): r.color for r in df.itertuples()}
    return codes, names, colors


def find_pairs():
    """Return {(grid,sample,target): [(reviewer, path), ...]} for multi-reviewer cells."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True)):
        m = NAME_RE.search(os.path.basename(f))
        if m:
            groups[(m.group(2), m.group(3), m.group(4))].append((m.group(1).lower(), f))
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def confusion(a, b, codes):
    """Confusion matrix over `codes`. Rows = reviewer A, cols = reviewer B.

    Only pixels where both reviewers assigned a legend class are counted.
    """
    idx = {c: i for i, c in enumerate(codes)}
    lut = np.full(max(codes) + 1, -1)
    for c, i in idx.items():
        lut[c] = i
    ai = lut[a]; bi = lut[b]
    valid = (ai >= 0) & (bi >= 0)
    n = len(codes)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (ai[valid], bi[valid]), 1)
    return cm, int(valid.sum())


def metrics_from_cm(cm):
    """Symmetric agreement metrics from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float)
    col = cm.sum(0).astype(float)
    total = cm.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((2 * tp + (row - tp) + (col - tp)) > 0,
                      2 * tp / (2 * tp + (row - tp) + (col - tp)), np.nan)
        iou = np.where((row + col - tp) > 0, tp / (row + col - tp), np.nan)
    present = (row + col) > 0
    oa = tp.sum() / total if total else np.nan
    pe = (row * col).sum() / (total * total) if total else np.nan
    kappa = (oa - pe) / (1 - pe) if total and (1 - pe) != 0 else np.nan
    return dict(overall_agreement=oa,
                macro_f1=np.nanmean(f1[present]) if present.any() else np.nan,
                mean_iou=np.nanmean(iou[present]) if present.any() else np.nan,
                kappa=kappa, f1=f1, iou=iou, support=row + col, n_valid=int(total))


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def plot_pair(gid, revA, revB, arrA, arrB, codes, names, colors, m, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    order = sorted(codes)
    lut = {c: i + 1 for i, c in enumerate(order)}   # 0 reserved for "other/nodata"
    def remap(a):
        out = np.zeros_like(a, dtype=np.int16)
        for c, i in lut.items():
            out[a == c] = i
        return out
    ra, rb = remap(arrA), remap(arrB)
    cmap = ListedColormap(["#ffffff"] + [colors[c] for c in order])
    norm = BoundaryNorm(np.arange(-0.5, len(order) + 1.5), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    axes[0].imshow(ra, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title(f"Reviewer: {revA}", fontsize=10)
    axes[1].imshow(rb, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(f"Reviewer: {revB}", fontsize=10)

    valid = (ra > 0) & (rb > 0)
    agree = np.zeros(ra.shape, np.uint8)
    agree[valid & (ra == rb)] = 1
    agree[valid & (ra != rb)] = 2
    acmap = ListedColormap(["#eeeeee", "#2ca02c", "#d62728"])
    axes[2].imshow(agree, cmap=acmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation="nearest")
    axes[2].set_title("Agreement", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    present = [c for c in order if (arrA == c).any() or (arrB == c).any()]
    class_handles = [Patch(facecolor=colors[c], edgecolor="k", label=f"{names[c]}") for c in present]
    agree_handles = [Patch(facecolor="#2ca02c", edgecolor="k", label="agree"),
                     Patch(facecolor="#d62728", edgecolor="k", label="disagree")]
    axes[2].legend(handles=agree_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.legend(handles=class_handles, loc="lower center", bbox_to_anchor=(0.5, 0.11),
               ncol=min(8, len(present)), fontsize=8)
    fig.suptitle(f"grid {gid}   {revA} vs {revB}\n"
                 f"agreement={m['overall_agreement']:.2f}  macroF1={m['macro_f1']:.2f}  "
                 f"mIoU={m['mean_iou']:.2f}  kappa={m['kappa']:.2f}", fontsize=9)
    _caption(fig, "Two reviewers' land-cover interpretations of the same CKIT-RF cell shown "
                  "side by side, with the rightmost panel marking where they agree in green and "
                  "disagree in red. The left and center panels share the class color legend at "
                  "the bottom, and the header reports overall agreement, macro F1, mean IoU, and "
                  "Cohen's kappa for this pair. Read it to see which classes and locations drive "
                  "the disagreement between the two interpreters.", top=0.93)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, codes, names, out_path, title="Inter-interpreter agreement (pooled over all pairs)",
                   decimals=0):
    """Pooled inter-interpreter confusion in the shared PA/UA style: cells are raw counts, color
    is the row proportion, with a PA column (row-conditional agreement) and reference support on
    the right, a UA row (column-conditional agreement) and predicted support on the bottom, and OA
    and kappa in the corner. There is no ground-truth axis here (both axes are interpreters), so PA
    is agreement given Reviewer A's label and UA is agreement given Reviewer B's label.
    """
    import matplotlib.pyplot as plt

    order = sorted(codes)
    present = [i for i, c in enumerate(order) if cm[i].sum() or cm[:, i].sum()]
    M = cm[np.ix_(present, present)].astype(float)
    labels = [names[order[i]] for i in present]
    n = len(labels)

    tp = np.diag(M)
    row = M.sum(1)                                         # reviewer A support (row totals)
    col = M.sum(0)                                         # reviewer B support (column totals)
    tot = M.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rn = M / np.where(row[:, None] > 0, row[:, None], np.nan)   # row proportion
        pa = np.where(row > 0, tp / row, np.nan)           # agreement given A's label
        ua = np.where(col > 0, tp / col, np.nan)           # agreement given B's label
    oa = tp.sum() / tot if tot else np.nan
    pe = (row * col).sum() / (tot * tot) if tot else np.nan
    kappa = (oa - pe) / (1 - pe) if tot and (1 - pe) != 0 else np.nan
    blues, greens = plt.get_cmap("Blues"), plt.get_cmap("Greens")

    # build an (n+1)x(n+1) rgba image: main block colored by row proportion, margins by agreement
    img = np.ones((n + 1, n + 1, 4))
    for i in range(n):
        for j in range(n):
            img[i, j] = blues(rn[i, j] if np.isfinite(rn[i, j]) else 0.0)
        img[i, n] = greens(pa[i] if np.isfinite(pa[i]) else 0.0)
    for j in range(n):
        img[n, j] = greens(ua[j] if np.isfinite(ua[j]) else 0.0)
    img[n, n] = greens(oa if np.isfinite(oa) else 0.0)

    fig, ax = plt.subplots(figsize=(0.8 * (n + 1) + 2, 0.8 * (n + 1) + 1.5))
    ax.imshow(img, aspect="auto")

    def txtcolor(v):
        return "white" if (np.isfinite(v) and v > 0.5) else "black"

    for i in range(n):
        for j in range(n):
            c = int(M[i, j])
            if c:
                ax.text(j, i, f"{c:,}", ha="center", va="center", fontsize=8, color=txtcolor(rn[i, j]))
    for i in range(n):                                     # PA column + reviewer A support
        t = f"{pa[i]*100:.{decimals}f}%" if np.isfinite(pa[i]) else "-"
        ax.text(n, i, f"{t}\nn={int(row[i]):,}", ha="center", va="center", fontsize=7.5,
                color=txtcolor(pa[i]))
    for j in range(n):                                     # UA row + reviewer B support
        t = f"{ua[j]*100:.{decimals}f}%" if np.isfinite(ua[j]) else "-"
        ax.text(j, n, f"{t}\nn={int(col[j]):,}", ha="center", va="center", fontsize=7.5,
                color=txtcolor(ua[j]))
    ax.text(n, n, f"OA {oa*100:.{decimals}f}%\nκ {kappa:.2f}", ha="center", va="center",
            fontsize=9, color=txtcolor(oa))

    ax.set_xticks(range(n + 1)); ax.set_xticklabels(labels + ["PA"], rotation=45, ha="left", fontsize=10)
    ax.set_yticks(range(n + 1)); ax.set_yticklabels(labels + ["UA"], fontsize=10)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Reviewer B (columns)", fontsize=12)
    ax.set_ylabel("Reviewer A (rows)", fontsize=12)
    ax.axhline(n - 0.5, color="0.4", lw=1.0); ax.axvline(n - 0.5, color="0.4", lw=1.0)
    ax.set_xticks(np.arange(-0.5, n + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n + 1, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=30)

    import textwrap
    cap = ("Inter-interpreter confusion pooled over all double-interpreted pairs. Cells are raw pixel "
           "counts colored by the row proportion. There is no ground-truth axis here, since both axes "
           "are interpreters, so PA is agreement given Reviewer A's label (with Reviewer A support n) "
           "and UA is agreement given Reviewer B's label (with Reviewer B support n); the corner gives "
           "overall agreement and Cohen's kappa.")
    wrapped = "\n".join(textwrap.wrap(cap, 108))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.03 + 0.028 * nlines, 1, 1])
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=9, color="0.3")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pairs")
    ap.add_argument("--no-figures", action="store_true", help="metrics only, skip figures")
    args = ap.parse_args()

    codes, names, colors = load_legend()
    pairs = find_pairs()
    print(f"multi-reviewer cells (pairs): {len(pairs)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    global_cm = np.zeros((len(codes), len(codes)), dtype=np.int64)
    rows = []
    for i, ((gid, samp, tgt), revs) in enumerate(sorted(pairs.items())):
        if args.limit is not None and i >= args.limit:
            break
        (revA, fA), (revB, fB) = revs[0], revs[1]
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            print(f"  skip grid {gid}: shape mismatch {a.shape} vs {b.shape}")
            continue
        cm, n_valid = confusion(a, b, codes)
        global_cm += cm
        m = metrics_from_cm(cm)
        rows.append(dict(grid=gid, sample=samp, target=tgt, revA=revA, revB=revB,
                         pair=f"{revA}-{revB}", n_valid=n_valid,
                         overall_agreement=round(m["overall_agreement"], 4),
                         macro_f1=round(m["macro_f1"], 4),
                         mean_iou=round(m["mean_iou"], 4),
                         kappa=round(m["kappa"], 4)))
        if not args.no_figures:
            plot_pair(gid, revA, revB, a, b, codes, names, colors, m,
                      os.path.join(OUT_DIR, f"{gid}_{revA}_vs_{revB}.png"))

    if not rows:
        raise SystemExit("no multi-reviewer pairs found.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_pair_metrics.csv"), index=False)

    # by reviewer pairing
    by = (df.groupby("pair")
            .agg(n=("grid", "count"),
                 overall_agreement=("overall_agreement", "mean"),
                 macro_f1=("macro_f1", "mean"),
                 mean_iou=("mean_iou", "mean"),
                 kappa=("kappa", "mean"))
            .round(3).sort_values("overall_agreement", ascending=False))
    by.to_csv(os.path.join(OUT_DIR, "by_reviewer_pair.csv"))

    # pooled confusion + metrics
    order = sorted(codes)
    cm_df = pd.DataFrame(global_cm, index=[names[c] for c in order], columns=[names[c] for c in order])
    cm_df.to_csv(os.path.join(OUT_DIR, "global_confusion_matrix.csv"))
    if not args.no_figures:
        plot_confusion(global_cm, codes, names, os.path.join(OUT_DIR, "global_confusion_matrix.png"))

    gm = metrics_from_cm(global_cm)
    with open(os.path.join(OUT_DIR, "global_metrics.txt"), "w") as fh:
        fh.write(f"multi-reviewer pairs: {len(rows)}\n")
        fh.write(f"pooled valid pixels: {gm['n_valid']:,}\n\n")
        fh.write(f"mean per-pair overall agreement: {df['overall_agreement'].mean():.4f}\n")
        fh.write(f"mean per-pair kappa:             {df['kappa'].mean():.4f}\n\n")
        fh.write(f"pooled overall agreement: {gm['overall_agreement']:.4f}\n")
        fh.write(f"pooled macro F1:          {gm['macro_f1']:.4f}\n")
        fh.write(f"pooled mean IoU:          {gm['mean_iou']:.4f}\n")
        fh.write(f"pooled kappa:             {gm['kappa']:.4f}\n\n")
        fh.write(f"{'class':<18}{'f1':>8}{'iou':>8}{'support(px)':>14}\n")
        for k, c in enumerate(order):
            if gm["support"][k] > 0:
                fh.write(f"{names[c]:<18}{gm['f1'][k]:>8.3f}{gm['iou'][k]:>8.3f}{int(gm['support'][k]):>14,}\n")

    print("\n" + "=" * 60)
    print(f"pairs scored: {len(rows)}")
    print(f"mean per-pair agreement: {df['overall_agreement'].mean():.3f}   "
          f"mean kappa: {df['kappa'].mean():.3f}")
    print(f"pooled agreement: {gm['overall_agreement']:.3f}  macroF1: {gm['macro_f1']:.3f}  "
          f"mIoU: {gm['mean_iou']:.3f}  kappa: {gm['kappa']:.3f}")
    print("\nby reviewer pairing:")
    print(by.to_string())
    print(f"\noutputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
