#!/usr/bin/env python3
"""Window-based accuracy assessment, Approach B (dominant pixel-pair per window),
adapted from Robert's binary window-sampling framework to our 10-class scheme.

Reference: disturbance_uncertainty/docs/window_sampling_methods.md (Approach B). His
landscapes are simulated and binary; ours are real interpreted maps vs. model maps with
10 classes. Structure mirrors his `window_sample_B` + `binary_confusion_from_samples`;
nothing is imported from his repo.

Setup
  - reference = interpreted CKIT-RF cells (rf_class), mapped to the common 10-class scheme
  - map field = the AlphaEarth model maps v2-v6 (already 1..10; 0 = background)
  - de-duplicate to one interpretation per location (grid+sample+target), random, seed 42,
    same location key as compare_interpreted_vs_model.py; all target years.

Window placement (NOT random sequential adsorption)
  Each cell is a complete 337x337 enumeration of a defined frame, so we tile it EXHAUSTIVELY
  with non-overlapping WxW windows. This gives inclusion probabilities known by construction
  (RSA does not). Partial windows at the right/bottom edge are dropped; discarded pixels are
  reported. W in {1,3,5,7,9}. W=1 recovers the per-pixel case and must reproduce the
  compare_interpreted_vs_model.py confusion exactly (asserted before proceeding).

Approach B
  For each window, tally all W^2 (map_class, ref_class) pairs over the 10x10 valid
  combinations and record the single most frequent cell as the window's one contribution to
  the confusion matrix. Pixels where either map or reference is invalid (background / unmapped)
  are not tallied; a window with no valid pixels contributes nothing.
  Ties: lowest map class code, then lowest reference class code. (Robert's binary tie rule —
  prefer (1,1),(0,0),(1,0),(0,1) — does not generalize to 10 classes; this ordering is the
  documented substitution.)

Outputs (reports/window_sampling_by_approach/Case_B_window_sampling/)
  - window_sampling_metrics.csv     version x W: OA, macro-F1, mean IoU, kappa, n_windows,
                                    windows_per_cell, edge_discarded_px/frac
  - window_sampling_confusion.csv   long-format confusion (version, W, ref, map, count)
  - window_sampling_metrics.png     metrics vs W per version, with windows/cell

NB: metrics are a diagnostic of how within-window aggregation changes the assessment; the
per-window design keeps inclusion probabilities exact but the effective sample size falls
as ~1/W^2 (reported as windows-per-cell).

Requires: rasterio, numpy, pandas, matplotlib
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreted_vs_model as C
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = "reports/window_sampling_by_approach/Case_B_window_sampling"
WS = [1, 3, 5, 7, 9]
N = 10
VERSIONS = ["v2", "v3", "v4", "v5", "v6"]
SEED = 42


def approachB_cm(ref, mdl, valid, W, n=N):
    """Dominant-pair confusion contribution for exhaustive WxW tiling of one cell.

    Returns (cm[ref, map], n_contributing_windows, edge_discarded_px, nominal_windows).
    """
    H, Wd = ref.shape
    nH, nW = H // W, Wd // W
    edge_disc = H * Wd - (nH * W) * (nW * W)
    cm = np.zeros((n, n), dtype=np.int64)
    if nH == 0 or nW == 0:
        return cm, 0, edge_disc, 0
    H2, W2 = nH * W, nW * W
    r = ref[:H2, :W2]; m = mdl[:H2, :W2]; v = valid[:H2, :W2]

    if W == 1:                                   # per-pixel special case
        rv, mv = r[v], m[v]
        np.add.at(cm, (rv - 1, mv - 1), 1)
        return cm, int(v.sum()), edge_disc, nH * nW

    # pair index p = (map-1)*n + (ref-1); ascending p == map asc then ref asc == tie rule
    mi = m.astype(np.int32); ri = r.astype(np.int32)
    pidx = np.where(v, (mi - 1) * n + (ri - 1), -1).reshape(nH, W, nW, W)
    counts = np.empty((n * n, nH, nW), dtype=np.int32)
    for p in range(n * n):
        counts[p] = (pidx == p).sum(axis=(1, 3))
    maxc = counts.max(axis=0)
    best = counts.argmax(axis=0)                 # first max -> lowest p -> tie rule
    win = maxc > 0
    bp = best[win]
    np.add.at(cm, (bp % n, bp // n), 1)          # ref = bp%n, map = bp//n
    return cm, int(win.sum()), edge_disc, nH * nW


def main():
    rf2common, names, colors = C.load_mappings()
    cells = sorted(glob.glob(os.path.join(C.RF_DIR, "**", SC.RF_GLOB), recursive=True))
    cells, ndup = C.dedupe_cells(cells, SEED)
    print(f"de-duplicated: {ndup} location(s); {len(cells)} cells (all target years, seed {SEED})")

    # cache interpreted (reference) common-class arrays once
    ref_cache = {}
    for f in cells:
        with rasterio.open(f) as ds:
            ref_cache[f] = C.to_common_rf(ds.read(1), rf2common)

    metric_rows, conf_rows = [], []
    for v in VERSIONS:
        tiles = [rasterio.open(t) for t in C.model_tiles(v)]
        cmW = {W: np.zeros((N, N), dtype=np.int64) for W in WS}
        nwin = {W: 0 for W in WS}
        disc = {W: 0 for W in WS}
        nominal = {W: 0 for W in WS}
        pixel_check = np.zeros((N, N), dtype=np.int64)
        for f in cells:
            ref = ref_cache[f]
            with rasterio.open(f) as ds:
                mdl = C.stitch_model_to_cell(ds, tiles)
            valid = (ref >= 1) & (ref <= N) & (mdl >= 1) & (mdl <= N)
            pcm, _ = C.confusion(ref, mdl)        # the exact function compare_* uses
            pixel_check += pcm
            for W in WS:
                cm, nw, dsc, nom = approachB_cm(ref, mdl, valid, W)
                cmW[W] += cm; nwin[W] += nw; disc[W] += dsc; nominal[W] += nom
        for t in tiles:
            t.close()

        # --- W=1 must reproduce the per-pixel confusion exactly ---
        ok = np.array_equal(cmW[1], pixel_check)
        print(f"[{v}] W=1 reproduces per-pixel confusion exactly: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"W=1 check FAILED for {v}; aborting.")

        total_px = sum(ref_cache[f].size for f in cells)
        for W in WS:
            gm = C.metrics_from_cm(cmW[W])
            metric_rows.append(dict(
                version=v, W=W,
                overall_accuracy=round(gm["overall_accuracy"], 4),
                macro_f1=round(gm["macro_f1"], 4),
                mean_iou=round(gm["mean_iou"], 4),
                kappa=round(gm["kappa"], 4),
                n_windows=nwin[W],
                windows_per_cell=round(nominal[W] / len(cells), 1),
                edge_discarded_px=disc[W],
                edge_discarded_frac=round(disc[W] / total_px, 4)))
            for i in range(N):
                for j in range(N):
                    if cmW[W][i, j]:
                        conf_rows.append(dict(version=v, W=W,
                                              ref_code=i + 1, ref_class=names[i + 1],
                                              map_code=j + 1, map_class=names[j + 1],
                                              count=int(cmW[W][i, j])))

    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(os.path.join(OUT, "window_sampling_metrics.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(OUT, "window_sampling_confusion.csv"), index=False)
    plot(mdf, os.path.join(OUT, "window_sampling_metrics.png"))

    print("\nApproach B metrics by version and window size:")
    print(mdf[["version", "W", "overall_accuracy", "macro_f1", "kappa",
               "windows_per_cell", "n_windows", "edge_discarded_frac"]].to_string(index=False))
    print("\nedge-discard (partial windows) by W (same for all versions):")
    ed = mdf[mdf.version == VERSIONS[0]][["W", "windows_per_cell", "edge_discarded_px", "edge_discarded_frac"]]
    print(ed.to_string(index=False))
    print(f"\noutputs -> {OUT}/window_sampling_*  (tie rule: lowest map code, then lowest ref code)")


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def plot(mdf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = {"v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd", "v5": "#ff7f0e", "v6": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in [(axes[0, 0], "overall_accuracy", "Overall accuracy"),
                              (axes[0, 1], "macro_f1", "Macro F1"),
                              (axes[1, 0], "kappa", "Cohen's kappa")]:
        for v in VERSIONS:
            s = mdf[mdf.version == v]
            ax.plot(s.W, s[metric], "o-", color=palette[v], label=v)
        ax.set_xlabel("window size W"); ax.set_ylabel(metric); ax.set_title(title)
        ax.set_xticks(WS); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    wc = mdf[mdf.version == VERSIONS[0]]
    ax.plot(wc.W, wc.windows_per_cell, "s-", color="k")
    ax.set_yscale("log"); ax.set_xticks(WS)
    ax.set_xlabel("window size W"); ax.set_ylabel("windows per cell (log)")
    ax.set_title("Effective sample size falls as ~1/W²")
    ax.grid(alpha=0.3)
    fig.suptitle("Approach B (dominant pixel-pair per window): metrics vs. window size\n"
                 "interpreted (reference) vs. model maps; exhaustive non-overlapping tiling", fontsize=12)
    _caption(fig, "The first three panels plot overall accuracy, macro F1, and Cohen's kappa against window size W, "
                  "with one colored line per model version v2 through v6, computed from the Approach B confusion "
                  "matrix in which each window contributes its single most frequent map and reference pixel pair. The "
                  "fourth panel shows windows per cell on a logarithmic axis, falling as roughly one over W squared "
                  "and identical across versions. Reading left to right shows how collapsing each window to its "
                  "dominant pair shifts the metrics as windows grow, while the effective sample size shrinks.", top=0.95)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
