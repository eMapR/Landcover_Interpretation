#!/usr/bin/env python3
"""Window-based accuracy assessment, Approach C (independent per-field label per window),
adapted from Robert's binary window-sampling framework to our 10-class scheme.

Reference: disturbance_uncertainty/docs/window_sampling_methods.md (Approach C). Structure
mirrors his `window_sample_C`; nothing is imported from his repo.

Setup (identical to Approach B, scripts/window_sampling_approachB.py)
  reference = interpreted CKIT-RF cells (rf_class -> common 10-class); map = model v2-v6;
  de-duplicate one interpretation per location (grid+sample+target), seed 42, all target years;
  each cell tiled EXHAUSTIVELY with non-overlapping WxW windows (no RSA); W in {1,3,5,7,9};
  partial edge windows dropped and reported. Pluralities are computed over pixels valid in
  BOTH fields (both class in 1..10), so a window with no jointly-valid pixels contributes nothing.

Approach C
  Derive a label for each field INDEPENDENTLY, then compare. Robert thresholds each field at
  >50% (a binary majority). With 10 classes a window frequently has no majority class, so we use
  PLURALITY: the most frequent class in the map field and the most frequent class in the
  reference field, recorded as (plurality_map, plurality_ref) — one sample per window. Ties:
  lowest class code (consistent with Approach B).
  DEVIATION FROM SOURCE: majority -> plurality. To keep this visible, we also report, per
  version x W, the fraction of windows whose plurality class was an actual majority (>50%), for
  the map and reference fields; where most windows have no majority (large W), the plurality is a
  weak window summary and the metrics must be read with that in mind.

B vs C
  B records the single most frequent (map,ref) PAIR; C labels each field separately. A window
  that is map 40% Forest/35% Wetland and ref 35% Forest/40% Wetland can agree under B (Forest-
  Forest the most common pair) yet disagree under C (plurality Forest vs plurality Wetland). We
  report n_windows where B != C per version x W — the quantity that isolates windows where
  within-window heterogeneity drives the assessment.

Verification
  W=1 must be identical to B at W=1 and to the per-pixel `compare_interpreted_vs_model.py`
  confusion. All three are asserted equal before proceeding.

Outputs (reports/window_sampling_by_approach/Case_C_window_sampling/)
  - window_sampling_metrics.csv     version x W: OA, macro-F1, mean IoU, kappa, n_windows,
                                    frac_map_majority, frac_ref_majority, n_bne, frac_bne,
                                    windows_per_cell, edge-discard
  - window_sampling_confusion.csv   long-format confusion for C (version, W, ref, map, count)
  - window_sampling_metrics.png     metrics / majority-fraction / B!=C vs W

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

OUT = "reports/window_sampling_by_approach/Case_C_window_sampling"
B_METRICS = "reports/window_sampling_by_approach/Case_B_window_sampling/window_sampling_metrics.csv"
WS = [1, 3, 5, 7, 9]
N = 10
VERSIONS = ["v2", "v3", "v4", "v5", "v6"]
SEED = 42


def compute_windows(ref, mdl, valid, W, n=N):
    """Per-cell WxW tiling; returns B and C confusion contributions plus C's extras.

    Returns dict: cmB, cmC (rows=ref, cols=map), nwin, bne (B!=C), mapmaj, refmaj
    (windows whose map/ref plurality is a strict majority), edge (discarded px), nominal.
    """
    H, Wd = ref.shape
    nH, nW = H // W, Wd // W
    edge = H * Wd - (nH * W) * (nW * W)
    cmB = np.zeros((n, n), np.int64); cmC = np.zeros((n, n), np.int64)
    base = dict(cmB=cmB, cmC=cmC, nwin=0, bne=0, mapmaj=0, refmaj=0, edge=edge, nominal=0)
    if nH == 0 or nW == 0:
        return base
    H2, W2 = nH * W, nW * W
    r = ref[:H2, :W2].astype(np.int32); m = mdl[:H2, :W2].astype(np.int32); v = valid[:H2, :W2]

    if W == 1:                                  # per-pixel: B == C == pixel pair
        rv, mv = r[v] - 1, m[v] - 1
        np.add.at(cmB, (rv, mv), 1)
        cmC = cmB.copy()
        nwin = int(v.sum())
        return dict(cmB=cmB, cmC=cmC, nwin=nwin, bne=0, mapmaj=nwin, refmaj=nwin,
                    edge=edge, nominal=nH * nW)

    nvalid = v.reshape(nH, W, nW, W).sum((1, 3))
    # C — independent per-field plurality (tie: lowest class code via argmax-first)
    mc = np.empty((n, nH, nW), np.int32); rc = np.empty((n, nH, nW), np.int32)
    for k in range(n):
        mc[k] = ((m == k + 1) & v).reshape(nH, W, nW, W).sum((1, 3))
        rc[k] = ((r == k + 1) & v).reshape(nH, W, nW, W).sum((1, 3))
    pl_map = mc.argmax(0); pl_ref = rc.argmax(0)
    max_map = mc.max(0); max_ref = rc.max(0)
    # B — dominant joint pair (tie: lowest map then lowest ref via ascending pair index)
    pidx = np.where(v, (m - 1) * n + (r - 1), -1).reshape(nH, W, nW, W)
    jc = np.empty((n * n, nH, nW), np.int32)
    for p in range(n * n):
        jc[p] = (pidx == p).sum((1, 3))
    best = jc.argmax(0)
    map_B = best // n; ref_B = best % n

    win = nvalid > 0
    np.add.at(cmC, (pl_ref[win], pl_map[win]), 1)
    np.add.at(cmB, (ref_B[win], map_B[win]), 1)
    bne = int(((map_B != pl_map) | (ref_B != pl_ref))[win].sum())
    mapmaj = int((max_map[win] * 2 > nvalid[win]).sum())   # strict > 50%
    refmaj = int((max_ref[win] * 2 > nvalid[win]).sum())
    return dict(cmB=cmB, cmC=cmC, nwin=int(win.sum()), bne=bne, mapmaj=mapmaj, refmaj=refmaj,
                edge=edge, nominal=nH * nW)


def main():
    rf2common, names, colors = C.load_mappings()
    cells = sorted(glob.glob(os.path.join(C.RF_DIR, "**", SC.RF_GLOB), recursive=True))
    cells, ndup = C.dedupe_cells(cells, SEED)
    print(f"de-duplicated: {ndup} location(s); {len(cells)} cells (all target years, seed {SEED})")
    os.makedirs(OUT, exist_ok=True)

    ref_cache = {}
    for f in cells:
        with rasterio.open(f) as ds:
            ref_cache[f] = C.to_common_rf(ds.read(1), rf2common)

    b_ref = pd.read_csv(B_METRICS) if os.path.exists(B_METRICS) else None

    metric_rows, conf_rows = [], []
    for v in VERSIONS:
        tiles = [rasterio.open(t) for t in C.model_tiles(v)]
        acc = {W: dict(cmB=np.zeros((N, N), np.int64), cmC=np.zeros((N, N), np.int64),
                       nwin=0, bne=0, mapmaj=0, refmaj=0, edge=0, nominal=0) for W in WS}
        pixel_check = np.zeros((N, N), np.int64)
        for f in cells:
            ref = ref_cache[f]
            with rasterio.open(f) as ds:
                mdl = C.stitch_model_to_cell(ds, tiles)
            valid = (ref >= 1) & (ref <= N) & (mdl >= 1) & (mdl <= N)
            pixel_check += C.confusion(ref, mdl)[0]
            for W in WS:
                d = compute_windows(ref, mdl, valid, W)
                a = acc[W]
                a["cmB"] += d["cmB"]; a["cmC"] += d["cmC"]
                for kk in ("nwin", "bne", "mapmaj", "refmaj", "edge", "nominal"):
                    a[kk] += d[kk]
        for t in tiles:
            t.close()

        # W=1: C == B == per-pixel confusion
        ok = np.array_equal(acc[1]["cmC"], acc[1]["cmB"]) and np.array_equal(acc[1]["cmC"], pixel_check)
        print(f"[{v}] W=1 identical (C == B == per-pixel): {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"W=1 check FAILED for {v}; aborting.")

        total_px = sum(ref_cache[f].size for f in cells)
        for W in WS:
            a = acc[W]
            gm = C.metrics_from_cm(a["cmC"])
            nb = a["nwin"]
            metric_rows.append(dict(
                version=v, W=W,
                overall_accuracy=round(gm["overall_accuracy"], 4),
                macro_f1=round(gm["macro_f1"], 4),
                mean_iou=round(gm["mean_iou"], 4),
                kappa=round(gm["kappa"], 4),
                n_windows=nb,
                frac_map_majority=round(a["mapmaj"] / nb, 4) if nb else np.nan,
                frac_ref_majority=round(a["refmaj"] / nb, 4) if nb else np.nan,
                n_bne=a["bne"],
                frac_bne=round(a["bne"] / nb, 4) if nb else np.nan,
                windows_per_cell=round(a["nominal"] / len(cells), 1),
                edge_discarded_px=a["edge"],
                edge_discarded_frac=round(a["edge"] / total_px, 4)))
            cmC = a["cmC"]
            for i in range(N):
                for j in range(N):
                    if cmC[i, j]:
                        conf_rows.append(dict(version=v, W=W, ref_code=i + 1, ref_class=names[i + 1],
                                              map_code=j + 1, map_class=names[j + 1], count=int(cmC[i, j])))

        # cross-check: our internal cmB should match the committed Approach B metrics
        if b_ref is not None:
            for W in WS:
                oa_b = C.metrics_from_cm(acc[W]["cmB"])["overall_accuracy"]
                row = b_ref[(b_ref.version == v) & (b_ref.W == W)]
                if len(row) and abs(round(oa_b, 4) - float(row.overall_accuracy.iloc[0])) > 1e-4:
                    print(f"  WARN cmB mismatch vs Case_B for {v} W={W}: {oa_b:.4f} vs {row.overall_accuracy.iloc[0]}")

    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(os.path.join(OUT, "window_sampling_metrics.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(OUT, "window_sampling_confusion.csv"), index=False)
    plot(mdf, os.path.join(OUT, "window_sampling_metrics.png"))

    print("\nApproach C metrics by version and window size:")
    print(mdf[["version", "W", "overall_accuracy", "macro_f1", "kappa",
               "frac_map_majority", "frac_ref_majority", "n_bne", "frac_bne", "n_windows"]].to_string(index=False))
    print(f"\noutputs -> {OUT}/  (majority->plurality deviation; tie rule lowest class code)")


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
    pal = {"v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd", "v5": "#ff7f0e", "v6": "#d62728"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    def lineplot(ax, col, title):
        for v in VERSIONS:
            s = mdf[mdf.version == v]
            ax.plot(s.W, s[col], "o-", color=pal[v], label=v)
        ax.set_xlabel("window size W"); ax.set_ylabel(col); ax.set_title(title)
        ax.set_xticks(WS); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    lineplot(axes[0, 0], "overall_accuracy", "Overall accuracy (Approach C)")
    lineplot(axes[0, 1], "macro_f1", "Macro F1")
    lineplot(axes[0, 2], "kappa", "Cohen's kappa")
    # majority fractions: ref is version-independent
    ax = axes[1, 0]
    for v in VERSIONS:
        s = mdf[mdf.version == v]
        ax.plot(s.W, s.frac_map_majority, "o-", color=pal[v], label=f"{v} map")
    refm = mdf[mdf.version == VERSIONS[0]]
    ax.plot(refm.W, refm.frac_ref_majority, "s--", color="k", label="reference (all v)")
    ax.set_xlabel("window size W"); ax.set_ylabel("fraction with true majority (>50%)")
    ax.set_title("Plurality that is an actual majority"); ax.set_xticks(WS)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    lineplot(axes[1, 1], "frac_bne", "Fraction of windows where B ≠ C")
    ax = axes[1, 2]
    wc = mdf[mdf.version == VERSIONS[0]]
    ax.plot(wc.W, wc.windows_per_cell, "s-", color="k")
    ax.set_yscale("log"); ax.set_xticks(WS)
    ax.set_xlabel("window size W"); ax.set_ylabel("windows per cell (log)")
    ax.set_title("Effective sample size ~1/W²"); ax.grid(alpha=0.3)
    fig.suptitle("Approach C (independent per-field plurality per window): metrics, majority "
                 "share, and B≠C vs. window size", fontsize=12)
    _caption(fig, "The top row plots overall accuracy, macro F1, and Cohen's kappa against window size W, one line "
                  "per version, under Approach C in which each window is labeled by the independent plurality class of "
                  "the map field and of the reference field. The bottom left panel shows the fraction of windows whose "
                  "plurality was an actual strict majority above 50 percent for each version's map and for the shared "
                  "reference, falling as W grows so that pluralities become weak summaries. The bottom middle panel "
                  "gives the fraction of windows where Approach B and Approach C disagree, isolating within-window "
                  "heterogeneity, and the bottom right panel shows windows per cell falling as roughly one over W "
                  "squared.", top=0.96)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
