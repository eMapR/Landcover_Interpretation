#!/usr/bin/env python3
"""Approach A from Robert's window-sampling framework as a DESIGN EXPERIMENT.

Approach A is pixel-level enumeration: every pixel in a placed window contributes one
(map, ref) pair to the confusion matrix. Under the exhaustive tiling used for B/C/D it
collapses to the full per-pixel comparison at any W, so it is only informative under
SAMPLING. Here we treat the window as the sampling unit and characterize the sampling
design's bias and precision against the census (we have every pixel, so the census OA /
macro-F1 / kappa are the known truth).

Setup: interpreted CKIT-RF cells = reference; model maps v2-v6 = map field; de-duplicated
one interpretation per location (grid+sample+target), default_rng seed 42, all target years.
The 154 cells are a simple random sample from the 21,561-cell grid_112_naip_brackets frame,
so the cell is the primary sampling unit.

Design: within each cell, place n windows of size WxW at random by random sequential
adsorption (uniform random top-left, reject if it overlaps a placed window: two WxW windows
overlap iff |dr|<W and |dc|<W; max_attempts = n*50), enumerate all W^2 pixels, and pool the
valid (map, ref) pairs across all cells into one confusion matrix -> one draw.

Sweep n in {1,2,5,10,25,50}, W in {1,3,5,7,9}, 200 independent draws at each (n, W). Per draw
record OA, macro-F1, kappa. Report the sampling distribution (mean, SD, 2.5/97.5 pct) against
the census value.

Quantities:
  Bias         mean(draws) - census (should be ~0; exact for W=1).
  Precision    SD of the sampling distribution vs n and W. At equal pixels sampled (n*W^2),
               W=1 should beat large W because within-window pixels are autocorrelated.
  Design effect  deff = Var(observed OA) / Var_binomial, where Var_binomial =
               p(1-p)/N_pixels uses the census OA p and the actual pooled pixel count N. deff
               is a VARIANCE ratio (its sqrt is the SD ratio); it says how much within-window
               autocorrelation inflates variance over independent pixels.
  Eff. sample size  N_pixels / deff  (effective independent pixels).

These are draws from a design whose properties we characterize -- NOT accuracy estimates.

Outputs (reports/window_sampling_by_approach/Case_A_window_sampling/)
  - approachA_design.csv     per version x n x W: census, mean/SD/CI of each metric, bias,
                             mean pixels, binomial SD, design effect, SD ratio, eff. n
  - sd_vs_cost.png           SD(OA) vs pixels sampled (n*W^2), one line per W, per version
  - design_effect_vs_W.png   design effect vs W, per version (+ effective sample size)
  - bias_vs_W.png            mean(OA) - census vs W, per version (unbiasedness check)

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreted_vs_model as C
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = "reports/window_sampling_by_approach/Case_A_window_sampling"
NS = [1, 2, 5, 10, 25, 50]
WS = [1, 3, 5, 7, 9]
N = 10
VERSIONS = ["v2", "v3", "v4", "v5", "v6"]
SEED = 42
DRAWS = 200


def place_windows(rng, H, Wd, W, n):
    """Random sequential adsorption: up to n non-overlapping WxW top-lefts."""
    rmax, cmax = H - W, Wd - W
    rs = np.empty(n, np.int32); cs = np.empty(n, np.int32)
    k = attempts = 0
    maxatt = n * 50
    while k < n and attempts < maxatt:
        r = rng.integers(0, rmax + 1); c = rng.integers(0, cmax + 1)
        if k == 0 or not np.any((np.abs(rs[:k] - r) < W) & (np.abs(cs[:k] - c) < W)):
            rs[k] = r; cs[k] = c; k += 1
        attempts += 1
    return rs[:k], cs[:k]


def gather(ref, mdl, valid, rs, cs, W):
    """Pooled 100-bin (ref,map) counts and valid-pixel count for the placed windows."""
    ar = np.arange(W)
    rows = rs[:, None, None] + ar[None, :, None]
    cols = cs[:, None, None] + ar[None, None, :]
    rw = ref[rows, cols].ravel(); mw = mdl[rows, cols].ravel(); vw = valid[rows, cols].ravel()
    rw = rw[vw]; mw = mw[vw]
    pairs = (rw.astype(np.int64) - 1) * N + (mw.astype(np.int64) - 1)   # cm[ref, map]
    return np.bincount(pairs, minlength=N * N), rw.size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--ns", nargs="+", type=int, default=NS)
    ap.add_argument("--ws", nargs="+", type=int, default=WS)
    args = ap.parse_args()

    rf2common, names, colors = C.load_mappings()
    cells = sorted(glob.glob(os.path.join(C.RF_DIR, "**", SC.RF_GLOB), recursive=True))
    cells, ndup = C.dedupe_cells(cells, SEED)
    print(f"de-duplicated: {ndup} location(s); {len(cells)} cells; draws={args.draws}")
    os.makedirs(OUT, exist_ok=True)

    ref_cache = {}
    for f in cells:
        with rasterio.open(f) as ds:
            ref_cache[f] = C.to_common_rf(ds.read(1), rf2common)

    rng = np.random.default_rng(SEED)
    rows = []
    for v in args.versions:
        tiles = [rasterio.open(t) for t in C.model_tiles(v)]
        mdl_cache, valid_cache, shape_cache = {}, {}, {}
        census_cm = np.zeros((N, N), np.int64)
        for f in cells:
            ref = ref_cache[f]
            with rasterio.open(f) as ds:
                mdl = C.stitch_model_to_cell(ds, tiles)
            valid = (ref >= 1) & (ref <= N) & (mdl >= 1) & (mdl <= N)
            mdl_cache[f], valid_cache[f], shape_cache[f] = mdl, valid, ref.shape
            census_cm += C.confusion(ref, mdl)[0]
        for t in tiles:
            t.close()
        cg = C.metrics_from_cm(census_cm)
        census = dict(oa=cg["overall_accuracy"], mf1=cg["macro_f1"], kappa=cg["kappa"])
        print(f"[{v}] census OA={census['oa']:.4f} macroF1={census['mf1']:.4f} kappa={census['kappa']:.4f}")

        for W in args.ws:
            for n in args.ns:
                oas = np.empty(args.draws); mf1s = np.empty(args.draws)
                kaps = np.empty(args.draws); npx = np.empty(args.draws)
                for d in range(args.draws):
                    flat = np.zeros(N * N, np.int64); tot = 0
                    for f in cells:
                        H, Wd = shape_cache[f]
                        rs, cs = place_windows(rng, H, Wd, W, n)
                        bc, cnt = gather(ref_cache[f], mdl_cache[f], valid_cache[f], rs, cs, W)
                        flat += bc; tot += cnt
                    m = C.metrics_from_cm(flat.reshape(N, N))
                    oas[d] = m["overall_accuracy"]; mf1s[d] = m["macro_f1"]
                    kaps[d] = m["kappa"]; npx[d] = tot
                meanN = float(npx.mean())
                sd_oa = float(oas.std(ddof=1))
                p = census["oa"]
                sd_binom = np.sqrt(p * (1 - p) / meanN) if 0 < p < 1 and meanN else np.nan
                deff = (sd_oa / sd_binom) ** 2 if sd_binom and sd_binom > 0 else np.nan
                rows.append(dict(
                    version=v, n=n, W=W, pixels_per_cell=n * W * W, mean_total_px=round(meanN, 1),
                    census_oa=round(p, 4), mean_oa=round(float(oas.mean()), 4),
                    sd_oa=round(sd_oa, 5), oa_lo=round(float(np.percentile(oas, 2.5)), 4),
                    oa_hi=round(float(np.percentile(oas, 97.5)), 4),
                    bias_oa=round(float(oas.mean()) - p, 5),
                    census_mf1=round(census["mf1"], 4), mean_mf1=round(float(mf1s.mean()), 4),
                    sd_mf1=round(float(mf1s.std(ddof=1)), 5),
                    census_kappa=round(census["kappa"], 4), mean_kappa=round(float(kaps.mean()), 4),
                    sd_kappa=round(float(kaps.std(ddof=1)), 5),
                    sd_binom_oa=round(float(sd_binom), 6), design_effect=round(float(deff), 3),
                    sd_ratio=round(float(np.sqrt(deff)), 3) if deff == deff else np.nan,
                    eff_sample_size=round(meanN / deff, 1) if deff and deff == deff else np.nan))
            print(f"  {v} W={W} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "approachA_design.csv"), index=False)
    make_plots(df, args.versions, os.path.join(OUT))

    print("\ndesign effect by version and W (averaged over n):")
    print(df.groupby(["version", "W"]).design_effect.mean().round(2).unstack().to_string())
    print("\nmax |bias| in OA across all cells:", df.bias_oa.abs().max())
    print(f"\noutputs -> {OUT}/  (sampling draws, NOT accuracy estimates)")


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def make_plots(df, versions, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wpal = {1: "#1f77b4", 3: "#2ca02c", 5: "#9467bd", 7: "#ff7f0e", 9: "#d62728"}
    vpal = {"v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd", "v5": "#ff7f0e", "v6": "#d62728"}

    # 1) SD(OA) vs pixels sampled per cell (n*W^2), line per W, panel per version
    fig, axes = plt.subplots(1, len(versions), figsize=(3.4 * len(versions), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, versions):
        s = df[df.version == v]
        for W in sorted(s.W.unique()):
            sw = s[s.W == W].sort_values("pixels_per_cell")
            ax.plot(sw.pixels_per_cell, sw.sd_oa, "o-", color=wpal[W], label=f"W={W}", ms=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("pixels sampled per cell (n·W²)"); ax.set_title(v)
        if ax is axes[0]:
            ax.set_ylabel("SD of OA across 200 draws")
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Precision at equal cost: SD of sampled OA vs pixels interpreted per cell\n"
                 "(at equal x, lower = more information per pixel — does W=1 dominate?)", fontsize=12)
    _caption(fig, "Each panel is one model version and plots the standard deviation of overall accuracy across "
                  "200 sampling draws against the number of pixels interpreted per cell, n times W squared, with one "
                  "line per window size W. Both axes are logarithmic, so at any fixed x position a lower line means "
                  "more information recovered per interpreted pixel. The W equal to 1 line sitting below the larger "
                  "windows indicates that single pixels beat wide windows at equal cost, since pixels within a window "
                  "are spatially autocorrelated.", top=0.9)
    fig.savefig(os.path.join(outdir, "sd_vs_cost.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # 2) design effect vs W (mean over n) + effective sample size
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g = df.groupby(["version", "W"]).design_effect.mean().reset_index()
    for v in versions:
        s = g[g.version == v]
        axes[0].plot(s.W, s.design_effect, "o-", color=vpal[v], label=v)
    axes[0].axhline(1, ls="--", color="k", lw=0.8)
    axes[0].set_xlabel("window size W"); axes[0].set_ylabel("design effect  Var_obs / Var_binomial")
    axes[0].set_title("Cost of autocorrelation: design effect vs W"); axes[0].set_xticks(WS)
    axes[0].legend(fontsize=8, frameon=False); axes[0].grid(alpha=0.3)
    # eff sample size vs nominal pixels, line per W, one representative version comparison
    for v in versions:
        s = df[df.version == v].groupby("W").agg(eff=("eff_sample_size", "mean"),
                                                 nom=("mean_total_px", "mean")).reset_index()
        axes[1].plot(s.W, s.eff / s.nom, "o-", color=vpal[v], label=v)
    axes[1].axhline(1, ls="--", color="k", lw=0.8)
    axes[1].set_xlabel("window size W"); axes[1].set_ylabel("effective / nominal pixels (1 / deff)")
    axes[1].set_title("Information retained per pixel interpreted"); axes[1].set_xticks(WS)
    axes[1].legend(fontsize=8, frameon=False); axes[1].grid(alpha=0.3)
    fig.suptitle("Design effect: how much within-window autocorrelation costs, by W and version",
                 fontsize=12)
    _caption(fig, "The left panel plots the design effect, the ratio of observed sampling variance to the binomial "
                  "variance expected from independent pixels, against window size W with one line per version, "
                  "averaged over n. The right panel plots the reciprocal of the design effect, the effective pixel "
                  "count divided by the nominal pixel count, which is the fraction of information retained per "
                  "interpreted pixel. The dashed reference line marks a design effect of 1, so lines rising above it "
                  "on the left, and falling below it on the right, show that wider windows and more autocorrelated "
                  "versions waste sampling effort.", top=0.94)
    fig.savefig(os.path.join(outdir, "design_effect_vs_W.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # 3) bias vs W (unbiasedness check)
    fig, ax = plt.subplots(figsize=(8, 5))
    gb = df.groupby(["version", "W"]).bias_oa.mean().reset_index()
    for v in versions:
        s = gb[gb.version == v]
        ax.plot(s.W, s.bias_oa, "o-", color=vpal[v], label=v)
    ax.axhline(0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("window size W"); ax.set_ylabel("mean sampled OA − census OA")
    ax.set_title("Unbiasedness: the sampling design recovers the census (bias ≈ 0 at all W)")
    ax.set_xticks(WS); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    _caption(fig, "This plots the bias in overall accuracy, the mean sampled overall accuracy across 200 draws minus "
                  "the known census overall accuracy, against window size W with one line per version. The dashed line "
                  "marks zero bias, the ideal. Lines hovering near zero at every W confirm that the window sampling "
                  "design recovers the census value without systematic error, so the precision costs of wide windows "
                  "seen in the other figures do not come with any bias penalty.")
    fig.savefig(os.path.join(outdir, "bias_vs_W.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
