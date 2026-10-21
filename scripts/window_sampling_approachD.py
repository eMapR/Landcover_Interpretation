#!/usr/bin/env python3
"""Window-based sampling, Approach D (proportional agreement scatter), PER CLASS.

Reference: disturbance_uncertainty/docs/window_sampling_methods.md (Approach D). Robert's D
summarizes each window by two continuous proportions (fraction of map / interpreter pixels that
are 'disturbed') and scatters them against the 1:1 line. We do this PER CLASS (not collapsed to
binary): for each class c and window, prop_map = fraction of the window's (jointly-valid) pixels
that are class c in the map, prop_ref = same in the reference. No confusion matrix / no OA — a
continuous view of whether each variant carries the right class abundance in the right area.

Setup (identical to Approach B/C)
  reference = interpreted cells (rf_class -> common 10-class); map = model v2-v6; dedup one
  interpretation per location (seed 42, all target years); exhaustive non-overlapping WxW tiling;
  proportions over pixels valid in BOTH fields; windows with no jointly-valid pixels dropped.
  W in {3,5,7,9} (W=1 is degenerate: proportions are only 0/1).

Zero-inflation
  For each class, windows where BOTH proportions are 0 (the class is absent from map and
  reference in that window) are DROPPED and counted — rare classes are heavily zero-inflated.
  Each panel is annotated with the retained window count.

Reading the result
  Per class per variant, how the scatter tightens toward 1:1 as W grows is the quantity of
  interest: a variant that scatters at W=3 but tracks 1:1 at W=9 has the right class abundance
  in the right general area but misplaces pixels locally. Quantified by RMSE around the 1:1 line
  (also MAE, bias, Pearson r), reported per version x W x class.

Outputs (reports/window_sampling_by_approach/Case_D_window_sampling/)
  - window_sampling_metrics.csv   version x W x class: n_retained, n_dropped, frac_dropped,
                                  rmse, mae, bias, corr
  - prop_scatter_<version>.png    10 classes (rows) x W (cols) proportional-agreement densities
  - tightness_vs_W.png            RMSE-to-1:1 vs W, per class, lines per version

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

OUT = "reports/window_sampling_by_approach/Case_D_window_sampling"
WS = [3, 5, 7, 9]
N = 10
VERSIONS = ["v2", "v3", "v4", "v5", "v6"]
SEED = 42
BINS = np.linspace(0.0, 1.0, 51)          # 50x50 density grid


def new_acc():
    return dict(n=0, ndrop=0, Sx=0.0, Sy=0.0, Sxx=0.0, Syy=0.0, Sxy=0.0,
                Sabs=0.0, Sdd=0.0, H=np.zeros((50, 50), np.int64))


def per_class_props(ref, mdl, valid, W, n=N):
    """Yield (class_index, prop_map array, prop_ref array, both_zero mask) per class,
    over windows with >=1 jointly-valid pixel, for one cell."""
    H, Wd = ref.shape
    nH, nW = H // W, Wd // W
    if nH == 0 or nW == 0:
        return
    H2, W2 = nH * W, nW * W
    r = ref[:H2, :W2].astype(np.int32); m = mdl[:H2, :W2].astype(np.int32); v = valid[:H2, :W2]
    nvalid = v.reshape(nH, W, nW, W).sum((1, 3)).astype(np.float64)
    win = nvalid > 0
    nv = nvalid[win]
    for c in range(n):
        mcc = ((m == c + 1) & v).reshape(nH, W, nW, W).sum((1, 3))[win]
        rcc = ((r == c + 1) & v).reshape(nH, W, nW, W).sum((1, 3))[win]
        yield c, mcc / nv, rcc / nv, (mcc == 0) & (rcc == 0)


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

    # acc[version][W][class]
    acc = {v: {W: {c: new_acc() for c in range(N)} for W in WS} for v in VERSIONS}
    for v in VERSIONS:
        tiles = [rasterio.open(t) for t in C.model_tiles(v)]
        for f in cells:
            ref = ref_cache[f]
            with rasterio.open(f) as ds:
                mdl = C.stitch_model_to_cell(ds, tiles)
            valid = (ref >= 1) & (ref <= N) & (mdl >= 1) & (mdl <= N)
            for W in WS:
                for c, x, y, both0 in per_class_props(ref, mdl, valid, W):
                    a = acc[v][W][c]
                    a["ndrop"] += int(both0.sum())
                    keep = ~both0
                    xk, yk = x[keep], y[keep]
                    if xk.size:
                        a["n"] += xk.size
                        a["Sx"] += xk.sum(); a["Sy"] += yk.sum()
                        a["Sxx"] += (xk * xk).sum(); a["Syy"] += (yk * yk).sum()
                        a["Sxy"] += (xk * yk).sum()
                        a["Sabs"] += np.abs(xk - yk).sum()
                        a["Sdd"] += ((xk - yk) ** 2).sum()
                        a["H"] += np.histogram2d(xk, yk, bins=[BINS, BINS])[0].astype(np.int64)
        for t in tiles:
            t.close()
        print(f"  {v} done", flush=True)

    # metrics
    rows = []
    for v in VERSIONS:
        for W in WS:
            for c in range(N):
                a = acc[v][W][c]; n = a["n"]
                if n == 0:
                    rows.append(dict(version=v, W=W, cls=names[c + 1], n_retained=0,
                                     n_dropped=a["ndrop"], frac_dropped=1.0,
                                     rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan))
                    continue
                sx, sy = a["Sx"] / n, a["Sy"] / n
                var_x = a["Sxx"] / n - sx * sx; var_y = a["Syy"] / n - sy * sy
                cov = a["Sxy"] / n - sx * sy
                corr = cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan
                rows.append(dict(version=v, W=W, cls=names[c + 1], n_retained=n,
                                 n_dropped=a["ndrop"],
                                 frac_dropped=round(a["ndrop"] / (n + a["ndrop"]), 4),
                                 rmse=round(np.sqrt(a["Sdd"] / n), 4),
                                 mae=round(a["Sabs"] / n, 4),
                                 bias=round(sx - sy, 4),
                                 corr=round(corr, 4)))
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(OUT, "window_sampling_metrics.csv"), index=False)

    for v in VERSIONS:
        scatter_grid(acc[v], mdf[mdf.version == v], names, os.path.join(OUT, f"prop_scatter_{v}.png"), v)
    tightness_plot(mdf, names, os.path.join(OUT, "tightness_vs_W.png"))

    print("\nApproach D — RMSE-to-1:1 (lower = tighter), by class, at W=3 and W=9:")
    piv3 = mdf[mdf.W == 3].pivot(index="cls", columns="version", values="rmse")
    piv9 = mdf[mdf.W == 9].pivot(index="cls", columns="version", values="rmse")
    print("W=3:\n", piv3.round(3).to_string())
    print("W=9:\n", piv9.round(3).to_string())
    print(f"\noutputs -> {OUT}/  (no confusion matrix / no OA by design)")


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def scatter_grid(acc_v, mdf_v, names, path, version):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    order = list(range(N))
    fig, axes = plt.subplots(N, len(WS), figsize=(2.6 * len(WS), 2.4 * N))
    for ci in order:
        for wj, W in enumerate(WS):
            ax = axes[ci, wj]
            a = acc_v[W][ci]
            Hm = a["H"].T                      # transpose so x=prop_map, y=prop_ref
            if Hm.sum() > 0:
                ax.imshow(np.where(Hm > 0, Hm, np.nan), origin="lower", extent=[0, 1, 0, 1],
                          aspect="auto", cmap="viridis", norm=LogNorm(vmin=1, vmax=max(Hm.max(), 2)))
            ax.plot([0, 1], [0, 1], "r-", lw=1)
            row = mdf_v[(mdf_v.W == W) & (mdf_v.cls == names[ci + 1])]
            n_ret = int(row.n_retained.iloc[0]) if len(row) else 0
            rmse = row.rmse.iloc[0] if len(row) else np.nan
            ax.text(0.03, 0.97, f"n={n_ret:,}\nrmse={rmse:.2f}" if n_ret else "n=0",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            if ci == 0:
                ax.set_title(f"W={W}", fontsize=9)
            if wj == 0:
                ax.set_ylabel(f"{names[ci+1]}\nprop_ref", fontsize=7)
            if ci == N - 1:
                ax.set_xlabel("prop_map", fontsize=7)
    fig.suptitle(f"Approach D per class — model {version}: prop_map vs prop_ref density "
                 f"(red = 1:1); both-zero windows dropped", fontsize=12)
    _caption(fig, "For one model version, this grid places the ten land cover classes down the rows and window sizes "
                  "W of 3, 5, 7, and 9 across the columns. Each cell is a log-scaled density of windows in the "
                  "proportion plane, where the horizontal axis is the fraction of a window's jointly valid pixels that "
                  "are the class in the map and the vertical axis is the same fraction in the reference, with the red "
                  "1:1 line marking perfect agreement and both-zero windows dropped. Panel annotations give the "
                  "retained window count and the RMSE to the 1:1 line, so density collapsing onto the red line as W "
                  "grows means the version carries the right class abundance in the right area.", top=0.985)
    fig.savefig(path, dpi=115, bbox_inches="tight"); plt.close(fig)


def tightness_plot(mdf, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = {"v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd", "v5": "#ff7f0e", "v6": "#d62728"}
    classes = [names[c + 1] for c in range(N)]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, cls in zip(axes.ravel(), classes):
        for v in VERSIONS:
            s = mdf[(mdf.version == v) & (mdf.cls == cls)].sort_values("W")
            ax.plot(s.W, s.rmse, "o-", color=pal[v], label=v)
        ax.set_title(cls, fontsize=10); ax.set_xlabel("W"); ax.set_ylabel("RMSE to 1:1")
        ax.set_xticks(WS); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Approach D: scatter tightness (RMSE to 1:1) vs window size, per class per variant\n"
                 "(falling = tightens with W: right abundance, locally misplaced pixels)", fontsize=12)
    _caption(fig, "Each of the ten panels is one land cover class and plots the RMSE of the window proportions to the "
                  "1:1 line against window size W, with one colored line per model version v2 through v6. Lower RMSE "
                  "means the map and reference class proportions agree more tightly within windows. A line that falls "
                  "as W increases marks a version that has the right class abundance over a general area but "
                  "misplaces pixels locally, so aggregating into larger windows brings the proportions into "
                  "agreement.", top=0.94)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
