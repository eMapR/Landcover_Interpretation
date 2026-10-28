#!/usr/bin/env python3
"""Spatial-tolerance diagnostic for inter-interpreter disagreement.

For each of the 69 pairs, per class, recompute agreement under RELAXED matching: a pixel
where reviewer A said class C counts as matched if C appears anywhere in the k x k
neighborhood of the same location in reviewer B's map. This is directional (dilate B for
A->B, dilate A for B->A) and asymmetric — it is NOT a confusion matrix and yields NO
overall accuracy. It is only a diagnostic of where disagreement is edge-driven.

Quantity of interest = per-class DELTA (relaxed - strict), i.e. how much agreement a
one-pixel tolerance recovers. A class that recovers a lot is dominated by boundary
misregistration; a class that barely moves is conceptual disagreement.

Chance floor / null: relaxed matching inflates agreement in heterogeneous areas, worse for
rare classes. So B's dilated masks are translated 3-5 px (decorrelating location while
preserving local class composition) and the relaxed match re-run; the null delta is the
recovery expected from window heterogeneity alone. We report DELTA ABOVE NULL.

Also run k=5 (5x5): a class whose recovery keeps climbing from 3x3 to 5x5 is misregistered
by more than a pixel — a different problem than sub-pixel boundary jitter.

Uncertainty: cluster (pair) bootstrap, seed 42. All randomness uses default_rng(42).

Classes whose pooled denominator is below a per-sensor minimum (MIN_DENOM_PX: 5000 px at 10 m,
500 px at 30 m) are kept in the CSV but left out of the figure and the console table.

Outputs (reports/interpreter_agreement/):
  - spatial_tolerance_delta.csv     per direction x window x class: strict, relaxed,
                                    delta, null_delta, delta_net (+95% CI), denom
  - spatial_tolerance_delta.png     per-class delta-above-null, 3x3 vs 5x5, both directions

NB: no relaxed overall accuracy is produced by design — this is an edge-driven-disagreement
diagnostic, not a corrected accuracy.

Requires: rasterio, numpy, pandas, scipy, matplotlib
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreters as CI
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = SC.out_path("reports/interpreter_agreement")
WINDOWS = [3, 5]
N_NULL = 4
# minimum pooled class denominator (px) to plot/print, chosen from the pixel area (m^2);
# classes strictly below are kept in the CSV but left out of the figure and console table
MIN_DENOM_PX = {
    100.0: 5000,   # 10 m (50 ha pooled)
    900.0: 500,    # 30 m, advisor 2026-10-28 (45 ha pooled)
}
rng_shift = np.random.default_rng(42)
rng_boot = np.random.default_rng(42)


def min_denom_px(pix_m2):
    """The MIN_DENOM_PX entry for a pixel area; ValueError for any pixel size not listed."""
    if pix_m2 not in MIN_DENOM_PX:
        raise ValueError(f"no minimum class denominator defined for pixel area {pix_m2:g} m^2 "
                         f"(known: {', '.join(f'{a:g}' for a in MIN_DENOM_PX)})")
    return MIN_DENOM_PX[pix_m2]


def rand_shift():
    mag = lambda: int(rng_shift.integers(3, 6))
    sgn = lambda: 1 if rng_shift.random() < 0.5 else -1
    return (mag() * sgn(), mag() * sgn())


SHIFTS = [rand_shift() for _ in range(N_NULL)]


def pair_counts(a, b, order):
    """Per-class counts for one pair, both directions and both windows (+ null).

    Returns nested dict of length-C arrays. Uses that dilation commutes with translation:
    dilate B once per class/window, then shift the dilated mask for the null.
    """
    C = len(order)
    out = {("AtoB", "den"): np.zeros(C), ("BtoA", "den"): np.zeros(C),
           ("AtoB", "strict"): np.zeros(C), ("BtoA", "strict"): np.zeros(C),
           ("AtoB", "nstrict"): np.zeros(C), ("BtoA", "nstrict"): np.zeros(C)}
    for w in WINDOWS:
        out[("AtoB", "relax", w)] = np.zeros(C)
        out[("BtoA", "relax", w)] = np.zeros(C)
        out[("AtoB", "nrelax", w)] = np.zeros(C)
        out[("BtoA", "nrelax", w)] = np.zeros(C)

    for k, code in enumerate(order):
        Ac = (a == code); Bc = (b == code)
        nA = int(Ac.sum()); nB = int(Bc.sum())
        out[("AtoB", "den")][k] = nA
        out[("BtoA", "den")][k] = nB
        if nA == 0 and nB == 0:
            continue
        out[("AtoB", "strict")][k] = int((Ac & Bc).sum())
        out[("BtoA", "strict")][k] = out[("AtoB", "strict")][k]     # A=C & B=C is symmetric
        # null strict: A=C & shifted(B=C)  and  B=C & shifted(A=C)
        ns_ab = np.mean([int((Ac & np.roll(Bc, s, (0, 1))).sum()) for s in SHIFTS]) if nA else 0
        ns_ba = np.mean([int((Bc & np.roll(Ac, s, (0, 1))).sum()) for s in SHIFTS]) if nB else 0
        out[("AtoB", "nstrict")][k] = ns_ab
        out[("BtoA", "nstrict")][k] = ns_ba
        for w in WINDOWS:
            dilB = ndimage.maximum_filter(Bc, size=w)
            dilA = ndimage.maximum_filter(Ac, size=w)
            out[("AtoB", "relax", w)][k] = int((Ac & dilB).sum())
            out[("BtoA", "relax", w)][k] = int((Bc & dilA).sum())
            out[("AtoB", "nrelax", w)][k] = np.mean([int((Ac & np.roll(dilB, s, (0, 1))).sum())
                                                     for s in SHIFTS])
            out[("BtoA", "nrelax", w)][k] = np.mean([int((Bc & np.roll(dilA, s, (0, 1))).sum())
                                                     for s in SHIFTS])
    return out


def stack_pairs(order):
    """Return (per-pair counts, pixel area in m^2 shared by every raster used)."""
    pairs = CI.find_pairs()
    recs, used = [], []
    for (gid, samp, tgt), revs in sorted(pairs.items()):
        (ra, fA), (rb, fB) = revs[0], revs[1]
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            continue
        # restrict to valid legend pixels
        valid = np.isin(a, order) & np.isin(b, order)
        a = np.where(valid, a, -1); b = np.where(valid, b, -1)
        recs.append(pair_counts(a, b, order))
        used += [fA, fB]
    return recs, SC.uniform_pixel_area_m2(used)


def pooled_delta_net(recs, idx, direction, window, k):
    """delta_net for one class index k, pooling the pairs in `idx`."""
    den = sum(recs[p][(direction, "den")][k] for p in idx)
    if den == 0:
        return np.nan
    strict = sum(recs[p][(direction, "strict")][k] for p in idx) / den
    relax = sum(recs[p][(direction, "relax", window)][k] for p in idx) / den
    nstrict = sum(recs[p][(direction, "nstrict")][k] for p in idx) / den
    nrelax = sum(recs[p][(direction, "nrelax", window)][k] for p in idx) / den
    return (relax - strict) - (nrelax - nstrict)


def main(boot=2000):
    os.makedirs(OUT, exist_ok=True)
    codes, names, _ = CI.load_legend()
    order = sorted(codes)
    recs, pix_m2 = stack_pairs(order)
    min_px = min_denom_px(pix_m2)
    n = len(recs)
    allidx = list(range(n))
    print(f"pairs: {n}   windows: {WINDOWS}   null shifts: {SHIFTS}")

    rows = []
    for di, direction in enumerate(["AtoB", "BtoA"]):
        for k, code in enumerate(order):
            den = sum(r[(direction, "den")][k] for r in recs)
            if den == 0:
                continue
            rec = dict(direction=direction, cls=names[code], code=code, denom_px=int(den))
            for w in WINDOWS:
                strict = sum(r[(direction, "strict")][k] for r in recs) / den
                relax = sum(r[(direction, "relax", w)][k] for r in recs) / den
                nstrict = sum(r[(direction, "nstrict")][k] for r in recs) / den
                nrelax = sum(r[(direction, "nrelax", w)][k] for r in recs) / den
                net = (relax - strict) - (nrelax - nstrict)
                b = np.array([pooled_delta_net(recs, rng_boot.integers(0, n, n), direction, w, k)
                              for _ in range(boot)])
                b = b[~np.isnan(b)]
                lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b.size else (np.nan, np.nan)
                rec[f"strict_{w}"] = round(strict, 4)
                rec[f"relaxed_{w}"] = round(relax, 4)
                rec[f"delta_{w}"] = round(relax - strict, 4)
                rec[f"null_delta_{w}"] = round(nrelax - nstrict, 4)
                rec[f"delta_net_{w}"] = round(net, 4)
                rec[f"net_lo_{w}"] = round(float(lo), 4)
                rec[f"net_hi_{w}"] = round(float(hi), 4)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "spatial_tolerance_delta.csv"), index=False)

    plot(df, os.path.join(OUT, "spatial_tolerance_delta.png"), min_px)

    print("\ndelta above null (A->B), classes with denom>=%d px:" % min_px)
    sub = df[(df.direction == "AtoB") & (df.denom_px >= min_px)].sort_values("delta_net_3", ascending=False)
    print(sub[["cls", "denom_px", "strict_3", "delta_net_3", "net_lo_3", "net_hi_3",
               "delta_net_5"]].to_string(index=False))
    print("\ninterpretation: high delta_net_3 = boundary-misregistration (edge-driven); "
          "~0 = conceptual disagreement; delta_net_5 >> delta_net_3 = misregistered > 1 px.")
    print(f"\noutputs -> {OUT}/spatial_tolerance_delta.csv/png  (no relaxed OA reported by design)")


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def plot(df, path, min_px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharey=True)   # stacked so each panel is wide and readable
    for ax, direction in zip(axes, ["AtoB", "BtoA"]):
        sub = df[(df.direction == direction) & (df.denom_px >= min_px)].copy()
        sub = sub.sort_values("delta_net_3", ascending=False)
        x = np.arange(len(sub)); wbar = 0.4
        err3 = [sub.delta_net_3 - sub.net_lo_3, sub.net_hi_3 - sub.delta_net_3]
        err5 = [sub.delta_net_5 - sub.net_lo_5, sub.net_hi_5 - sub.delta_net_5]
        ax.bar(x - wbar / 2, sub.delta_net_3, wbar, yerr=err3, capsize=2, color="#1f77b4", label="3x3")
        ax.bar(x + wbar / 2, sub.delta_net_5, wbar, yerr=err5, capsize=2, color="#ff7f0e", label="5x5")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(sub.cls, rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(f"{direction[0]} → {direction[-1]}  (dilate {'B' if direction=='AtoB' else 'A'})",
                     fontsize=12)
        ax.set_ylabel("net agreement recovered above null", fontsize=12)
        ax.legend(frameon=False, fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(False)
    # clean single-line title only; descriptive caption lives in chapter3_captions.md
    fig.suptitle("Agreement Recovery Under Spatial Tolerance", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
