#!/usr/bin/env python3
"""Directional asymmetry of inter-interpreter disagreement — does a reviewer
systematically over-assign a class relative to whoever they were compared with?

Across all 69 double-interpreted pairs, for each pair the full directed confusion matrix
(rows = reviewer A, cols = reviewer B; alphabetical A/B) is built in PIXELS (area, not
patch count). Pooling per reviewer R and class C:

  claim_R[C]      = pixels where R said C and the partner said something else
  claim_partner[C]= pixels where the partner said C and R said something else

  over-assignment index = log2( (claim_R + 1) / (claim_partner + 1) )
      > 0  -> R over-assigns C (claims it where partners do not)
      < 0  -> R under-assigns C

Uncertainty is a cluster (pair) bootstrap — the resampling unit is the pair, seed 42.
A per-reviewer directed class-pair table shows WHICH boundary each leaning rides on
(e.g. R says X where the partner says Y).

Outputs (reports/interpreter_agreement/):
  - reviewer_class_overassignment.csv   reviewer x class: index, 95% CI, area, significance
  - reviewer_directed_classpairs.csv    per reviewer: strongest directed class-pair leanings
  - reviewer_overassignment_heatmap.png  reviewer x class log2 index (significant marked)

Requires: rasterio, numpy, pandas, matplotlib
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreters as CI
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = "reports/interpreter_agreement"
rng = np.random.default_rng(42)


def build_pair_cms():
    """Return (order, names, list of (revA, revB, cm), pix_ha) with cm[i,j]=px A=code_i,B=code_j.

    pix_ha is the pixel area shared by every raster used (pooled px -> ha needs one pixel size).
    """
    codes, names, _ = CI.load_legend()
    order = sorted(codes)
    pairs = CI.find_pairs()
    out, used = [], []
    for (gid, samp, tgt), revs in sorted(pairs.items()):
        (ra, fA), (rb, fB) = revs[0], revs[1]
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            continue
        cm, _ = CI.confusion(a, b, order)
        out.append((ra, rb, cm))
        used += [fA, fB]
    return order, names, out, SC.uniform_pixel_area_m2(used) / SC.M2_PER_HA


def pool_claims(cms, reviewers, C):
    """Return claim_self, claim_partner, dir_count per reviewer for a set of (A,B,cm)."""
    self_ = {r: np.zeros(C) for r in reviewers}
    other = {r: np.zeros(C) for r in reviewers}
    dirc = {r: np.zeros((C, C)) for r in reviewers}   # dirc[R][x,y]=px R=x, partner=y
    for A, B, cm in cms:
        row, col, diag = cm.sum(1), cm.sum(0), np.diag(cm)
        self_[A] += row - diag;  other[A] += col - diag
        self_[B] += col - diag;  other[B] += row - diag
        dirc[A] += cm
        dirc[B] += cm.T
    for r in reviewers:
        np.fill_diagonal(dirc[r], 0)
    return self_, other, dirc


def main(boot=2000):
    order, names, cms, pix_ha = build_pair_cms()
    C = len(order)
    reviewers = sorted({r for A, B, _ in cms for r in (A, B)})
    n_pairs = {r: sum(1 for A, B, _ in cms if r in (A, B)) for r in reviewers}
    print(f"pairs: {len(cms)}   reviewers: {reviewers}   pairs/reviewer: {n_pairs}")

    self_, other, dirc = pool_claims(cms, reviewers, C)

    # cluster (pair) bootstrap of the log2 index
    idx_all = np.array([np.arange(len(cms))])
    boot_idx = {r: np.full((boot, C), np.nan) for r in reviewers}
    for bi in range(boot):
        pick = [cms[k] for k in rng.integers(0, len(cms), len(cms))]
        s_b, o_b, _ = pool_claims(pick, reviewers, C)
        for r in reviewers:
            boot_idx[r][bi] = np.log2((s_b[r] + 1) / (o_b[r] + 1))

    rows = []
    for r in reviewers:
        for k, code in enumerate(order):
            cs, co = self_[r][k], other[r][k]
            if cs + co < 50:                      # skip near-absent classes for this reviewer
                continue
            b = boot_idx[r][:, k]; b = b[~np.isnan(b)]
            lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b.size else (np.nan, np.nan)
            idx = float(np.log2((cs + 1) / (co + 1)))
            rows.append(dict(reviewer=r, cls=names[code], code=code,
                             claim_self_ha=round(cs * pix_ha, 1),
                             claim_partner_ha=round(co * pix_ha, 1),
                             log2_index=round(idx, 3),
                             ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
                             significant=bool(lo > 0 or hi < 0), n_pairs=n_pairs[r]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "reviewer_class_overassignment.csv"), index=False)

    # per-reviewer strongest directed class-pair leanings (area-weighted, smoothed ratio)
    dp = []
    for r in reviewers:
        d = dirc[r]
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                a, b2 = d[i, j], d[j, i]
                if a + b2 < 200:
                    continue
                dp.append(dict(reviewer=r, says=names[order[i]], partner_says=names[order[j]],
                               px_R_over=int(a), px_partner_over=int(b2),
                               area_R_over_ha=round(a * pix_ha, 1),
                               log2_ratio=round(float(np.log2((a + 1) / (b2 + 1))), 3)))
    dpdf = pd.DataFrame(dp).sort_values(["reviewer", "log2_ratio"], ascending=[True, False])
    dpdf.to_csv(os.path.join(OUT, "reviewer_directed_classpairs.csv"), index=False)

    heatmap(df, reviewers, os.path.join(OUT, "reviewer_overassignment_heatmap.png"))

    print("\nreviewer over-assignment index (log2; + = over-assigns; * = 95% CI excludes 0):")
    piv = df.pivot(index="reviewer", columns="cls", values="log2_index")
    print(piv.round(2).to_string())
    print("\nsignificant over-assignments (index > 0, CI excludes 0):")
    sig = df[(df.log2_index > 0) & df.significant].sort_values("log2_index", ascending=False)
    for r in sig.itertuples():
        print(f"  {r.reviewer:8} {r.cls:14} idx={r.log2_index:+.2f} "
              f"[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]  ({r.claim_self_ha} vs {r.claim_partner_ha} ha)")
    print(f"\noutputs -> {OUT}/reviewer_*")


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def heatmap(df, reviewers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = df.pivot(index="reviewer", columns="cls", values="log2_index")
    sigp = df.pivot(index="reviewer", columns="cls", values="significant")
    # order columns by overall stable land cover then disturbance
    col_order = [c for c in ["Water", "Forest", "Agriculture", "Grass/Shrub", "Wetland",
                             "Urban", "Other", "Harvest", "Development", "Insect/Disease",
                             "Beaver", "Unknown"] if c in piv.columns]
    piv = piv[col_order]; sigp = sigp[col_order]
    M = piv.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(1.0 * len(col_order) + 2, 0.7 * len(piv) + 2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(col_order))); ax.set_xticklabels(col_order, rotation=45, ha="right")
    # anonymize reviewers to letters (alphabetical), matching figure 3.5's reviewer_letter_key.csv
    letters = {rev: chr(ord("A") + i) for i, rev in enumerate(sorted(piv.index))}
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([letters[r] for r in piv.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                star = "*" if bool(sigp.to_numpy()[i, j]) else ""
                ax.text(j, i, f"{M[i,j]:+.2f}{star}", ha="center", va="center", fontsize=7,
                        color="black" if abs(M[i, j]) < 0.6 * vmax else "white")
    ax.set_xlabel("class"); ax.set_ylabel("reviewer")
    # single clean title only; descriptive caption lives in chapter3_captions.md
    ax.set_title("Reviewer Class Over-Assignment", fontsize=15, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="log2 over-assignment")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
