#!/usr/bin/env python3
"""Geometry of inter-interpreter disagreement, per DIRECTED class pair.

On the 69 double-interpreted cells (no de-duplication — the replicates are the point),
for each ordered class pair (A, B) where reviewer_a said A and reviewer_b said B
(direction kept, NOT symmetrized), build the binary disagreement mask per cell, run
8-connected component labeling, and measure each patch:

  area        pixel count (and hectares from the raster transform; 1 px = 0.01 ha at 10 m)
  perimeter   4-connected boundary-edge count (a single pixel = 4; not a vertex trace)
  pa_ratio    perimeter / area
  shape_index P / (2*sqrt(pi*A)) — normalized against a circle of equal area, so it
              separates shape from size (raw P/A falls with size at constant shape)

reviewer_a / reviewer_b are the two reviewers ordered alphabetically, fixed per cell.

Reference distribution: the same geometry on the AGREEMENT areas (both reviewers concur,
class_a == class_b), so disagreement geometry is read against the geometry of the classes
themselves (a thin ribbon around a large agreed patch != a patch the size of the features).

Traps handled:
  - edge-of-cell artifact: patches touching the cell border have truncated perimeters and
    biased shape; flagged via `touches_edge`, and summaries reported with and without them.
  - thin ribbons: at 10 m a 1-2 px ribbon makes the discrete perimeter a poor length
    estimate and shape index unstable; the per-patch pixel width (2A/P) distribution is
    reported so we know how much sits at the resolution limit.

Outputs (reports/interpreter_agreement/geometry/):
  - patch_geometry.csv            long format, one row per patch (disagreement + agreement)
  - patchpair_summary.csv         per directed class pair x {all, interior}: n, area median/IQR,
                                  shape median, fraction of disagreement area < / >= 0.1 ha
  - width_distribution.csv        pixel-width summary of disagreement patches
  - area_ecdf_focus.png           area ECDFs for the known high-disagreement boundaries
  - shape_index_ecdf_focus.png    shape-index ECDFs for the same boundaries
  - width_ecdf.png                pixel-width ECDF (resolution-limit check)

Usage:
    python scripts/disagreement_geometry.py

Requires: rasterio, numpy, pandas, matplotlib, scipy
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreters as CI  # find_pairs / load_legend
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = "reports/interpreter_agreement/geometry"
STRUCT = np.ones((3, 3), int)      # 8-connectivity
AREA_THRESH_HA = 0.1               # 0.1 ha = 10 px at 10 m
rng = np.random.default_rng(42)    # only used if any sampling is needed


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def per_pixel_perimeter(mask):
    """4-connected boundary-edge count per pixel (outside-array counts as boundary)."""
    m = mask.astype(np.int16)
    nin = np.zeros_like(m)
    nin[1:, :] += m[:-1, :]
    nin[:-1, :] += m[1:, :]
    nin[:, 1:] += m[:, :-1]
    nin[:, :-1] += m[:, 1:]
    return (4 - nin) * m           # 0 off-mask; single pixel -> 4


def patch_rows(mask, cell_id, ra, rb, ca, cb, kind, names, pix_ha):
    """Label `mask` (8-conn) and return one dict per connected patch."""
    lab, n = ndimage.label(mask, structure=STRUCT)
    if n == 0:
        return []
    area = np.bincount(lab.ravel())[1:].astype(float)                 # per label
    perim_pp = per_pixel_perimeter(mask)
    perim = np.bincount(lab.ravel(), weights=perim_pp.ravel())[1:]
    # edge-touching labels
    border = np.zeros(mask.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    edge_labels = set(np.unique(lab[border])) - {0}
    rows = []
    for lid in range(1, n + 1):
        A = area[lid - 1]; P = perim[lid - 1]
        rows.append(dict(
            cell_id=cell_id, reviewer_a=ra, reviewer_b=rb,
            class_a=names[ca], class_b=names[cb], patch_id=lid,
            area_px=int(A), area_ha=round(A * pix_ha, 4),
            perimeter=int(P), pa_ratio=round(P / A, 4),
            shape_index=round(P / (2 * np.sqrt(np.pi * A)), 4),
            width_px=round(2 * A / P, 4) if P else np.nan,
            touches_edge=bool(lid in edge_labels), kind=kind))
    return rows


def main():
    codes, names, colors = CI.load_legend()
    codeset = set(codes)
    pairs = CI.find_pairs()
    print(f"double-interpreted cells: {len(pairs)} (no de-duplication)")
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for (gid, samp, tgt), revs in sorted(pairs.items()):
        (ra, fA), (rb, fB) = revs[0], revs[1]      # alphabetical; direction fixed
        cell_id = f"{gid}_s{samp}_t{tgt}"
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            continue
        pix_ha = SC.uniform_pixel_area_m2([fA, fB]) / SC.M2_PER_HA
        valid = np.isin(a, codes) & np.isin(b, codes)

        # disagreement: directed class pairs (A != B)
        dis = valid & (a != b)
        for (A, B) in {(int(x), int(y)) for x, y in zip(a[dis], b[dis])}:
            rows += patch_rows((a == A) & (b == B), cell_id, ra, rb, A, B, "disagreement", names, pix_ha)

        # agreement reference: both concur on class C
        agr = valid & (a == b)
        for C in {int(x) for x in a[agr]}:
            rows += patch_rows((a == C) & (b == C), cell_id, ra, rb, C, C, "agreement", names, pix_ha)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "patch_geometry.csv"), index=False)
    print(f"patches: {len(df)}  (disagreement {int((df.kind=='disagreement').sum())}, "
          f"agreement {int((df.kind=='agreement').sum())})")

    write_summaries(df, names)
    make_plots(df, names, colors)
    print(f"\noutputs -> {OUT}/")


def _pair_stats(sub):
    a = sub.area_ha.to_numpy()
    tot = a.sum()
    under = a[a < AREA_THRESH_HA].sum()
    return dict(
        n_patches=len(sub),
        area_median_ha=round(float(np.median(a)), 4),
        area_q25_ha=round(float(np.percentile(a, 25)), 4),
        area_q75_ha=round(float(np.percentile(a, 75)), 4),
        shape_index_median=round(float(np.median(sub.shape_index)), 3),
        width_px_median=round(float(np.median(sub.width_px.dropna())), 3),
        frac_area_under_0p1ha=round(float(under / tot), 4) if tot else np.nan,
        frac_area_over_0p1ha=round(float(1 - under / tot), 4) if tot else np.nan,
    )


def write_summaries(df, names):
    dis = df[df.kind == "disagreement"]
    out = []
    for subset, mask in [("all", np.ones(len(dis), bool)),
                         ("interior", ~dis.touches_edge.to_numpy())]:
        d = dis[mask]
        for (ca, cb), sub in d.groupby(["class_a", "class_b"]):
            out.append(dict(class_a=ca, class_b=cb, subset=subset, **_pair_stats(sub)))
    summ = pd.DataFrame(out).sort_values(["subset", "n_patches"], ascending=[True, False])
    summ.to_csv(os.path.join(OUT, "patchpair_summary.csv"), index=False)

    # pixel-width distribution of disagreement patches
    w = dis.width_px.dropna().to_numpy()
    wr = []
    for label, mask in [("all", np.ones(len(dis), bool)),
                        ("interior", ~dis.touches_edge.to_numpy())]:
        ww = dis.width_px.to_numpy()[mask]
        ww = ww[~np.isnan(ww)]
        wr.append(dict(subset=label, n=len(ww),
                       width_median=round(float(np.median(ww)), 3),
                       frac_le_1px=round(float((ww <= 1.0).mean()), 4),
                       frac_le_2px=round(float((ww <= 2.0).mean()), 4),
                       frac_gt_2px=round(float((ww > 2.0).mean()), 4)))
    pd.DataFrame(wr).to_csv(os.path.join(OUT, "width_distribution.csv"), index=False)

    # console: focus boundaries
    print("\nfocus boundaries (disagreement, subset=all): "
          "n | area median [IQR] ha | shape med | %area <0.1ha")
    for title, ca, cb in FOCUS_LABELS():
        for A, B in [(ca, cb), (cb, ca)]:
            sub = dis[(dis.class_a == names[A]) & (dis.class_b == names[B])]
            if len(sub):
                s = _pair_stats(sub)
                print(f"  {names[A]:>12} -> {names[B]:<12} "
                      f"n={s['n_patches']:>5}  {s['area_median_ha']:.3f} "
                      f"[{s['area_q25_ha']:.3f},{s['area_q75_ha']:.3f}]  "
                      f"shp={s['shape_index_median']:.2f}  "
                      f"<0.1ha={100*s['frac_area_under_0p1ha']:.0f}%")


BEAVER = 62


def FOCUS_LABELS():
    # (title, code_a, code_b) undirected boundaries known to drive disagreement
    return [("Forest<->Wetland", 3, 5),
            ("Agriculture<->Grass/Shrub", 1, 2),
            ("Grass/Shrub<->Forest", 2, 3),
            ("Grass/Shrub<->Wetland", 2, 5),
            ("Development<->Urban", 30, 0)]


def classic(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def ecdf(a):
    a = np.sort(np.asarray(a, float))
    return a, np.arange(1, a.size + 1) / a.size


def make_plots(df, names, colors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dis = df[df.kind == "disagreement"]
    agr = df[df.kind == "agreement"]
    focus = FOCUS_LABELS()

    def panels(value_col, xlabel, logx, fname, title, caption):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()
        for ax, (ttl, ca, cb) in zip(axes, focus):
            for A, B, style in [(ca, cb, "-"), (cb, ca, "--")]:
                sub = dis[(dis.class_a == names[A]) & (dis.class_b == names[B])]
                if len(sub) >= 3:
                    xs, ys = ecdf(sub[value_col]);
                    ax.plot(xs, ys, style, lw=1.6, label=f"{names[A]}→{names[B]} (n={len(sub)})")
            ref = agr[agr.class_a.isin([names[ca], names[cb]])]
            if len(ref) >= 3:
                xs, ys = ecdf(ref[value_col])
                ax.plot(xs, ys, color="0.4", lw=1.4, ls=":", label=f"agreement ref (n={len(ref)})")
            if logx:
                ax.set_xscale("log")
            ax.set_title(ttl, fontsize=13)
            ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel("cumulative fraction", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=9.5, frameon=False)
            classic(ax)
        # 6th panel: Beaver <-> anything
        ax = axes[5]
        sub = dis[(dis.class_a == names[BEAVER]) | (dis.class_b == names[BEAVER])]
        if len(sub) >= 3:
            xs, ys = ecdf(sub[value_col]); ax.plot(xs, ys, "-", color="#CC66CA", lw=1.6,
                                                   label=f"Beaver↔any (n={len(sub)})")
        refb = agr[agr.class_a == names[BEAVER]]
        if len(refb) >= 3:
            xs, ys = ecdf(refb[value_col]); ax.plot(xs, ys, color="0.4", lw=1.4, ls=":",
                                                    label=f"agreement Beaver (n={len(refb)})")
        if logx:
            ax.set_xscale("log")
        ax.set_title("Beaver ↔ anything", fontsize=13)
        ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel("cumulative fraction", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=9.5, frameon=False); classic(ax)
        fig.suptitle(title, fontsize=17, fontweight="bold")
        if caption:                                             # descriptive caption lives in the manuscript caption file
            _caption(fig, caption, top=0.93)
        else:
            fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=140, bbox_inches="tight")
        plt.close(fig)

    # area_ecdf_focus is Chapter 3 figure 3.4b; clean title, no in-figure caption (caption is in the
    # manuscript caption file). other panels() figures below keep their descriptive captions.
    panels("area_ha", "patch area (ha, log)", True, "area_ecdf_focus.png",
           "Contested-Patch Area Distribution", "")
    panels("shape_index", "shape index  P / (2√(πA))", False, "shape_index_ecdf_focus.png",
           "Disagreement patch shape index by directed class pair (vs. agreement reference)",
           "Each panel plots the empirical cumulative distribution of the disagreement-patch "
           "shape index, defined as perimeter divided by 2 times the square root of pi times "
           "area, for one high-disagreement class boundary, with solid and dashed lines for the "
           "two directed orderings and a dotted gray reference curve for the agreement patches. "
           "The shape index equals 1 for a circle and rises for elongated or convoluted patches, "
           "so curves shifted right of the agreement reference indicate ribbon-like disagreement "
           "along class edges. The sixth panel pools all Beaver disagreement patches.")

    # pixel-width ECDF (resolution-limit check)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    allw = dis.width_px.dropna()
    inw = dis[~dis.touches_edge].width_px.dropna()
    for data, lab, c in [(allw, f"all disagreement (n={len(allw)})", "#1f77b4"),
                         (inw, f"interior only (n={len(inw)})", "#d62728")]:
        xs, ys = ecdf(data); ax.plot(xs, ys, lw=1.8, color=c, label=lab)
    for xv in (1, 2):
        ax.axvline(xv, color="0.6", lw=0.8, ls="--")
    ax.set_xlim(0, 8)
    ax.set_xlabel("patch pixel width  (2A / P)"); ax.set_ylabel("cumulative fraction")
    ax.set_title("Disagreement patch width — resolution-limit check\n(dashed: 1 and 2 pixels)")
    ax.legend(fontsize=8, frameon=False); classic(ax)
    _caption(fig, "Empirical cumulative distribution of disagreement-patch pixel width, computed "
                  "as twice the area over the perimeter, for all disagreement patches in blue and "
                  "for interior patches that do not touch the cell edge in red. The dashed "
                  "vertical lines mark widths of 1 and 2 pixels, the resolution limit at 10 m "
                  "where the discrete perimeter and shape index become unreliable. A large share "
                  "of mass at or below 2 pixels means much of the disagreement sits in thin "
                  "ribbons near the pixel resolution limit.")
    fig.savefig(os.path.join(OUT, "width_ecdf.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
