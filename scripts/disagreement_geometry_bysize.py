#!/usr/bin/env python3
"""Area-weighted and size-conditional geometry of inter-interpreter disagreement.

The pooled shape index (~1.13-1.20) is dominated by single-pixel specks (a 1x1 patch has
shape index = 4/(2*sqrt(pi)) = 1.128). This re-does the geometry weighting by AREA rather
than patch count, and conditions on patch size, to characterize the patches that actually
hold the disagreement area (on Sentinel-2, the >0.1 ha patches carry 55-83% of it).

The small/large split comes from the pixel area (DG.SIZE_SPLITS): 0.1 ha at 10 m (size bins
lt_0.1ha / ge_0.1ha) and 5 px = 0.45 ha at 30 m (lt_5px / ge_5px); "small" is strictly below.
Any other pixel size is an error.

For each directed class pair it reports, for {all, small, large}:
  - count-median and AREA-WEIGHTED-median shape index and pixel width (2A/P)
  - median extent = area / bounding-box area (compactness): ribbons have low extent + high
    shape index + ~1-2 px width; blobs have high extent + low shape index + several px width

Then, for Grass/Shrub<->Wetland (the worst case on Sentinel-2, ~83% of area in >0.1 ha patches), it pulls
the 10 largest disagreement patches (undirected GS/Wetland zone) and renders each: the two
interpretations side by side with the disagreement patch outlined, so a wide margin band is
distinguishable from a solid interior block.

No de-duplication. Sampling (if any) uses numpy default_rng(42). Classic-theme plots.

Outputs (reports/interpreter_agreement/geometry/):
  - size_conditional_summary.csv
  - shape_index_area_weighted_ecdf.png
  - gs_wetland_top10.csv
  - gs_wetland_top10.png

Requires: rasterio, numpy, pandas, matplotlib, scipy
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreters as CI
import disagreement_geometry as DG   # per_pixel_perimeter, classic, ecdf, FOCUS_LABELS, STRUCT
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = DG.OUT
GS, WET = 2, 5
rng = np.random.default_rng(42)


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def wmedian(values, weights):
    """Weighted median of `values` weighted by `weights`."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    order = np.argsort(v); v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])]) if c[-1] > 0 else np.nan


def build_patches(pairs, names, codes):
    """Recompute disagreement patches with extent (area / bbox area).

    Returns (patch DataFrame, pixel area in m^2); ValueError if cells have different pixel areas.
    """
    rows = []
    pix_areas = set()                                # m^2; the size split needs one pixel size
    for (gid, samp, tgt), revs in sorted(pairs.items()):
        (ra, fA), (rb, fB) = revs[0], revs[1]
        cid = f"{gid}_s{samp}_t{tgt}"
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            continue
        pix_m2 = SC.uniform_pixel_area_m2([fA, fB])
        pix_areas.add(pix_m2)
        pix_ha = pix_m2 / SC.M2_PER_HA
        valid = np.isin(a, codes) & np.isin(b, codes)
        dis = valid & (a != b)
        for (A, B) in {(int(x), int(y)) for x, y in zip(a[dis], b[dis])}:
            mask = (a == A) & (b == B)
            lab, n = ndimage.label(mask, structure=DG.STRUCT)
            if n == 0:
                continue
            area = np.bincount(lab.ravel())[1:].astype(float)
            perim = np.bincount(lab.ravel(),
                                weights=DG.per_pixel_perimeter(mask).ravel())[1:]
            objs = ndimage.find_objects(lab)
            border = np.zeros(mask.shape, bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            edge_labels = set(np.unique(lab[border])) - {0}
            for lid in range(1, n + 1):
                Ap, P = area[lid - 1], perim[lid - 1]
                sl = objs[lid - 1]
                bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
                rows.append(dict(cell_id=cid, class_a=names[A], class_b=names[B],
                                 area_px=int(Ap), area_ha=Ap * pix_ha,
                                 shape_index=P / (2 * np.sqrt(np.pi * Ap)),
                                 width_px=2 * Ap / P if P else np.nan,
                                 extent=Ap / bbox, touches_edge=bool(lid in edge_labels)))
    if len(pix_areas) != 1:
        raise ValueError(f"cells have different pixel areas: {sorted(pix_areas)} m^2")
    return pd.DataFrame(rows), pix_areas.pop()


def size_summary(df, split):
    """Per directed focus pair x size bin: count- and area-weighted geometry."""
    focus = []
    for _, ca, cb in DG.FOCUS_LABELS():
        focus += [(ca, cb), (cb, ca)]
    name_of = {c: n for c, n in zip(*[[], []])}  # placeholder
    out = []
    groups = {"ALL disagreement": df}
    # focus directed pairs by display name
    codes_names = df  # already display names in class_a/class_b
    focus_names = []
    leg = CI.load_legend()[1]
    for ca, cb in focus:
        focus_names.append((leg[ca], leg[cb]))
    focus_names.append(("Beaver", "*any*"))

    def rowset(sub, label, sizebin):
        if len(sub) == 0:
            return
        w = sub.area_ha.to_numpy()
        out.append(dict(
            pair=label, size_bin=sizebin, n_patches=len(sub),
            total_area_ha=round(float(w.sum()), 2),
            med_shape=round(float(np.median(sub.shape_index)), 3),
            aw_med_shape=round(wmedian(sub.shape_index, w), 3),
            med_width=round(float(np.median(sub.width_px.dropna())), 3),
            aw_med_width=round(wmedian(sub.width_px.fillna(0), w), 3),
            med_extent=round(float(np.median(sub.extent)), 3),
            aw_med_extent=round(wmedian(sub.extent, w), 3)))

    for (na, nb) in focus_names:
        if nb == "*any*":
            sub_all = df[(df.class_a == na) | (df.class_b == na)]
            label = f"{na} <-> any"
        else:
            sub_all = df[(df.class_a == na) & (df.class_b == nb)]
            label = f"{na} -> {nb}"
        small = DG.is_small(sub_all, split)
        for sizebin, sub in [("all", sub_all),
                             (f"lt_{split['tag']}", sub_all[small]),
                             (f"ge_{split['tag']}", sub_all[~small])]:
            rowset(sub, label, sizebin)
    # pooled ALL
    small = DG.is_small(df, split)
    for sizebin, sub in [("all", df),
                         (f"lt_{split['tag']}", df[small]),
                         (f"ge_{split['tag']}", df[~small])]:
        rowset(sub, "ALL disagreement", sizebin)
    return pd.DataFrame(out)


def aw_ecdf_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leg = CI.load_legend()[1]
    focus = [(leg[ca], leg[cb]) for _, ca, cb in DG.FOCUS_LABELS()] + [("Beaver", None)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    for ax, (na, nb) in zip(axes, focus):
        if nb is None:
            sub = df[(df.class_a == "Beaver") | (df.class_b == "Beaver")]
            ttl = "Beaver <-> any"
        else:
            sub = df[((df.class_a == na) & (df.class_b == nb)) |
                     ((df.class_a == nb) & (df.class_b == na))]
            ttl = f"{na} <-> {nb}"
        if len(sub) >= 3:
            v = sub.shape_index.to_numpy(); w = sub.area_ha.to_numpy()
            order = np.argsort(v); v, w = v[order], w[order]
            ax.plot(v, np.cumsum(w) / w.sum(), "-", color="#1f77b4", lw=1.8,
                    label="area-weighted")
            xs, ys = DG.ecdf(sub.shape_index)
            ax.plot(xs, ys, ":", color="0.4", lw=1.4, label="count-weighted")
        ax.axvline(1.128, color="0.6", lw=0.8, ls="--")  # single-pixel value
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("shape index"); ax.set_ylabel("cumulative fraction")
        ax.set_xlim(1.0, 3.5); ax.legend(fontsize=7, frameon=False); DG.classic(ax)
    # chapter 3 figure 3.4c; clean title, no in-figure caption (caption is in the manuscript caption file)
    fig.suptitle("Contested-Patch Shape Complexity", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def render_top10(pairs, names, colors, csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    # find largest undirected GS/Wetland disagreement zones across all cells
    cand = []
    files = {}
    for (gid, samp, tgt), revs in sorted(pairs.items()):
        (ra, fA), (rb, fB) = revs[0], revs[1]
        cid = f"{gid}_s{samp}_t{tgt}"; files[cid] = (ra, fA, rb, fB)
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            continue
        pix_ha = SC.uniform_pixel_area_m2([fA, fB]) / SC.M2_PER_HA
        mask = ((a == GS) & (b == WET)) | ((a == WET) & (b == GS))
        lab, n = ndimage.label(mask, structure=DG.STRUCT)
        if n == 0:
            continue
        area = np.bincount(lab.ravel())[1:]
        perim = np.bincount(lab.ravel(), weights=DG.per_pixel_perimeter(mask).ravel())[1:]
        objs = ndimage.find_objects(lab)
        for lid in range(1, n + 1):
            Ap, P = int(area[lid - 1]), perim[lid - 1]
            sl = objs[lid - 1]
            bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
            cand.append(dict(cell_id=cid, lid=lid, area_px=Ap, area_ha=round(Ap * pix_ha, 3),
                             shape_index=round(P / (2 * np.sqrt(np.pi * Ap)), 3),
                             width_px=round(2 * Ap / P, 3), extent=round(Ap / bbox, 3)))
    cdf = pd.DataFrame(cand).sort_values("area_px", ascending=False).head(10).reset_index(drop=True)
    cdf.to_csv(csv_path, index=False)
    print("\nGrass/Shrub<->Wetland — 10 largest disagreement patches:")
    print(cdf[["cell_id", "area_ha", "width_px", "shape_index", "extent"]].to_string(index=False))

    # colormap over RF classes
    order = sorted(names)
    lut = {c: i + 1 for i, c in enumerate(order)}
    cmap = ListedColormap(["#ffffff"] + [colors[c] for c in order])
    norm = BoundaryNorm(np.arange(-0.5, len(order) + 1.5), cmap.N)

    def remap(arr):
        o = np.zeros_like(arr, np.int16)
        for c, i in lut.items():
            o[arr == c] = i
        return o

    fig, axes = plt.subplots(10, 2, figsize=(7.2, 34))
    for i, r in cdf.iterrows():
        ra, fA, rb, fB = files[r.cell_id]
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        mask = ((a == GS) & (b == WET)) | ((a == WET) & (b == GS))
        lab, _ = ndimage.label(mask, structure=DG.STRUCT)
        patch = lab == r.lid
        for j, (arr, rev) in enumerate([(a, ra), (b, rb)]):
            ax = axes[i, j]
            ax.imshow(remap(arr), cmap=cmap, norm=norm, interpolation="nearest")
            ax.contour(patch, levels=[0.5], colors="red", linewidths=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{r.area_ha} ha\nw={r.width_px}px sh={r.shape_index}\next={r.extent}",
                              fontsize=7)
            ax.set_title(f"{rev}" + (f"  [{r.cell_id}]" if j == 0 else ""), fontsize=8)
    handles = [Patch(facecolor=colors[c], edgecolor="k", label=names[c])
               for c in order if c in (GS, WET, 3, 1, 0, 4)]
    handles.append(Patch(edgecolor="red", facecolor="none", label="disagreement patch"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Grass/Shrub <-> Wetland: 10 largest disagreement patches\n"
                 "(two interpretations side by side, patch outlined in red)", fontsize=11)
    _caption(fig, "The ten largest Grass/Shrub versus Wetland disagreement patches, one per row, "
                  "each showing the two reviewers' interpretations side by side with the "
                  "disagreement patch outlined in red and the RF class colors keyed by the bottom "
                  "legend. Row labels give the patch area in hectares along with its pixel width, "
                  "shape index, and extent (area over bounding-box area). Read it to tell a wide "
                  "solid interior block of genuine class disagreement from a thin margin ribbon "
                  "hugging the edge of an otherwise agreed patch.", top=0.93)
    fig.savefig(png_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    codes, names, colors = CI.load_legend()
    pairs = CI.find_pairs()
    print(f"double-interpreted cells: {len(pairs)} (no de-duplication)")
    os.makedirs(OUT, exist_ok=True)

    df, pix_m2 = build_patches(pairs, names, codes)
    split = DG.size_split(pix_m2)
    print(f"disagreement patches: {len(df)}")

    summ = size_summary(df, split)
    summ.to_csv(os.path.join(OUT, "size_conditional_summary.csv"), index=False)
    print(f"\nsize-conditional geometry (>={split['ge_text']} patches):")
    show = summ[summ.size_bin == f"ge_{split['tag']}"][
        ["pair", "n_patches", "total_area_ha", "med_width", "med_shape", "med_extent",
         "aw_med_width", "aw_med_shape"]]
    print(show.to_string(index=False))

    aw_ecdf_plot(df, os.path.join(OUT, "shape_index_area_weighted_ecdf.png"))
    render_top10(pairs, names, colors,
                 os.path.join(OUT, "gs_wetland_top10.csv"),
                 os.path.join(OUT, "gs_wetland_top10.png"))
    print(f"\noutputs -> {OUT}/")


if __name__ == "__main__":
    main()
