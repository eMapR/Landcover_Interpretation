#!/usr/bin/env python3
"""Visualize the largest change/stable interpreter disagreements: the two reviewers' interpreted
cells side by side, with the disagreed area outlined.

Reads the top rows of change_stable_pixels_long.csv (by area), finds the two reviewer rasters for
each cell, and renders reviewer A next to reviewer B in the RF interpreted colors, outlining the
pixels where one reviewer assigned the stable class and the other the change class for that row.

Outputs -> reports/interpreter_agreement/change_stable_conflicts/examples/

Requires: rasterio, numpy, pandas, matplotlib
"""

import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

RF_DIR = SC.RF_DIR
RF_LEGEND = "data/reference/label_lookup.csv"
BASE = SC.out_path("reports/interpreter_agreement/change_stable_conflicts")
LONG_CSV = os.path.join(BASE, "change_stable_pixels_long.csv")
OUT = os.path.join(BASE, "examples")
N_TOP = 10
LEGEND_GAP = 0.15                                      # inches above and below the legend
DPI = 150
NAME_RE = re.compile(rf"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)", re.I)


def _caption(fig, text, top=1.0, width=125, reserve=0.0):
    """Lay out the axes above a wrapped caption. `reserve` inches are kept free between the caption
    and the axes (the figure must already be that much taller); the rest of the layout is in inches
    of the original height, so reserve=0 gives the original layout."""
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    h = fig.get_figheight()
    h0 = h - reserve
    fig.tight_layout(rect=[0, (h0 * (0.02 + 0.035 * nlines) + reserve) / h, 1,
                           1 - h0 * (1 - top) / h])
    return fig.text(0.5, 0.01 * h0 / h, wrapped, ha="center", va="bottom", fontsize=8,
                    color="0.35")


def _inches(artist, fig):
    """Rendered bounding box of an artist in figure inches (x0, y0, x1, y1)."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi, bb.y1 / fig.dpi


def load_legend():
    df = pd.read_csv(RF_LEGEND)
    code2name = {int(r.code): r.display_name for r in df.itertuples()}
    code2rgb = {int(r.code): mcolors.to_rgb(r.color) for r in df.itertuples()}
    name2code = {r.display_name: int(r.code) for r in df.itertuples()}
    return code2name, code2rgb, name2code


def pair_index():
    """(grid,sample,target) -> {reviewer: path} for the double-interpreted cells."""
    groups = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True)):
        m = NAME_RE.search(os.path.basename(f))
        if m:
            groups[(m.group(2), m.group(3), m.group(4))][m.group(1).lower()] = f
    return groups


def colorize(arr, code2rgb):
    rgb = np.ones(arr.shape + (3,))                     # white for anything unmapped
    for code, c in code2rgb.items():
        rgb[arr == code] = c
    return rgb


def main():
    os.makedirs(OUT, exist_ok=True)
    code2name, code2rgb, name2code = load_legend()
    idx = pair_index()

    df = pd.read_csv(LONG_CSV, dtype={"grid": str, "sample": str, "target": str})
    top = df.sort_values("area_ha", ascending=False).head(N_TOP).reset_index(drop=True)

    for rank, row in top.iterrows():
        key = (row.grid, row["sample"], row.target)     # row.sample is the pandas method, not the col
        paths = idx.get(key, {})
        if row.revA not in paths or row.revB not in paths:
            print(f"  missing raster for {key} {row.revA}/{row.revB}; skip")
            continue
        with rasterio.open(paths[row.revA]) as ds:
            a = ds.read(1)
            px_m = abs(ds.transform.a) * ds.crs.linear_units_factor[1]   # pixel width in metres
        with rasterio.open(paths[row.revB]) as ds:
            b = ds.read(1)
        if a.shape != b.shape:
            print(f"  shape mismatch {key}; skip")
            continue
        sc, cc = name2code[row.stable_class], name2code[row.change_class]
        # the disagreed pixels for this row's class pair, either direction
        conflict = ((a == sc) & (b == cc)) | ((a == cc) & (b == sc))

        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        for ax, arr, rev in [(axes[0], a, row.revA), (axes[1], b, row.revB)]:
            ax.imshow(colorize(arr, code2rgb), interpolation="nearest")
            # outline the disagreed area on both panels
            ax.contour(conflict.astype(float), levels=[0.5], colors="black", linewidths=1.4)
            ax.set_title(f"reviewer {rev}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

        present = sorted(set(np.unique(a)).union(np.unique(b)) & set(code2name))
        handles = [Patch(facecolor=code2rgb[c], edgecolor="0.4", label=code2name[c]) for c in present]
        handles.append(Patch(facecolor="none", edgecolor="black", label="disagreed area (outlined)"))
        leg = fig.legend(handles=handles, loc="upper center", ncol=min(7, len(handles)),
                         fontsize=8, borderaxespad=0)
        # legend gets its own strip between the maps and the caption: grow the figure by the
        # legend's measured height plus a gap on each side, so the maps keep their size
        fig.canvas.draw()
        _, ly0, _, ly1 = _inches(leg, fig)
        # whole output pixels, so the maps and text above snap to the same pixels as before
        reserve = np.ceil(((ly1 - ly0) + 2 * LEGEND_GAP) * DPI) / DPI
        h0 = fig.get_figheight()
        fig.set_size_inches(fig.get_figwidth(), h0 + reserve)
        km = a.shape[1] * px_m / 1000
        fig.suptitle(f"#{rank + 1}  cell {row.grid} ({km:.1f} km, target {row.target})  ·  "
                     f"{row.revA} vs {row.revB}  ·  outlined: {row.stable_class} vs "
                     f"{row.change_class}  ({row.area_ha:.0f} ha, one called stable, one called "
                     f"change)", fontsize=10, y=1 - 0.02 * h0 / (h0 + reserve))
        cap = _caption(fig, "One of the largest change-versus-stable interpreter disagreements, showing "
                 "the two reviewers' interpreted maps for the same cell side by side in the RF land "
                 "cover colors. The black outline marks the pixels where one reviewer assigned the "
                 "stable class and the other assigned the paired change class, drawn identically on "
                 "both panels. Compare the two panels within the outlined area to see how the "
                 "reviewers split on that class pair, with the cell, class pair, and disagreed area "
                 "in hectares given in the header.", top=0.93, reserve=reserve)
        fig.canvas.draw()
        maps_y0 = min(_inches(ax, fig)[1] for ax in axes)    # drawn (aspect-shrunk) map bottoms
        leg.set_bbox_to_anchor((0.5, (maps_y0 - LEGEND_GAP) / fig.get_figheight()),
                               transform=fig.transFigure)
        fig.canvas.draw()
        lx0, ly0, lx1, ly1 = _inches(leg, fig)
        cap_y1 = _inches(cap, fig)[3]
        for ax in axes:                                       # no legend over a map column
            ax_x0, _, ax_x1, _ = _inches(ax, fig)
            if lx1 > ax_x0 and lx0 < ax_x1 and ly1 > maps_y0 - 0.02:
                print(f"  WARN rank {rank + 1}: legend overlaps a map panel")
        if ly0 < cap_y1 + 0.02:
            print(f"  WARN rank {rank + 1}: legend overlaps the caption")
        if lx0 < 0 or lx1 > fig.get_figwidth():
            print(f"  WARN rank {rank + 1}: legend wider than the figure")
        print(f"  legend fit rank {rank + 1}: map bottom {maps_y0:.2f} in, legend "
              f"{ly0:.2f}-{ly1:.2f} in, caption top {cap_y1:.2f} in")
        out = os.path.join(OUT, f"rank{rank + 1:02d}_grid{row.grid}_"
                                f"{row.stable_class}_vs_{row.change_class}.png".replace("/", "-"))
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {os.path.basename(out)}")

    print(f"\nwrote {OUT}/ ({min(N_TOP, len(top))} examples)")


if __name__ == "__main__":
    main()
