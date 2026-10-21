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
NAME_RE = re.compile(rf"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)", re.I)


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


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
        fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), fontsize=8,
                   bbox_to_anchor=(0.5, 0.20))
        km = a.shape[1] * 10 / 1000
        fig.suptitle(f"#{rank + 1}  cell {row.grid} ({km:.1f} km, target {row.target})  ·  "
                     f"{row.revA} vs {row.revB}  ·  outlined: {row.stable_class} vs "
                     f"{row.change_class}  ({row.area_ha:.0f} ha, one called stable, one called "
                     f"change)", fontsize=10)
        _caption(fig, "One of the largest change-versus-stable interpreter disagreements, showing "
                 "the two reviewers' interpreted maps for the same cell side by side in the RF land "
                 "cover colors. The black outline marks the pixels where one reviewer assigned the "
                 "stable class and the other assigned the paired change class, drawn identically on "
                 "both panels. Compare the two panels within the outlined area to see how the "
                 "reviewers split on that class pair, with the cell, class pair, and disagreed area "
                 "in hectares given in the header.", top=0.93)
        out = os.path.join(OUT, f"rank{rank + 1:02d}_grid{row.grid}_"
                                f"{row.stable_class}_vs_{row.change_class}.png".replace("/", "-"))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {os.path.basename(out)}")

    print(f"\nwrote {OUT}/ ({min(N_TOP, len(top))} examples)")


if __name__ == "__main__":
    main()
