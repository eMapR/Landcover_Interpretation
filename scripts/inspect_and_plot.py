#!/usr/bin/env python3
"""Inspect and visualize the Random Forest classified land cover rasters.

The rasters are single-band integer class maps where each pixel value is a land
cover / disturbance class code. Codes are decoded via the legend in
``data/reference/label_lookup.csv``.

Examples:
    # print metadata + class histogram for one raster
    python scripts/inspect_and_plot.py --stats <path/to/rf_class_*.tif>

    # global class distribution across every downloaded raster (writes a CSV)
    python scripts/inspect_and_plot.py --stats-all

    # plot a single classified map with a labelled legend
    python scripts/inspect_and_plot.py --plot <path/to/rf_class_*.tif>

    # montage of the first N maps
    python scripts/inspect_and_plot.py --montage 9

Requires: rasterio, numpy, matplotlib, pandas
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio
import sensor_config as SC  # RF_DIR, out_path (LCI_RF_DIR / LCI_OUT_DIR)

warnings.filterwarnings("ignore")

RASTER_DIR = SC.RF_DIR
LEGEND_CSV = "data/reference/label_lookup.csv"
OUT_DIR = SC.out_path("outputs")


def load_legend(path=LEGEND_CSV):
    """Return {code: {'name','display_name','color','type'}} from the legend CSV."""
    df = pd.read_csv(path)
    return {int(r.code): {"name": r.name, "display_name": r.display_name,
                          "color": r.color, "type": r.type}
            for r in df.itertuples()}


def find_rasters(pattern=None):
    files = sorted(glob.glob(os.path.join(RASTER_DIR, "**", "rf_class*.tif"),
                             recursive=True))
    if pattern:
        files = [f for f in files if pattern in f]
    return files


def stats_one(path, legend):
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = dict(width=src.width, height=src.height, count=src.count,
                    dtype=src.dtypes[0], crs=str(src.crs), res=src.res,
                    nodata=src.nodata)
    print(f"file:   {os.path.basename(path)}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    vals, counts = np.unique(arr, return_counts=True)
    total = arr.size
    print("  class distribution:")
    for v, c in zip(vals.tolist(), counts.tolist()):
        info = legend.get(v, {"display_name": f"code {v}"})
        print(f"    {v:>3}  {info['display_name']:<16} {c:>7} px  ({100*c/total:5.1f}%)")


def stats_all(legend):
    files = find_rasters()
    if not files:
        sys.exit(f"no rasters found under {RASTER_DIR} (run a fetch script first)")
    px = {code: 0 for code in legend}
    present = {code: 0 for code in legend}  # rasters containing this class
    total_px = 0
    for f in files:
        with rasterio.open(f) as src:
            arr = src.read(1)
        total_px += arr.size
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            px[v] = px.get(v, 0) + c
            present[v] = present.get(v, 0) + 1

    rows = []
    for code, info in legend.items():
        rows.append(dict(code=code, display_name=info["display_name"],
                         type=info["type"], pixels=px.get(code, 0),
                         pct_of_pixels=round(100 * px.get(code, 0) / total_px, 3),
                         rasters_present=present.get(code, 0)))
    df = pd.DataFrame(rows).sort_values("pixels", ascending=False)
    print(f"rasters: {len(files)}   total pixels: {total_px:,}\n")
    print(df.to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "class_distribution.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


def _caption(fig, text, top=1.0, width=95):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.04 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def _plot_axis(ax, arr, legend):
    """Render a class array onto ax using the legend colors. Returns codes shown."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    codes = sorted(legend)
    colors = [legend[c]["color"] for c in codes]
    # remap raster values -> 0..N-1 index so colors line up regardless of gaps
    lut = {c: i for i, c in enumerate(codes)}
    idx = np.vectorize(lambda v: lut.get(v, -1))(arr).astype(float)
    idx[idx < 0] = np.nan
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(codes) + 0.5), cmap.N)
    ax.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return set(np.unique(arr).tolist())


def plot_one(path, legend):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    with rasterio.open(path) as src:
        arr = src.read(1)
    fig, ax = plt.subplots(figsize=(8, 7))
    present = _plot_axis(ax, arr, legend)
    ax.set_title(os.path.basename(path), fontsize=8)
    handles = [Patch(facecolor=legend[c]["color"], edgecolor="k",
                     label=f"{c} {legend[c]['display_name']}")
               for c in sorted(legend) if c in present]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, title="class")
    _caption(fig, "A single Random Forest classified land-cover raster, with each pixel colored by "
             "its class code as decoded through the legend on the right, which lists only the classes "
             "present in this map. The map shows the spatial layout of land-cover and disturbance "
             "classes for this interpreted cell. Read the colors against the legend to see where "
             "each class falls.", width=90)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0] + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_montage(n, legend):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    files = find_rasters()[:n]
    if not files:
        sys.exit(f"no rasters found under {RASTER_DIR}")
    cols = int(np.ceil(np.sqrt(len(files))))
    rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    present = set()
    for ax, f in zip(axes, files):
        present |= _plot_axis(ax, rasterio.open(f).read(1), legend)
        # short label: grid id + sample
        ax.set_title(os.path.basename(f).split("_reviewer_")[-1][:22], fontsize=6)
    for ax in axes[len(files):]:
        ax.axis("off")
    handles = [Patch(facecolor=legend[c]["color"], edgecolor="k",
                     label=f"{c} {legend[c]['display_name']}")
               for c in sorted(legend) if c in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)),
               fontsize=8, title="class", bbox_to_anchor=(0.5, 0.08))
    _caption(fig, "A montage of the first several Random Forest classified land-cover rasters, one "
             "per panel, each colored by class code through the shared legend below. The panels give "
             "a quick overview of the range of land-cover and disturbance patterns across the "
             "interpreted cells. Scan the panels to compare class composition and spatial texture "
             "from cell to cell.", width=110)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"montage_{len(files)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", metavar="TIF", help="print metadata + histogram for one raster")
    ap.add_argument("--stats-all", action="store_true", help="class distribution across all rasters")
    ap.add_argument("--plot", metavar="TIF", help="save a labelled plot of one raster")
    ap.add_argument("--montage", type=int, metavar="N", help="save a montage of the first N rasters")
    ap.add_argument("--legend", default=LEGEND_CSV, help="legend CSV (default: data/reference/label_lookup.csv)")
    args = ap.parse_args()

    legend = load_legend(args.legend)

    if args.stats:
        stats_one(args.stats, legend)
    elif args.stats_all:
        stats_all(legend)
    elif args.plot:
        plot_one(args.plot, legend)
    elif args.montage:
        plot_montage(args.montage, legend)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
