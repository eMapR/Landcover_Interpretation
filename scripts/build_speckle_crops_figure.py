#!/usr/bin/env python3
"""Figure 2.9, the classified-map speckle crop, regenerated from the current 180-cell pipeline.

Panels of the same ground location (cell 31320), all pixel-aligned (337 x 337 at 10 m, EPSG:5070),
colored with the standard 10-class palette:
  base version    (figure_2_9_speckle_crops):          spectral baseline, then v2..v6
  with-reference  (figure_2_9_speckle_crops_with_ref): interpreted reference, spectral baseline, v2..v6

Data sources (all local, no Earth Engine):
  - Interpreted reference: data/raw/rf_class_maps/*_s2_31320/*.tif, the adjudicated "Interpreted (RF)"
    reference (see compare_interpreted_vs_model.py). Stored as CKIT class ids; remapped to the model's
    10 codes via CROSSWALK.
  - Spectral baseline (spec_all): data/raw/spectral_transferability_10class_percell/<bracket>/
    pred_specall_<bracket>_cell<id>.tif, already in 10-class codes.
  - Embeddings v2..v6: data/raw/transfer_predictions/<bracket>/pred_<bracket>_cell<id>.tif, bands 1..5.

Run: python scripts/build_speckle_crops_figure.py
Requires: rasterio, numpy, matplotlib
"""

import glob
import importlib.util
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
import sensor_config as SC  # RF_DIR, out_path (LCI_RF_DIR / LCI_OUT_DIR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = f"{ROOT}/data/raw/transfer_predictions"
SPEC_DIR = f"{ROOT}/data/raw/spectral_transferability_10class_percell"
RF_MAP_DIR = os.path.join(ROOT, SC.RF_DIR)
OUT = SC.out_path("manuscript_formatting/figures", ROOT)
CROP_CELL = "31320"
BRACKET = "2018_2020"
VBAND = {"v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5}      # transfer_predictions band order
NAME10 = {1: "Harvest", 2: "Development", 3: "Forest", 4: "Urban", 5: "Water",
          6: "Agriculture", 7: "Grass/Shrub", 8: "Wetland", 9: "Beaver", 10: "Insect/Disease"}
# CKIT reference id -> model code (for the interpreted reference raster)
CROSSWALK = {0: 4, 1: 6, 2: 7, 3: 3, 4: 5, 5: 8, 20: 1, 30: 2, 50: 10, 62: 9}
LUT = np.zeros(63, np.uint8)
for _k, _v in CROSSWALK.items():
    LUT[_k] = _v


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("C", "compare_interpreted_vs_model.py")
C10 = C.load_mappings()[2]                                  # canonical 10-class palette {code: color}
# color lookup: index 0 = nodata (white), 1..10 = class colors
CLUT = np.ones((11, 4))
for code, col in C10.items():
    CLUT[code] = to_rgb(col) + (1.0,)
CMAP = ListedColormap([CLUT[i] for i in range(11)])


def _interpreted_reference():
    """Adjudicated interpreted reference for the crop cell, CKIT ids remapped to the 10 model codes."""
    files = glob.glob(f"{RF_MAP_DIR}/*_s2_{CROP_CELL}/*.tif")
    if not files:
        raise SystemExit(f"STOP: interpreted reference for cell {CROP_CELL} not found under {RF_MAP_DIR}.")
    with rasterio.open(files[0]) as s:
        raw = s.read(1)
    return LUT[np.clip(raw, 0, 62)]


def _spectral_baseline():
    """spec_all spectral-baseline classification for the crop cell (already 10-class codes)."""
    files = glob.glob(f"{SPEC_DIR}/*/pred_specall_*cell{CROP_CELL}.tif")
    if not files:
        raise SystemExit(f"STOP: spec_all prediction for cell {CROP_CELL} not found under {SPEC_DIR}.")
    with rasterio.open(files[0]) as s:
        return s.read(1)


def _draw(panels, res, stem, title, caption):
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels) + 0.4, 5.4))
    for ax, (name, arr) in zip(axes, panels):
        ax.imshow(arr, cmap=CMAP, vmin=0, vmax=10, interpolation="nearest")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    # scale bar on the first panel: 1 km = 100 px at 10 m
    h, w = panels[0][1].shape
    km_px = 1000 / res
    y0 = h - 22
    axes[0].add_patch(Rectangle((12, y0), km_px, 6, facecolor="black", edgecolor="white", lw=0.6))
    axes[0].text(12 + km_px / 2, y0 - 6, "1 km", ha="center", va="bottom", fontsize=8, color="black")

    present = sorted({int(c) for _, arr in panels for c in np.unique(arr) if c > 0})
    handles = [Patch(facecolor=CLUT[c], edgecolor="0.4", label=NAME10[c]) for c in present]
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.32, wspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 0.16))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.015, "\n".join(textwrap.wrap(caption, 135)), ha="center", va="bottom",
             fontsize=9, color="0.3")

    png, pdf = f"{OUT}/{stem}.png", f"{OUT}/{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png} and {pdf}")


def main():
    os.makedirs(OUT, exist_ok=True)

    crop_f = glob.glob(f"{PRED_DIR}/*/pred_*_cell{CROP_CELL}.tif")
    if not crop_f:
        raise SystemExit(f"STOP: crop cell {CROP_CELL} not found in current predictions.")
    with rasterio.open(crop_f[0]) as s:
        bands = {v: s.read(b) for v, b in VBAND.items()}
        res = s.res[0]                                       # metres per pixel (10 m)

    spec = _spectral_baseline()
    ref = _interpreted_reference()
    emb_panels = [(v, bands[v]) for v in VBAND]

    # base version: spectral baseline + embeddings
    _draw([("spectral", spec)] + emb_panels, res, "figure_2_9_speckle_crops",
          f"Same Location Classified by the Spectral Baseline and Embedding Configurations (Cell {CROP_CELL})",
          f"The same NAIP grid cell (cell {CROP_CELL}) classified by the spectral baseline (spec_all) and by "
          "each embedding configuration, v2 through v6, with the standard land-cover class colors. The scale "
          "bar is 1 km. The dot-product configuration v6 fragments into salt-and-pepper speckle, while the "
          "spectral baseline and the baseline-preserving embedding configurations stay spatially coherent.")

    # with-reference version: interpreted reference + spectral baseline + embeddings
    _draw([("interpreted reference", ref), ("spectral", spec)] + emb_panels,
          res, "figure_2_9_speckle_crops_with_ref",
          f"Interpreted Reference, Spectral Baseline, and Embedding Configurations (Cell {CROP_CELL})",
          f"The adjudicated interpreted reference for cell {CROP_CELL} alongside the spectral baseline (spec_all) "
          "and each embedding configuration, v2 through v6, with the standard land-cover class colors. The scale "
          "bar is 1 km. Against the reference, the spectral baseline and the baseline-preserving embedding "
          "configurations stay spatially coherent, while the dot-product configuration v6 dissolves into speckle.")


if __name__ == "__main__":
    main()
