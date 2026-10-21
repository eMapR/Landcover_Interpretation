"""Beaver example figures: for each reviewer, the interpreted cell with the largest connected beaver
patch, shown as a full-cell map plus a zoom on the patch. Exploratory; outputs go to the gitignored
reports/beaver_examples/ so they stay visible in VS Code without committing.

Run: python scripts/beaver_example_figure.py
"""

import glob
import importlib.util
import os
import re

import numpy as np
import rasterio
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = SC.out_path("reports/beaver_examples", ROOT)
ALREADY_SHOWN = ("10333", "41")                            # grid, sample already rendered separately
CONN = np.ones((3, 3), int)                                # 8-connectivity
NAME = {1: "Harvest", 2: "Development", 3: "Forest", 4: "Urban", 5: "Water", 6: "Agriculture",
        7: "Grass/Shrub", 8: "Wetland", 9: "Beaver", 10: "Insect/Disease"}


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, os.path.join(ROOT, "scripts", p))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


bmc = _load("bmc", "build_transfer_confusion.py")
C = _load("C", "compare_interpreted_vs_model.py")
C10 = C.load_mappings()[2]
CLUT = np.ones((11, 4))                                     # 0 = excluded -> white
for c, col in C10.items():
    CLUT[c] = to_rgb(col) + (1.0,)


def largest_beaver_patch(a):
    mask = a == 62                                          # ckit beaver
    if not mask.any():
        return 0, None
    lab, n = ndimage.label(mask, CONN)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return int(sizes.max()), (lab == (int(np.argmax(sizes)) + 1))


def _overlaps(fig, artists, box):
    """True if any artist's rendered extent intersects the box patch's extent (display coords)."""
    r = fig.canvas.get_renderer()
    bb = box.get_window_extent(r)
    return any(a.get_window_extent(r).overlaps(bb) for a in artists)


def render(path, rev, grid, samp, tgt):
    with rasterio.open(path) as s:
        a = s.read(1)
        pix_ha = SC.pixel_area_m2(s.transform, s.crs) / SC.M2_PER_HA
        res = abs(s.transform.a)                            # m per pixel; figure sizes below set in m
    _, patch = largest_beaver_patch(a)
    ref10 = bmc._REF_LUT[np.where((a >= 0) & (a <= 62), a, 0)]
    img = CLUT[ref10]
    H, W = a.shape
    rows, cols = np.where(patch)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    npx = int(patch.sum())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(img, interpolation="nearest")
    pad = round(60 / res)
    box = ax0.add_patch(Rectangle((c0 - pad, r0 - pad), (c1 - c0) + 2 * pad, (r1 - r0) + 2 * pad,
                                  fill=False, edgecolor="black", lw=2))
    ax0.set_title(f"Interpreted cell {grid} (sample {samp}, {rev}, {tgt})\n"
                  "black box = largest beaver patch", fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([])
    bar_px, bar_x0 = 1000 / res, round(100 / res)           # 1 km scale bar
    bar = ax0.add_patch(Rectangle((bar_x0, H - round(180 / res)), bar_px, max(1, round(50 / res)),
                                  facecolor="black", edgecolor="white", lw=0.5))
    lbl = ax0.text(bar_x0 + bar_px / 2, H - round(220 / res), "1 km", ha="center", va="bottom",
                   fontsize=8)

    z = round(450 / res)
    zr0, zr1, zc0, zc1 = max(r0 - z, 0), min(r1 + z, H), max(c0 - z, 0), min(c1 + z, W)
    ax1.imshow(img[zr0:zr1, zc0:zc1], interpolation="nearest")
    ax1.contour(patch[zr0:zr1, zc0:zc1].astype(float), levels=[0.5], colors="black", linewidths=1.8)
    ax1.set_title(f"Zoom on the largest beaver patch\n{npx} px (~{npx * pix_ha:.1f} ha)", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])

    present = sorted({int(c) for c in np.unique(ref10) if c > 0})
    handles = [Patch(facecolor=CLUT[c], edgecolor="0.4", label=NAME[c]) for c in present]
    handles += [plt.Line2D([], [], color="black", lw=1.8, label="largest beaver patch outline")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if _overlaps(fig, (bar, lbl), box):                     # patch box under the bar: move it bottom-right
        bar.set_x(W - 1 - bar_x0 - bar_px)
        lbl.set_x(W - 1 - bar_x0 - bar_px / 2)
        if _overlaps(fig, (bar, lbl), box):
            print(f"  WARN: scale bar overlaps the patch box in both corners ({rev} cell {grid})")
    out = f"{OUT}/beaver_example_{rev}_cell{grid}_sample{samp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, npx, pix_ha


def main():
    os.makedirs(OUT, exist_ok=True)
    # per reviewer: (max_patch_px, grid, sample, target, path)
    best = {}
    for f in sorted(glob.glob(os.path.join(ROOT, SC.RF_DIR, "**", SC.RF_GLOB), recursive=True)):
        m = re.search(r"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_.*target_(\d+)", os.path.basename(f), re.I)
        if not m:
            continue
        rev, grid, samp, tgt = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
        if (grid, samp) == ALREADY_SHOWN:                  # skip the one already rendered
            continue
        with rasterio.open(f) as s:
            mx, _ = largest_beaver_patch(s.read(1))
        if mx > 0 and mx > best.get(rev, (0,))[0]:
            best[rev] = (mx, grid, samp, tgt, f)

    print("largest beaver patch per reviewer (excluding the already-shown cell):")
    for rev in sorted(best):
        mx, grid, samp, tgt, f = best[rev]
        out, npx, pix_ha = render(f, rev, grid, samp, tgt)
        print(f"  {rev:<8} sample {samp:>4} grid {grid:>6} target {tgt}  patch {npx} px "
              f"(~{npx*pix_ha:.1f} ha)  -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
