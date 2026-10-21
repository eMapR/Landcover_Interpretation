"""Change-class example figures: for a given change class, the interpreted cells with the largest
connected patch of that class, one per reviewer where possible, filled to five with the largest
remaining patches. Each figure is a full-cell map plus a zoom on the patch. Exploratory; outputs go to
the gitignored reports/<class>_examples/ so they stay visible in VS Code without committing.

Run: python scripts/change_class_examples.py --class insect_disease
     python scripts/change_class_examples.py --class development
"""

import argparse
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
CONN = np.ones((3, 3), int)                                # 8-connectivity
# change class -> (ckit label id, display name)
CLASSES = {"harvest": (20, "Harvest"), "development": (30, "Development"),
           "insect_disease": (50, "Insect/Disease"), "beaver": (62, "Beaver")}
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


def largest_patch(a, ckit):
    mask = a == ckit
    if not mask.any():
        return 0, None
    lab, n = ndimage.label(mask, CONN)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return int(sizes.max()), (lab == (int(np.argmax(sizes)) + 1))


def render(path, ckit, disp, rev, grid, samp, tgt, out_dir):
    with rasterio.open(path) as s:
        a = s.read(1)
        pix_ha = SC.pixel_area_m2(s.transform, s.crs) / SC.M2_PER_HA
    _, patch = largest_patch(a, ckit)
    ref10 = bmc._REF_LUT[np.where((a >= 0) & (a <= 62), a, 0)]
    img = CLUT[ref10]
    H, W = a.shape
    rows, cols = np.where(patch)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    npx = int(patch.sum())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(img, interpolation="nearest")
    pad = 6
    ax0.add_patch(Rectangle((c0 - pad, r0 - pad), (c1 - c0) + 2 * pad, (r1 - r0) + 2 * pad,
                            fill=False, edgecolor="black", lw=2))
    ax0.set_title(f"Interpreted cell {grid} (sample {samp}, {rev}, {tgt})\n"
                  f"black box = largest {disp} patch", fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.add_patch(Rectangle((10, H - 18), 100, 5, facecolor="black", edgecolor="white", lw=0.5))
    ax0.text(60, H - 22, "1 km", ha="center", va="bottom", fontsize=8)

    z = 45
    zr0, zr1, zc0, zc1 = max(r0 - z, 0), min(r1 + z, H), max(c0 - z, 0), min(c1 + z, W)
    ax1.imshow(img[zr0:zr1, zc0:zc1], interpolation="nearest")
    ax1.contour(patch[zr0:zr1, zc0:zc1].astype(float), levels=[0.5], colors="black", linewidths=1.8)
    ax1.set_title(f"Zoom on the largest {disp} patch\n{npx} px (~{npx * pix_ha:.1f} ha)", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])

    present = sorted({int(c) for c in np.unique(ref10) if c > 0})
    handles = [Patch(facecolor=CLUT[c], edgecolor="0.4", label=NAME[c]) for c in present]
    handles += [plt.Line2D([], [], color="black", lw=1.8, label=f"largest {disp} patch outline")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    tag = disp.lower().replace("/", "_")
    out = f"{out_dir}/{tag}_example_{rev}_cell{grid}_sample{samp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, npx, pix_ha


def select(cells, k=5):
    """One per reviewer (largest) first, then fill with the next-largest cells, up to k."""
    cells.sort(reverse=True)
    chosen, used_rev, used_cell = [], set(), set()
    for c in cells:                                        # pass 1: reviewer diversity
        if len(chosen) >= k:
            break
        if c[1] not in used_rev:
            chosen.append(c); used_rev.add(c[1]); used_cell.add((c[2], c[3]))
    for c in cells:                                        # pass 2: fill with largest remaining
        if len(chosen) >= k:
            break
        if (c[2], c[3]) not in used_cell:
            chosen.append(c); used_cell.add((c[2], c[3]))
    return chosen[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True, choices=list(CLASSES))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ckit, disp = CLASSES[args.cls]
    out_dir = SC.out_path(f"reports/{args.cls}_examples", ROOT)
    os.makedirs(out_dir, exist_ok=True)

    cells, per_rev = [], {}
    for f in sorted(glob.glob(os.path.join(ROOT, SC.RF_DIR, "**", SC.RF_GLOB), recursive=True)):
        m = re.search(r"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_.*target_(\d+)", os.path.basename(f), re.I)
        if not m:
            continue
        rev = m.group(1).lower()
        per_rev.setdefault(rev, 0)
        with rasterio.open(f) as s:
            mx, _ = largest_patch(s.read(1), ckit)
        if mx > 0:
            per_rev[rev] += 1
            cells.append((mx, rev, m.group(2), m.group(3), m.group(4), f))

    print(f"{disp} (CKIT {ckit}): reviewers with the class -> "
          + ", ".join(f"{r}={per_rev[r]}" for r in sorted(per_rev) if per_rev[r]))
    print(f"reviewers without any {disp}: "
          + (", ".join(r for r in sorted(per_rev) if not per_rev[r]) or "none"))
    for mx, rev, grid, samp, tgt, f in select(cells, args.k):
        out, npx, pix_ha = render(f, ckit, disp, rev, grid, samp, tgt, out_dir)
        print(f"  {rev:<8} sample {samp:>4} grid {grid:>6} target {tgt}  patch {npx} px "
              f"(~{npx*pix_ha:.1f} ha)  -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
