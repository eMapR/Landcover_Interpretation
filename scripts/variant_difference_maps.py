#!/usr/bin/env python3
"""Pairwise difference maps between the classified embedding-model variants (v2, v3, v4, v5).

v6 is excluded on purpose: its per-pixel dot-product speckle makes every comparison read as
noise. That leaves four smooth variants and six unordered pairs (v2_vs_v3 ... v4_vs_v5). By
convention the lower version number is always A and the higher is B.

Per pixel, in the 10-class model scheme (1=Harvest, 2=Development, 9=Beaver, 10=Insect/Disease
are change; 3-8 are stable), each pixel falls in one category:
  0 background     either map is 0 (no classified data)
  1 agree          same class in both, rendered in that class's color
  2 stable/stable  both stable but different classes, grey
  3 A-change/B-stable   A calls change where B calls stable
  4 B-change/A-stable   B calls change where A calls stable
  5 change/change  both change but different change classes
Categories 3 and 4 are kept distinct: "A sees disturbance B missed" and "B sees disturbance A
missed" are different findings and are not collapsed.

For each pair this writes, under reports/variant_difference_maps/<pair>/:
  - difference_map.png            full-map render, decimated (factor stated in the caption)
  - category_stats.csv            pixel count and area per category, full resolution
  - cat5_change_pairs.csv         change/change mismatches by directed change-class pair
  - cat3_Achange_by_class.csv     A-change/B-stable by the change class A assigned
  - cat4_Bchange_by_class.csv     B-change/A-stable by the change class B assigned
  - zoom_topNN_*.png              the 10 largest change-involved disagreement patches
  - overlay_topNN_*.png           the 10 largest such patches that intersect an interpreted cell

Scale: the mosaic is 72,577 x 48,386 at 10 m (two horizontal tiles). Stats are computed at full
resolution from windowed reads; only the overview render and the patch-labeling grid are
downsampled, at stated factors.

Requires: rasterio, numpy, pandas, scipy, matplotlib
"""

import argparse
import glob
import os
import random
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

warnings.filterwarnings("ignore")

MODEL_DIR = "data/raw/model_maps"
RF_DIR = SC.RF_DIR
CROSSWALK = "data/reference/class_crosswalk.csv"
LEGEND = "data/reference/model_maps_10class_legend.csv"
OUT_ROOT = SC.out_path("reports/variant_difference_maps")

VARIANTS = ["v2", "v3", "v4", "v5"]                     # v6 excluded (per-pixel speckle)
CHANGE_CODES = [1, 2, 9, 10]                            # harvest, development, beaver, insect/disease

RENDER_DS = 16                                          # full-map overview decimation (160 m)
LABEL_DS = 8                                            # patch-labeling grid (80 m); divides tile edge
BLOCK = 2048                                            # row-block height for streaming (mult. of both DS)
STRUCT = np.ones((3, 3), int)                           # 8-connectivity for connected components
# a labeling block counts toward a patch only if it is at least this dense in change-involved
# disagreement. change-involved disagreement is widespread (the variants disagree on change over
# large areas), so without a density floor 8-connectivity merges the whole diffuse field into one
# mosaic-spanning blob. this floor keeps "patches" to concentrated cores. stated in the caption.
MIN_BLOCK_PX = 16                                       # >= 25% of an 80 m (64 px) block
MAX_CROP = 2500                                         # cap on a crop dimension so reads stay bounded
N_OVERLAY = 5                                           # patch overlays and cell examples per pair

# category render codes and their colors; agree pixels keep the class code 1..10
CAT_GREY, CAT_A, CAT_B, CAT_CC = 11, 12, 13, 14
CAT_COLORS = {CAT_GREY: "#6e6e6e", CAT_A: "#ff00ff", CAT_B: "#00e0ff", CAT_CC: "#ff2a00"}
CAT_LABEL = {
    0: "background", 1: "agree", 2: "stable/stable mismatch",
    3: "A-change / B-stable", 4: "B-change / A-stable", 5: "change/change mismatch",
}


def _caption(fig, text, top=1.0, width=125):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


# ----------------------------------------------------------------------------- reference data
def load_legend():
    leg = pd.read_csv(LEGEND)
    names, colors = {}, {}
    for r in leg.itertuples():
        if int(r.code) > 0:
            names[int(r.code)] = r.display_name
            colors[int(r.code)] = r.color
    return names, colors


def load_crosswalk():
    cw = pd.read_csv(CROSSWALK)
    rf2common = {}
    for r in cw.itertuples():
        if pd.notna(r.model_code) and pd.notna(r.rf_code) and int(r.model_code) > 0:
            rf2common[int(r.rf_code)] = int(r.model_code)
    return rf2common


def _mute(color, frac=0.55):
    # blend a class color toward white so agree areas read as a faint landscape and the
    # saturated disagreement colors stand out on top
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


# change-focused recolor: stable classes read as one gray, only change carries color
CHANGE_GRAY = "#8a8a8a"                                 # all stable agree + stable/stable mismatch
WATER_GRAY = "#565656"                                  # water is a slightly darker gray


def build_cmaps(names, colors):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # difference-map colormap: 0 black, 1..10 muted class colors, 11..14 disagreement categories
    diff_cols = ["#000000"] + [_mute(colors[c]) for c in range(1, 11)] + \
                [CAT_COLORS[c] for c in (CAT_GREY, CAT_A, CAT_B, CAT_CC)]
    diff_cmap = ListedColormap(diff_cols)
    diff_norm = BoundaryNorm(np.arange(-0.5, 15.5), diff_cmap.N)
    # class colormap for the A|B|interpreted panels: 0 white, 1..10 full class colors
    cls_cmap = ListedColormap(["#ffffff"] + [colors[c] for c in range(1, 11)])
    cls_norm = BoundaryNorm(np.arange(-0.5, 11.5), cls_cmap.N)
    return diff_cmap, diff_norm, cls_cmap, cls_norm


def build_change_cmap(colors):
    """Change-focused colormap over the same render codes: stable -> gray (water darker), only the
    agreed change classes and the change-involved disagreement categories keep color."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # codes: 0 bg, 1..10 dominant agree class, 11 stable/stable, 12 A-change, 13 B-change, 14 chg/chg
    agree = []
    for c in range(1, 11):
        if c in CHANGE_CODES:                          # agreed change class -> its saturated color
            agree.append(colors[c])
        elif c == 5:                                   # water -> darker gray
            agree.append(WATER_GRAY)
        else:                                          # every other stable class -> medium gray
            agree.append(CHANGE_GRAY)
    cols = ["#000000"] + agree + [CHANGE_GRAY,         # 11 stable/stable mismatch -> gray too
                                  CAT_COLORS[CAT_A], CAT_COLORS[CAT_B], CAT_COLORS[CAT_CC]]
    cmap = ListedColormap(cols)
    return cmap, BoundaryNorm(np.arange(-0.5, 15.5), cmap.N)


# ----------------------------------------------------------------------------- mosaic i/o
def open_tiles(version):
    """Return [(dataset, col_offset, width)] for a variant, ordered left to right."""
    folder = os.path.join(MODEL_DIR, f"classified_maps_10class_{version}")
    paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not paths:
        raise SystemExit(f"no tiles in {folder}")
    tiles, off = [], 0
    for p in paths:
        ds = rasterio.open(p)
        tiles.append((ds, off, ds.width))
        off += ds.width
    return tiles


def mosaic_pixel_m2(tiles):
    """Area (m^2) of one mosaic pixel, from the first tile's transform and CRS."""
    ds = tiles[0][0]
    return SC.pixel_area_m2(ds.transform, ds.crs)


def mosaic_dims(tiles):
    w = sum(t[2] for t in tiles)
    h = tiles[0][0].height
    return h, w


def read_window(tiles, r0, r1, c0, c1):
    """Stitch a window [r0:r1, c0:c1] of the mosaic across its horizontal tiles."""
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    for ds, off, w in tiles:
        lo, hi = max(c0, off), min(c1, off + w)
        if lo >= hi:
            continue
        arr = ds.read(1, window=((r0, r1), (lo - off, hi - off)))
        out[:, lo - c0:hi - c0] = arr
    return out


# ----------------------------------------------------------------------------- categorization
def categorize(A, B, ischange):
    """Return (rendercode, masks) for aligned variant windows A (lower) and B (higher)."""
    valid = (A > 0) & (B > 0)
    agree = valid & (A == B)
    dis = valid & (A != B)
    achg = ischange[A]
    bchg = ischange[B]
    cat2 = dis & ~achg & ~bchg                          # both stable, different
    cat3 = dis & achg & ~bchg                           # A change, B stable
    cat4 = dis & ~achg & bchg                           # B change, A stable
    cat5 = dis & achg & bchg                            # both change, different
    rc = np.zeros(A.shape, dtype=np.uint8)
    rc[agree] = A[agree]
    rc[cat2] = CAT_GREY
    rc[cat3] = CAT_A
    rc[cat4] = CAT_B
    rc[cat5] = CAT_CC
    return rc, dict(valid=valid, agree=agree, cat2=cat2, cat3=cat3, cat4=cat4, cat5=cat5)


def _block_sum(mask, ds):
    # sum a boolean mask over ds x ds blocks; pad the ragged right and bottom edges with zeros so
    # the reshape is exact, then reduce. returns per-block counts (0..ds*ds)
    h, w = mask.shape
    ph, pw = (-h) % ds, (-w) % ds
    if ph or pw:
        mask = np.pad(mask, ((0, ph), (0, pw)))
    hb, wb = mask.shape[0] // ds, mask.shape[1] // ds
    return mask.reshape(hb, ds, wb, ds).sum(axis=(1, 3)).astype(np.uint16)


def stream_pair(tilesA, tilesB, H, W, max_rows=None):
    """Single streaming pass over a pair. Returns render grid, label-count grid, and stats."""
    ischange = np.zeros(256, dtype=bool)
    ischange[CHANGE_CODES] = True

    rH, rW = -(-H // RENDER_DS), -(-W // RENDER_DS)      # ceil division
    lH, lW = -(-H // LABEL_DS), -(-W // LABEL_DS)
    # per 160 m render block, count each of the 10 agree classes and the 4 disagreement categories,
    # so the block can be colored by what actually dominates it rather than by priority
    blockcnt = np.zeros((rH, rW, 14), dtype=np.uint16)   # 0..9 agree class 1..10, 10..13 cat2..cat5
    labelcount = np.zeros((lH, lW), dtype=np.uint32)     # exact cat3/4/5 pixels per 80 m block

    cat_px = np.zeros(6, dtype=np.int64)                 # categories 0..5
    cat5_pair = np.zeros((11, 11), dtype=np.int64)       # directed (A_class, B_class)
    cat3_cls = np.zeros(11, dtype=np.int64)              # by A change class
    cat4_cls = np.zeros(11, dtype=np.int64)              # by B change class

    row_end = min(H, max_rows) if max_rows else H
    for r0 in range(0, row_end, BLOCK):
        r1 = min(r0 + BLOCK, row_end)
        A = read_window(tilesA, r0, r1, 0, W)
        B = read_window(tilesB, r0, r1, 0, W)
        rc, m = categorize(A, B, ischange)

        # full-resolution category counts
        cat_px[0] += (~m["valid"]).sum()
        cat_px[1] += m["agree"].sum()
        cat_px[2] += m["cat2"].sum()
        cat_px[3] += m["cat3"].sum()
        cat_px[4] += m["cat4"].sum()
        cat_px[5] += m["cat5"].sum()

        # breakdowns
        if m["cat5"].any():
            a5, b5 = A[m["cat5"]], B[m["cat5"]]
            np.add.at(cat5_pair, (a5, b5), 1)
        if m["cat3"].any():
            np.add.at(cat3_cls, A[m["cat3"]], 1)
        if m["cat4"].any():
            np.add.at(cat4_cls, B[m["cat4"]], 1)

        # overview render: accumulate per-block class and category counts (composition, not
        # priority) so each 160 m block can later be colored by its dominant content
        rr = r0 // RENDER_DS
        for k in range(1, 11):                          # agree, split by the agreed class
            am = m["agree"] & (A == k)
            if am.any():
                bc = _block_sum(am, RENDER_DS)
                blockcnt[rr:rr + bc.shape[0], :bc.shape[1], k - 1] += bc
        for j, cat in enumerate(("cat2", "cat3", "cat4", "cat5")):
            if m[cat].any():
                bc = _block_sum(m[cat], RENDER_DS)
                blockcnt[rr:rr + bc.shape[0], :bc.shape[1], 10 + j] += bc

        # labeling grid: exact cat3/4/5 pixel count per 80 m block
        ci = m["cat3"] | m["cat4"] | m["cat5"]
        if ci.any():
            lr, lc = np.nonzero(ci)
            np.add.at(labelcount, ((r0 + lr) // LABEL_DS, lc // LABEL_DS), 1)

        print(f"    rows {r0:>6}-{r1:<6} of {row_end}", flush=True)

    # color each block by what dominates it: compare total agree vs total disagreement pixels; if
    # disagreement wins, use the leading disagreement category, else use the dominant agree class.
    # blocks with no valid pixels stay background. no priority bias, so the map reflects the true
    # per-block composition
    agree_tot = blockcnt[:, :, :10].sum(2)
    dis_tot = blockcnt[:, :, 10:].sum(2)
    valid = agree_tot + dis_tot
    render = np.zeros((rH, rW), dtype=np.uint8)          # 0 background
    dom_agree = blockcnt[:, :, :10].argmax(2).astype(np.uint8) + 1        # 1..10
    dom_dis = blockcnt[:, :, 10:].argmax(2).astype(np.uint8)             # 0..3 -> codes 11..14
    dis_wins = (valid > 0) & (dis_tot > agree_tot)
    agr_wins = (valid > 0) & ~dis_wins
    render[agr_wins] = dom_agree[agr_wins]
    render[dis_wins] = CAT_GREY + dom_dis[dis_wins]      # CAT_GREY=11, so 11..14

    stats = dict(cat_px=cat_px, cat5_pair=cat5_pair, cat3_cls=cat3_cls, cat4_cls=cat4_cls)
    return render, labelcount, stats


# ----------------------------------------------------------------------------- outputs
def write_stats(out_dir, stats, names, A_name, B_name, pix_m2):
    pix_ha, pix_km2 = pix_m2 / SC.M2_PER_HA, pix_m2 / SC.M2_PER_KM2
    cat_px = stats["cat_px"]
    valid = cat_px[1:].sum()
    rows = []
    for c in range(6):
        px = int(cat_px[c])
        rows.append(dict(category=c, label=CAT_LABEL[c].replace("A", A_name).replace("B", B_name)
                         if c in (3, 4) else CAT_LABEL[c],
                         pixels=px, area_ha=round(px * pix_ha, 2),
                         area_km2=round(px * pix_km2, 4),
                         pct_of_valid=round(100 * px / valid, 4) if valid and c > 0 else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "category_stats.csv"), index=False)

    # cat5 directed change-class pairs
    p = stats["cat5_pair"]
    r5 = [dict(A_class=names[a], B_class=names[b], pixels=int(p[a, b]),
               area_ha=round(int(p[a, b]) * pix_ha, 2))
          for a in CHANGE_CODES for b in CHANGE_CODES if a != b and p[a, b] > 0]
    pd.DataFrame(sorted(r5, key=lambda d: -d["pixels"])).to_csv(
        os.path.join(out_dir, "cat5_change_pairs.csv"), index=False)

    # cat3 / cat4 by detecting variant's change class
    for arr, col, fn in [(stats["cat3_cls"], A_name, "cat3_Achange_by_class.csv"),
                         (stats["cat4_cls"], B_name, "cat4_Bchange_by_class.csv")]:
        rr = [dict(change_class=names[c], detected_by=col, pixels=int(arr[c]),
                   area_ha=round(int(arr[c]) * pix_ha, 2))
              for c in CHANGE_CODES if arr[c] > 0]
        pd.DataFrame(sorted(rr, key=lambda d: -d["pixels"])).to_csv(
            os.path.join(out_dir, fn), index=False)
    return cat_px


def render_full_map(out_dir, render, diff_cmap, diff_norm, names, colors, A_name, B_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(13, 9.6))
    ax.imshow(render, cmap=diff_cmap, norm=diff_norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    # disagreement categories (saturated) as an in-map legend
    dis_handles = [
        Patch(facecolor=CAT_COLORS[CAT_A], edgecolor="k", label=f"{A_name} change / {B_name} stable"),
        Patch(facecolor=CAT_COLORS[CAT_B], edgecolor="k", label=f"{B_name} change / {A_name} stable"),
        Patch(facecolor=CAT_COLORS[CAT_CC], edgecolor="k", label="change / change mismatch"),
        Patch(facecolor=CAT_COLORS[CAT_GREY], edgecolor="k", label="stable / stable mismatch"),
    ]
    leg = ax.legend(handles=dis_handles, loc="lower left", fontsize=8, framealpha=0.9,
                    title="disagreement (saturated)")
    leg.get_title().set_fontsize(8)
    ax.add_artist(leg)
    # agree blocks (muted) are colored by the agreed class; give every class its own swatch below
    agree_handles = [Patch(facecolor=_mute(colors[c]), edgecolor="0.5", label=names[c])
                     for c in range(1, 11)]
    fig.legend(handles=agree_handles, loc="lower center", ncol=10, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.09),
               title="agree (muted): both variants assign this land-cover class")
    ax.set_title(f"{A_name} vs {B_name}  difference map  (each {RENDER_DS*10} m block colored by "
                 f"its majority: agree vs disagreement, then the leading category; no priority bias. "
                 f"stats at full 10 m resolution)", fontsize=9)
    _caption(fig, f"Full-mosaic overview comparing the {A_name} and {B_name} classified maps, with "
             f"each {RENDER_DS*10} m block colored by its dominant content: muted land-cover colors "
             f"where the two variants agree on the class, and saturated colors where they disagree. "
             f"The saturated categories separate {A_name}-calls-change / {B_name}-stable, "
             f"{B_name}-calls-change / {A_name}-stable, change / change mismatch, and stable / stable "
             f"mismatch, as listed in the legends. Scan for saturated regions to find where the two "
             f"variants diverge on disturbance, keeping in mind the map is decimated while the "
             f"category statistics are computed at full 10 m resolution.", top=1.0)
    fig.savefig(os.path.join(out_dir, "difference_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    render_change_map(out_dir, render, names, colors, A_name, B_name)


def render_change_map(out_dir, render, names, colors, A_name, B_name):
    """Change-focused recolor of the same block render: stable is gray (water darker), color only
    for the agreed change classes and the change-involved disagreement categories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cmap, norm = build_change_cmap(colors)
    fig, ax = plt.subplots(figsize=(13, 9.6))
    ax.imshow(render, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    dis_handles = [
        Patch(facecolor=CAT_COLORS[CAT_A], edgecolor="k", label=f"{A_name} change / {B_name} stable"),
        Patch(facecolor=CAT_COLORS[CAT_B], edgecolor="k", label=f"{B_name} change / {A_name} stable"),
        Patch(facecolor=CAT_COLORS[CAT_CC], edgecolor="k", label="change / change mismatch"),
        Patch(facecolor=CHANGE_GRAY, edgecolor="k", label="stable (agree or mismatch)"),
        Patch(facecolor=WATER_GRAY, edgecolor="k", label="water"),
    ]
    leg = ax.legend(handles=dis_handles, loc="lower left", fontsize=8, framealpha=0.9,
                    title="change-involved disagreement (stable = gray)")
    leg.get_title().set_fontsize(8)
    ax.add_artist(leg)
    # the agreed change classes keep their own color; give each a swatch below
    agree_handles = [Patch(facecolor=colors[c], edgecolor="0.4", label=names[c]) for c in CHANGE_CODES]
    fig.legend(handles=agree_handles, loc="lower center", ncol=len(CHANGE_CODES), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.09),
               title="agreed change class (both variants assign the same change class)")
    ax.set_title(f"{A_name} vs {B_name}  change-focused difference map  (stable in gray, water "
                 f"darker; color only for agreed change and change disagreement; {RENDER_DS*10} m "
                 f"blocks)", fontsize=9)
    _caption(fig, f"Change-focused recolor of the same {A_name} versus {B_name} block render: all "
             f"stable land cover is drawn in gray, with water a darker gray, so only the agreed change "
             f"classes and the change-involved disagreement categories carry saturated color. The "
             f"legend separates {A_name}-change / {B_name}-stable, {B_name}-change / {A_name}-stable, "
             f"and change / change mismatch from the agreed change classes. Use this view to isolate "
             f"where the two variants disagree about disturbance against a muted stable background, at "
             f"{RENDER_DS*10} m block resolution.", top=1.0)
    fig.savefig(os.path.join(out_dir, "difference_map_change.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def label_patches(labelcount, pix_ha):
    """8-connected components of the change-involved disagreement, with exact areas.

    Labeling is done on the 80 m grid, but each block carries the exact number of full-res
    cat3/4/5 pixels it contains, so a component's summed count is its exact full-resolution
    area. Bounding boxes are returned in full-resolution pixel coordinates.
    """
    lab, n = ndimage.label(labelcount >= MIN_BLOCK_PX, structure=STRUCT)
    if n == 0:
        return []
    areas = ndimage.sum(labelcount, lab, index=np.arange(1, n + 1))   # exact cat3/4/5 pixels
    slices = ndimage.find_objects(lab)
    patches = []
    for i, sl in enumerate(slices):
        sy, sx = sl
        patches.append(dict(
            pid=i + 1, area_px=int(areas[i]), area_ha=round(int(areas[i]) * pix_ha, 2),
            r0=sy.start * LABEL_DS, r1=sy.stop * LABEL_DS,
            c0=sx.start * LABEL_DS, c1=sx.stop * LABEL_DS))
    patches.sort(key=lambda d: -d["area_px"])
    return patches


def crop_window(patch, H, W, margin=120):
    r0 = max(0, patch["r0"] - margin); r1 = min(H, patch["r1"] + margin)
    c0 = max(0, patch["c0"] - margin); c1 = min(W, patch["c1"] + margin)
    # cap the extent so a sprawling patch never triggers a multi-gigabyte read; large patches are
    # shown centred on their bounding box at the cap size
    if r1 - r0 > MAX_CROP:
        cr = (patch["r0"] + patch["r1"]) // 2
        r0 = max(0, cr - MAX_CROP // 2); r1 = min(H, r0 + MAX_CROP)
    if c1 - c0 > MAX_CROP:
        cc = (patch["c0"] + patch["c1"]) // 2
        c0 = max(0, cc - MAX_CROP // 2); c1 = min(W, c0 + MAX_CROP)
    return r0, r1, c0, c1


def render_zoom(out_dir, rank, patch, tilesA, tilesB, H, W, diff_cmap, diff_norm, names, colors,
                A_name, B_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ischange = np.zeros(256, dtype=bool); ischange[CHANGE_CODES] = True
    r0, r1, c0, c1 = crop_window(patch, H, W)
    A = read_window(tilesA, r0, r1, c0, c1)
    B = read_window(tilesB, r0, r1, c0, c1)
    rc, _ = categorize(A, B, ischange)

    fig, ax = plt.subplots(figsize=(7, 8.4))
    ax.imshow(rc, cmap=diff_cmap, norm=diff_norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"#{rank}   {A_name} vs {B_name}   {patch['area_ha']:,} ha change-involved "
                 f"disagreement patch\none of the pair's 10 largest patches where the variants "
                 f"disagree and at least one calls change; full 10 m resolution, "
                 f"{(c1 - c0) * 10 / 1000:.0f} x {(r1 - r0) * 10 / 1000:.0f} km crop", fontsize=8.5)
    # same two-part legend as the overview: saturated disagreement categories, muted agree classes
    dis_handles = [
        Patch(facecolor=CAT_COLORS[CAT_A], edgecolor="k", label=f"{A_name} change / {B_name} stable"),
        Patch(facecolor=CAT_COLORS[CAT_B], edgecolor="k", label=f"{B_name} change / {A_name} stable"),
        Patch(facecolor=CAT_COLORS[CAT_CC], edgecolor="k", label="change / change mismatch"),
        Patch(facecolor=CAT_COLORS[CAT_GREY], edgecolor="k", label="stable / stable mismatch"),
    ]
    agree_handles = [Patch(facecolor=_mute(colors[c]), edgecolor="0.5", label=names[c])
                     for c in range(1, 11)]
    fig.legend(handles=dis_handles, loc="lower center", bbox_to_anchor=(0.5, 0.15), ncol=2,
               fontsize=7.5, frameon=False, title="disagreement (saturated)")
    fig.legend(handles=agree_handles, loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=5,
               fontsize=7, frameon=False, title="agree (muted): both variants assign this class")
    _caption(fig, f"Full 10 m resolution crop of one of the ten largest patches where the {A_name} "
             f"and {B_name} maps disagree and at least one calls change, ranked #{rank} at "
             f"{patch['area_ha']:,} ha. Pixels are colored with the same scheme as the overview: "
             f"muted where the two variants agree on the class, and saturated for the four "
             f"disagreement categories in the legend. Read the saturated colors to see which variant "
             f"is calling disturbance and which is calling stable inside this patch.", top=0.95)
    fig.savefig(os.path.join(out_dir, f"zoom_top{rank:02d}_{patch['area_px']}px.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_overlay(out_dir, rank, patch, tilesA, tilesB, H, W, cells, rf2common,
                   cls_cmap, cls_norm, names, colors, A_name, B_name, mosaic_tf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    r0, r1, c0, c1 = crop_window(patch, H, W)
    A = read_window(tilesA, r0, r1, c0, c1)
    B = read_window(tilesB, r0, r1, c0, c1)

    # interpreted cells share the mosaic's 10 m EPSG:5070 grid exactly, so place them by direct
    # index arithmetic (no reproject, which would extrapolate the edge value across the window)
    interp = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    ox, oy = mosaic_tf.c, mosaic_tf.f
    for cell in cells:
        b = cell["bounds"]
        cc0 = round((b[0] - ox) / 10.0)                 # cell's column offset in the mosaic
        cr0 = round((oy - b[3]) / 10.0)                 # cell's row offset (b[3] = top)
        with rasterio.open(cell["path"]) as ds:
            ch, cw = ds.height, ds.width
            orr0, orr1 = max(r0, cr0), min(r1, cr0 + ch)
            occ0, occ1 = max(c0, cc0), min(c1, cc0 + cw)
            if orr0 >= orr1 or occ0 >= occ1:
                continue
            sub = ds.read(1, window=((orr0 - cr0, orr1 - cr0), (occ0 - cc0, occ1 - cc0)))
        common = np.zeros_like(sub)
        for rf_code, cc in rf2common.items():
            common[sub == rf_code] = cc
        dst = interp[orr0 - r0:orr1 - r0, occ0 - c0:occ1 - c0]
        interp[orr0 - r0:orr1 - r0, occ0 - c0:occ1 - c0] = np.where(common > 0, common, dst)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, arr, t in [(axes[0], A, f"variant {A_name}"), (axes[1], B, f"variant {B_name}"),
                       (axes[2], interp, "interpreted reference")]:
        ax.imshow(arr, cmap=cls_cmap, norm=cls_norm, interpolation="nearest")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=colors[c], edgecolor="k", label=f"{c} {names[c]}")
               for c in range(1, 11)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, bbox_to_anchor=(0.5, 0.09))
    fig.suptitle(f"#{rank}  {patch['area_ha']} ha change-involved disagreement intersecting an "
                 f"interpreted cell", fontsize=10)
    _caption(fig, f"Three panels show the same crop for variant {A_name}, variant {B_name}, and the "
             f"interpreted reference, all in the shared 10-class color scheme, for a "
             f"{patch['area_ha']} ha change-involved disagreement patch that overlaps an interpreted "
             f"cell. The interpreted panel is only populated where an interpreted cell falls in the "
             f"crop, so the surrounding area is blank. Compare the two variant panels against the "
             f"interpreted reference to judge which variant is closer to the human interpretation "
             f"where they disagree.", top=0.93)
    fig.savefig(os.path.join(out_dir, f"overlay_top{rank:02d}_{patch['area_px']}px.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def cell_extent(cell, mosaic_tf):
    # cell's pixel window in the mosaic; the interpreted rasters share the 10 m grid exactly
    c0 = round((cell["bounds"][0] - mosaic_tf.c) / 10.0)
    r0 = round((mosaic_tf.f - cell["bounds"][3]) / 10.0)
    return r0, r0 + cell["height"], c0, c0 + cell["width"]


def select_cell_examples(tilesA, tilesB, cells, mosaic_tf, n=5):
    """Rank interpreted cells by the change-involved A/B disagreement inside the cell, top n.

    These become the cell-sized 3-panel examples, so the interpreted reference fills its panel
    instead of sitting as a speck inside a much larger patch crop.
    """
    ischange = np.zeros(256, dtype=bool); ischange[CHANGE_CODES] = True
    scored = []
    for cell in cells:
        r0, r1, c0, c1 = cell_extent(cell, mosaic_tf)
        A = read_window(tilesA, r0, r1, c0, c1)
        B = read_window(tilesB, r0, r1, c0, c1)
        _, m = categorize(A, B, ischange)
        score = int((m["cat3"] | m["cat4"] | m["cat5"]).sum())
        scored.append((score, cell))
    scored.sort(key=lambda s: -s[0])
    return scored[:n]


def render_cell_overlay(out_dir, rank, score, cell, tilesA, tilesB, mosaic_tf, rf2common,
                        cls_cmap, cls_norm, names, colors, A_name, B_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pix_ha = mosaic_pixel_m2(tilesA) / SC.M2_PER_HA
    r0, r1, c0, c1 = cell_extent(cell, mosaic_tf)
    A = read_window(tilesA, r0, r1, c0, c1)
    B = read_window(tilesB, r0, r1, c0, c1)
    with rasterio.open(cell["path"]) as ds:
        rf = ds.read(1)
    interp = np.zeros_like(rf, dtype=np.uint8)
    for rf_code, cc in rf2common.items():
        interp[rf == rf_code] = cc

    gid = re.search(r"grid_(\d+)", cell["name"]).group(1)
    km = cell["width"] * 10 / 1000
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, arr, t in [(axes[0], A, f"variant {A_name}"), (axes[1], B, f"variant {B_name}"),
                       (axes[2], interp, "interpreted reference")]:
        ax.imshow(arr, cmap=cls_cmap, norm=cls_norm, interpolation="nearest")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=colors[c], edgecolor="k", label=f"{c} {names[c]}")
               for c in range(1, 11)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, bbox_to_anchor=(0.5, 0.09))
    fig.suptitle(f"interpreted cell {gid}  ({km:.1f} x {km:.1f} km, same extent in all three "
                 f"panels)  ·  {score * pix_ha:.0f} ha change-involved {A_name}/{B_name} "
                 f"disagreement in the cell", fontsize=10)
    _caption(fig, f"Three panels show one whole interpreted cell at the same extent for variant "
             f"{A_name}, variant {B_name}, and the interpreted reference, in the shared 10-class "
             f"color scheme. This cell was selected for holding a large amount of change-involved "
             f"{A_name} versus {B_name} disagreement, so the interpreted reference fills its panel "
             f"rather than sitting as a speck in a larger crop. Compare the two variant panels to the "
             f"interpreted panel to see which variant matches the human interpretation where the two "
             f"variants disagree on disturbance.", top=0.93)
    fig.savefig(os.path.join(out_dir, f"overlay_cell{rank:02d}_grid{gid}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _interp_from_cell(cell, rf2common):
    with rasterio.open(cell["path"]) as ds:
        rf = ds.read(1)
    interp = np.zeros_like(rf, dtype=np.uint8)
    for rf_code, cc in rf2common.items():
        interp[rf == rf_code] = cc
    return interp


def build_cell_panels(cells, rf2common, cls_cmap, cls_norm, names, colors, n=20):
    """One figure per selected interpreted cell: all five variants plus the interpreted reference,
    cropped to the cell extent. Cells are ranked by how much the four smooth variants (v2-v5)
    disagree inside the cell (v6 is excluded from the ranking since its per-pixel speckle disagrees
    almost everywhere, but it is shown in the panels). No streaming; only windowed reads.
    """
    allv = VARIANTS + ["v6"]
    tiles = {v: open_tiles(v) for v in allv}
    mosaic_tf = tiles["v2"][0][0].transform
    out_dir = os.path.join(OUT_ROOT, "cell_all_variants")
    os.makedirs(out_dir, exist_ok=True)

    scored = []
    for cell in cells:
        r0, r1, c0, c1 = cell_extent(cell, mosaic_tf)
        smooth = np.stack([read_window(tiles[v], r0, r1, c0, c1) for v in VARIANTS])
        valid = (smooth >= 1).all(0) & (smooth <= 10).all(0)
        disagree = valid & ~(smooth == smooth[0]).all(0)      # v2-v5 not unanimous
        scored.append((int(disagree.sum()), cell))
    scored.sort(key=lambda s: -s[0])

    for rank, (score, cell) in enumerate(scored[:n], 1):
        render_cell_panel(out_dir, rank, score, cell, tiles, allv, mosaic_tf, rf2common,
                          cls_cmap, cls_norm, names, colors)
    for v in allv:
        for ds, _, _ in tiles[v]:
            ds.close()
    print(f"wrote {out_dir}/  ({min(n, len(scored))} cells, 6-panel each)")


def render_cell_panel(out_dir, rank, score, cell, tiles, allv, mosaic_tf, rf2common,
                      cls_cmap, cls_norm, names, colors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pix_ha = mosaic_pixel_m2(tiles[allv[0]]) / SC.M2_PER_HA
    r0, r1, c0, c1 = cell_extent(cell, mosaic_tf)
    panels = [(v, read_window(tiles[v], r0, r1, c0, c1)) for v in allv]
    panels.append(("interpreted reference", _interp_from_cell(cell, rf2common)))
    gid = re.search(r"grid_(\d+)", cell["name"]).group(1)
    km = cell["width"] * 10 / 1000

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6))
    for ax, (label, arr) in zip(axes.ravel(), panels):
        # aspect="auto" so each image fills its cell rather than overflowing into the row above's
        # title; keeps the panel titles clear of the images
        ax.imshow(arr, cmap=cls_cmap, norm=cls_norm, interpolation="nearest", aspect="auto")
        ax.set_title(label, fontsize=11, pad=6); ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=colors[c], edgecolor="k", label=f"{c} {names[c]}")
               for c in range(1, 11)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=8, bbox_to_anchor=(0.5, 0.08))
    fig.suptitle(f"interpreted cell {gid}  ({km:.1f} x {km:.1f} km, same extent in every panel)  "
                 f"·  all five variants vs the interpreted reference  ·  ranked #{rank} by v2-v5 "
                 f"in-cell disagreement ({score * pix_ha:.0f} ha)", fontsize=11)
    _caption(fig, "Six panels show one interpreted cell at the same extent: the five embedding "
             "variants v2 through v6 and the interpreted reference, all in the shared 10-class color "
             "scheme. The cell is ranked by how much the four smooth variants v2 to v5 disagree "
             "inside it, and v6 is shown but excluded from the ranking since its per-pixel speckle "
             "disagrees almost everywhere. Compare each variant panel against the interpreted "
             "reference to see which variants match the human interpretation and how the v6 speckle "
             "differs from the smoother variants.", top=0.94)
    fig.savefig(os.path.join(out_dir, f"cell{rank:02d}_grid{gid}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- interpreted cells
def location_key(path):
    m = re.search(rf"grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)", os.path.basename(path))
    return (m.group(1), m.group(2), m.group(3)) if m else (os.path.basename(path),)


def deduped_cells(seed=42):
    paths = sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True))
    groups = defaultdict(list)
    for p in paths:
        groups[location_key(p)].append(p)
    rng = random.Random(seed)
    kept = []
    for k in sorted(groups):
        v = sorted(groups[k])
        kept.append(rng.choice(v) if len(v) > 1 else v[0])
    cells = []
    for p in sorted(kept):
        with rasterio.open(p) as ds:
            cells.append(dict(path=p, name=os.path.splitext(os.path.basename(p))[0],
                              bounds=tuple(ds.bounds), width=ds.width, height=ds.height))
    return cells


def patch_bounds_5070(patch, mosaic_tf):
    left = mosaic_tf.c + patch["c0"] * 10.0
    right = mosaic_tf.c + patch["c1"] * 10.0
    top = mosaic_tf.f - patch["r0"] * 10.0
    bot = mosaic_tf.f - patch["r1"] * 10.0
    return left, bot, right, top


def intersects_any_cell(patch, cells, mosaic_tf):
    pl, pb, pr, pt = patch_bounds_5070(patch, mosaic_tf)
    for c in cells:
        b = c["bounds"]
        if not (b[2] <= pl or b[0] >= pr or b[3] <= pb or b[1] >= pt):
            return True
    return False


# ----------------------------------------------------------------------------- driver
def run_pair(A_name, B_name, cells, rf2common, names, colors, cmaps, max_rows, n_top):
    diff_cmap, diff_norm, cls_cmap, cls_norm = cmaps
    out_dir = os.path.join(OUT_ROOT, f"{A_name}_vs_{B_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {A_name} vs {B_name} ===", flush=True)

    tilesA = open_tiles(A_name)
    tilesB = open_tiles(B_name)
    H, W = mosaic_dims(tilesA)
    mosaic_tf = tilesA[0][0].transform
    pix_m2 = mosaic_pixel_m2(tilesA)
    pix_ha = pix_m2 / SC.M2_PER_HA

    render, labelcount, stats = stream_pair(tilesA, tilesB, H, W, max_rows=max_rows)
    np.save(os.path.join(out_dir, "render_grid.npy"), render)   # cache so map appearance can be
    cat_px = write_stats(out_dir, stats, names, A_name, B_name, pix_m2)  # re-rendered without re-streaming
    render_full_map(out_dir, render, diff_cmap, diff_norm, names, colors, A_name, B_name)

    patches = label_patches(labelcount, pix_ha)
    print(f"  change-involved disagreement patches: {len(patches)}", flush=True)

    # top-N largest patches overall -> zoomed crops
    top = patches[:n_top]
    for rank, p in enumerate(top, 1):
        render_zoom(out_dir, rank, p, tilesA, tilesB, H, W, diff_cmap, diff_norm, names, colors,
                    A_name, B_name)

    # how many of the overall top-N intersect an interpreted cell
    n_top_hit = sum(intersects_any_cell(p, cells, mosaic_tf) for p in top)

    # separately: the 5 largest patches that DO intersect an interpreted cell -> 3-panel overlays
    hits = [p for p in patches if intersects_any_cell(p, cells, mosaic_tf)][:N_OVERLAY]
    for rank, p in enumerate(hits, 1):
        render_overlay(out_dir, rank, p, tilesA, tilesB, H, W, cells, rf2common,
                       cls_cmap, cls_norm, names, colors, A_name, B_name, mosaic_tf)

    # plus 5 cell-sized 3-panel examples: interpreted cells with the most in-cell A/B disagreement
    examples = select_cell_examples(tilesA, tilesB, cells, mosaic_tf, n=N_OVERLAY)
    for rank, (score, cell) in enumerate(examples, 1):
        render_cell_overlay(out_dir, rank, score, cell, tilesA, tilesB, mosaic_tf, rf2common,
                            cls_cmap, cls_norm, names, colors, A_name, B_name)

    # cache the rendered patches (bounding boxes) so the zoom and overlay crops can be redrawn
    # from raw tiles without another streaming pass
    cache = [dict(role="zoom", rank=i + 1, **{k: p[k] for k in
                  ("pid", "area_px", "area_ha", "r0", "r1", "c0", "c1")})
             for i, p in enumerate(top)]
    cache += [dict(role="overlay", rank=i + 1, **{k: p[k] for k in
                   ("pid", "area_px", "area_ha", "r0", "r1", "c0", "c1")})
              for i, p in enumerate(hits)]
    pd.DataFrame(cache).to_csv(os.path.join(out_dir, "patches_cache.csv"), index=False)

    for ds, _, _ in tilesA + tilesB:
        ds.close()

    with open(os.path.join(out_dir, "notes.txt"), "w") as fh:
        fh.write(f"{A_name} vs {B_name}\n")
        fh.write(f"total change-involved disagreement patches: {len(patches)}\n")
        fh.write(f"largest patch: {top[0]['area_ha'] if top else 0} ha\n")
        fh.write(f"overall top-{n_top} patches intersecting an interpreted cell: "
                 f"{n_top_hit} of {len(top)}\n")
        fh.write(f"patches intersecting a cell (any size): "
                 f"{sum(intersects_any_cell(p, cells, mosaic_tf) for p in patches)}\n")
    print(f"  overall top-{n_top} intersecting an interpreted cell: {n_top_hit} of {len(top)}",
          flush=True)

    summary = dict(pair=f"{A_name}_vs_{B_name}",
                   agree_pct=round(100 * cat_px[1] / cat_px[1:].sum(), 3),
                   cat2_stable_mismatch_ha=round(cat_px[2] * pix_ha, 1),
                   cat3_Achange_ha=round(cat_px[3] * pix_ha, 1),
                   cat4_Bchange_ha=round(cat_px[4] * pix_ha, 1),
                   cat5_changechange_ha=round(cat_px[5] * pix_ha, 1),
                   n_patches=len(patches),
                   largest_patch_ha=top[0]["area_ha"] if top else 0.0,
                   top_hits_cell=n_top_hit)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="specific pairs like v2_vs_v3 (default: all six)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stop after this many mosaic rows (smoke test only)")
    ap.add_argument("--n-top", type=int, default=10, help="patches per zoom/overlay ranking")
    ap.add_argument("--plots-only", action="store_true",
                    help="re-render difference_map.png from cached render_grid.npy, no streaming")
    ap.add_argument("--zooms-only", action="store_true",
                    help="re-render zoom and overlay crops from cached patches_cache.csv, no streaming")
    ap.add_argument("--cell-panels", action="store_true",
                    help="build cell_all_variants/: 6-panel (5 variants + interpreted) per cell, no streaming")
    ap.add_argument("--cell-panels-n", type=int, default=20,
                    help="number of cells for --cell-panels (default 20, ranked by v2-v5 disagreement)")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    names, colors = load_legend()
    rf2common = load_crosswalk()
    cmaps = build_cmaps(names, colors)

    all_pairs = [(VARIANTS[i], VARIANTS[j])
                 for i in range(len(VARIANTS)) for j in range(i + 1, len(VARIANTS))]
    if args.pairs:
        want = set(args.pairs)
        all_pairs = [(a, b) for a, b in all_pairs if f"{a}_vs_{b}" in want]

    if args.cell_panels:
        # 6-panel per cell (all five variants plus the interpreted reference); windowed reads only
        _, _, cls_cmap, cls_norm = cmaps
        build_cell_panels(deduped_cells(seed=42), rf2common, cls_cmap, cls_norm, names, colors,
                          n=args.cell_panels_n)
        return

    if args.plots_only:
        # re-draw only the overview from the cached grid, so legend or color tweaks are instant
        diff_cmap, diff_norm = cmaps[0], cmaps[1]
        for A_name, B_name in all_pairs:
            out_dir = os.path.join(OUT_ROOT, f"{A_name}_vs_{B_name}")
            grid_path = os.path.join(out_dir, "render_grid.npy")
            if not os.path.exists(grid_path):
                print(f"  no cached grid for {A_name}_vs_{B_name}; run the full pass first")
                continue
            render_full_map(out_dir, np.load(grid_path), diff_cmap, diff_norm, names, colors,
                            A_name, B_name)
            print(f"  re-rendered {out_dir}/difference_map.png")
        return

    if args.zooms_only:
        # redraw zoom and overlay crops from the cached patch boxes, reading only small windows
        diff_cmap, diff_norm, cls_cmap, cls_norm = cmaps
        cells = deduped_cells(seed=42)
        for A_name, B_name in all_pairs:
            out_dir = os.path.join(OUT_ROOT, f"{A_name}_vs_{B_name}")
            cache_path = os.path.join(out_dir, "patches_cache.csv")
            if not os.path.exists(cache_path):
                print(f"  no patch cache for {A_name}_vs_{B_name}; run the full pass first")
                continue
            df = pd.read_csv(cache_path)
            tilesA, tilesB = open_tiles(A_name), open_tiles(B_name)
            H, W = mosaic_dims(tilesA)
            mosaic_tf = tilesA[0][0].transform
            for row in df.itertuples():
                if row.role == "overlay" and row.rank > N_OVERLAY:
                    continue                            # keep only the top-N patch overlays
                patch = dict(pid=row.pid, area_px=int(row.area_px), area_ha=row.area_ha,
                             r0=int(row.r0), r1=int(row.r1), c0=int(row.c0), c1=int(row.c1))
                if row.role == "zoom":
                    render_zoom(out_dir, row.rank, patch, tilesA, tilesB, H, W, diff_cmap, diff_norm,
                                names, colors, A_name, B_name)
                else:
                    render_overlay(out_dir, row.rank, patch, tilesA, tilesB, H, W, cells, rf2common,
                                   cls_cmap, cls_norm, names, colors, A_name, B_name, mosaic_tf)
            # cell-sized 3-panel examples (recomputed; the scoring is cheap windowed reads)
            for rank, (score, cell) in enumerate(
                    select_cell_examples(tilesA, tilesB, cells, mosaic_tf, n=N_OVERLAY), 1):
                render_cell_overlay(out_dir, rank, score, cell, tilesA, tilesB, mosaic_tf, rf2common,
                                    cls_cmap, cls_norm, names, colors, A_name, B_name)
            for ds, _, _ in tilesA + tilesB:
                ds.close()
            print(f"  re-rendered {out_dir} zoom/overlay/cell crops")
        return

    cells = deduped_cells(seed=42)
    print(f"interpreted cells (deduped, seed 42): {len(cells)}", flush=True)

    summaries = []
    for A_name, B_name in all_pairs:
        summaries.append(run_pair(A_name, B_name, cells, rf2common, names, colors, cmaps,
                                  args.max_rows, args.n_top))

    if summaries:
        sdf = pd.DataFrame(summaries)
        sdf.to_csv(os.path.join(OUT_ROOT, "pairs_summary.csv"), index=False)
        print("\n" + "=" * 78)
        print(f"interpreted cells (deduped, seed 42): {len(cells)}")
        print(sdf.to_string(index=False))
        print(f"\nwrote {OUT_ROOT}/pairs_summary.csv")


if __name__ == "__main__":
    main()
