#!/usr/bin/env python3
"""Change/stable conflicts between interpreters: one reviewer called a pixel change and the other
called it stable (and vice versa).

Using the RF interpreted scheme, the change classes are Harvest (20), Development (30),
Insect/Disease (50) and Beaver (62), and the stable classes are Urban (0), Agriculture (1),
Grass/Shrub (2), Forest (3), Water (4), Wetland (5) and Other (13). Fire (40) has zero pixels.
Unknown (10) is unattributed disturbance, not a stable class and not one of the four attributed
change classes, so an Unknown-vs-stable disagreement is one reviewer flagging disturbance without
attributing it; it is excluded from the conflict count and reported separately.

This is the change-vs-stable counterpart of the change/change analysis and is expected to be much
larger, since the inter-interpreter matrix shows most change-class disagreement goes to stable
classes, not to other change classes.

For every double-interpreted cell (two reviewers on the same grid/sample/target, reviewer order
alphabetical and therefore arbitrary) this finds pixels where one reviewer assigned an attributed
change class and the other a stable class, and reports:
  - total conflict pixels and area, as a fraction of all disagreement and of all change-labeled px
  - directed A-class -> B-class pair counts, plus the symmetrized stable<->change counts
  - which cells and reviewer pairs the conflicts occur in (long-format table)
  - 8-connected component patches: count, median and max area

Outputs -> reports/interpreter_agreement/change_stable_conflicts/
  - change_stable_pixels_long.csv   one row per (cell, revA, revB, A_class, B_class)
  - ordered_pairs.csv               directed and symmetrized stable/change class-pair totals
  - change_stable_patches.csv       one row per connected-component patch
  - summary.txt                     the headline numbers in plain text
  - COLUMNS.md                      data dictionary

Requires: rasterio, numpy, pandas, scipy
"""

import glob
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

warnings.filterwarnings("ignore")

RF_DIR = SC.RF_DIR
RF_LEGEND = "data/reference/label_lookup.csv"
OUT = SC.out_path("reports/interpreter_agreement/change_stable_conflicts")

NAME_RE = re.compile(rf"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)", re.I)
CHANGE = [20, 30, 50, 62]                               # harvest, development, insect/disease, beaver
STABLE = [0, 1, 2, 3, 4, 5, 13]                         # urban, ag, grass/shrub, forest, water, wetland, other
UNKNOWN = 10                                            # unattributed disturbance, reported separately
STRUCT = np.ones((3, 3), int)                           # 8-connectivity


def load_names():
    df = pd.read_csv(RF_LEGEND)
    return {int(r.code): r.display_name for r in df.itertuples()}, [int(c) for c in df.code]


def find_pairs():
    """{(grid,sample,target): [(reviewer, path), ...]} for cells with >1 reviewer, alphabetical."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True)):
        m = NAME_RE.search(os.path.basename(f))
        if m:
            groups[(m.group(2), m.group(3), m.group(4))].append((m.group(1).lower(), f))
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def write_data_dictionary(n_long, n_patches, pix_ha):
    side_m = (pix_ha * SC.M2_PER_HA) ** 0.5
    names = load_names()[0]
    chg = ", ".join(f"{names[c]} ({c})" for c in CHANGE)
    stb = ", ".join(f"{names[c]} ({c})" for c in STABLE)
    txt = f"""# change_stable_conflicts data dictionary

These files flag pixels where one interpreter called a pixel an attributed change class and the
other called it a stable class (in either direction). Change classes are {chg}. Stable classes are
{stb}. Unknown (10) is unattributed disturbance and is neither stable nor an attributed change
class, so Unknown-vs-stable is reported separately in `summary.txt`, not counted as a conflict;
Fire (40) has zero pixels. See `summary.txt` for the headline totals and `ordered_pairs.csv` for
the directed and symmetrized class-pair totals.

## change_stable_pixels_long.csv  ({n_long} rows)

One row per (cell, reviewer pair, directed class pair). It depicts, for each double-interpreted
cell that has any change/stable conflict, how many pixels each ordered A-class -> B-class conflict
covers, where exactly one of the two classes is stable and the other is an attributed change
class. A cell with several conflicting class pairs gets several rows. Reviewer A/B ordering is
alphabetical and therefore arbitrary, so a single directed pair carries no meaning on its own; the
`stable_class` and `change_class` columns give the reviewer-order-independent view, and the
symmetrized totals are in `ordered_pairs.csv`.

| column | meaning |
|---|---|
| grid | grid cell id (the physical cell) |
| sample | interpretation sample index for that cell |
| target | interpreted target year |
| revA | reviewer A (alphabetically first of the pair) |
| revB | reviewer B (alphabetically second) |
| A_class | the class reviewer A assigned to these pixels (stable or change) |
| B_class | the class reviewer B assigned to the same pixels (the other kind) |
| class_pair | `A_class->B_class`, the directed conflict label |
| stable_class | the stable class of the pair, regardless of which reviewer assigned it |
| change_class | the attributed change class of the pair, regardless of reviewer |
| pixels | number of conflict pixels of this class pair in this cell |
| area_ha | pixels x {pix_ha:g} ha (one {side_m:g} m pixel = {pix_ha:g} ha) |

## change_stable_patches.csv  ({n_patches} rows)

One row per connected-component patch of the change/stable conflict mask, labeled per cell with
8-connectivity. It depicts the spatial grouping of the conflicts: whether they are a handful of
large blobs or many scattered pixels. A patch is a spatially contiguous run of conflict pixels
within one cell and reviewer pair, so patch counts and areas distinguish "a few large disagreed
zones" from "boundary or salt-and-pepper speckle."

| column | meaning |
|---|---|
| grid | grid cell id the patch is in |
| revA | reviewer A (alphabetically first) |
| revB | reviewer B (alphabetically second) |
| patch_id | patch label within this cell and reviewer pair (1-based) |
| pixels | number of conflict pixels in the patch |
| area_ha | pixels x {pix_ha:g} ha |
"""
    with open(os.path.join(OUT, "COLUMNS.md"), "w") as fh:
        fh.write(txt)


def main():
    os.makedirs(OUT, exist_ok=True)
    names, legend_codes = load_names()
    legend_arr = np.zeros(256, dtype=bool); legend_arr[legend_codes] = True
    change_arr = np.zeros(256, dtype=bool); change_arr[CHANGE] = True
    stable_arr = np.zeros(256, dtype=bool); stable_arr[STABLE] = True

    pairs = find_pairs()
    # pooled pixel counts -> ha needs one pixel size across every raster in the run
    pix_ha = SC.uniform_pixel_area_m2([f for revs in pairs.values() for _, f in revs]) / SC.M2_PER_HA
    print(f"double-interpreted cells (pairs, first two reviewers): {len(pairs)}")

    long_rows, patch_rows = [], []
    directed = defaultdict(int)                          # (a_class, b_class) -> pixels
    tot_conflict = tot_disagree = tot_changelab = tot_unknown = 0
    n_cells_with_conflict = 0

    for (gid, samp, tgt), revs in sorted(pairs.items()):
        (revA, fA), (revB, fB) = revs[0], revs[1]
        with rasterio.open(fA) as s:
            a = s.read(1)
        with rasterio.open(fB) as s:
            b = s.read(1)
        if a.shape != b.shape:
            print(f"  skip grid {gid}: shape mismatch")
            continue

        valid = legend_arr[a] & legend_arr[b]
        achg, bchg = change_arr[a], change_arr[b]
        astab, bstab = stable_arr[a], stable_arr[b]
        disagree = valid & (a != b)
        change_lab = valid & (achg | bchg)
        conflict = (achg & bstab) | (astab & bchg)      # one attributed change, one stable
        unknown_stable = ((a == UNKNOWN) & bstab) | ((b == UNKNOWN) & astab)

        tot_disagree += int(disagree.sum())
        tot_changelab += int(change_lab.sum())
        tot_unknown += int(unknown_stable.sum())
        nconf = int(conflict.sum())
        tot_conflict += nconf
        if nconf == 0:
            continue
        n_cells_with_conflict += 1

        # per class-pair pixel counts within this cell
        av, bv = a[conflict], b[conflict]
        pair_counts = defaultdict(int)
        for ca, cb in zip(av.tolist(), bv.tolist()):
            pair_counts[(ca, cb)] += 1
        for (ca, cb), cnt in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
            directed[(ca, cb)] += cnt
            sc = names[ca] if ca in STABLE else names[cb]     # stable member of the pair
            cc = names[cb] if ca in STABLE else names[ca]     # attributed-change member
            long_rows.append(dict(
                grid=gid, sample=samp, target=tgt, revA=revA, revB=revB,
                A_class=names[ca], B_class=names[cb], class_pair=f"{names[ca]}->{names[cb]}",
                stable_class=sc, change_class=cc, pixels=cnt, area_ha=round(cnt * pix_ha, 3)))

        # connected-component patches of the conflict mask (8-connectivity)
        lab, n = ndimage.label(conflict, structure=STRUCT)
        if n:
            areas = np.bincount(lab.ravel())[1:]
            for k, ar in enumerate(areas, 1):
                patch_rows.append(dict(grid=gid, revA=revA, revB=revB, patch_id=k,
                                       pixels=int(ar), area_ha=round(int(ar) * pix_ha, 3)))

    # long-format pixel table
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(OUT, "change_stable_pixels_long.csv"), index=False)

    # directed and symmetrized stable/change pair totals
    dir_rows = []
    for (ca, cb), cnt in directed.items():
        dir_rows.append(dict(A_class=names[ca], B_class=names[cb],
                             class_pair=f"{names[ca]}->{names[cb]}",
                             pixels=cnt, area_ha=round(cnt * pix_ha, 3)))
    sym = defaultdict(int)
    for (ca, cb), cnt in directed.items():
        sym[frozenset((ca, cb))] += cnt
    for key, cnt in sym.items():
        c = sorted(key)                                 # (stable_code, change_code) either order
        sc = names[c[0]] if c[0] in STABLE else names[c[1]]
        cc = names[c[1]] if c[0] in STABLE else names[c[0]]
        dir_rows.append(dict(A_class=sc, B_class=cc,
                             class_pair=f"{sc}<->{cc} (symmetrized)",
                             pixels=cnt, area_ha=round(cnt * pix_ha, 3)))
    pd.DataFrame(sorted(dir_rows, key=lambda d: -d["pixels"])).to_csv(
        os.path.join(OUT, "ordered_pairs.csv"), index=False)

    # patches
    patch_df = pd.DataFrame(patch_rows)
    patch_df.to_csv(os.path.join(OUT, "change_stable_patches.csv"), index=False)

    write_data_dictionary(len(long_df), len(patch_df), pix_ha)

    # summary
    frac_dis = tot_conflict / tot_disagree if tot_disagree else float("nan")
    frac_chg = tot_conflict / tot_changelab if tot_changelab else float("nan")
    if len(patch_df):
        p_n, p_med, p_max = len(patch_df), patch_df.pixels.median(), patch_df.pixels.max()
    else:
        p_n = p_med = p_max = 0
    # the biggest symmetrized stable<->change pairs, for the headline
    top = sorted(sym.items(), key=lambda kv: -kv[1])[:5]
    top_str = "; ".join(
        f"{(names[sorted(k)[0]] if sorted(k)[0] in STABLE else names[sorted(k)[1]])}"
        f"<->{(names[sorted(k)[1]] if sorted(k)[0] in STABLE else names[sorted(k)[0]])} "
        f"{v * pix_ha:.1f} ha" for k, v in top)

    lines = [
        "change/stable conflicts between interpreters",
        "(one reviewer called change, the other called stable; either direction)",
        "",
        f"double-interpreted cells (pairs): {len(pairs)}",
        f"cells with at least one conflict: {n_cells_with_conflict}",
        "",
        f"total conflict pixels: {tot_conflict:,}  ({tot_conflict * pix_ha:,.1f} ha)",
        f"  as a fraction of all disagreement pixels ({tot_disagree:,}): {frac_dis:.4%}",
        f"  as a fraction of all change-labeled pixels ({tot_changelab:,}): {frac_chg:.4%}",
        "",
        f"Unknown-vs-stable pixels (reported separately, NOT counted as conflicts): "
        f"{tot_unknown:,}  ({tot_unknown * pix_ha:,.1f} ha)",
        "",
        f"largest symmetrized stable<->change pairs: {top_str}",
        "",
        f"connected-component patches (8-connectivity): {p_n}",
        f"  median patch area: {p_med * pix_ha:.3f} ha ({int(p_med) if p_n else 0} px)",
        f"  max patch area:    {p_max * pix_ha:.3f} ha ({int(p_max) if p_n else 0} px)",
        "",
        "convention: reviewer A/B ordering is alphabetical and arbitrary, so directed pair counts",
        "(A_class->B_class) carry no meaning on their own; the `stable_class`/`change_class` columns",
        "and the symmetrized counts in ordered_pairs.csv are the reviewer-order-independent numbers.",
        "",
        "context: this is the dominant form of change-class disagreement. Compare with",
        "change_change_conflicts/ (both reviewers called change but disagreed on type), which is",
        "far smaller.",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)
    print(f"\nwrote {OUT}/ (long, ordered_pairs, patches, summary, COLUMNS)")


if __name__ == "__main__":
    main()
