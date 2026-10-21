#!/usr/bin/env python3
"""Extract per-cell grid geometry from the CKIT-RF classified rasters for a GEE-pinned export.

For per-cell confusion matrices to be valid, each GEE prediction raster must align pixel for pixel
with the CKIT-RF reference raster for that cell: same CRS, resolution, origin, and dimensions, with
no post-hoc resampling. This walks the rf_class_*.tif rasters and, per cell, records the CRS, the
affine transform in GEE crsTransform order, and the grid dimensions, so a GEE export can pin its
output grid to each cell. It also validates the north-up 10 m geometry and the CKIT label encoding.

The script refuses to write the CSV if any raster carries an unmapped label value or any
double-interpreted cell has mismatched grids across reviewers, so the export never depends on a
file with an unresolved encoding or alignment problem.

Output: exports/gee/ckit_cell_grids.csv
"""

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
import sensor_config as SC  # RF_DIR, out_path (LCI_RF_DIR / LCI_OUT_DIR)

warnings.filterwarnings("ignore")

RF_DIR = SC.RF_DIR
OUT_DIR = SC.out_path("exports/gee")
CSV_PATH = os.path.join(OUT_DIR, "ckit_cell_grids.csv")

FNAME_RE = re.compile(r"grid_(\d+)_sample_")
PAD = 5

# ckit label_id -> 10-class schema code; two values are valid but excluded from the 10-class matrix
CROSSWALK = {0: 4, 1: 6, 2: 7, 3: 3, 4: 5, 5: 8, 20: 1, 30: 2, 50: 10, 62: 9}
EXCLUDED_LABELS = {10, 13}                              # unknown abstention, other_no_change
ALLOWED_LABELS = set(CROSSWALK) | EXCLUDED_LABELS


def cell_id_from(path):
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    # int then zero-pad to width 5 so integer-typed or unpadded sources normalize identically
    return str(int(m.group(1))).zfill(PAD)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rasters = sorted(glob.glob(os.path.join(RF_DIR, "**", "rf_class*.tif"), recursive=True))

    rows = []                                           # one dict per geometry-valid raster
    geom_excluded = []                                  # non-north-up or non-10 m grids, kept out of the csv
    label_problems = {}                                 # file -> sorted list of unmapped values
    unparsed = []

    for path in rasters:
        cid = cell_id_from(path)
        base = os.path.basename(path)
        if cid is None:
            unparsed.append(base)
            continue
        with rasterio.open(path) as ds:
            t = ds.transform
            a, b, c, d, e, f = t.a, t.b, t.c, t.d, t.e, t.f
            crs = ds.crs.to_string() if ds.crs else "MISSING"
            w, h = ds.width, ds.height
            # assert north-up and 10 m; a violating raster is out of scope for a 10 m pin, so warn
            # and keep it out of the csv rather than write a bad row
            if not (a == 10 and e == -10 and b == 0 and d == 0):
                geom_excluded.append((base, dict(a=a, b=b, d=d, e=e, w=w, h=h)))
                continue
            vals = np.unique(ds.read(1))
        unmapped = sorted(int(v) for v in vals if int(v) not in ALLOWED_LABELS)
        if unmapped:
            label_problems[base] = unmapped
        rows.append(dict(cell_id=cid, crs=crs, x_scale=a, x_shear=b, x_trans=c,
                         y_shear=d, y_scale=e, y_trans=f, width=w, height=h, file=base))

    df = pd.DataFrame(rows)

    # double-interpreted cells: same cell_id from more than one raster
    dup_counts = df.cell_id.value_counts()
    double_ids = sorted(dup_counts[dup_counts > 1].index)
    geom_cols = ["crs", "x_scale", "x_shear", "x_trans", "y_shear", "y_scale", "y_trans",
                 "width", "height"]
    mismatched = []                                     # cell_ids whose duplicate grids differ
    for cid in double_ids:
        sub = df[df.cell_id == cid][geom_cols].drop_duplicates()
        if len(sub) > 1:
            mismatched.append(cid)

    # ---- summary ----
    size_dist = df.groupby(["width", "height"]).size().sort_values(ascending=False)
    print("=" * 70)
    print("CKIT-RF per-cell grid geometry")
    print("=" * 70)
    print(f"rasters found: {len(rasters)}   geometry-valid (10 m, north-up): {len(df)}   "
          f"excluded for bad geometry: {len(geom_excluded)}   unparsed: {len(unparsed)}")
    print(f"unique cells (geometry-valid): {df.cell_id.nunique()}")
    print("\ngrid size (width x height) -> raster count:")
    for (w, h), n in size_dist.items():
        print(f"  {w} x {h}: {n}")
    if len(size_dist) > 1:
        print("  NOTE: grid sizes vary across cells, so the per-cell width and height must be "
              "pinned individually; a single fixed 337x337 export would misalign some cells.")
    print(f"\ndouble-interpreted cells (two rasters, one cell_id): {len(double_ids)}")
    print(f"double-interpreted cells with MISMATCHED grids: {len(mismatched)}"
          f"{'  -> ' + ', '.join(mismatched) if mismatched else ''}")

    print(f"\nnon-north-up or non-10 m grids excluded from the csv: {len(geom_excluded)}")
    for base, g in geom_excluded:
        print(f"  GEOM (excluded): {base}  a={g['a']} e={g['e']} b={g['b']} d={g['d']} "
              f"size={g['w']}x{g['h']}")

    print(f"\nrasters with unmapped label values: {len(label_problems)}")
    for base, vals in label_problems.items():
        print(f"  LABELS: {base}  unmapped={vals}")
    if unparsed:
        print(f"\nunparsed filenames: {unparsed}")

    # ---- gates: do not write a CSV the export would rely on if either check fails ----
    if label_problems or mismatched:
        print("\nSTOP: not writing the CSV.")
        if label_problems:
            print(f"  {len(label_problems)} raster(s) carry label values outside the crosswalk and "
                  f"{{10, 13}}. Resolve the encoding mismatch first.")
        if mismatched:
            print(f"  {len(mismatched)} double-interpreted cell(s) have grids that differ across "
                  f"reviewers. Resolve the alignment first.")
        raise SystemExit(1)

    # keep one row per cell_id; matched grids are interchangeable, so first occurrence is fine
    out = (df.sort_values(["cell_id", "file"])
             .drop_duplicates(subset="cell_id", keep="first")
             .sort_values("cell_id"))
    out = out[["cell_id", "crs", "x_scale", "x_shear", "x_trans", "y_shear", "y_scale",
               "y_trans", "width", "height", "file"]]
    out.to_csv(CSV_PATH, index=False)
    print(f"\nlabel check passed (all values in the crosswalk or {{10, 13}}); "
          f"all double-interpreted grids match.")
    print(f"wrote {CSV_PATH}  ({len(out)} rows, one per unique cell)")


if __name__ == "__main__":
    main()
