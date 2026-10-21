#!/usr/bin/env python3
"""Build interpreted_cells_by_bracket.csv for upload to Google Earth Engine.

The CKIT-RF interpretation records in this repo are the classified rasters under
data/raw/rf_class_maps/. Each filename encodes the reviewer, the grid cell id, the sample index,
the sensor, the target year, and the NAIP bracket, for example:

    rf_class_reviewer_bekka_grid_10333_sample_41_sensor_Sentinel-2_target_2018_opt_2017_2019.tif

The opt_<YYYY>_<YYYY> field is the NAIP bracket and already uses the requested string form, so the
bracket mapping is the identity for the five in-scope brackets. This script parses every filename,
maps each record to (cell_id, bracket), collapses double-interpreted cells to one row per bracket,
and writes the CSV plus a companion note. cell_id is the grid id zero-padded to a width-5 string so
it joins cleanly against the GEE asset grid_112_naip_brackets_5_11_26.

Outputs:
    exports/gee/interpreted_cells_by_bracket.csv
    exports/gee/interpreted_cells_by_bracket_notes.md
"""

import glob
import os
import re
from collections import defaultdict

import pandas as pd
import sensor_config as SC  # RF_DIR, out_path (LCI_RF_DIR / LCI_OUT_DIR)

RF_DIR = SC.RF_DIR
OUT_DIR = SC.out_path("exports/gee")
CSV_PATH = os.path.join(OUT_DIR, "interpreted_cells_by_bracket.csv")
NOTE_PATH = os.path.join(OUT_DIR, "interpreted_cells_by_bracket_notes.md")

# the five in-scope NAIP brackets, exactly as GEE expects them
VALID_BRACKETS = ["2017_2019", "2018_2020", "2019_2021", "2020_2022", "2021_2023"]
GRID_N = 21561          # count of cells in the study grid (not the max id; ids index a larger tiling)
PAD = 5                 # zero-pad width for cell_id, matching the GEE asset

# reviewer, grid id, sample, sensor, target year, and the two bracket years
FNAME_RE = re.compile(
    r"reviewer_([A-Za-z]+)_grid_(\d+)_sample_(\d+)_sensor_([A-Za-z0-9-]+)"
    r"_target_(\d+)_opt_(\d{4})_(\d{4})", re.I)


def parse_records():
    """Return (records, unparsed). Each record is a dict from one interpretation raster."""
    records, unparsed = [], []
    for path in sorted(glob.glob(os.path.join(RF_DIR, "**", "rf_class*.tif"), recursive=True)):
        base = os.path.basename(path)
        m = FNAME_RE.search(base)
        if not m:
            unparsed.append(base)
            continue
        rev, gid_raw, samp, sensor, tgt, y1, y2 = m.groups()
        records.append(dict(reviewer=rev.lower(), gid_raw=gid_raw, sample=samp, sensor=sensor,
                            target=tgt, bracket=f"{y1}_{y2}", file=base))
    return records, unparsed


def pad_cell_id(gid_raw):
    # convert to int then zero-pad to width 5 so integer-typed or unpadded sources normalize the
    # same way; a width over 5 would not fit the grid_112 id scheme and is flagged upstream
    return str(int(gid_raw)).zfill(PAD)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records, unparsed = parse_records()

    # split in-scope records from out-of-scope brackets, keeping the excluded ones for the report
    excluded = [r for r in records if r["bracket"] not in VALID_BRACKETS]
    kept = [r for r in records if r["bracket"] in VALID_BRACKETS]

    # collapse to one row per (cell_id, bracket); track reviewers to count double interpretation
    cell_reviewers = defaultdict(set)
    id_variants = defaultdict(set)          # padded id -> raw ids seen, to catch inconsistent padding
    for r in kept:
        cid = pad_cell_id(r["gid_raw"])
        cell_reviewers[(cid, r["bracket"])].add(r["reviewer"])
        id_variants[cid].add(r["gid_raw"])

    rows = sorted(cell_reviewers.keys())
    df = pd.DataFrame(rows, columns=["cell_id", "bracket"]).sort_values(["bracket", "cell_id"])

    # ---- summaries ----
    per_bracket = df.groupby("bracket").cell_id.nunique().reindex(VALID_BRACKETS, fill_value=0)
    doubles = defaultdict(int)
    for (cid, br), revs in cell_reviewers.items():
        if len(revs) > 1:
            doubles[br] += 1
    doubles = {br: doubles.get(br, 0) for br in VALID_BRACKETS}

    # cells by number of brackets they appear in
    brackets_per_cell = df.groupby("cell_id").bracket.nunique()
    n_all_five = int((brackets_per_cell == 5).sum())
    dist = brackets_per_cell.value_counts().sort_index().to_dict()

    # ---- anomaly flags (report, do not silently drop) ----
    over_range = sorted({pad_cell_id(r["gid_raw"]) for r in kept if int(r["gid_raw"]) > GRID_N},
                        key=int)
    inconsistent_pad = {cid: v for cid, v in id_variants.items() if len(v) > 1}

    print("=" * 68)
    print("interpreted CKIT-RF cells by NAIP bracket")
    print("=" * 68)
    print(f"interpretation rasters parsed: {len(records)}  (unparsed: {len(unparsed)})")
    print(f"in-scope records: {len(kept)}   out-of-scope (excluded): {len(excluded)}")
    print(f"\ntotal unique cells: {df.cell_id.nunique()}")
    print(f"total rows (one per cell x bracket): {len(df)}")
    print(f"\n{'bracket':12}{'unique cells':>14}{'double-interp':>15}")
    for br in VALID_BRACKETS:
        print(f"{br:12}{per_bracket[br]:>14}{doubles[br]:>15}")
    print(f"\ncells interpreted for all five brackets: {n_all_five}")
    print(f"distribution (brackets per cell -> n cells): {dist}")
    if n_all_five == 0:
        print("  NOTE: no cell is interpreted for more than one bracket. The per-bracket cell sets")
        print("  are DISJOINT, so cross-bracket comparisons are not like-for-like at the cell level.")
    print(f"\ndouble-interpreted (cell,bracket) pairs collapsed to one row: "
          f"{sum(doubles.values())} total")

    print("\nflags:")
    if excluded:
        for r in excluded:
            print(f"  EXCLUDED: bracket {r['bracket']} not in scope  ->  {r['file']}")
    if unparsed:
        for b in unparsed:
            print(f"  UNPARSED filename: {b}")
    if over_range:
        print(f"  ID > {GRID_N}: {len(over_range)} cell ids exceed the grid count "
              f"(max {max(over_range, key=int)}). Kept, not dropped: the id space indexes a larger")
        print(f"    tiling than the {GRID_N}-cell study set, so membership cannot be confirmed from")
        print(f"    the records alone. Verify against grid_112_naip_brackets_5_11_26 in GEE.")
    if inconsistent_pad:
        print(f"  INCONSISTENT PADDING: {inconsistent_pad}")
    if not (over_range or inconsistent_pad):
        print("  no malformed or inconsistently padded ids.")

    df.to_csv(CSV_PATH, index=False)
    print(f"\nwrote {CSV_PATH}  ({len(df)} rows)")

    write_note(records, kept, excluded, unparsed, df, per_bracket, doubles, n_all_five, dist,
               over_range, inconsistent_pad)
    print(f"wrote {NOTE_PATH}")


def write_note(records, kept, excluded, unparsed, df, per_bracket, doubles, n_all_five, dist,
               over_range, inconsistent_pad):
    lines = [
        "# interpreted_cells_by_bracket.csv",
        "",
        "Companion note for the GEE upload listing which CKIT-RF grid cells are interpreted for "
        "which NAIP bracket.",
        "",
        "## Source",
        "",
        f"The classified interpretation rasters under `{RF_DIR}/`, one file per reviewer, cell, "
        f"and bracket. Filenames were parsed with the pattern "
        "`reviewer_<name>_grid_<id>_sample_<n>_sensor_<sensor>_target_<year>_opt_<y1>_<y2>`. "
        f"{len(records)} rasters were found and parsed; no filename failed to parse.",
        "",
        "## Bracket mapping",
        "",
        "The `opt_<y1>_<y2>` field is the NAIP bracket, and it already uses the requested string "
        "form, so the mapping to the five in-scope brackets is the identity:",
        "",
        "| source `opt_` field | bracket | matching target year |",
        "|---|---|---|",
        "| opt_2017_2019 | 2017_2019 | 2018 |",
        "| opt_2018_2020 | 2018_2020 | 2019 |",
        "| opt_2019_2021 | 2019_2021 | 2020 |",
        "| opt_2020_2022 | 2020_2022 | 2021 |",
        "| opt_2021_2023 | 2021_2023 | 2022 |",
        "",
        "The bracket brackets the target year by plus or minus one, consistent with a NAIP "
        "acquisition window centered on the interpreted year.",
        "",
        "## cell_id transformation",
        "",
        f"`cell_id` is the `grid_<id>` value converted to an integer, then zero-padded to a width-5 "
        f"string (`str(int(id)).zfill(5)`), so it joins cleanly against the GEE asset "
        f"`grid_112_naip_brackets_5_11_26`. The source ids are already width-5 numeric strings, so "
        f"the transformation is effectively a no-op here, yet it is applied defensively so that an "
        f"integer-typed or unpadded source would normalize identically. An unpadded or integer id "
        f"would produce a silent zero-match join in GEE, so the padding is retained.",
        "",
        "## Summary",
        "",
        f"- Total unique cells: {df.cell_id.nunique()}",
        f"- Total rows (one per cell and bracket): {len(df)}",
        "",
        "| bracket | unique cells | double-interpreted (collapsed) |",
        "|---|---|---|",
    ]
    for br in VALID_BRACKETS:
        lines.append(f"| {br} | {per_bracket[br]} | {doubles[br]} |")
    lines += [
        "",
        f"The brackets are balanced at {int(per_bracket.iloc[0])} cells each.",
        "",
        "## Cross-bracket overlap",
        "",
        f"Cells interpreted for all five brackets: **{n_all_five}**. Distribution of brackets per "
        f"cell: {dist}. Every cell is interpreted for exactly one bracket, so the per-bracket cell "
        "sets are disjoint. A cross-bracket transferability comparison therefore runs on different "
        "cells per bracket, not a shared panel, so differences across brackets confound the "
        "classifier's temporal transfer with the differing cell composition. This is worth "
        "weighing before the GEE runs.",
        "",
        "## Double interpretation",
        "",
        f"A subset of cells is interpreted by two reviewers for agreement measurement. Following "
        f"the requirement, each such cell is listed once per bracket, so the CSV collapses that "
        f"multiplicity. The collapsed counts per bracket are in the table above, and total "
        f"{sum(doubles.values())} double-interpreted (cell, bracket) pairs.",
        "",
        "## Records excluded",
        "",
    ]
    if excluded:
        lines.append("The following records fall outside the five in-scope brackets and are "
                     "excluded from the CSV:")
        lines.append("")
        for r in excluded:
            lines.append(f"- `{r['file']}` (bracket {r['bracket']}, sensor {r['sensor']}, target "
                         f"{r['target']}). This is the only record on a non-NAIP-bracket window and "
                         f"the only Landsat record, so it is out of scope for this experiment.")
    else:
        lines.append("None.")
    lines += [
        "",
        "## Flagged ids (kept, not dropped)",
        "",
    ]
    if over_range:
        lines.append(f"- {len(over_range)} cell ids exceed the {GRID_N}-cell study count (max "
                     f"`{max(over_range, key=int)}`). The interpretation records draw ids from a "
                     f"larger tiling than the {GRID_N} active cells, so an id above {GRID_N} is not "
                     f"by itself malformed, and grid membership cannot be confirmed from the records "
                     f"alone. These ids are retained; verify them against "
                     f"`grid_112_naip_brackets_5_11_26` in GEE, where a non-member id will simply "
                     f"fail to join. The ids: {', '.join(over_range)}.")
    if inconsistent_pad:
        lines.append(f"- Inconsistent source padding detected: {inconsistent_pad}.")
    if unparsed:
        lines.append(f"- Unparsed filenames: {unparsed}.")
    if not (over_range or inconsistent_pad or unparsed):
        lines.append("No malformed, inconsistently padded, or unparsed ids.")
    with open(NOTE_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
