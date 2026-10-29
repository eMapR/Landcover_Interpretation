#!/usr/bin/env python3
"""Read-only diagnostic: score each Prithvi model against the Landsat references with the Prithvi
label year = reference target year + offset, offset in -2..+2.

Same reference selection and scoring as collapsed_5class_confusion.py --prithvi: seeded dedup
(deduped_cells, seed 42), argmax via prithvi_loader, centre-based lookup onto the reference grid,
zero-input mask "any_year" taken from the chip of the offset label year, 4x4 census (Beaver,
Unknown, Fire excluded; agriculture out of the matrix), metrics_f1_zero. A cell is skipped for an
offset when no output exists for that year. Also reports offset 0 restricted to the cells that
have all five offsets.

Outputs -> $LCI_OUT_DIR/reports/prithvi_year_offset/ (refuses to run without LCI_OUT_DIR, so it
cannot write into the Sentinel-2 reports/):
  - year_offset_metrics.csv          per model x offset (+ offset 0 on the all-5 cells)
  - year_offset_summary.md           the same as a Markdown table, with offset coverage
  - cell_offset_availability.csv     which offsets have a Prithvi output, per reference cell

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs LCI_OUT_DIR=check_landsat \
        /vol/v1/bekka/venvs/lci/bin/python scripts/prithvi_year_offset.py
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio

import collapsed_5class_confusion as C   # imports only; writes nothing
import prithvi_loader as PL

OFFSETS = [-2, -1, 0, 1, 2]
MASK = "any_year"


def load(ref_path, model, year):
    """load_for_reference() with an explicit label year; None when the output does not exist."""
    cell_id, _, _ = PL.parse_reference(ref_path)
    path = PL.prithvi_path(model, year, cell_id)
    if not os.path.exists(path):
        return None
    probs, transform = PL.read_probs(path)
    codes = PL.collapsed_argmax(probs)
    zero = PL.zero_input_mask(year, cell_id, transform, rule=MASK)
    with rasterio.open(ref_path) as ref_ds:
        return (PL.align_to_reference(codes, transform, ref_ds),
                PL.align_to_reference(zero, transform, ref_ds, fill=False))


def main():
    if C.SC.OUT_DIR is None:
        sys.exit("refusing to run: set LCI_OUT_DIR so outputs cannot overwrite the Sentinel-2 reports/")
    if C.SC.SENSOR != "Landsat":
        sys.exit(f"refusing to run: LCI_SENSOR is {C.SC.SENSOR!r}, expected 'Landsat'")
    out = C.SC.out_path("reports/prithvi_year_offset")
    os.makedirs(out, exist_ok=True)
    cells = C.deduped_cells()
    print(f"de-duplicated reference cells (seed {C.SEED}): {len(cells)}")
    # cms[(model, offset)][cell] = 4x4 matrix
    cms = {(m, o): {} for m in PL.PRITHVI_MODELS for o in OFFSETS}
    avail = []
    for i, cell in enumerate(cells):
        cell_id, target, reviewer = PL.parse_reference(cell)
        with rasterio.open(cell) as ds:
            rf = ds.read(1)
        row = dict(cell_id=cell_id, target_year=target, reviewer=reviewer)
        for o in OFFSETS:
            year = target + o
            loaded = {m: load(cell, m, year) for m in PL.PRITHVI_MODELS}
            n_none = sum(v is None for v in loaded.values())
            if n_none:
                if n_none != len(loaded):
                    raise RuntimeError(f"{cell} year {year}: output for only some models")
                row[f"off{o:+d}"] = False
                continue
            row[f"off{o:+d}"] = True
            for m, (codes, zero) in loaded.items():
                covered = codes != PL.NODATA
                cm, _, _ = C.prithvi_cell_tallies(rf, codes, covered & ~zero)
                cms[(m, o)][cell] = cm
        avail.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cells)}")
    av = pd.DataFrame(avail)
    av.to_csv(os.path.join(out, "cell_offset_availability.csv"), index=False)
    all5 = set(np.array(cells)[av[[f"off{o:+d}" for o in OFFSETS]].all(1).values])
    print(f"cells with all five offsets: {len(all5)}")

    rows = []
    for m in PL.PRITHVI_MODELS:
        for o, subset in [(o, None) for o in OFFSETS] + [(0, "all5")]:
            d = cms[(m, o)]
            if subset:
                d = {c: v for c, v in d.items() if c in all5}
            M = sum(d.values()) if d else np.zeros((4, 4), np.int64)
            mt = C.metrics_f1_zero(M)
            rows.append(dict(
                model=m, offset=o, cells=("all-5 subset" if subset else "all available"),
                cells_with_output=len(d), cells_scored=sum(int(v.sum() > 0) for v in d.values()),
                valid_px=int(M.sum()),
                harvest_recall=mt["recall[2]"], harvest_precision=mt["precision[2]"],
                harvest_F1=mt["F1[2]"], development_F1=mt["F1[3]"],
                insect_disease_F1=mt["F1[4]"], kappa=mt["kappa"],
                harvest_support=int(mt["support[2]"]), development_support=int(mt["support[3]"]),
                insect_disease_support=int(mt["support[4]"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "year_offset_metrics.csv"), index=False)

    fmt = lambda v: f"{v:.3f}" if np.isfinite(v) else "n/a"
    md = ["# Prithvi label-year offset diagnostic (Landsat, mask any_year)", "",
          f"Reference: {C.RF_DIR}, seeded dedup seed {C.SEED}, {len(cells)} cells. "
          f"Label year = target year + offset. Cells with all five offsets: {len(all5)}.", "",
          "cells = cells with an output for that year / cells contributing valid pixels.", "",
          "| model | offset | cell set | cells | valid px | Harv R | Harv P | Harv F1 | Dev F1 | I/D F1 | kappa |",
          "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['offset']:+d} | {r['cells']} | {r['cells_with_output']} / "
                  f"{r['cells_scored']} | {r['valid_px']:,} | {fmt(r['harvest_recall'])} | "
                  f"{fmt(r['harvest_precision'])} | {fmt(r['harvest_F1'])} | "
                  f"{fmt(r['development_F1'])} | {fmt(r['insect_disease_F1'])} | {fmt(r['kappa'])} |")
    md += ["", "Offset coverage (cells with an output):",
           *[f"- {o:+d}: {int(av[f'off{o:+d}'].sum())}" for o in OFFSETS]]
    with open(os.path.join(out, "year_offset_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
