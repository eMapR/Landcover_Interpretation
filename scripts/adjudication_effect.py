#!/usr/bin/env python3
"""Stage 2: quantify the effect of adjudication on the interpreted reference over the 72
multiply-interpreted cells. Adjudication is whole-cell reviewer selection, so the adjudicated
reference for a cell is the chosen reviewer's raster.

Metrics (each under ten_class and five_class, long-format CSVs to reports/adjudication_effect/):
  A  contested share of the adjudicated reference, per class (two sub-metrics)
  C  dilution: contested pixels over the 72-cell and the 180-cell valid denominators
  selection_frequency  cells selected and valid-pixel share per interpreter (own small csv)
  one_sided_unknown    where one interpreter said Unknown and another said a class, by that class

Rules: exclude Unknown (10) pairwise (drop a pixel from all arms if ANY arm is Unknown); class 0
(Urban) retained; Other (13) folds into Stable for five-class; the one 3-interpreter cell keeps all
three arms.

Run: python scripts/adjudication_effect.py
Requires: numpy, pandas, rasterio
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_DIR = os.path.join(ROOT, SC.RF_DIR)
OUT = SC.out_path("reports/adjudication_effect", ROOT)
NAME_RE = re.compile(rf"reviewer_([a-z]+)_grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)_opt_(\d{{4}}_\d{{4}})", re.I)

leg = pd.read_csv(os.path.join(ROOT, "data/reference/label_lookup.csv"))
NAMES = {int(r.code): r.display_name for r in leg.itertuples()}

# native code -> five-class integer (1..5); Unknown(10) -> 0 (dropped)
FIVE_LUT = np.zeros(63, dtype=int)
for c in (0, 1, 2, 3, 4, 5, 13):
    FIVE_LUT[c] = 1                                            # Stable (incl. Other)
FIVE_LUT[20], FIVE_LUT[30], FIVE_LUT[50], FIVE_LUT[62] = 2, 3, 4, 5
FIVE_NAME = {1: "Stable", 2: "Harvest", 3: "Development", 4: "Insect/Disease", 5: "Beaver"}
TEN_CODES = [0, 1, 2, 3, 4, 5, 13, 20, 30, 50, 62]            # native classes, Unknown excluded


def label(scheme, code):
    return FIVE_NAME[code] if scheme == "five" else NAMES[code]


def classes(scheme):
    return list(FIVE_NAME) if scheme == "five" else TEN_CODES


def read(path):
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.int32)


def load_cells():
    cells = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True)):
        m = NAME_RE.search(os.path.basename(f))
        if m:
            cells[(m.group(2).zfill(5), m.group(3), m.group(4))][m.group(1).lower()] = f
    return cells


def cell_arrays(revs):
    """Return sorted reviewer names, stacked native arrays, valid (pairwise non-Unknown) mask."""
    names_arms = sorted(revs)
    stack = np.stack([read(revs[r]) for r in names_arms])      # (k, H, W) native codes
    valid = ~(stack == 10).any(0)                              # drop pixel if ANY arm is Unknown
    return names_arms, stack, valid


def hand_check(key, revs, chosen):
    g, s, t = key
    names_arms, stack, valid = cell_arrays(revs)
    k = len(names_arms)
    print(f"\n  CELL grid {g} sample {s} target {t}  ({k} interpreters: {names_arms}); chosen = {chosen}")
    print(f"    shape {stack.shape[1:]}, total pixels {stack[0].size:,}")
    print(f"    pixels with ANY arm Unknown (dropped, pairwise): {int((stack == 10).any(0).sum()):,}")
    print(f"    valid (paired) pixels: {int(valid.sum()):,}")
    for scheme in ("ten", "five"):
        arms_s = FIVE_LUT[stack] if scheme == "five" else stack
        agree = np.ones(valid.shape, bool)
        for i in range(1, k):
            agree &= arms_s[i] == arms_s[0]
        contested = valid & ~agree
        print(f"    [{scheme}] contested (arms disagree, valid): {int(contested.sum()):,}"
              f"   ({100 * contested.sum() / max(1, valid.sum()):.2f}% of valid)")
    # one-sided Unknown pairwise
    tot = 0
    for i in range(k):
        for j in range(i + 1, k):
            ai, aj = stack[i], stack[j]
            one = ((ai == 10) & (aj != 10)) | ((aj == 10) & (ai != 10))
            tot += int(one.sum())
    print(f"    one-sided Unknown pixels (pairwise, one arm Unknown, other a class): {tot:,}")


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells()
    multiply = {k: v for k, v in cells.items() if len(v) >= 2}
    single = {k: v for k, v in cells.items() if len(v) == 1}

    truth = pd.read_csv(os.path.join(ROOT, "exports/truth_selections.csv"), dtype=str, keep_default_na=False)
    choice = {(str(r.grid_id).zfill(5), str(r.sample_id)): r.reviewer.strip().lower() for r in truth.itertuples()}

    # ---- two-cell hand-check (one 2-interp, the 3-interp cell) ----
    two_keys = sorted(k for k, v in multiply.items() if len(v) == 2)
    three_key = next(k for k, v in multiply.items() if len(v) >= 3)
    print("=" * 78 + "\nHAND-CHECK (verify contested-pixel and pairwise-Unknown logic)\n" + "=" * 78)
    for key in (two_keys[0], three_key):
        hand_check(key, multiply[key], choice[(key[0], key[1])])

    # ---- accumulators ----
    adj_total = {s: defaultdict(int) for s in ("ten", "five")}
    adj_contested = {s: defaultdict(int) for s in ("ten", "five")}
    total_contested = {"ten": 0, "five": 0}
    onesided = {s: defaultdict(int) for s in ("ten", "five")}
    onesided_total = 0
    dropped_unknown = 0
    valid_72 = 0
    sel_cells, sel_pixels = defaultdict(int), defaultdict(int)

    for key, revs in multiply.items():
        g, s, t = key
        names_arms, stack, valid = cell_arrays(revs)
        chosen = choice[(g, s)]
        adj = stack[names_arms.index(chosen)]
        dropped_unknown += int((stack == 10).any(0).sum())
        valid_72 += int(valid.sum())
        sel_cells[chosen] += 1
        sel_pixels[chosen] += int(valid.sum())
        for scheme in ("ten", "five"):
            arms_s = FIVE_LUT[stack] if scheme == "five" else stack
            adj_s = FIVE_LUT[adj] if scheme == "five" else adj
            agree = np.ones(valid.shape, bool)
            for i in range(1, len(names_arms)):
                agree &= arms_s[i] == arms_s[0]
            contested = valid & ~agree
            total_contested[scheme] += int(contested.sum())
            for c in classes(scheme):
                cm = valid & (adj_s == c)
                adj_total[scheme][c] += int(cm.sum())
                adj_contested[scheme][c] += int((cm & contested).sum())
        # one-sided Unknown, pairwise
        for i in range(len(names_arms)):
            for j in range(i + 1, len(names_arms)):
                ai, aj = stack[i], stack[j]
                one = ((ai == 10) & (aj != 10)) | ((aj == 10) & (ai != 10))
                other = np.where(ai == 10, aj, ai)[one]
                onesided_total += int(one.sum())
                for scheme in ("ten", "five"):
                    oc = FIVE_LUT[other] if scheme == "five" else other
                    v, cnt = np.unique(oc, return_counts=True)
                    for vv, nn in zip(v, cnt):
                        onesided[scheme][int(vv)] += int(nn)

    # ---- 180-cell valid denominator (single cells: non-Unknown of the one arm) ----
    valid_180 = valid_72
    for key, revs in single.items():
        arr = read(next(iter(revs.values())))
        valid_180 += int((arr != 10).sum())

    # ---- write CSVs ----
    rows_A = []
    for scheme in ("ten", "five"):
        for c in classes(scheme):
            dt, dc = adj_total[scheme][c], adj_contested[scheme][c]
            rows_A.append(dict(scheme=f"{scheme}_class", metric="contested_share_of_adjudicated",
                               **{"class": label(scheme, c)}, interpreter="NA",
                               numerator=dc, denominator=dt, percentage=round(100 * dc / dt, 3) if dt else np.nan))
            tc = total_contested[scheme]
            rows_A.append(dict(scheme=f"{scheme}_class", metric="contested_composition",
                               **{"class": label(scheme, c)}, interpreter="NA",
                               numerator=dc, denominator=tc, percentage=round(100 * dc / tc, 3) if tc else np.nan))
    pd.DataFrame(rows_A).to_csv(os.path.join(OUT, "metric_A_contested_share.csv"), index=False)

    rows_C = []
    for scheme in ("ten", "five"):
        tc = total_contested[scheme]
        rows_C.append(dict(scheme=f"{scheme}_class", metric="dilution_72cell", **{"class": "NA"},
                           interpreter="NA", numerator=tc, denominator=valid_72,
                           percentage=round(100 * tc / valid_72, 4)))
        rows_C.append(dict(scheme=f"{scheme}_class", metric="dilution_180cell", **{"class": "NA"},
                           interpreter="NA", numerator=tc, denominator=valid_180,
                           percentage=round(100 * tc / valid_180, 4)))
    pd.DataFrame(rows_C).to_csv(os.path.join(OUT, "metric_C_dilution.csv"), index=False)

    rows_U = []
    for scheme in ("ten", "five"):
        for c in sorted(onesided[scheme]):
            n = onesided[scheme][c]
            rows_U.append(dict(scheme=f"{scheme}_class", metric="one_sided_unknown",
                               **{"class": label(scheme, c)}, interpreter="NA",
                               numerator=n, denominator=onesided_total,
                               percentage=round(100 * n / onesided_total, 3) if onesided_total else np.nan))
    pd.DataFrame(rows_U).to_csv(os.path.join(OUT, "one_sided_unknown.csv"), index=False)

    sel_rows = [dict(interpreter=r, cells_selected=sel_cells[r], valid_pixels=sel_pixels[r],
                     pixel_share_pct=round(100 * sel_pixels[r] / valid_72, 3))
                for r in sorted(sel_cells, key=lambda x: -sel_cells[x])]
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "selection_frequency.csv"), index=False)

    # ---- printed accounting + the manuscript summary (Metric A, five-class) ----
    print("\n" + "=" * 78 + "\nVALID-PIXEL ACCOUNTING\n" + "=" * 78)
    print(f"  pixels dropped by pairwise-Unknown rule (72 cells): {dropped_unknown:,}")
    print(f"  valid (paired) pixels, 72 multiply cells: {valid_72:,}")
    print(f"  valid pixels, full 180-cell set:          {valid_180:,}")
    print(f"  one-sided-Unknown pixels (pairwise total): {onesided_total:,}")

    print("\nSELECTION FREQUENCY (per interpreter, over 72 adjudicated cells)")
    print(pd.DataFrame(sel_rows).to_string(index=False))

    print("\nMETRIC C  DILUTION")
    print(pd.DataFrame(rows_C)[["scheme", "metric", "numerator", "denominator", "percentage"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("MANUSCRIPT SUMMARY  -  METRIC A, FIVE-CLASS COLLAPSE")
    print("contested share = % of adjudicated-class pixels that were contested")
    print("=" * 78)
    a5 = [r for r in rows_A if r["scheme"] == "five_class" and r["metric"] == "contested_share_of_adjudicated"]
    comp5 = {r["class"]: r for r in rows_A if r["scheme"] == "five_class" and r["metric"] == "contested_composition"}
    print(f"{'class':16} {'adj pixels':>12} {'contested':>10} {'% contested':>12} {'% of all contested':>20}")
    for r in a5:
        cc = comp5[r["class"]]
        print(f"{r['class']:16} {r['denominator']:>12,} {r['numerator']:>10,} "
              f"{r['percentage']:>11}% {cc['percentage']:>19}%")

    print(f"\nwrote CSVs -> {OUT}/  (metric_A_contested_share, metric_C_dilution, one_sided_unknown, selection_frequency)")


if __name__ == "__main__":
    main()
