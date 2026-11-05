#!/usr/bin/env python3
"""Read-only diagnostic: does the choice of interpretation at double-interpreted Landsat locations
change the Prithvi scores?

pick A = deduped_cells() (default_rng seed 42); pick B = the same, except every location with two or
more interpretations uses a different one than pick A (with three or more, the next one in sorted
order after pick A's, cyclically). Scored as scripts/prithvi_year_offset.py: prithvi_loader argmax,
centre lookup, zero-input mask "any_year" from the label-year chip, 4x4 census, metrics_f1_zero,
cell bootstrap (seed 42, 2000 draws) via collapsed_5class_confusion.bootstrap_cis. Offsets -1 and 0
are explicit (label year = target + offset), independent of prithvi_loader.PRITHVI_YEAR_RULE:
offset -1 is the confirmed rule (target_minus_1), offset 0 the old "target" rule.

This is the sensitivity analysis for the reference selection used by collapsed_5class_confusion.py
--prithvi (reproducible seeded selection, seed 42, kept per decision). Landsat only. Refuses to run
without LCI_OUT_DIR; writes only to $LCI_OUT_DIR/reports/prithvi_dedup_check/:
  - multi_interpreted_locations.csv   the double-interpreted locations, reviewer agreement, class px
  - pick_A_vs_B_metrics.csv           per model x offset x metric: pick A, pick B, diff, A's CI
  - dedup_check_summary.md            tables, largest differences, model rankings

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs LCI_OUT_DIR=check_dedup \
        /vol/v1/bekka/venvs/lci/bin/python scripts/prithvi_dedup_check.py
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collapsed_5class_confusion as C   # noqa: E402  (imports only; writes nothing)
import prithvi_loader as PL              # noqa: E402

OFFSETS = [-1, 0]
MASK = "any_year"
METRICS = [("OA", "OA"), ("kappa", "kappa"), ("macro_F1", "macro-F1"),
           ("recall[2]", "Harvest R"), ("precision[2]", "Harvest P"), ("F1[2]", "Harvest F1"),
           ("F1[3]", "Dev F1"), ("F1[4]", "I/D F1")]


def load(ref_path, model, year):
    """As prithvi_year_offset.load(): (codes, zero) on the reference grid, or None."""
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


def groups():
    paths = sorted(glob.glob(os.path.join(C.RF_DIR, "**", C.SC.RF_GLOB), recursive=True))
    g = defaultdict(list)
    for p in paths:
        g[C.location_key(p)].append(p)
    return {k: sorted(v) for k, v in g.items()}


def score(cells):
    """{(model, offset): [per-cell 4x4]} and per-offset matched cell sets."""
    cms = {(m, o): [] for m in PL.PRITHVI_MODELS for o in OFFSETS}
    matched = {o: set() for o in OFFSETS}
    for cell in cells:
        _, target, _ = PL.parse_reference(cell)
        with rasterio.open(cell) as ds:
            rf = ds.read(1)
        for o in OFFSETS:
            loaded = {m: load(cell, m, target + o) for m in PL.PRITHVI_MODELS}
            n_none = sum(v is None for v in loaded.values())
            if n_none:
                if n_none != len(loaded):
                    raise RuntimeError(f"{cell} offset {o}: output for only some models")
                continue
            matched[o].add(cell)
            for m, (codes, zero) in loaded.items():
                cm, _, _ = C.prithvi_cell_tallies(rf, codes, (codes != PL.NODATA) & ~zero)
                if cm.sum() > 0:          # as main_prithvi: only cells with valid pixels
                    cms[(m, o)].append(cm)
    return cms, matched


def read_on(path, ref_ds):
    """Collapsed reference of `path` on ref_ds's grid (centre lookup; 0 off its footprint)."""
    with rasterio.open(path) as ds:
        a, t = C.collapse_reference(ds.read(1)), ds.transform
    return PL.align_to_reference(a, t, ref_ds, fill=0)


def main():
    if C.SC.SENSOR != "Landsat":
        sys.exit("run with LCI_SENSOR=Landsat")
    if C.SC.OUT_DIR is None:
        sys.exit("set LCI_OUT_DIR (outputs go to $LCI_OUT_DIR/reports/prithvi_dedup_check/)")
    out = C.SC.out_path("reports/prithvi_dedup_check")
    os.makedirs(out, exist_ok=True)
    g = groups()
    pick_a = C.deduped_cells()
    a_set = set(pick_a)
    multi = {k: v for k, v in g.items() if len(v) > 1}
    swap, multi_rows = {}, []
    for k, v in sorted(multi.items()):
        (a,) = [p for p in v if p in a_set]
        b = v[(v.index(a) + 1) % len(v)]
        swap[a] = b
        multi_rows.append(dict(grid=k[0], sample=k[1], target=int(k[2]), n_interp=len(v),
                               reviewers="+".join(sorted(PL.parse_reference(p)[2] for p in v)),
                               pick_A=PL.parse_reference(a)[2], pick_B=PL.parse_reference(b)[2],
                               ref_A=os.path.basename(a), ref_B=os.path.basename(b)))
    pick_b = sorted(swap.get(p, p) for p in pick_a)
    assert len(pick_a) == len(pick_b) == len(g)
    lines = [f"# Dedup choice check (Landsat, mask {MASK})", "",
             f"Reference rasters {sum(map(len, g.values()))}, locations {len(g)}, "
             f"locations with >1 interpretation: {len(multi)} "
             f"(max {max(map(len, multi.values()))} per location).", ""]

    # ---- 1 + 2: the multi-interpreted locations
    agree_rows = []
    for r in multi_rows:
        pa = [p for p in g[(r["grid"], r["sample"], str(r["target"]))] if os.path.basename(p) == r["ref_A"]][0]
        pb = [p for p in g[(r["grid"], r["sample"], str(r["target"]))] if os.path.basename(p) == r["ref_B"]][0]
        with rasterio.open(pa) as ds, rasterio.open(pb) as ds_b:
            same_grid = ds_b.transform == ds.transform
            ra = read_on(pa, ds)
            rb = read_on(pb, ds)
        cell_id = PL.parse_reference(pa)[0]
        both = (ra >= 1) & (rb >= 1)
        row = dict(r, cell_id=cell_id, same_grid=bool(same_grid), both_valid_px=int(both.sum()),
                   agree_px=int((both & (ra == rb)).sum()))
        row["agreement"] = row["agree_px"] / row["both_valid_px"] if row["both_valid_px"] else np.nan
        for name, ref in (("A", ra), ("B", rb)):
            row[f"excluded_{name}"] = int((ref == 0).sum())
            for c, lab in ((2, "harvest"), (3, "development"), (4, "insect_disease"), (5, "beaver")):
                row[f"{lab}_{name}"] = int((ref == c).sum())
        for o in OFFSETS:
            row[f"prithvi_off{o:+d}"] = os.path.exists(PL.prithvi_path(PL.PRITHVI_MODELS[0],
                                                                      r["target"] + o, cell_id))
        agree_rows.append(row)
    ad = pd.DataFrame(agree_rows)
    ad.to_csv(os.path.join(out, "multi_interpreted_locations.csv"), index=False)

    lines += ["## 1. Multi-interpreted locations", "",
              "| reviewer pair | locations | Prithvi match off -1 | Prithvi match off 0 |",
              "|---|---:|---:|---:|"]
    for pair, d in list(ad.groupby("reviewers")) + [("all", ad)]:
        lines.append(f"| {pair} | {len(d)} | {int(d['prithvi_off-1'].sum())} | "
                     f"{int(d['prithvi_off+0'].sum())} |")
    lines += ["", f"All on the same grid as each other: {bool(ad['same_grid'].all())}. "
              f"Pick A reviewer counts: {ad['pick_A'].value_counts().to_dict()}.", ""]

    lines += ["## 2. Reviewer disagreement on those locations (5-class collapse, whole raster)", "",
              "Agreement = share of pixels where both maps are in classes 1-5 (Unknown/Fire excluded) "
              "with the same class. Pixel counts are per reviewer (first / second in the pair, "
              "alphabetical).", "",
              "| reviewer pair | locs | both-valid px | agreement | Harvest | Development | "
              "Insect/Disease |", "|---|---:|---:|---:|---:|---:|---:|"]
    # re-key by reviewer name rather than A/B
    per_rev = []
    for _, r in ad.iterrows():
        first, second = r["reviewers"].split("+")
        key = {r["pick_A"]: "A", r["pick_B"]: "B"}
        per_rev.append({**{f"{lab}_{i}": r[f"{lab}_{key[n]}"]
                           for i, n in ((1, first), (2, second))
                           for lab in ("harvest", "development", "insect_disease")},
                        "reviewers": r["reviewers"], "both": r["both_valid_px"], "agree": r["agree_px"]})
    pr = pd.DataFrame(per_rev)
    for pair, d in list(pr.groupby("reviewers")) + [("all (first/second within pair)", pr)]:
        s = d.sum(numeric_only=True)
        lines.append(f"| {pair} | {len(d)} | {int(s['both']):,} | {s['agree'] / s['both']:.3f} | "
                     + " | ".join(f"{int(s[f'{lab}_1']):,} / {int(s[f'{lab}_2']):,}"
                                  for lab in ("harvest", "development", "insect_disease")) + " |")
    s = ad.sum(numeric_only=True)
    lines += ["", f"Pick A vs pick B totals on these {len(ad)} locations: "
              + ", ".join(f"{lab} {int(s[f'{lab}_A']):,} vs {int(s[f'{lab}_B']):,}"
                          for lab in ("harvest", "development", "insect_disease", "beaver")) + ".",
              f"Per-location agreement: median {ad['agreement'].median():.3f}, "
              f"min {ad['agreement'].min():.3f}.", ""]

    # ---- 3: scores, pick A vs pick B
    res = {}
    for name, cells in (("A", pick_a), ("B", pick_b)):
        print(f"scoring pick {name} ...")
        res[name] = score(cells)
    rows = []
    for m in PL.PRITHVI_MODELS:
        for o in OFFSETS:
            ca, cb = np.array(res["A"][0][(m, o)]), np.array(res["B"][0][(m, o)])
            pa, pb = C.metrics_f1_zero(ca.sum(0)), C.metrics_f1_zero(cb.sum(0))
            bci = C.bootstrap_cis(ca, len(ca), metrics_fn=C.metrics_f1_zero)
            for k, lab in METRICS:
                lo, hi = bci[k]
                d = pb[k] - pa[k]
                rows.append(dict(model=m, offset=o, metric=lab, n_A=len(ca), n_B=len(cb),
                                 A=pa[k], B=pb[k], diff_B_minus_A=d, A_ci_lo=lo, A_ci_hi=hi,
                                 A_ci_width=hi - lo, abs_diff_over_width=abs(d) / (hi - lo)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "pick_A_vs_B_metrics.csv"), index=False)

    for o in OFFSETS:
        lines += [f"## 3. Offset {o:+d}: pick A vs pick B (all scored cells)", "",
                  f"Matched cells: pick A {len(res['A'][1][o])}, pick B {len(res['B'][1][o])}. "
                  "Cell: A -> B (B - A; width of A's 95% bootstrap CI; |B - A| / width).", "",
                  "| model | n | " + " | ".join(l for _, l in METRICS) + " |",
                  "|---|---:|" + "---|" * len(METRICS)]
        for m in PL.PRITHVI_MODELS:
            d = df[(df.model == m) & (df.offset == o)].set_index("metric")
            lines.append(f"| {m} | {d['n_A'].iloc[0]} | " + " | ".join(
                f"{d.at[l, 'A']:.3f} → {d.at[l, 'B']:.3f} ({d.at[l, 'diff_B_minus_A']:+.3f}; "
                f"{d.at[l, 'A_ci_width']:.3f}; {d.at[l, 'abs_diff_over_width']:.2f})"
                for _, l in METRICS) + " |")
        lines.append("")

    # ---- 4: conclusion inputs
    big = df[df.abs_diff_over_width > 0.25]
    lines += ["## 4. Summary", "",
              f"Largest |B - A| / CI width: {df.abs_diff_over_width.max():.2f} "
              f"({df.loc[df.abs_diff_over_width.idxmax(), ['model', 'offset', 'metric']].to_dict()}).",
              f"Entries above 0.25: {len(big)} of {len(df)}."]
    lines += [f"- {r.model} off {r.offset:+d} {r.metric}: {r.A:.3f} -> {r.B:.3f} "
              f"({r.diff_B_minus_A:+.3f}, {r.abs_diff_over_width:.2f} of width {r.A_ci_width:.3f})"
              for r in big.itertuples()]
    lines += ["", "Model ranking (best first) per metric; '=' where pick A and pick B agree:", ""]
    changed = 0
    for o in OFFSETS:
        for _, lab in METRICS:
            d = df[(df.offset == o) & (df.metric == lab)]
            ra = list(d.sort_values("A", ascending=False).model)
            rb = list(d.sort_values("B", ascending=False).model)
            same = ra == rb
            changed += not same
            short = lambda r: " > ".join(x.replace("_encoder_q1_w2", "").replace("_q1_w2", "") for x in r)
            gap = ""
            if not same:   # spread of A values among the models whose order changed
                moved = [x for x, y in zip(ra, rb) if x != y]
                v = d.set_index("model").loc[moved, "A"]
                gap = f" (pick A spread among swapped models {v.max() - v.min():.4f})"
            lines.append(f"- off {o:+d} {lab}: {'=' if same else 'CHANGED'} A: {short(ra)}"
                         + ("" if same else f" | B: {short(rb)}{gap}"))
    lines += ["", f"Rankings changed: {changed} of {len(OFFSETS) * len(METRICS)}.", "",
              "Offset -1 is the confirmed year rule (target_minus_1); offset 0 is the old 'target' rule."]
    with open(os.path.join(out, "dedup_check_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
