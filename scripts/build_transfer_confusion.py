#!/usr/bin/env python3
"""Per-class confusion matrices for the classifier temporal-transferability experiment.

A Random Forest trained once on 2018/2020 AlphaEarth embeddings was applied to five NAIP brackets
(2017_2019, 2018_2020, 2019_2021, 2020_2022, 2021_2023), restricted to the CKIT-RF interpreted
cells for each bracket (36 disjoint cells per bracket). 2018_2020 is the in-sample control. This
compares the GEE-exported per-cell predictions against the CKIT-RF interpreted reference and
accumulates a 10x10 confusion per (variant, bracket).

Predictions live in data/raw/transfer_predictions/<bracket>/pred_<bracket>_cell<id>.tif, 5 bands in
the fixed order band1=v2, band2=v3, band3=v4, band4=v5, band5=v6, values 1 to 10. The CKIT-RF
reference rasters live in data/raw/rf_class_maps/, single band, int32, CKIT label_id codes remapped
to the 10-class schema. The predictions were exported pinned to each CKIT raster's grid, so a
prediction and its reference must align pixel for pixel; a grid mismatch means the pinning failed,
and the pair is skipped rather than resampled.

Outputs -> reports/transfer_confusion/

Requires: rasterio, numpy, pandas
"""

import argparse
import glob
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
import sensor_config as SC  # RF_DIR, out_path (LCI_RF_DIR / LCI_OUT_DIR)

warnings.filterwarnings("ignore")

PRED_DIR = "data/raw/transfer_predictions"
RF_DIR = SC.RF_DIR
OUT = SC.out_path("reports/transfer_confusion")
SEED = 42
BRACKETS = ["2017_2019", "2018_2020", "2019_2021", "2020_2022", "2021_2023"]
CONTROL = "2018_2020"
BANDS = {1: "v2", 2: "v3", 3: "v4", 4: "v5", 5: "v6"}   # fixed band-to-variant order
MIN_SUP = 100                                           # px floor for a per-class metric
MIN_CELLS = 5                                           # cells a class must appear in for reliable metrics

# ckit label_id -> 10-class schema code, applied to the reference only
CROSSWALK = {0: 4, 1: 6, 2: 7, 3: 3, 4: 5, 5: 8, 20: 1, 30: 2, 50: 10, 62: 9}
EXCLUDE = {10, 13}                                      # unknown abstention, other_no_change: drop the pixel
ALLOWED = set(CROSSWALK) | EXCLUDE
NAMES = {1: "harvest", 2: "development", 3: "forest", 4: "urban", 5: "water", 6: "ag",
         7: "grass_shrub", 8: "wetland", 9: "beaver", 10: "insect_disease"}
LABELS = [NAMES[c] for c in range(1, 11)]

# reference remap lut: crosswalk -> schema, excluded/unmapped -> 0 (dropped downstream)
_REF_LUT = np.zeros(63, np.uint8)
for _k, _v in CROSSWALK.items():
    _REF_LUT[_k] = _v


def pad(gid):
    return str(int(gid)).zfill(5)


def build_reference_index():
    """cell_id -> sorted list of (reviewer, path) for the 10 m Sentinel-2 reference rasters."""
    idx = defaultdict(list)
    rx = re.compile(r"reviewer_([A-Za-z]+)_grid_(\d+)_sample_", re.I)
    for p in sorted(glob.glob(os.path.join(RF_DIR, "**", "rf_class*.tif"), recursive=True)):
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        # skip non-10 m rasters (e.g. the out-of-scope 30 m landsat record) so they never match
        with rasterio.open(p) as ds:
            if not (ds.transform.a == 10 and ds.transform.e == -10):
                continue
        idx[pad(m.group(2))].append((m.group(1).lower(), p))
    return {k: sorted(v) for k, v in idx.items()}


def choose_references(ref_index):
    """Pick one reference per cell_id; random for double-interpreted cells (default_rng seed 42).

    Draws are made in sorted cell_id order so the selection is deterministic and independent of
    filesystem order. Returns (chosen, n_double).
    """
    rng = np.random.default_rng(SEED)
    chosen, n_double = {}, 0
    for cid in sorted(ref_index):
        rp = ref_index[cid]
        if len(rp) > 1:
            n_double += 1
            chosen[cid] = rp[int(rng.integers(len(rp)))][1]
        else:
            chosen[cid] = rp[0][1]
    return chosen, n_double


def load_truth(truth_csv):
    """cell_id -> chosen reviewer, from the adjudicated truth selections."""
    df = pd.read_csv(truth_csv, dtype=str, keep_default_na=False)
    return {pad(r.grid_id): r.reviewer.strip().lower() for r in df.itertuples()}


def choose_references_truth(ref_index, truth):
    """Pick the adjudicated reviewer's reference per cell_id. Returns (chosen, n_multi, missing, mismatch)."""
    chosen, n_multi, missing, mismatch = {}, 0, [], []
    for cid in sorted(ref_index):
        revpaths = ref_index[cid]
        want = truth.get(cid)
        if want is None:                                    # indexed cell absent from the truth set
            missing.append(cid)
            continue
        match = [p for r, p in revpaths if r == want]
        if not match:                                       # truth names a reviewer with no raster here
            mismatch.append((cid, want, [r for r, _ in revpaths]))
            continue
        chosen[cid] = match[0]
        if len(revpaths) > 1:
            n_multi += 1
    return chosen, n_multi, missing, mismatch


def grids_match(a, b, atol=1e-6):
    if a.width != b.width or a.height != b.height:
        return False, "size"
    if str(a.crs) != str(b.crs):
        return False, "crs"
    ta, tb = a.transform, b.transform
    if not np.allclose([ta.a, ta.b, ta.c, ta.d, ta.e, ta.f],
                       [tb.a, tb.b, tb.c, tb.d, tb.e, tb.f], atol=atol):
        return False, "transform"
    return True, ""


# descriptive titles per source; the bracket or cap detail goes in the caption below, not the title
DISPLAY = {"harvest": "Harvest", "development": "Development", "forest": "Forest", "urban": "Urban",
           "water": "Water", "ag": "Agriculture", "grass_shrub": "Grass/Shrub", "wetland": "Wetland",
           "beaver": "Beaver", "insect_disease": "Insect/Disease"}
LABELS_DISPLAY = [DISPLAY[NAMES[c]] for c in range(1, 11)]
LABELS5 = ["Stable", "Harvest", "Development", "Insect/Disease", "Beaver"]
VARIANT_TITLE = {
    "v2": "AlphaEarth Embedding v2, baseline + delta",
    "v3": "AlphaEarth Embedding v3, stacked years",
    "v4": "AlphaEarth Embedding v4, delta only",
    "v5": "AlphaEarth Embedding v5, baseline + dot",
    "v6": "AlphaEarth Embedding v6, dot only",
    "spec_all": "Spectral Composite, all sensors all indices",
    "specall": "Spectral Composite, all sensors all indices",
    "cap50": "Change-cap, 50 training points per change class",
    "cap100": "Change-cap, 100 training points per change class",
    "cap150": "Change-cap, 150 training points per change class",
    "cap200": "Change-cap, 200 training points per change class",
}


def _cm_caption(variant, bracket, k):
    scheme = "Ten-class" if k == 10 else "Five-class"
    if str(variant).startswith("cap"):
        ctx = f"pooled over 180 cells, change-class training cap of {str(variant)[3:]} points"
    elif bracket == "pooled":
        ctx = "pooled over all cells"
    elif bracket in BRACKETS:
        ctx = f"NAIP bracket {bracket.replace('_', '-')}" + (
            " (in-sample control)" if bracket == CONTROL else "")
    else:
        ctx = str(bracket)
    return (f"{scheme} confusion matrix, {ctx}, against the interpreted reference. Reference classes "
            "are on the rows and predicted classes on the columns. Cells are raw pixel counts colored "
            "by the row proportion, so the diagonal shade is the producer's accuracy. The PA column "
            "is producer's accuracy (recall) with the reference support n; the UA row is user's "
            "accuracy (precision) with the predicted support n; the corner gives overall accuracy and "
            "Cohen's kappa.")


def render_cm_png(M, mt, variant, bracket, path, labels=None):
    """Heatmap of one confusion matrix (k=5 or 10): cells are raw counts colored by the row
    proportion, with a PA column and reference support, a UA row and predicted support, and OA and
    kappa in the corner. Descriptive title, a caption band below the matrix, and print-size fonts.
    Metrics are computed from M, so mt may be None (kept for call compatibility).
    """
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = M.astype(float)
    k = M.shape[0]
    labels = labels if labels is not None else (LABELS_DISPLAY if k == 10 else LABELS5)
    tp = np.diag(M); row = M.sum(1); col = M.sum(0); tot = M.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = M / np.where(row[:, None] > 0, row[:, None], np.nan)   # row proportion
        pa = np.where(row > 0, tp / row, np.nan)                    # producer's (recall)
        ua = np.where(col > 0, tp / col, np.nan)                    # user's (precision)
    oa = tp.sum() / tot if tot else np.nan
    pe = (row * col).sum() / (tot * tot) if tot else np.nan
    kappa = (oa - pe) / (1 - pe) if tot and (1 - pe) else np.nan
    blues, greens = plt.get_cmap("Blues"), plt.get_cmap("Greens")

    img = np.ones((k + 1, k + 1, 4))
    for i in range(k):
        for j in range(k):
            img[i, j] = blues(rn[i, j] if np.isfinite(rn[i, j]) else 0.0)
        img[i, k] = greens(pa[i] if np.isfinite(pa[i]) else 0.0)
    for j in range(k):
        img[k, j] = greens(ua[j] if np.isfinite(ua[j]) else 0.0)
    img[k, k] = greens(oa if np.isfinite(oa) else 0.0)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(img, aspect="auto")

    def txtcolor(v):
        return "white" if (np.isfinite(v) and v > 0.5) else "black"

    csz = 8.5 if k == 10 else 11
    for i in range(k):
        for j in range(k):
            c = int(M[i, j])
            if c:
                ax.text(j, i, f"{c:,}", ha="center", va="center", fontsize=csz, color=txtcolor(rn[i, j]))
    for i in range(k):                                   # PA column + reference support
        t = f"{pa[i]*100:.0f}%" if np.isfinite(pa[i]) else "-"
        ax.text(k, i, f"{t}\nn={int(row[i]):,}", ha="center", va="center", fontsize=8, color=txtcolor(pa[i]))
    for j in range(k):                                   # UA row + predicted support
        t = f"{ua[j]*100:.0f}%" if np.isfinite(ua[j]) else "-"
        ax.text(j, k, f"{t}\nn={int(col[j]):,}", ha="center", va="center", fontsize=8, color=txtcolor(ua[j]))
    ax.text(k, k, f"OA {oa*100:.0f}%\nκ {kappa:.2f}", ha="center", va="center", fontsize=9,
            color=txtcolor(oa))

    ax.set_xticks(range(k + 1)); ax.set_xticklabels(labels + ["PA"], rotation=45, ha="left", fontsize=11)
    ax.set_yticks(range(k + 1)); ax.set_yticklabels(labels + ["UA"], fontsize=11)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("prediction (columns)", fontsize=12)
    ax.set_ylabel("reference (rows)", fontsize=12)
    ax.axhline(k - 0.5, color="0.4", lw=1.0); ax.axvline(k - 0.5, color="0.4", lw=1.0)
    ax.set_xticks(np.arange(-0.5, k + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k + 1, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    ax.set_title(VARIANT_TITLE.get(variant, str(variant)), fontsize=15, fontweight="bold", pad=30)

    # caption band below the matrix
    wrapped = "\n".join(textwrap.wrap(_cm_caption(variant, bracket, k), 108))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.03 + 0.028 * nlines, 1, 1])
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=9, color="0.3")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metrics(M):
    M = M.astype(float)
    tp = np.diag(M); row = M.sum(1); col = M.sum(0); tot = M.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(row > 0, tp / row, np.nan)        # producer's accuracy
        precision = np.where(col > 0, tp / col, np.nan)      # user's accuracy
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
        iou = np.where((row + col - tp) > 0, tp / (row + col - tp), np.nan)
    present = row > 0
    oa = tp.sum() / tot if tot else np.nan
    pe = (row * col).sum() / (tot * tot) if tot else np.nan
    kappa = (oa - pe) / (1 - pe) if tot and (1 - pe) != 0 else np.nan
    return dict(OA=oa, kappa=kappa,
                macro_F1=np.nanmean(f1[present]) if present.any() else np.nan,
                mean_IoU=np.nanmean(iou[present]) if present.any() else np.nan,
                precision=precision, recall=recall, f1=f1, iou=iou, support=row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default=None,
                    help="use the adjudicated reviewer per cell from this CSV instead of a random pick")
    ap.add_argument("--out", default=None, help="output folder (default depends on --truth)")
    args = ap.parse_args()

    global OUT
    ref_index = build_reference_index()
    if args.truth:
        truth = load_truth(args.truth)
        chosen_ref, n_double, missing, mismatch = choose_references_truth(ref_index, truth)
        ref_desc = (f"the adjudicated reviewer from {os.path.basename(args.truth)} "
                    f"(notebooks/adjudicate_truth.ipynb)")
        OUT = args.out or SC.out_path("reports/transfer_confusion_adjudicated")
        os.makedirs(OUT, exist_ok=True)
        print(f"reference cells indexed: {len(ref_index)}   multi-interpreted using the adjudicated "
              f"choice: {n_double}  -> reference from {args.truth}")
        if missing:
            print(f"  note: {len(missing)} indexed cell(s) not in the truth set (not scored here): "
                  f"{missing[:10]}")
        if mismatch:
            print(f"  STOP: {len(mismatch)} cell(s) whose truth reviewer has no matching raster:")
            for cid, want, have in mismatch:
                print(f"    {cid}: truth={want}, rasters={have}")
            raise SystemExit(1)
    else:
        chosen_ref, n_double = choose_references(ref_index)
        ref_desc = f"random with a fixed seed (numpy.default_rng({SEED}))"
        OUT = args.out or OUT
        os.makedirs(OUT, exist_ok=True)
        print(f"reference cells indexed: {len(ref_index)}   double-interpreted (>=2 reviewers): "
              f"{n_double}  -> one chosen at random per cell (default_rng seed {SEED})")

    # per (variant, bracket) accumulator
    cms = {(v, b): np.zeros((10, 10), np.int64) for v in BANDS.values() for b in BRACKETS}
    cells_used = defaultdict(set)                       # bracket -> set of cell_ids scored
    cells_present = {b: np.zeros(11, int) for b in BRACKETS}   # bracket -> per-class cell count
    skipped = {"missing_ref": [], "grid_mismatch": [], "band_count": []}
    unmapped = defaultdict(int)                         # ref value outside crosswalk/exclude -> px count
    bracket_mismatch = []

    for bracket in BRACKETS:
        preds = sorted(glob.glob(os.path.join(PRED_DIR, bracket, f"pred_{bracket}_cell*.tif")))
        for pp in preds:
            m = re.search(r"cell(\d+)\.tif$", os.path.basename(pp))
            cid = pad(m.group(1))
            if cid not in chosen_ref:
                skipped["missing_ref"].append((bracket, cid))
                continue
            rp = chosen_ref[cid]
            # soft check: the chosen reference's bracket should match the prediction bracket
            mo = re.search(r"opt_(\d{4}_\d{4})", os.path.basename(rp))
            if mo and mo.group(1) != bracket:
                bracket_mismatch.append((bracket, cid, mo.group(1)))
            with rasterio.open(pp) as pds, rasterio.open(rp) as rds:
                if pds.count != 5:
                    skipped["band_count"].append((bracket, cid, pds.count))
                    continue
                ok, why = grids_match(pds, rds)
                if not ok:
                    skipped["grid_mismatch"].append(
                        (bracket, cid, why,
                         (pds.width, pds.height, tuple(round(x, 3) for x in pds.transform[:6])),
                         (rds.width, rds.height, tuple(round(x, 3) for x in rds.transform[:6]))))
                    continue
                ref_raw = rds.read(1)
                # flag any reference value outside the crosswalk and the exclude set
                for val in np.unique(ref_raw):
                    iv = int(val)
                    if iv not in ALLOWED:
                        unmapped[(bracket, cid, iv)] += int((ref_raw == val).sum())
                safe = np.where((ref_raw >= 0) & (ref_raw <= 62), ref_raw, 0)
                ref = _REF_LUT[safe]                    # excluded/unmapped -> 0
                # count this cell once per reference class present (design-aware support)
                for k in range(1, 11):
                    if (ref == k).any():
                        cells_present[bracket][k] += 1
                for band, variant in BANDS.items():
                    pred = pds.read(band)
                    valid = (ref >= 1) & (ref <= 10) & (pred >= 1) & (pred <= 10)
                    if valid.any():
                        np.add.at(cms[(variant, bracket)], (ref[valid] - 1, pred[valid] - 1), 1)
                cells_used[bracket].add(cid)

    # ---- report gate: unmapped reference values are an encoding error ----
    if unmapped:
        print("\nUNMAPPED reference values found (outside crosswalk and {10,13}):")
        agg = defaultdict(int)
        for (bracket, cid, iv), n in unmapped.items():
            agg[iv] += n
            print(f"  bracket {bracket} cell {cid}: value {iv} x {n} px")
        print(f"  totals by value: {dict(agg)}")

    # ---- metrics, matrices, long table ----
    long_rows = []
    print(f"\n{'variant':6}{'bracket':11}{'cells':>6}{'pixels':>10}{'OA':>7}{'macF1':>7}"
          f"{'mIoU':>7}{'kappa':>7}{'minSupCls':>22}")
    for bracket in BRACKETS:
        for variant in BANDS.values():
            M = cms[(variant, bracket)]
            pd.DataFrame(M, index=LABELS, columns=LABELS).to_csv(
                os.path.join(OUT, f"cm_{variant}_{bracket}.csv"))
            mt = metrics(M)
            render_cm_png(M, mt, variant, bracket, os.path.join(OUT, f"cm_{variant}_{bracket}.png"))
            support = mt["support"]
            tot = int(M.sum())
            # lowest-support present class, to keep low support visible in the running summary
            pres = np.where(support > 0)[0]
            if len(pres):
                lk = pres[np.argmin(support[pres])]
                minsup = f"{NAMES[lk+1]}={int(support[lk])}"
            else:
                minsup = "none"
            print(f"{variant:6}{bracket:11}{len(cells_used[bracket]):>6}{tot:>10,}"
                  f"{mt['OA']:>7.3f}{mt['macro_F1']:>7.3f}{mt['mean_IoU']:>7.3f}"
                  f"{mt['kappa']:>7.3f}{minsup:>22}")
            for k in range(10):
                c = k + 1
                sup = int(support[k])
                cp = int(cells_present[bracket][c])
                # low support is design-aware: a class in fewer than MIN_CELLS cells, or under the
                # pixel floor, has an unreliable per-class metric since pixels within a cell are
                # autocorrelated
                low = cp < MIN_CELLS or sup < MIN_SUP
                long_rows.append(dict(
                    variant=variant, bracket=bracket, control=(bracket == CONTROL),
                    class_code=c, class_name=NAMES[c],
                    precision=_r(mt["precision"][k]), recall=_r(mt["recall"][k]),
                    f1=_r(mt["f1"][k]), iou=_r(mt["iou"][k]), support=sup,
                    cells_present=cp, low_support=low,
                    OA=round(mt["OA"], 5), macro_F1=round(mt["macro_F1"], 5),
                    mean_IoU=round(mt["mean_IoU"], 5), kappa=round(mt["kappa"], 5),
                    n_cells=len(cells_used[bracket]), total_pixels=tot))
    pd.DataFrame(long_rows).to_csv(os.path.join(OUT, "transfer_metrics_long.csv"), index=False)

    # ---- skip report ----
    print("\nskipped pairs:")
    print(f"  missing reference: {len(skipped['missing_ref'])}  {skipped['missing_ref'][:5]}")
    print(f"  grid mismatch:     {len(skipped['grid_mismatch'])}")
    for s in skipped["grid_mismatch"][:10]:
        print(f"    {s}")
    print(f"  band count != 5:   {len(skipped['band_count'])}  {skipped['band_count'][:5]}")
    if bracket_mismatch:
        print(f"  reference/prediction bracket mismatch: {len(bracket_mismatch)} {bracket_mismatch[:5]}")

    write_note(n_double, cells_used, skipped, unmapped, bracket_mismatch, ref_desc)
    print(f"\nwrote {OUT}/ (25 matrices, transfer_metrics_long.csv, note)")


def _r(x):
    return round(float(x), 5) if np.isfinite(x) else ""


def write_note(n_double, cells_used, skipped, unmapped, bracket_mismatch, ref_desc):
    low = []
    # low-support classes are the same across variants within a bracket, so read one variant
    long = pd.read_csv(os.path.join(OUT, "transfer_metrics_long.csv"))
    for bracket in BRACKETS:
        sub = long[(long.bracket == bracket) & (long.variant == "v2")]
        lc = sub[sub.low_support][["class_name", "support", "cells_present"]]
        low.append((bracket, [f"{r.class_name} ({r.support} px, {r.cells_present} cells)"
                              for r in lc.itertuples()]))

    lines = [
        "# transfer_confusion",
        "",
        "Per-class confusion matrices and accuracy metrics for the classifier temporal-"
        "transferability experiment: a Random Forest trained once on 2018/2020 AlphaEarth "
        "embeddings, applied to five NAIP brackets and compared against the CKIT-RF interpreted "
        "reference. Generated by `scripts/build_transfer_confusion.py`.",
        "",
        "## Inputs",
        "",
        "- Predictions: `data/raw/transfer_predictions/<bracket>/pred_<bracket>_cell<id>.tif`, 5 "
        "bands in the fixed order band1=v2, band2=v3, band3=v4, band4=v5, band5=v6, values 1 to 10.",
        "- Reference: `data/raw/rf_class_maps/rf_class_..._grid_<id>_...tif`, single band, int32, "
        "CKIT label_id codes.",
        "",
        "## Crosswalk (reference only)",
        "",
        "CKIT label_id to the 10-class schema: "
        + ", ".join(f"{k}->{v}" for k, v in CROSSWALK.items()) + ".",
        "",
        f"Excluded reference values (no 10-class equivalent, pixel dropped): "
        f"{sorted(EXCLUDE)} (10 = unknown abstention, 13 = other_no_change). Any reference value "
        f"outside the crosswalk and the exclude set is treated as an encoding error, counted, and "
        f"reported.",
        "",
        "## Multi-interpreted cells",
        "",
        f"Some cell ids have two or three CKIT reference rasters (multiple reviewers). One reference "
        f"per cell is used, selected as {ref_desc}. {n_double} multi-interpreted cell(s) contributed "
        f"their selected reference to these matrices.",
        "",
        "## Disjoint-cell caveat",
        "",
        "The five brackets use disjoint cell sets (36 cells each, no cell shared across brackets), "
        "so a bracket-to-bracket difference confounds the classifier's temporal transfer with the "
        "differing cell composition and landscape difficulty. These are five independent "
        "assessments, not a controlled transfer curve. 2018_2020 is the in-sample control, yet it "
        "is scored on its own 36 cells, not the cells of the other brackets, so it is not a shared "
        "baseline either. Read each bracket on its own terms, and read the per-class support "
        "before trusting a per-class number.",
        "",
        "## Support and low-support classes",
        "",
        "At 36 cells per bracket several classes have very few reference pixels, and their "
        "per-class metrics are then unreliable. `transfer_metrics_long.csv` carries both `support` "
        "(reference pixel count per class) and `cells_present` (how many of the cells contain the "
        f"class). The `low_support` flag is design-aware: it fires when a class appears in fewer "
        f"than {MIN_CELLS} cells or has under {MIN_SUP} px, since pixels within a cell are "
        "autocorrelated, so a large pixel count in one or two cells is still one or two "
        "observations. Reference support is the same across variants within a bracket, since the "
        "reference does not change. Low-support classes per bracket (v2, representative):",
        "",
    ]
    for bracket, lc in low:
        lines.append(f"- {bracket}: {', '.join(lc) if lc else 'none flagged'}")
    lines += ["", "## Skipped or flagged", ""]
    lines.append(f"- Missing reference: {len(skipped['missing_ref'])}.")
    lines.append(f"- Grid mismatch (pinning failed, not resampled): {len(skipped['grid_mismatch'])}.")
    lines.append(f"- Prediction band count not 5: {len(skipped['band_count'])}.")
    lines.append(f"- Reference/prediction bracket mismatch: {len(bracket_mismatch)}.")
    lines.append(f"- Unmapped reference values: {len(unmapped)} (cell, value) occurrences.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `cm_<variant>_<bracket>.csv`: 25 matrices (5 variants x 5 brackets), 10x10 raw "
                 "counts, reference on rows (diagonal is producer's accuracy), prediction on columns.")
    lines.append("- `transfer_metrics_long.csv`: one row per (variant, bracket, class) with per-class "
                 "precision, recall, F1, IoU, support, and low_support, plus the aggregate OA, "
                 "macro-F1, mean IoU, and kappa repeated on each row, and a `control` flag "
                 "(true for 2018_2020).")
    with open(os.path.join(OUT, "note.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
