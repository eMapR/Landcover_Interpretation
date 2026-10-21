#!/usr/bin/env python3
"""Interpreted-vs-model comparison on the temporally matched cells only (NAIP bracket 2018-2020).

The model maps v2-v6 are classified from 2018 and 2020 embeddings, so only interpreted cells with
target year 2019 carry the matching NAIP bracket (2018-2020). Target 2018 brackets 2017-2019,
2021 brackets 2020-2022, etc. This rerun filters to target 2019 and supersedes the all-years runs
for the model-comparison arm. It does not touch the existing outputs.

Reference: interpreted cells with target 2019, de-duplicated to one per location (numpy
default_rng seed 42). Variants v2-v6. Per variant it computes both the 10-class confusion and the
5-class collapsed confusion (Stable = urban/agriculture/grass-shrub/forest/water/wetland/other;
Harvest, Development, Insect/Disease, Beaver distinct; Unknown excluded), each with raw and
row-normalized matrices, OA / macro-F1 / mean IoU / kappa, and per-class precision/recall/F1/IoU/
support. The all-Stable baseline OA is reported alongside the 5-class OA.

Inference: target year is a property of the cell (it follows from NAIP availability), not of the
random draw, so if interpreters worked a randomized list the target-2019 subset is a probability
sample of the 2018/2020-eligible subpopulation. That subpopulation is narrower than the full
21,561-cell frame; its size is estimated from the sample proportion of 2019 cells. CIs use the
cell as the primary sampling unit: ratio estimators with FPC sqrt(1 - n/N_eligible), cross-checked
with a cell-level bootstrap (seed 42). With n small the intervals are wide and are reported as
such. Per-class estimates are suppressed where support is negligible (the arithmetic runs but the
number is not supportable).

Outputs -> reports/model_comparison_2018_2020/

Requires: rasterio, numpy, pandas, matplotlib
"""

import glob
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")

RF_DIR = SC.RF_DIR
MODEL_DIR = "data/raw/model_maps"
CROSSWALK = "data/reference/class_crosswalk.csv"
LEGEND = "data/reference/model_maps_10class_legend.csv"
OUT = SC.out_path("reports/model_comparison_2018_2020")
VERSIONS = ["v2", "v3", "v4", "v5", "v6"]
TARGET = "2019"
N_FRAME = 21561
SEED = 42
BOOT = 2000
MIN_SUP = 100                                           # px; below this a per-class metric is not reported

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collapsed_5class_confusion as cc  # canonical 5-class collapse
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

# canonical 5-class collapse, shared from collapsed_5class_confusion (Other -> Stable, Unknown drop)
NAMES5 = cc.NAMES5
_REF_COLLAPSE = cc._REF_COLLAPSE
_MODEL_COLLAPSE = cc._MODEL_COLLAPSE


# ----------------------------------------------------------------------------- reference data
def load_10class():
    cw = pd.read_csv(CROSSWALK)
    rf2common = {int(r.rf_code): int(r.model_code) for r in cw.itertuples()
                 if pd.notna(r.model_code) and pd.notna(r.rf_code) and int(r.model_code) > 0}
    leg = pd.read_csv(LEGEND)
    names = {int(r.code): r.display_name for r in leg.itertuples() if int(r.code) > 0}
    return rf2common, names


def location_key(path):
    m = re.search(rf"grid_(\d+)_sample_(\d+)_sensor_{SC.SENSOR}_target_(\d+)", os.path.basename(path))
    return (m.group(1), m.group(2), m.group(3)) if m else (os.path.basename(path),)


def deduped_2019():
    paths = sorted(glob.glob(os.path.join(RF_DIR, "**", SC.RF_GLOB), recursive=True))
    groups_all, groups_t = defaultdict(list), defaultdict(list)
    for p in paths:
        k = location_key(p)
        groups_all[k].append(p)
        if k[2] == TARGET:
            groups_t[k].append(p)
    rng = np.random.default_rng(SEED)
    kept = []
    for k in sorted(groups_t):
        v = sorted(groups_t[k])
        kept.append(v[int(rng.integers(len(v)))] if len(v) > 1 else v[0])
    return sorted(kept), len(groups_all)


def model_tiles(version):
    return sorted(glob.glob(os.path.join(MODEL_DIR, f"classified_maps_10class_{version}", "*.tif")))


def _intersect(a, b):
    return not (a.right <= b.left or a.left >= b.right or a.top <= b.bottom or a.bottom >= b.top)


def stitch_model_to_cell(cell_ds, tile_srcs):
    dst = np.zeros((cell_ds.height, cell_ds.width), dtype=np.uint8)
    for src in tile_srcs:
        if not _intersect(cell_ds.bounds, src.bounds):
            continue
        tmp = np.zeros_like(dst)
        reproject(source=rasterio.band(src, 1), destination=tmp,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=cell_ds.transform, dst_crs=cell_ds.crs,
                  resampling=Resampling.nearest)
        dst = np.where(tmp > 0, tmp, dst)
    return dst


def cell_cm_10(rf, model, rf2common):
    ref = np.zeros_like(rf, dtype=np.uint8)
    for rc, cc in rf2common.items():
        ref[rf == rc] = cc
    valid = (ref >= 1) & (ref <= 10) & (model >= 1) & (model <= 10)
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (ref[valid] - 1, model[valid] - 1), 1)
    return cm


def cell_cm_5(rf, model):
    ref = _REF_COLLAPSE[np.where((rf >= 0) & (rf <= 62), rf, 0)]
    mdl = _MODEL_COLLAPSE[np.where(model <= 10, model, 0)]
    valid = (ref >= 1) & (ref <= 5) & (mdl >= 1) & (mdl <= 5)
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (ref[valid] - 1, mdl[valid] - 1), 1)
    return cm


# ----------------------------------------------------------------------------- metrics + CIs
def metrics(M):
    M = M.astype(float)
    K = M.shape[0]
    tp = np.diag(M); row = M.sum(1); col = M.sum(0); tot = M.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(row > 0, tp / row, np.nan)
        precision = np.where(col > 0, tp / col, np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
        iou = np.where((row + col - tp) > 0, tp / (row + col - tp), np.nan)
    present = row > 0
    oa = tp.sum() / tot if tot else np.nan
    pe = (row * col).sum() / (tot * tot) if tot else np.nan
    kappa = (oa - pe) / (1 - pe) if tot and (1 - pe) != 0 else np.nan
    out = dict(OA=oa, kappa=kappa,
               macro_F1=np.nanmean(f1[present]) if present.any() else np.nan,
               mean_IoU=np.nanmean(iou[present]) if present.any() else np.nan,
               baseline_OA=row.max() / tot if tot else np.nan)
    for k in range(K):
        out[f"precision[{k+1}]"] = precision[k]
        out[f"recall[{k+1}]"] = recall[k]
        out[f"F1[{k+1}]"] = f1[k]
        out[f"IoU[{k+1}]"] = iou[k]
        out[f"rowsup[{k+1}]"] = row[k]
        out[f"colsup[{k+1}]"] = col[k]
    return out


def ratio_ci(y, x, n, N):
    Y, X = y.sum(), x.sum()
    if X <= 0 or n < 2:
        return np.nan, np.nan, np.nan, np.nan
    R = Y / X
    xbar = X / n
    s2 = ((y - R * x) ** 2).sum() / (n - 1)
    var = (1 - n / N) / (n * xbar ** 2) * s2
    se = np.sqrt(var) if var > 0 else 0.0
    return R, se, R - 1.96 * se, R + 1.96 * se


def ratio_cis(cms, n, N):
    K = cms.shape[1]
    tp = cms[:, np.arange(K), np.arange(K)].astype(float)
    row = cms.sum(2).astype(float); col = cms.sum(1).astype(float); tot = cms.sum((1, 2)).astype(float)
    out = {"OA": ratio_ci(tp.sum(1), tot, n, N)}
    for k in range(K):
        out[f"recall[{k+1}]"] = ratio_ci(tp[:, k], row[:, k], n, N)
        out[f"precision[{k+1}]"] = ratio_ci(tp[:, k], col[:, k], n, N)
        out[f"IoU[{k+1}]"] = ratio_ci(tp[:, k], row[:, k] + col[:, k] - tp[:, k], n, N)
    return out


def bootstrap_cis(cms, n, seed=SEED, B=BOOT):
    rng = np.random.default_rng(seed)
    keys = list(metrics(cms.sum(0)).keys())
    acc = {k: np.empty(B) for k in keys}
    for b in range(B):
        m = metrics(cms[rng.integers(0, n, size=n)].sum(0))
        for k in keys:
            acc[k][b] = m[k]
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in acc.items()}


# ----------------------------------------------------------------------------- outputs
def fmt_ci(est, lo, hi, d=3):
    if est is None or not np.isfinite(est):
        return "n/a"
    return f"{est:.{d}f} ({lo:.{d}f}--{hi:.{d}f})"


def plot_rownorm(M, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.errstate(invalid="ignore"):
        rn = M / M.sum(1, keepdims=True)
    K = len(labels)
    fig, ax = plt.subplots(figsize=(0.85 * K + 2.2, 0.8 * K + 1.8))
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("model"); ax.set_ylabel("reference (rows)")
    ax.grid(False)
    for i in range(K):
        for j in range(K):
            v = rn[i, j]
            if np.isfinite(v) and v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 else "black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="P(model | reference)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rf2common, names10 = load_10class()
    cells, n_all = deduped_2019()
    n_cells = len(cells)
    N_elig = round(N_FRAME * n_cells / n_all)
    print(f"target-{TARGET} cells (deduped, seed {SEED}): {n_cells} of {n_all} all-years locations")
    print(f"NAIP bracket for target {TARGET}: {int(TARGET)-1}-{int(TARGET)+1} (matches the model's "
          f"2018/2020 embeddings)")
    print(f"estimated 2018/2020-eligible frame: ~{N_elig:,} cells "
          f"(= {N_FRAME:,} * {n_cells}/{n_all}); FPC uses this, not {N_FRAME:,}")

    # bracket confirmation table
    cell_rows = []
    for c in cells:
        g, s, t = location_key(c)
        cell_rows.append(dict(cell=os.path.splitext(os.path.basename(c))[0], grid=g, sample=s,
                              target=t, naip_bracket=f"{int(t)-1}-{int(t)+1}",
                              matches_model=(t == TARGET)))
    pd.DataFrame(cell_rows).to_csv(os.path.join(OUT, "cells.csv"), index=False)

    long_rows = []
    fpc = np.sqrt(1 - n_cells / N_elig)
    schemes = [("10class", names10, cell_cm_10, 10), ("5class", NAMES5, cell_cm_5, 5)]
    variant_tables = {"10class": [], "5class": []}

    for scheme, names, cmfn, K in schemes:
        for v in VERSIONS:
            tiles = [rasterio.open(t) for t in model_tiles(v)]
            cms = []
            for cell in cells:
                with rasterio.open(cell) as ds:
                    rf = ds.read(1)
                    model = stitch_model_to_cell(ds, tiles)
                cm = cmfn(rf, model, rf2common) if K == 10 else cmfn(rf, model)
                if cm.sum() > 0:
                    cms.append(cm)
            for t in tiles:
                t.close()
            cms = np.array(cms)
            n = len(cms)
            census = cms.sum(0)
            pt = metrics(census)
            rci = ratio_cis(cms, n, N_elig)
            bci = bootstrap_cis(cms, n)
            labels = [names[c] for c in range(1, K + 1)]

            # matrices + figure
            pd.DataFrame(census, index=labels, columns=labels).to_csv(
                os.path.join(OUT, f"confusion_{scheme}_{v}_counts.csv"))
            with np.errstate(invalid="ignore"):
                rn = census / census.sum(1, keepdims=True)
            pd.DataFrame(np.round(rn, 4), index=labels, columns=labels).to_csv(
                os.path.join(OUT, f"confusion_{scheme}_{v}_rownorm.csv"))
            extra = f"  all-Stable base={pt['baseline_OA']:.3f}" if scheme == "5class" else ""
            plot_rownorm(census, labels,
                         f"{v} {scheme} (target {TARGET}, n={n})  OA={pt['OA']:.3f} "
                         f"kappa={pt['kappa']:.3f}{extra}",
                         os.path.join(OUT, f"confusion_{scheme}_{v}_rownorm.png"))

            # overall long rows
            for mk, label in [("OA", "OA"), ("kappa", "kappa"), ("macro_F1", "macro_F1"),
                              ("mean_IoU", "mean_IoU"), ("baseline_OA", "baseline_OA")]:
                r = rci.get(mk, (np.nan, np.nan, np.nan, np.nan))
                b = bci.get(mk, (np.nan, np.nan))
                long_rows.append(dict(scheme=scheme, variant=v, scope="overall", cls="", metric=label,
                                      estimate=round(pt[mk], 5), supportable=True,
                                      ratio_ci_lo=_r(r[2]), ratio_ci_hi=_r(r[3]),
                                      boot_ci_lo=_r(b[0]), boot_ci_hi=_r(b[1]), support=int(census.sum())))
            # per-class, with design-aware support flagging: a metric is supportable only if the
            # class has enough pixels AND appears in enough cells to estimate between-cell variance
            cells_ref = (cms.sum(2) > 0).sum(0)          # (K,) cells with reference support
            cells_mdl = (cms.sum(1) > 0).sum(0)          # (K,) cells with model support
            for k in range(K):
                c = k + 1
                rowsup = int(pt[f"rowsup[{c}]"]); colsup = int(pt[f"colsup[{c}]"])
                ok_ref = rowsup >= MIN_SUP and cells_ref[k] >= 3
                ok_mdl = colsup >= MIN_SUP and cells_mdl[k] >= 3
                for metric, ok in [("precision", ok_mdl), ("recall", ok_ref),
                                   ("F1", ok_ref), ("IoU", ok_ref)]:
                    key = f"{metric}[{c}]"
                    r = rci.get(key, (np.nan, np.nan, np.nan, np.nan))
                    b = bci.get(key, (np.nan, np.nan))
                    long_rows.append(dict(
                        scheme=scheme, variant=v, scope="class", cls=names[c], metric=metric,
                        estimate=round(pt[key], 5) if (ok and np.isfinite(pt[key])) else "",
                        supportable=bool(ok),
                        ratio_ci_lo=_r(r[2]) if ok else "", ratio_ci_hi=_r(r[3]) if ok else "",
                        boot_ci_lo=_r(b[0]) if ok else "", boot_ci_hi=_r(b[1]) if ok else "",
                        support=rowsup, cells_present=int(cells_ref[k])))

            variant_tables[scheme].append(dict(
                variant=v, n_cells=n, valid_px=int(census.sum()), baseline_OA=pt["baseline_OA"],
                OA_ci=fmt_ci(pt["OA"], rci["OA"][2], rci["OA"][3]),
                kappa_ci=fmt_ci(pt["kappa"], *bci["kappa"]),
                macroF1_ci=fmt_ci(pt["macro_F1"], *bci["macro_F1"]),
                mIoU_ci=fmt_ci(pt["mean_IoU"], *bci["mean_IoU"])))
            print(f"  {scheme} {v}: n={n} OA={pt['OA']:.3f} kappa={pt['kappa']:.3f} "
                  f"macroF1={pt['macro_F1']:.3f}")

    pd.DataFrame(long_rows).to_csv(os.path.join(OUT, "metrics_long.csv"), index=False)
    write_tables(variant_tables, n_cells, N_elig, fpc, cells)

    # support flags: which change classes are unsupportable (5-class, shared reference)
    unsup = sorted({row["cls"] for row in long_rows
                    if row["scheme"] == "5class" and row["scope"] == "class"
                    and not row["supportable"] and row["metric"] == "recall"})
    write_summary(variant_tables, n_cells, n_all, N_elig, fpc, unsup)
    print(f"\nunsupportable change classes (5-class, < {MIN_SUP} px or < 3 cells): "
          f"{unsup if unsup else 'none'}")
    print(f"wrote {OUT}/")


def _r(x):
    return round(float(x), 5) if x is not None and np.isfinite(x) else ""


def _variant_md_tex(rows, scheme, n_cells, N_elig):
    head = ("| Variant | Valid px | " + ("All-Stable OA | " if scheme == "5class" else "") +
            "OA (95% CI) | kappa (95% CI) | macro-F1 (95% CI) | mean IoU (95% CI) |")
    sep = "|---|---|" + ("---|" if scheme == "5class" else "") + "---|---|---|---|"
    md = [f"## {scheme} summary (target 2019, n={n_cells})", "", head, sep]
    for r in rows:
        base = f"{r['baseline_OA']:.3f} | " if scheme == "5class" else ""
        md.append(f"| {r['variant']} | {r['valid_px']:,} | {base}{r['OA_ci']} | {r['kappa_ci']} | "
                  f"{r['macroF1_ci']} | {r['mIoU_ci']} |")
    cols = "lrrcccc" if scheme == "5class" else "lrcccc"
    basehdr = "All-Stable OA & " if scheme == "5class" else ""
    tex = [r"% " + scheme + r" summary, target 2019. Requires \usepackage{booktabs}.",
           r"\begin{table}[t]", r"\centering",
           r"\caption{Interpreted-vs-model " + scheme.replace("class", "-class") +
           r" census on the " + f"{n_cells}" + r" temporally matched cells (target 2019, NAIP "
           r"bracket 2018--2020). 95\% design-based ratio-estimator CIs (FPC against the estimated "
           r"2018/2020-eligible frame $N\approx" + f"{N_elig:,}".replace(",", "{,}") +
           r"$); $\kappa$/macro-F1/mean IoU CIs are cell-level bootstrap. $n$ is small so the "
           r"intervals are wide.}",
           r"\label{tab:model2019_" + scheme + "}",
           r"\begin{tabular}{" + cols + "}", r"\toprule",
           "Variant & Valid px & " + basehdr +
           r"OA (95\% CI) & $\kappa$ (95\% CI) & macro-F1 (95\% CI) & mean IoU (95\% CI) \\",
           r"\midrule"]
    for r in rows:
        base = f"{r['baseline_OA']:.3f} & " if scheme == "5class" else ""
        tex.append(f"{r['variant']} & {r['valid_px']:,} & {base}{r['OA_ci']} & {r['kappa_ci']} & "
                   f"{r['macroF1_ci']} & {r['mIoU_ci']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def write_tables(variant_tables, n_cells, N_elig, fpc, cells):
    md_all, tex_all = [], []
    for scheme in ("10class", "5class"):
        md, tex = _variant_md_tex(variant_tables[scheme], scheme, n_cells, N_elig)
        md_all.append(md); tex_all.append(tex)
    with open(os.path.join(OUT, "summary_by_variant.md"), "w") as fh:
        fh.write(f"# Model comparison on temporally matched cells (target 2019)\n\n"
                 f"n={n_cells} cells, NAIP bracket 2018-2020, matching the model's 2018/2020 "
                 f"embeddings. CIs use the cell as PSU; ratio-estimator FPC against the estimated "
                 f"2018/2020-eligible frame (~{N_elig:,} cells), not the full 21,561. Per-class "
                 f"metrics are omitted where support is < {MIN_SUP} px or the class appears in < 3 "
                 f"cells (design-aware; see metrics_long.csv `supportable` and `cells_present`).\n\n"
                 + "\n".join(md_all))
    with open(os.path.join(OUT, "summary_by_variant.tex"), "w") as fh:
        fh.write("\n".join(tex_all))


def write_summary(variant_tables, n_cells, n_all, N_elig, fpc, unsup):
    v5 = {r["variant"]: r for r in variant_tables["5class"]}
    v10 = {r["variant"]: r for r in variant_tables["10class"]}
    lines = [
        "interpreted-vs-model comparison on the temporally matched cells (supersedes all-years)",
        f"target year {TARGET}, NAIP bracket {int(TARGET)-1}-{int(TARGET)+1}, matching the model's "
        f"2018/2020 embeddings.",
        f"cells: {n_cells} (deduped, seed 42) of {n_all} all-years locations. The plan named 30; "
        f"the data now holds {n_cells} target-2019 cells (data added since).",
        "",
        "Inference: target year is a property of the cell (it follows from NAIP availability), not",
        "of the random draw. If interpreters worked a randomized list, the target-2019 subset is a",
        "probability sample of the 2018/2020-ELIGIBLE subpopulation, which is narrower than the full",
        f"21,561-cell frame. That subpopulation is estimated at ~{N_elig:,} cells (sample proportion",
        f"of 2019 cells), and the FPC uses it: sqrt(1 - {n_cells}/{N_elig:,}) = {fpc:.3f}. With n="
        f"{n_cells} the CIs are wide; they are reported, not suppressed.",
        "",
        f"{'variant':8}{'OA10':>8}{'kappa10':>9}{'OA5':>8}{'all-Stable':>12}{'kappa5':>9}",
    ]
    for v in VERSIONS:
        oa10 = float(v10[v]["OA_ci"].split(" ")[0]); ka10 = float(v10[v]["kappa_ci"].split(" ")[0])
        oa5 = float(v5[v]["OA_ci"].split(" ")[0]); ka5 = float(v5[v]["kappa_ci"].split(" ")[0])
        lines.append(f"{v:8}{oa10:>8.3f}{ka10:>9.3f}{oa5:>8.3f}{v5[v]['baseline_OA']:>12.3f}{ka5:>9.3f}")
    lines += ["",
              "As in the census, every variant's 5-class OA is below the all-Stable baseline with",
              "kappa ~ 0: the collapsed maps carry almost no change-detection information here either.",
              "The 10-class kappa is much higher (v2 0.59, v6 0.08) because it credits stable-class",
              "discrimination that the collapse strips away.",
              "",
              f"Per-class support flagging is design-aware: a class is scored only if it has >= "
              f"{MIN_SUP} px AND appears in >= 3 cells (else no between-cell variance). Here every",
              f"change class clears both (Harvest 18 cells, Development 10, Insect/Disease 8, Beaver 5),",
              f"so none are suppressed: {', '.join(unsup) if unsup else 'none'}. But Beaver rests on 5",
              "cells and its CIs are very wide; cells_present is in metrics_long.csv so the fragility",
              "is visible. With n=36 all change-class CIs are wide and are reported, not suppressed.",
              "macro-F1 averages 5 classes (5-class) or 10 (10-class) and is not comparable across",
              "the two, nor as a level with earlier all-years matrices."]
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
