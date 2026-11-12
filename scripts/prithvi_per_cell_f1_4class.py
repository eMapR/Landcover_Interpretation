#!/usr/bin/env python3
"""Landsat: per-grid-cell macro-F1 under the four-class scheme for the four Prithvi models.

Landsat/Prithvi counterpart of scripts/per_cell_f1_5class.py (Sentinel-2 only). Four classes, since
Beaver is excluded for Prithvi: Stable, Harvest, Development, Insect/Disease. For each Prithvi model
(scripts/prithvi_loader.py) and each scored grid cell, the cell's 4x4 confusion matrix is built
against the interpreted Landsat reference, and the cell's score is the unweighted mean of the
per-class F1 over the classes included in the cell (macro-F1 over included classes).

Scheme, cells and pixels (identical to collapsed_5class_confusion.py --prithvi, zero-input mask
any_year): label year = reference target year - 1; Beaver, Unknown/Fire and agricultural change
excluded; only reference pixels with a Prithvi pixel under their center, minus the zero-input mask;
the 312 scored cells with seeded reviewer selection (seed 42). The per-cell matrices come from
prithvi_class_ci_vs_interpreter.model_cells() (the evaluation's own functions); per-class F1 is
prithvi_per_cell_change_f1.cell_f1(): 2TP / (reference px + predicted px), 0 when TP = 0. All four
models share the same cells, so there is no separate common cell set.

Include rule: a class counts in a cell's macro-F1 if it is in the reference OR the prediction
(default, used for the figures), or only if it is in the reference (ref_only). Both rules are
reported in the CSVs and note.md. Stable scores about 1 in nearly every cell and the change classes
about 0, so a cell's macro-F1 sits near 1/n for n included classes; under the default rule a model
that predicts less change includes fewer classes and gets a higher per-cell macro-F1. Under the
reference-only rule every model has the same cell composition, so those rows are the fairer model
comparison.

Checks, all before anything is written (the script stops on any difference):
  1. per-cell matrices summed over the cells == confusion_<model>_masked_counts.csv (the folder of
     --eval-csv), entry by entry, for every model;
  2. ref_support_px, pred_support_px and n_cells == every any_year row of eval_class_metrics.csv;
  3. every scored cell is listed in scored_cells.csv;
  4. pooled macro-F1 (cc.metrics_f1_zero on the pooled matrix) == the any_year macro_F1 estimate in
     metrics_long.csv, to 5 decimals, for every model;
  5. per-cell macro-F1 under the reference-only rule == cc.metrics_f1_zero(cell)["macro_F1"] on every
     (model, cell).

Outputs -> $LCI_OUT_DIR/reports/prithvi_per_cell_f1_4class/
  - prithvi_per_cell_f1_4class.csv     one row per (model, cell): macro-F1 and included classes
                                       under both rules, Stable-only flags, per-class F1
  - prithvi_per_cell_f1_summary.csv    per (include rule, model, subset all_cells / excl_stable_only)
  - prithvi_f1_violin.png              per-cell macro-F1 by model (default rule)
  - prithvi_f1_by_n_classes.png        reference classes per cell; macro-F1 by classes included
  - note.md

Refuses to run without --eval-csv, without LCI_OUT_DIR, with LCI_SENSOR other than Landsat, with a
zero-input rule other than any_year or a year rule other than target_minus_1, or when the output
folder is inside <repo>/reports, /vol/v1/HLS_Prithvi, /home/server/hpc/emapr or the Prithvi root
(check_out_path).

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs LCI_OUT_DIR=reports_landsat python scripts/prithvi_per_cell_f1_4class.py --eval-csv reports_landsat/reports/collapsed_5class_confusion/eval_class_metrics.csv

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collapsed_5class_confusion as cc            # noqa: E402  metrics_f1_zero, names, _r5
import prithvi_class_ci_vs_interpreter as PCI      # noqa: E402  model_cells, check_out_path, fit-check
import prithvi_per_cell_change_f1 as PCF           # noqa: E402  cell_f1, evaluation checks, helpers
import prithvi_loader as PL                        # noqa: E402

SC = cc.SC
OUT_SUB = "reports/prithvi_per_cell_f1_4class"
MASK = PCI.MASK                                    # any_year
MODELS = PCI.MODELS
SHORT = PCI.SHORT
COLOR = PCI.COLOR
NAMES = cc.NAMES5
CLASSES = cc.P_CLASSES                             # [1, 2, 3, 4]
F1_COLS = ["f1_stable", "f1_harvest", "f1_development", "f1_insect_disease"]
RULES = PCF.RULES                                  # ref_or_pred (default), ref_only
RULE_TEXT = {"ref_or_pred": "it is in the reference or the prediction",
             "ref_only": "it is in the reference"}
SUFFIX = {"ref_or_pred": "", "ref_only": "_ref_only"}
SUBSETS = ["all_cells", "excl_stable_only"]
GUIDES = [(2, "1/2"), (3, "1/3"), (4, "1/4")]      # 1/n guide lines
DPI = 300
NBSP = PCF.NBSP
JITTER_SEED = 0
STATS = ["mean", "median", "q25", "q75", "min", "max"]


# ----------------------------------------------------------------------------- per-cell values
def cell_macro(cm, rule):
    """(macro-F1, included 0-based classes, per-class F1 list) for one cell's 4x4 under `rule`.
    Per-class F1 from PCF.cell_f1 (NaN when the class is in neither reference nor prediction)."""
    vals = [PCF.cell_f1(cm, k) for k in range(len(CLASSES))]
    f1 = [v[0] for v in vals]
    inc = [k for k, (_, _, ref, pred) in enumerate(vals)
           if (ref + pred > 0 if rule == "ref_or_pred" else ref > 0)]
    return (float(np.mean([f1[k] for k in inc])) if inc else np.nan), inc, f1


def per_cell_rows(per, selected):
    rows = []
    for m in MODELS:
        for key, cm in per[m]:
            ref_path = selected[key]
            _, target, reviewer = PL.parse_reference(ref_path)
            row = dict(model=SHORT[m], grid=key[0], sample=key[1], target_year=int(target),
                       label_year=PL.label_year(int(target)), reviewer=reviewer,
                       reference=os.path.basename(ref_path), cell_px=int(cm.sum()),
                       n_ref_classes=int((cm.sum(1) > 0).sum()))
            for rule in RULES:
                macro, inc, f1 = cell_macro(cm, rule)
                sfx = SUFFIX[rule]
                row[f"macro_f1{sfx}"] = macro                 # full precision; rounded at write
                row[f"n_classes_included{sfx}"] = len(inc)
                row[f"stable_only{sfx}"] = inc == [0]
            row.update({c: f for c, f in zip(F1_COLS, f1)})
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- checks 4 and 5
def check_pooled_macro(per, eval_dir):
    """Check 4: metrics_f1_zero macro-F1 of the pooled matrix == metrics_long.csv any_year macro_F1.
    Returns (n checked, mismatches, {model: full pooled macro-F1})."""
    path = os.path.join(eval_dir, "metrics_long.csv")
    if not os.path.isfile(path):
        return 0, [f"{path} not found"], {}
    ml = pd.read_csv(path)
    ml = ml[(ml["mask"] == MASK) & (ml.scope == "overall") & (ml.metric == "macro_F1")]
    bad, pooled = [], {}
    for m in MODELS:
        full = float(cc.metrics_f1_zero(np.stack([cm for _, cm in per[m]]).sum(0))["macro_F1"])
        pooled[m] = full
        sel = ml[ml.variant == m]
        if len(sel) != 1:
            bad.append(f"{m}: {len(sel)} any_year macro_F1 rows in {path}")
        elif abs(round(full, 5) - float(sel.estimate.iloc[0])) > 5e-7:
            bad.append(f"{SHORT[m]} macro_F1: rebuilt {round(full, 5)} vs metrics_long "
                       f"{sel.estimate.iloc[0]}")
    return len(MODELS), bad, pooled


def check_ref_only_macro(per):
    """Check 5: reference-only per-cell macro-F1 == cc.metrics_f1_zero(cell)["macro_F1"]."""
    bad, n = [], 0
    for m in MODELS:
        for key, cm in per[m]:
            mine = cell_macro(cm, "ref_only")[0]
            ev = cc.metrics_f1_zero(cm)["macro_F1"]
            n += 1
            if not (np.isfinite(ev) and abs(mine - ev) < 1e-12):
                bad.append(f"{SHORT[m]} {key}: ref_only macro {mine} vs metrics_f1_zero {ev}")
    return n, bad


def agri_px(eval_dir):
    """Predicted agricultural-change px (any_year) from the evaluation's side table, or None."""
    path = os.path.join(eval_dir, "side_agricultural_change.csv")
    if not os.path.isfile(path):
        return None
    t = pd.read_csv(path)
    return int(t[t["mask"] == MASK].predicted_agricultural_change_px.sum())


# ----------------------------------------------------------------------------- summary
def _stats(x):
    x = np.asarray(x, float)
    if not x.size:
        return {s: np.nan for s in STATS}
    return dict(mean=x.mean(), median=float(np.median(x)), q25=float(np.quantile(x, 0.25)),
                q75=float(np.quantile(x, 0.75)), min=x.min(), max=x.max())


def summary_full(df, pooled):
    """Per (rule, model, subset), full precision (rounded once when written or printed)."""
    rows = []
    for rule in RULES:
        sfx = SUFFIX[rule]
        for m in MODELS:
            g = df[df.model == SHORT[m]]
            n_so = int(g[f"stable_only{sfx}"].sum())
            for sub in SUBSETS:
                h = g if sub == "all_cells" else g[~g[f"stable_only{sfx}"]]
                ninc = h[f"n_classes_included{sfx}"]
                rows.append(dict(include_rule=rule, model=SHORT[m], subset=sub, n_cells=len(h),
                                 n_stable_only=n_so,
                                 **{f"n_incl_{k}": int((ninc == k).sum()) for k in range(1, 5)},
                                 **_stats(h[f"macro_f1{sfx}"]), pooled_macro_f1=pooled[m]))
    return pd.DataFrame(rows)


def _rng_txt(vals):
    """'a' or 'a to b' from full values, each rounded once to 3 decimals."""
    lo, hi = f"{min(vals):.3f}", f"{max(vals):.3f}"
    return lo if lo == hi else f"{lo} to {hi}"


def medians_by_n(df, rule):
    """{n included: [median per model]} for the n values every model has."""
    sfx = SUFFIX[rule]
    out = {}
    for k in range(1, 5):
        meds = []
        for m in MODELS:
            g = df[(df.model == SHORT[m]) & (df[f"n_classes_included{sfx}"] == k)]
            if len(g):
                meds.append(float(g[f"macro_f1{sfx}"].median()))
        if len(meds) == len(MODELS):
            out[k] = meds
    return out


# ----------------------------------------------------------------------------- figures
def _guides(fig, ax, label_x_in, tr, texts, prefix):
    """Dotted 1/n lines across `ax`, labels in the margin right of it (outside the plot area)."""
    for n, lab in GUIDES:
        ax.axhline(1.0 / n, ls=":", lw=0.9, color="0.45", zorder=1)
        y_disp = ax.transData.transform((0, 1.0 / n))[1]
        y_in = y_disp / fig.dpi
        t = fig.text(label_x_in, y_in, lab, transform=tr, ha="left", va="center", fontsize=9.5,
                     color="0.35")
        texts.append((f"{prefix} guide {lab}", t))


def _measure_caption(fig, caption, width_in, fs):
    cap_txt = PCI._wrap_measure(fig, caption, width_in, fs)
    probe = fig.text(0, 0, cap_txt, fontsize=fs)
    h = probe.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    probe.remove()
    return cap_txt, h


def _fit_check(fig, texts, areas, label):
    return PCI.check_overlaps(fig, texts, [], label) + PCF._texts_vs_areas(fig, texts, areas, label)


def violin_fig(out_dir, df, summ, n_cells):
    plt = PCF._plt()
    from matplotlib.lines import Line2D
    rule = "ref_or_pred"
    s_all = summ[(summ.include_rule == rule) & (summ.subset == "all_cells")]
    s_ro = summ[(summ.include_rule == "ref_only") & (summ.subset == "all_cells")]
    meds = medians_by_n(df, rule)
    so_scores = df[df.stable_only].macro_f1
    so_txt = _rng_txt(so_scores) if len(so_scores) else "-"
    n_so = [int(s_all[s_all.model == SHORT[m]].n_stable_only.iloc[0]) for m in MODELS]
    med_txt = "; ".join(f"{k} classes {_rng_txt(v)}" for k, v in meds.items() if k > 1)
    caption = (f"Per-cell macro-F1 (the mean of the per-class F1 over the classes included in the "
               f"cell; a class is included when {RULE_TEXT[rule]}) for the four Prithvi models on "
               f"the {n_cells} scored cells (Landsat, four classes, zero-input mask any_year). One "
               f"point per cell, hollow where only Stable is included (these cells score "
               f"{so_txt}); bar = median, diamond = mean. Stable scores about 1 in nearly every cell "
               f"and the change classes about 0, so a cell's macro-F1 sits near 1/n for n included "
               f"classes (dotted lines; median per model: {med_txt}). A model that predicts less "
               f"change includes fewer classes and gets a higher per-cell macro-F1 under this rule. "
               f"Under the reference-only rule every model has the same cell composition, which "
               f"makes it the fairer model comparison (mean "
               f"{_rng_txt(s_ro['mean'])} across models; see note.md).")

    W = 10.5
    L_IN, R_IN = 0.85, 0.75
    AX_W, AX_H = W - L_IN - R_IN, 4.6
    TITLE_IN, GAP_TOP = 0.55, 0.3
    XT_IN, LEG_IN, CAP_GAP, CAP_PAD, CAP_FS = 0.95, 0.42, 0.15, 0.12, 9
    fig = plt.figure(figsize=(W, 4.0), dpi=DPI)
    cap_txt, cap_h = _measure_caption(fig, caption, W - 0.6, CAP_FS)
    H = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XT_IN + AX_H + GAP_TOP + TITLE_IN
    fig.set_size_inches(W, H)
    ax_y0 = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XT_IN
    tr = fig.dpi_scale_trans
    ax = fig.add_axes([L_IN / W, ax_y0 / H, AX_W / W, AX_H / H])
    rng = np.random.default_rng(JITTER_SEED)
    for i, m in enumerate(MODELS):
        g = df[df.model == SHORT[m]]
        d = g.macro_f1.values.astype(float)
        so = g.stable_only.values
        if len(d) >= 2 and np.ptp(d) > 0:
            body = ax.violinplot([d], positions=[i], showextrema=False, widths=0.8)["bodies"][0]
            body.set_facecolor(COLOR[m]); body.set_alpha(0.25); body.set_edgecolor("0.5")
        xs = i + rng.uniform(-0.17, 0.17, len(d))
        ax.scatter(xs[~so], d[~so], s=9, color=COLOR[m], alpha=0.5, edgecolor="none", zorder=3)
        ax.scatter(xs[so], d[so], s=14, facecolor="none", edgecolor=COLOR[m], linewidths=0.8,
                   zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(d)] * 2, color="black", lw=2, zorder=4)
        ax.plot(i, np.mean(d), marker="D", color="black", ms=6, zorder=5, ls="")
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{SHORT[m]}\n(n={n_cells};\n{k} Stable-only)" for m, k in zip(MODELS, n_so)])
    ax.tick_params(axis="x", labelsize=10, length=0, pad=6)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Per-Cell Macro-F1 (Four Classes)", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    texts = []
    fig.canvas.draw()
    _guides(fig, ax, L_IN + AX_W + 0.1, tr, texts, "violin")
    title = fig.text(W / 2, H - GAP_TOP / 2 - TITLE_IN / 2,
                     f"Per-Cell Macro-F1 by Prithvi Model (Landsat, Four Classes, {n_cells} Cells)",
                     transform=tr, ha="center", va="center", fontsize=14, fontweight="bold")
    handles = [Line2D([], [], color="black", lw=2, label="median"),
               Line2D([], [], color="black", marker="D", ls="", ms=6, label="mean"),
               Line2D([], [], color="0.4", marker="o", ls="", ms=5, alpha=0.7,
                      markeredgecolor="none", label="cell"),
               Line2D([], [], color="0.4", marker="o", ls="", ms=5, markerfacecolor="none",
                      label="Stable-only cell"),
               Line2D([], [], color="0.45", ls=":", lw=0.9, label="1/n")]
    leg = fig.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=10,
                     bbox_to_anchor=(W / 2, CAP_PAD + cap_h + CAP_GAP + LEG_IN / 2),
                     bbox_transform=tr, columnspacing=1.8)
    cap = fig.text(W / 2, CAP_PAD, cap_txt, transform=tr, ha="center", va="bottom",
                   fontsize=CAP_FS, color="0.3")
    fig.canvas.draw()
    texts += [("title", title), ("legend", leg), ("caption", cap), ("y label", ax.yaxis.label)]
    texts += [(f"x tick '{t.get_text().splitlines()[0]}'", t) for t in PCF._drawn_ticklabels(ax.xaxis)]
    texts += [(f"y tick '{t.get_text()}'", t) for t in PCF._drawn_ticklabels(ax.yaxis)]
    warns = _fit_check(fig, texts, [("plot area", ax)], "violin")
    path = os.path.join(out_dir, "prithvi_f1_violin.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return warns, path


def by_n_fig(out_dir, df, n_cells):
    plt = PCF._plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rule = "ref_or_pred"
    ref_counts = df[df.model == SHORT[MODELS[0]]].n_ref_classes.value_counts()
    ks = [k for k in range(1, 5)]
    meds = medians_by_n(df, rule)
    caption = (f"Left: number of reference classes per cell on the {n_cells} scored cells (Landsat, "
               f"four classes); the same for every model. Right: per-cell macro-F1 by the number of "
               f"classes included in the cell (a class is included when {RULE_TEXT[rule]}), per "
               f"model, with the number of cells per model under each group in model order "
               f"({' / '.join(SHORT[m] for m in MODELS)}); boxes show the quartiles and median. "
               f"The scores fall in bands near 1/n (dotted lines; median per model: "
               + "; ".join(f"{k} {'class' if k == 1 else 'classes'} {_rng_txt(v)}"
                           for k, v in meds.items())
               + "), because Stable scores about 1 and the change classes about 0. A model that "
               "predicts change in more cells has more cells with 3 or 4 classes and so a lower "
               "per-cell macro-F1, whatever its accuracy on the classes themselves.")

    W = 14.0
    L_IN, LP_W, MID_GAP, R_IN = 0.85, 2.9, 1.05, 0.75
    RP_W = W - L_IN - LP_W - MID_GAP - R_IN
    AX_H = 4.3
    TITLE_IN, PT_IN, GAP_TOP = 0.55, 0.45, 0.15
    XT_IN, LEG_IN, CAP_GAP, CAP_PAD, CAP_FS = 0.95, 0.42, 0.15, 0.12, 9
    fig = plt.figure(figsize=(W, 4.0), dpi=DPI)
    cap_txt, cap_h = _measure_caption(fig, caption, W - 0.6, CAP_FS)
    H = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XT_IN + AX_H + PT_IN + GAP_TOP + TITLE_IN
    fig.set_size_inches(W, H)
    ax_y0 = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XT_IN
    tr = fig.dpi_scale_trans
    axl = fig.add_axes([L_IN / W, ax_y0 / H, LP_W / W, AX_H / H])
    rx0 = L_IN + LP_W + MID_GAP
    axr = fig.add_axes([rx0 / W, ax_y0 / H, RP_W / W, AX_H / H])

    # left: reference class composition
    cnt = [int(ref_counts.get(k, 0)) for k in ks]
    axl.bar(ks, cnt, color="#4d4d4d", edgecolor="white", width=0.65)
    axl.set_xticks(ks)
    axl.set_xticklabels([f"{k}\n({c})" for k, c in zip(ks, cnt)])
    axl.set_xlim(0.4, 4.6)
    axl.set_ylim(0, max(cnt) * 1.08)
    axl.set_xlabel("Reference Classes in the Cell\n(Cells in Parentheses)", fontsize=11)
    axl.set_ylabel("Cells", fontsize=11)

    # right: macro-F1 by classes included, per model
    offs = [-0.3, -0.1, 0.1, 0.3]
    rng = np.random.default_rng(JITTER_SEED)
    tick_lab = []
    for k in ks:
        ns = []
        for m, o in zip(MODELS, offs):
            d = df[(df.model == SHORT[m]) & (df.n_classes_included == k)].macro_f1.values.astype(float)
            ns.append(len(d))
            if not len(d):
                continue
            bp = axr.boxplot([d], positions=[k + o], widths=0.15, patch_artist=True,
                             showfliers=False, whis=(0, 100), manage_ticks=False)
            for b in bp["boxes"]:
                b.set_facecolor(COLOR[m]); b.set_alpha(0.35); b.set_edgecolor(COLOR[m])
            for part in ("whiskers", "caps"):
                for ln in bp[part]:
                    ln.set_color(COLOR[m])
            for ln in bp["medians"]:
                ln.set_color("black"); ln.set_linewidth(1.6)
            axr.scatter(k + o + rng.uniform(-0.05, 0.05, len(d)), d, s=6, color=COLOR[m],
                        alpha=0.45, edgecolor="none", zorder=3)
        tick_lab.append(f"{k} {'class' if k == 1 else 'classes'}\n" + " / ".join(str(n) for n in ns))
    axr.set_xticks(ks)
    axr.set_xticklabels(tick_lab)
    axr.set_xlim(0.45, 4.55)
    axr.set_ylim(0, 1.04)
    axr.set_xlabel("Classes Included in the Cell (Reference or Prediction)", fontsize=11)
    axr.set_ylabel("Per-Cell Macro-F1", fontsize=11)
    for a in (axl, axr):
        a.tick_params(axis="x", labelsize=10, length=0, pad=6)
        a.tick_params(axis="y", labelsize=10)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    texts = []
    fig.canvas.draw()
    _guides(fig, axr, rx0 + RP_W + 0.1, tr, texts, "right panel")
    py = ax_y0 + AX_H + PT_IN / 2
    t1 = fig.text(L_IN + LP_W / 2, py, "Reference Classes per Cell", transform=tr, ha="center",
                  va="center", fontsize=12, fontweight="bold")
    t2 = fig.text(rx0 + RP_W / 2, py, "Macro-F1 by Classes Included, per Model", transform=tr,
                  ha="center", va="center", fontsize=12, fontweight="bold")
    title = fig.text(W / 2, H - GAP_TOP / 2 - TITLE_IN / 2,
                     f"Per-Cell Macro-F1 and Cell Composition (Landsat, Four Classes, {n_cells} Cells)",
                     transform=tr, ha="center", va="center", fontsize=14, fontweight="bold")
    handles = [Patch(facecolor=COLOR[m], alpha=0.6, edgecolor=COLOR[m], label=SHORT[m])
               for m in MODELS] + [Line2D([], [], color="0.45", ls=":", lw=0.9, label="1/n")]
    leg = fig.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=10,
                     bbox_to_anchor=(W / 2, CAP_PAD + cap_h + CAP_GAP + LEG_IN / 2),
                     bbox_transform=tr, columnspacing=1.8)
    cap = fig.text(W / 2, CAP_PAD, cap_txt, transform=tr, ha="center", va="bottom",
                   fontsize=CAP_FS, color="0.3")
    fig.canvas.draw()
    texts += [("title", title), ("left panel title", t1), ("right panel title", t2),
              ("legend", leg), ("caption", cap)]
    for nm, a in (("left", axl), ("right", axr)):
        texts += [(f"{nm} x label", a.xaxis.label), (f"{nm} y label", a.yaxis.label)]
        texts += [(f"{nm} x tick '{t.get_text().splitlines()[0]}'", t)
                  for t in PCF._drawn_ticklabels(a.xaxis)]
        texts += [(f"{nm} y tick '{t.get_text()}'", t) for t in PCF._drawn_ticklabels(a.yaxis)]
    warns = _fit_check(fig, texts, [("left plot area", axl), ("right plot area", axr)], "by_n")
    path = os.path.join(out_dir, "prithvi_f1_by_n_classes.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return warns, path


# ----------------------------------------------------------------------------- note
def write_note(out_dir, df, summ, n_cells, n_dedup, n_listed, agri, checks, warns):
    def f3(v):
        return "-" if not np.isfinite(v) else f"{v:.3f}"

    def table(rule):
        s = summ[summ.include_rule == rule]
        L = ["| Model | Cells | Stable-only | 1 / 2 / 3 / 4 classes | Mean | Median | Q25 | Q75 | "
             "Mean excl. Stable-only | Median excl. Stable-only |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for m in MODELS:
            a = s[(s.model == SHORT[m]) & (s.subset == "all_cells")].iloc[0]
            e = s[(s.model == SHORT[m]) & (s.subset == "excl_stable_only")].iloc[0]
            L.append(f"| {SHORT[m]} | {a.n_cells} | {a.n_stable_only} | "
                     f"{a.n_incl_1} / {a.n_incl_2} / {a.n_incl_3} / {a.n_incl_4} | {f3(a['mean'])} | "
                     f"{f3(a['median'])} | {f3(a.q25)} | {f3(a.q75)} | {f3(e['mean'])} | "
                     f"{f3(e['median'])} |")
        return L

    s_def = summ[(summ.include_rule == "ref_or_pred") & (summ.subset == "all_cells")]
    s_ro = summ[(summ.include_rule == "ref_only") & (summ.subset == "all_cells")]
    meds = medians_by_n(df, "ref_or_pred")
    so_def = df[df.stable_only].macro_f1
    so_ro = df[df.stable_only_ref_only].macro_f1_ref_only
    most = s_def.loc[s_def.n_stable_only.idxmax()]
    least = s_def.loc[s_def.n_stable_only.idxmin()]
    comp = s_ro.iloc[0]
    L = ["# prithvi_per_cell_f1_4class", "",
         "Per-grid-cell macro-F1 under the four-class scheme (Stable, Harvest, Development, "
         "Insect/Disease) for the four Prithvi models on Landsat, treating each grid cell as one "
         "sample. Generated by `scripts/prithvi_per_cell_f1_4class.py`; Landsat/Prithvi counterpart "
         "of `scripts/per_cell_f1_5class.py` (Sentinel-2 only).", "",
         "## Method", "",
         "- Scheme and pixels as in `collapsed_5class_confusion.py --prithvi`: label year = reference "
         "target year - 1; Beaver, Unknown/Fire and agricultural change excluded; reference pixels "
         "with a Prithvi pixel under their center, minus the zero-input mask any_year.",
         f"- Agricultural change: " + (f"no model predicts it ({agri} px in "
                                         "side_agricultural_change.csv, any_year), so excluding it "
                                         "from the matrix removes nothing." if agri is not None else
                                         "side_agricultural_change.csv not found; excluded from the "
                                         "matrix as in the evaluation."),
         f"- Cells: {n_dedup} de-duplicated reference cells (seeded reviewer selection, seed "
         f"{cc.SEED}); {n_listed} have Prithvi output (scored_cells.csv); **{n_cells}** have at least "
         "one usable pixel and are scored. All four models are scored on the same cells, so there is "
         "no separate common cell set.",
         "- Per-class F1 = 2TP / (reference pixels + predicted pixels) of the class in the cell, 0 "
         "when there is no true positive (`prithvi_per_cell_change_f1.cell_f1`). A cell's macro-F1 is "
         "the unweighted mean of the per-class F1 over the classes included in the cell.",
         "- Include rule: a class is included when it is in the reference or the prediction "
         "(default, used for the figures), or only when it is in the reference (reference-only). "
         "The reference-only macro-F1 is the same as the evaluation's `metrics_f1_zero` macro-F1 for "
         "the cell (check 5).", "",
         "## How to read per-cell macro-F1 (the 1/n effect)", "",
         "Stable scores about 1 in nearly every cell and the change classes about 0, so a cell's "
         "macro-F1 is close to 1/n, where n is the number of classes included: about 1 when only "
         "Stable is included, about 0.5 with two classes, 0.33 with three and 0.25 with four. Median "
         "per model under the default rule: "
         + "; ".join(f"{k} {'class' if k == 1 else 'classes'} {_rng_txt(v)}" for k, v in meds.items())
         + ". The per-cell macro-F1 therefore mostly counts the classes in a cell.", "",
         "Under the default rule a predicted change class is included even where the reference has "
         "none, so a model that predicts less change includes fewer classes and gets a higher "
         f"per-cell macro-F1. For example {most.model} has {most.n_stable_only} Stable-only cells "
         f"(mean {f3(most['mean'])}) and {least.model} has {least.n_stable_only} (mean "
         f"{f3(least['mean'])}). Under the reference-only rule every model has the same cell "
         f"composition ({comp.n_incl_1} / {comp.n_incl_2} / {comp.n_incl_3} / {comp.n_incl_4} cells "
         "with 1 / 2 / 3 / 4 reference classes), so the reference-only rows are the fairer model "
         f"comparison: mean {_rng_txt(s_ro['mean'])} across models.", "",
         "Stable-only cells: under the default rule only Stable is in the reference and the "
         f"prediction, and these cells score {_rng_txt(so_def) if len(so_def) else '-'}. Under the "
         "reference-only rule they are the cells whose reference is all Stable; they score the "
         f"Stable F1, {_rng_txt(so_ro) if len(so_ro) else '-'}, lowered by any change the model "
         "predicts there.", "",
         "## Default rule (reference or prediction)", ""]
    L += table("ref_or_pred")
    L += ["", "## Reference-only rule", ""]
    L += table("ref_only")
    L += ["", "Cells = scored cells; 1 / 2 / 3 / 4 classes = cells by number of included classes; "
          "excl. Stable-only = the same statistics without the Stable-only cells.", "",
          "Pooled macro-F1 of the evaluation (any_year, for context): "
          + ", ".join(f"{SHORT[m]} {f3(float(s_def[s_def.model == SHORT[m]].pooled_macro_f1.iloc[0]))}"
                      for m in MODELS) + ".", "",
          "## Checks against the evaluation", ""]
    L += [f"- {c}" for c in checks]
    L += ["", "## Figure checks", ""]
    L += [f"- {w}" for w in warns] if warns else [
        "- Fit-check (every text against every other text, the canvas edge and every plot area): "
        "0 WARN."]
    L += ["", "## Outputs", "",
          "- `prithvi_per_cell_f1_4class.csv`: one row per (model, cell): macro_f1, "
          "n_classes_included and stable_only for the default rule, the same with suffix _ref_only "
          "for the reference-only rule, n_ref_classes, cell_px, per-class F1 (f1_stable, f1_harvest, "
          "f1_development, f1_insect_disease; empty where the class is in neither reference nor "
          "prediction), and the cell's grid, sample, target and label year, reviewer and reference.",
          "- `prithvi_per_cell_f1_summary.csv`: per include rule, model and subset (all_cells, "
          "excl_stable_only): cell counts, Stable-only count, cells by number of included classes, "
          "mean, median, quartiles, min, max, and the evaluation's pooled macro-F1.",
          "- `prithvi_f1_violin.png`, `prithvi_f1_by_n_classes.png` (default rule).", ""]
    with open(os.path.join(out_dir, "note.md"), "w") as fh:
        fh.write("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", required=True,
                    help="required: eval_class_metrics.csv written by collapsed_5class_confusion.py "
                         "--prithvi; its folder must also hold the masked confusion count CSVs, "
                         "scored_cells.csv and metrics_long.csv")
    args = ap.parse_args()
    if SC.OUT_DIR is None:
        sys.exit("refusing to run: set LCI_OUT_DIR so outputs cannot overwrite the Sentinel-2 reports/")
    if SC.SENSOR != "Landsat":
        sys.exit(f"refusing to run: LCI_SENSOR is {SC.SENSOR!r}, expected 'Landsat'")
    if PL.PRITHVI_ZERO_INPUT != MASK or PL.PRITHVI_YEAR_RULE != "target_minus_1":
        sys.exit(f"refusing to run: zero-input rule {PL.PRITHVI_ZERO_INPUT!r} / year rule "
                 f"{PL.PRITHVI_YEAR_RULE!r}, expected {MASK!r} / 'target_minus_1'")
    try:
        out_dir = PCI.check_out_path(SC.out_path(OUT_SUB))
    except ValueError as e:
        sys.exit(f"refusing to run: {e}")
    if not os.path.isfile(args.eval_csv):
        sys.exit(f"refusing to run: --eval-csv {args.eval_csv} not found")
    eval_dir = os.path.dirname(os.path.abspath(args.eval_csv))

    per, selected = PCI.model_cells()
    n_cells = len(per[MODELS[0]])
    if any([k for k, _ in per[m]] != [k for k, _ in per[MODELS[0]]] for m in MODELS):
        sys.exit("STOP: models are scored on different cells")

    n1, bad1 = PCF.check_counts(per, eval_dir)
    n2, bad2, _ = PCF.check_eval(per, args.eval_csv)
    n3, bad3, n_listed = PCF.check_scored(per, selected, eval_dir)
    n4, bad4, pooled = check_pooled_macro(per, eval_dir)
    n5, bad5 = check_ref_only_macro(per)
    bad = bad1 + bad2 + bad3 + bad4 + bad5
    if bad:
        print("STOP: rebuilt per-cell matrices do not reproduce the evaluation:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    checks = [
        f"1. Per-cell 4x4 matrices summed over the {n_cells} cells equal "
        f"confusion_<model>_masked_counts.csv entry by entry for all {len(MODELS)} models "
        f"({n1} entries).",
        f"2. Reference and predicted supports and n_cells equal all {n2} {MASK} rows of "
        "eval_class_metrics.csv.",
        f"3. All {n3} scored cells are listed in scored_cells.csv ({n_listed} rows: cells with "
        "Prithvi output, including those with no usable pixel).",
        f"4. Pooled macro-F1 (metrics_f1_zero on the pooled matrix) equals the {MASK} macro_F1 "
        f"estimate in metrics_long.csv to 5 decimals for all {n4} models.",
        f"5. Per-cell macro-F1 under the reference-only rule equals metrics_f1_zero macro_F1 on all "
        f"{n5} (model, cell) cases (to 1e-12)."]
    for c in checks:
        print("check " + c)

    df = per_cell_rows(per, selected)
    summ = summary_full(df, pooled)
    agri = agri_px(eval_dir)

    # everything computed and checked: write
    os.makedirs(out_dir, exist_ok=True)
    w1, p1 = violin_fig(out_dir, df, summ, n_cells)
    w2, p2 = by_n_fig(out_dir, df, n_cells)
    warns = w1 + w2
    for w in warns:
        print(w)
    out = df.copy()
    for c in ["macro_f1", "macro_f1_ref_only"] + F1_COLS:
        out[c] = out[c].map(lambda v: cc._r5(v) if np.isfinite(v) else np.nan)
    out.to_csv(os.path.join(out_dir, "prithvi_per_cell_f1_4class.csv"), index=False)
    s_out = summ.copy()
    for c in STATS:
        s_out[c] = s_out[c].round(4)
    s_out["pooled_macro_f1"] = s_out.pooled_macro_f1.map(cc._r5)
    s_out.to_csv(os.path.join(out_dir, "prithvi_per_cell_f1_summary.csv"), index=False)
    write_note(out_dir, df, summ, n_cells, len(selected), n_listed, agri, checks, warns)

    for rule in RULES:
        print(f"\n===== include rule {rule} =====")
        print(summ[summ.include_rule == rule].drop(columns="include_rule").to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(warns)} WARN line(s) from the figure fit-check")
    print(f"outputs -> {out_dir}/")
    for p in (p1, p2):
        print(f"  {p}")


if __name__ == "__main__":
    main()
