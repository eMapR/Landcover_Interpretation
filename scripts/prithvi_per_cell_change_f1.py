#!/usr/bin/env python3
"""Landsat: per-grid-cell F1 of the three change classes for the four Prithvi models.

Landsat/Prithvi counterpart of scripts/per_cell_change_f1.py (Sentinel-2 only). For each Prithvi
model (scripts/prithvi_loader.py), each scored grid cell and each change class (Harvest, Development,
Insect/Disease), the cell's 4x4 confusion matrix is built against the interpreted Landsat reference
and that class's F1 in the cell is computed. The distribution of interest, per (model, change class),
is the set of per-cell F1 values across the cells that include the class. Stable is kept in the
matrix but not analyzed.

Scheme, cells and pixels (identical to collapsed_5class_confusion.py --prithvi, zero-input mask
any_year): label year = reference target year - 1; Stable, Harvest, Development, Insect/Disease;
Beaver, Unknown/Fire and agricultural change excluded; only reference pixels with a Prithvi pixel under
their center, minus the zero-input mask; the 312 scored cells with seeded reviewer selection (seed 42).
The per-cell matrices come from prithvi_class_ci_vs_interpreter.model_cells(), which calls the
evaluation's own functions (cc.deduped_cells, PL.load_for_reference, cc.prithvi_cell_tallies).
All four models share the same cells by construction, so there is no separate common cell set.

Per-cell F1 = 2TP / (reference px + predicted px) of the class in the cell (0 when TP = 0). On every
cell where the class is in the reference this equals cc.metrics_f1_zero (checked); metrics_f1_zero
leaves a prediction-only cell undefined, so it is not used directly.

Include rule: a cell counts for class X if X is in the reference OR the prediction (default,
--include ref_or_pred; a prediction-only cell is commission, F1 = 0), or only if X is in the reference
(--include ref_only). The figures use the chosen rule; the summary CSV and note.md report both.
Small cells (< 5 px of the class in reference + prediction) are counted, not removed.

Checks, all before anything is written (the script stops on any difference):
  1. per-cell matrices summed over the cells == confusion_<model>_masked_counts.csv (the folder of
     --eval-csv), entry by entry, for every model;
  2. ref_support_px, pred_support_px and n_cells == every any_year row of eval_class_metrics.csv;
  3. per-cell F1 == cc.metrics_f1_zero on every cell where the class is in the reference;
  4. every scored cell is listed in scored_cells.csv (same folder).

Outputs -> $LCI_OUT_DIR/reports/prithvi_per_cell_change_f1/
  - prithvi_per_cell_change_f1.csv     one row per (model, change class, cell with the class in
                                       reference or prediction)
  - prithvi_change_f1_summary.csv      per (include rule, change class, model): N, composition,
                                       small cells, mean/median/quartiles, F1 = 0 / 1 shares, pooled F1
  - prithvi_change_f1_violins.png      per change class, the four models' per-cell F1 with N
  - prithvi_change_f1_heatmap.png      mean per-cell F1 (median and n in each tile)
  - note.md

Refuses to run without --eval-csv, without LCI_OUT_DIR, with LCI_SENSOR other than Landsat, with a
zero-input rule other than any_year or a year rule other than target_minus_1, or when the output
folder is inside <repo>/reports, /vol/v1/HLS_Prithvi, /home/server/hpc/emapr or the Prithvi root
(check_out_path).

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs LCI_OUT_DIR=reports_landsat python scripts/prithvi_per_cell_change_f1.py --eval-csv reports_landsat/reports/collapsed_5class_confusion/eval_class_metrics.csv

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collapsed_5class_confusion as cc            # noqa: E402  metrics_f1_zero, names, _r5
import prithvi_class_ci_vs_interpreter as PCI      # noqa: E402  model_cells, check_out_path, figure checks
import prithvi_loader as PL                        # noqa: E402

SC = cc.SC
OUT_SUB = "reports/prithvi_per_cell_change_f1"
MASK = PCI.MASK                                    # any_year
MODELS = PCI.MODELS
SHORT = PCI.SHORT
COLOR = PCI.COLOR
NAMES = cc.NAMES5
P_LABELS = cc.P_LABELS                             # Stable, Harvest, Development, Insect/Disease
CHANGE = [2, 3, 4]                                 # Harvest, Development, Insect/Disease
SMALL_PX = 5                                       # small cell: reference + predicted px < 5
RULES = ["ref_or_pred", "ref_only"]
RULE_TEXT = {"ref_or_pred": "the class is in the reference or the prediction",
             "ref_only": "the class is in the reference"}
DPI = 300
NBSP = " "
JITTER_SEED = 0


def cell_f1(cm, k):
    """(f1, tp, ref px, pred px) of 0-based class k in one cell's 4x4: 2TP/(ref+pred), 0 if no TP."""
    tp, ref, pred = int(cm[k, k]), int(cm[k, :].sum()), int(cm[:, k].sum())
    return (2.0 * tp / (ref + pred) if ref + pred > 0 else np.nan), tp, ref, pred


# ----------------------------------------------------------------------------- checks
def check_counts(per, eval_dir):
    """Check 1: pooled per-cell matrices == confusion_<model>_masked_counts.csv, entry by entry."""
    bad = []
    for m in MODELS:
        path = os.path.join(eval_dir, f"confusion_{m}_masked_counts.csv")
        if not os.path.isfile(path):
            bad.append(f"{path} not found")
            continue
        ev = pd.read_csv(path, index_col=0)
        if list(ev.index) != P_LABELS or list(ev.columns) != P_LABELS:
            bad.append(f"{path}: labels {list(ev.index)} / {list(ev.columns)}, expected {P_LABELS}")
            continue
        pooled = np.stack([cm for _, cm in per[m]]).sum(0)
        for i, ri in enumerate(P_LABELS):
            for j, cj in enumerate(P_LABELS):
                if int(pooled[i, j]) != int(ev.iloc[i, j]):
                    bad.append(f"{SHORT[m]} [{ri}, {cj}]: rebuilt {int(pooled[i, j])} vs "
                               f"counts CSV {int(ev.iloc[i, j])}")
    return 16 * len(MODELS), bad


def check_eval(per, eval_csv):
    """Check 2: supports and n_cells == every any_year row of eval_class_metrics.csv.
    Returns (rows checked, mismatches, {(model, code): eval F1})."""
    ev = pd.read_csv(eval_csv)
    ev = ev[ev["mask"] == MASK]
    bad, n, pooled_f1 = [], 0, {}
    for m in MODELS:
        stack = np.stack([cm for _, cm in per[m]])
        pooled = stack.sum(0)
        for c in cc.P_CLASSES:
            k = c - 1
            sel = ev[(ev.model == m) & (ev["class"] == NAMES[c])]
            if len(sel) != 1:
                bad.append(f"{m} {NAMES[c]}: {len(sel)} rows in {eval_csv}")
                continue
            e = sel.iloc[0]
            mine = dict(ref_support_px=int(pooled[k].sum()), pred_support_px=int(pooled[:, k].sum()),
                        n_cells=len(stack))
            for col, v in mine.items():
                if pd.isna(e[col]) or int(e[col]) != v:
                    bad.append(f"{SHORT[m]} {NAMES[c]} {col}: rebuilt {v} vs eval {e[col]!r}")
            pooled_f1[(m, c)] = float(e["F1"]) if not pd.isna(e["F1"]) else np.nan
            n += 1
    if n != len(ev):
        bad.append(f"checked {n} rows but {eval_csv} has {len(ev)} {MASK} rows")
    return n, bad, pooled_f1


def check_f1_rule(per):
    """Check 3: 2TP/(ref+pred) == cc.metrics_f1_zero F1 on every cell with the class in the reference."""
    bad, n = [], 0
    for m in MODELS:
        for key, cm in per[m]:
            mf = cc.metrics_f1_zero(cm)
            for c in CHANGE:
                f1, _, ref, _ = cell_f1(cm, c - 1)
                if ref == 0:
                    continue
                n += 1
                if not (np.isfinite(mf[f"F1[{c}]"]) and abs(f1 - mf[f"F1[{c}]"]) < 1e-12):
                    bad.append(f"{SHORT[m]} {key} {NAMES[c]}: 2TP/(ref+pred) {f1} vs "
                               f"metrics_f1_zero {mf[f'F1[{c}]']}")
    return n, bad


def check_scored(per, selected, eval_dir):
    """Check 4: every scored cell's reference file is listed in scored_cells.csv."""
    path = os.path.join(eval_dir, "scored_cells.csv")
    if not os.path.isfile(path):
        return 0, [f"{path} not found"], 0
    listed = set(pd.read_csv(path)["reference"])
    keys = [k for k, _ in per[MODELS[0]]]
    bad = [f"{os.path.basename(selected[k])} not in {path}" for k in keys
           if os.path.basename(selected[k]) not in listed]
    return len(keys), bad, len(listed)


# ----------------------------------------------------------------------------- tables
def per_cell_rows(per, selected):
    """One row per (model, change class, cell with the class in reference or prediction)."""
    rows = []
    for m in MODELS:
        for key, cm in per[m]:
            ref_path = selected[key]
            _, target, reviewer = PL.parse_reference(ref_path)
            for c in CHANGE:
                f1, tp, ref, pred = cell_f1(cm, c - 1)
                if ref + pred == 0:
                    continue
                rows.append(dict(
                    model=SHORT[m], change_class=NAMES[c], grid=key[0], sample=key[1],
                    target_year=int(target), label_year=PL.label_year(int(target)), reviewer=reviewer,
                    f1=cc._r5(f1), tp_px=tp, ref_pixels_of_class=ref, pred_pixels_of_class=pred,
                    class_px=ref + pred, small_cell=bool(ref + pred < SMALL_PX),
                    present_via=("both" if ref and pred else "reference" if ref else "prediction"),
                    in_ref_only_set=bool(ref > 0), reference=os.path.basename(ref_path)))
    return pd.DataFrame(rows)


def subset(df, rule, m, c):
    g = df[(df.model == SHORT[m]) & (df.change_class == NAMES[c])]
    return g[g.in_ref_only_set] if rule == "ref_only" else g


def summary_rows(df, pooled_f1):
    rows = []
    for rule in RULES:
        for c in CHANGE:
            for m in MODELS:
                g = subset(df, rule, m, c)
                f = g.f1.astype(float)
                n = len(g)
                rows.append(dict(
                    include_rule=rule, change_class=NAMES[c], model=SHORT[m], n_cells=n,
                    n_both=int((g.present_via == "both").sum()),
                    n_ref_only=int((g.present_via == "reference").sum()),
                    n_pred_only=int((g.present_via == "prediction").sum()),
                    n_small_lt5=int(g.small_cell.sum()), n_small_le2=int((g.class_px <= 2).sum()),
                    mean_f1=round(float(f.mean()), 4) if n else np.nan,
                    median_f1=round(float(f.median()), 4) if n else np.nan,
                    q25_f1=round(float(f.quantile(0.25)), 4) if n else np.nan,
                    q75_f1=round(float(f.quantile(0.75)), 4) if n else np.nan,
                    frac_f1_zero=round(float((f == 0).mean()), 4) if n else np.nan,
                    frac_f1_one=round(float((f == 1).mean()), 4) if n else np.nan,
                    n_f1_zero=int((f == 0).sum()), n_f1_one=int((f == 1).sum()),
                    pooled_f1=cc._r5(pooled_f1[(m, c)]),
                    # full precision for note.md (rounded once there); not written to the CSV
                    full_mean_f1=float(f.mean()) if n else np.nan,
                    full_median_f1=float(f.median()) if n else np.nan))
    return pd.DataFrame(rows)


FULL_COLS = ["full_mean_f1", "full_median_f1"]     # in-memory only, dropped from the summary CSV


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _drawn_ticklabels(axis):
    """Tick labels matplotlib actually draws: non-empty and with the tick inside the view limits
    (get_ticklabels also returns labels for ticks just outside the range, which are not drawn)."""
    lo, hi = sorted(axis.get_view_interval())
    eps = 1e-9 * max(1.0, abs(hi - lo))
    return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
            if t.get_text() and lo - eps <= v <= hi + eps]


def _texts_vs_areas(fig, texts, areas, label):
    """WARN for any text intersecting a plot area (axes box: data, violins, markers, heat tiles)."""
    r = fig.canvas.get_renderer()
    warns = []
    for n, t in texts:
        bb = t.get_window_extent(r)
        for an, a in areas:
            if bb.overlaps(a.get_window_extent(r)):
                warns.append(f"WARN [{label}] {n} overlaps {an}")
    return warns


def violins(out_dir, df, rule, n_cells):
    plt = _plt()
    from matplotlib.lines import Line2D

    W = 14.5
    L_IN, R_IN, GAP_IN = 0.85, 0.25, 0.45
    PW = (W - L_IN - R_IN - 2 * GAP_IN) / 3
    AX_H = 4.2
    SUP_IN, PT_IN = 0.6, 0.45
    XT_IN, LEG_IN, CAP_GAP, CAP_PAD, CAP_FS = 0.75, 0.42, 0.15, 0.12, 9

    caption = (f"Per-cell F1 of each change class for the four Prithvi models on the {n_cells} scored "
               f"cells (Landsat interpreted reference, pixels with a Prithvi pixel under the reference "
               f"pixel center, zero-input mask any_year). Per-cell F1 = "
               + NBSP.join("2TP / (reference px + predicted px)".split(" "))
               + f" of the class in the cell. A cell is included for a class when "
               f"{RULE_TEXT[rule]}; n under each model is the number of included cells"
               + ("; a cell where only the model predicts the class (commission) scores 0, so the "
                  "spike at 0 is largely false alarms" if rule == "ref_or_pred" else "")
               + f". One point per cell (jittered sideways), hollow when the cell has fewer than "
               f"{SMALL_PX}{NBSP}px of the class in reference + prediction; bar = median, diamond = "
               f"mean.")
    fig = plt.figure(figsize=(W, 4.0), dpi=DPI)
    cap_txt = PCI._wrap_measure(fig, caption, W - 0.6, CAP_FS)
    probe = fig.text(0, 0, cap_txt, fontsize=CAP_FS)
    cap_h = probe.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    probe.remove()
    H = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XT_IN + AX_H + PT_IN + SUP_IN
    fig.set_size_inches(W, H)
    ax_y0 = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XT_IN
    tr = fig.dpi_scale_trans
    rng = np.random.default_rng(JITTER_SEED)

    axes, texts = [], []
    for pi, c in enumerate(CHANGE):
        x0 = L_IN + pi * (PW + GAP_IN)
        ax = fig.add_axes([x0 / W, ax_y0 / H, PW / W, AX_H / H])
        axes.append((f"{NAMES[c]} plot area", ax))
        ns = []
        for i, m in enumerate(MODELS):
            g = subset(df, rule, m, c)
            d = g.f1.astype(float).values
            small = g.small_cell.values
            ns.append(len(d))
            if len(d) >= 2 and np.ptp(d) > 0:
                body = ax.violinplot([d], positions=[i], showextrema=False, widths=0.8)["bodies"][0]
                body.set_facecolor(COLOR[m]); body.set_alpha(0.25); body.set_edgecolor("0.5")
            if len(d):
                xs = i + rng.uniform(-0.15, 0.15, len(d))
                ax.scatter(xs[~small], d[~small], s=10, color=COLOR[m], alpha=0.55,
                           edgecolor="none", zorder=3)
                ax.scatter(xs[small], d[small], s=12, facecolor="none", edgecolor=COLOR[m],
                           linewidths=0.8, zorder=3)
                ax.plot([i - 0.3, i + 0.3], [np.median(d)] * 2, color="black", lw=2, zorder=4)
                ax.plot(i, np.mean(d), marker="D", color="black", ms=6, zorder=5, ls="")
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([f"{SHORT[m]}\n(n={n})" for m, n in zip(MODELS, ns)])
        ax.tick_params(axis="x", labelsize=10, length=0, pad=6)
        ax.tick_params(axis="y", labelsize=10)
        if pi == 0:
            ax.set_ylabel("Per-Cell F1", fontsize=12)
        else:
            ax.set_yticklabels([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(False)
        t = fig.text(x0 + PW / 2, ax_y0 + AX_H + PT_IN / 2, NAMES[c], transform=tr, ha="center",
                     va="center", fontsize=13, fontweight="bold")
        texts.append((f"panel title {NAMES[c]}", t))

    sup = fig.text(W / 2, H - SUP_IN / 2, f"Per-Cell F1 of the Change Classes by Prithvi Model "
                   f"(Landsat, {n_cells} Cells)", transform=tr, ha="center", va="center",
                   fontsize=15, fontweight="bold")
    handles = [Line2D([], [], color="black", lw=2, label="median"),
               Line2D([], [], color="black", marker="D", ls="", ms=6, label="mean"),
               Line2D([], [], color="0.4", marker="o", ls="", ms=5, alpha=0.7,
                      markeredgecolor="none", label="cell"),
               Line2D([], [], color="0.4", marker="o", ls="", ms=5, markerfacecolor="none",
                      label=f"cell with < {SMALL_PX} px of the class")]
    leg = fig.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=10,
                     bbox_to_anchor=(W / 2, CAP_PAD + cap_h + CAP_GAP + LEG_IN / 2),
                     bbox_transform=tr, columnspacing=2.0)
    cap = fig.text(W / 2, CAP_PAD, cap_txt, transform=tr, ha="center", va="bottom",
                   fontsize=CAP_FS, color="0.3")
    fig.canvas.draw()
    texts += [("title", sup), ("legend", leg), ("caption", cap)]
    for an, ax in axes:
        p = an.replace(" plot area", "")
        texts += [(f"{p} x tick '{t.get_text().splitlines()[0]}'", t)
                  for t in _drawn_ticklabels(ax.xaxis)]
        texts += [(f"{p} y tick '{t.get_text()}'", t) for t in _drawn_ticklabels(ax.yaxis)]
        if ax.yaxis.label.get_text():
            texts.append((f"{p} y label", ax.yaxis.label))
    warns = PCI.check_overlaps(fig, texts, [], "violins") + _texts_vs_areas(fig, texts, axes, "violins")
    path = os.path.join(out_dir, "prithvi_change_f1_violins.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return warns, path


def heatmap(out_dir, summ, rule, n_cells):
    plt = _plt()
    s = summ[summ.include_rule == rule]
    mean = np.full((len(CHANGE), len(MODELS)), np.nan)
    med = np.full_like(mean, np.nan)
    ncell = np.zeros(mean.shape, int)
    for ci, c in enumerate(CHANGE):
        for mi, m in enumerate(MODELS):
            r = s[(s.change_class == NAMES[c]) & (s.model == SHORT[m])].iloc[0]
            # full precision, rounded once when the tile text is formatted
            mean[ci, mi], med[ci, mi], ncell[ci, mi] = r.full_mean_f1, r.full_median_f1, r.n_cells

    W = 10.4
    L_IN, TW, TH = 1.55, 1.65, 1.0
    CB_GAP, CB_W, R_IN = 0.3, 0.22, 1.1
    AX_W, AX_H = TW * len(MODELS), TH * len(CHANGE)
    TITLE_IN, GAP_TOP, XT_IN, CAP_GAP, CAP_PAD, CAP_FS = 0.55, 0.2, 0.45, 0.15, 0.12, 9
    caption = (f"Mean per-cell F1 of each change class (rows) for each Prithvi model (columns) on the "
               f"{n_cells} scored cells (Landsat, zero-input mask any_year), with the median and the "
               f"number of included cells n in each tile. A cell is included when "
               f"{RULE_TEXT[rule]}. The median is 0 in "
               f"{'every' if np.all(med == 0) else 'almost every' if np.mean(med == 0) > 0.5 else 'some'}"
               f" tile because most included cells score 0, so the color shows the mean; read n with "
               f"it, since the tiles rest on very different numbers of cells.")
    fig = plt.figure(figsize=(W, 4.0), dpi=DPI)
    cap_txt = PCI._wrap_measure(fig, caption, W - 0.6, CAP_FS)
    probe = fig.text(0, 0, cap_txt, fontsize=CAP_FS)
    cap_h = probe.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    probe.remove()
    H = CAP_PAD + cap_h + CAP_GAP + XT_IN + AX_H + GAP_TOP + TITLE_IN
    fig.set_size_inches(W, H)
    ax_y0 = CAP_PAD + cap_h + CAP_GAP + XT_IN
    tr = fig.dpi_scale_trans
    ax = fig.add_axes([L_IN / W, ax_y0 / H, AX_W / W, AX_H / H])
    cax = fig.add_axes([(L_IN + AX_W + CB_GAP) / W, ax_y0 / H, CB_W / W, AX_H / H])
    vmax = max(0.05, float(np.nanmax(mean)))
    im = ax.imshow(mean, cmap="Greens", vmin=0, vmax=vmax, aspect="auto")
    tiles = []
    for ci in range(len(CHANGE)):
        for mi in range(len(MODELS)):
            t = ax.text(mi, ci, f"mean {mean[ci, mi]:.3f}\nmedian {med[ci, mi]:.3f}\n"
                                f"n = {ncell[ci, mi]}", ha="center", va="center", fontsize=9.5,
                        linespacing=1.3, color="white" if mean[ci, mi] > 0.55 * vmax else "black")
            tiles.append(((ci, mi), t))
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_yticks(range(len(CHANGE)))
    ax.set_yticklabels([NAMES[c] for c in CHANGE])
    ax.tick_params(labelsize=11, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean Per-Cell F1", fontsize=11)
    cb.ax.tick_params(labelsize=9.5)
    title = fig.text(W / 2, H - GAP_TOP / 2 - TITLE_IN / 2, f"Mean Per-Cell F1 by Change Class and "
                     f"Prithvi Model (Landsat, {n_cells} Cells)", transform=tr, ha="center",
                     va="center", fontsize=13.5, fontweight="bold")
    cap = fig.text(W / 2, CAP_PAD, cap_txt, transform=tr, ha="center", va="bottom",
                   fontsize=CAP_FS, color="0.3")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    outer = [("title", title), ("caption", cap), ("colorbar label", cb.ax.yaxis.label)]
    outer += [(f"x tick '{t.get_text()}'", t) for t in _drawn_ticklabels(ax.xaxis)]
    outer += [(f"y tick '{t.get_text()}'", t) for t in _drawn_ticklabels(ax.yaxis)]
    outer += [(f"colorbar tick '{t.get_text()}'", t) for t in _drawn_ticklabels(cb.ax.yaxis)]
    tile_texts = [(f"tile {NAMES[CHANGE[ci]]}/{SHORT[MODELS[mi]]}", t) for (ci, mi), t in tiles]
    warns = PCI.check_overlaps(fig, outer + tile_texts, [], "heatmap")
    warns += _texts_vs_areas(fig, outer, [("heatmap area", ax), ("colorbar", cax)], "heatmap")
    margin = 0.06 * fig.dpi
    for ((ci, mi), t), (n, _) in zip(tiles, tile_texts):
        bb = t.get_window_extent(r)
        (x0, y0), (x1, y1) = ax.transData.transform([(mi - 0.5, ci + 0.5), (mi + 0.5, ci - 0.5)])
        if not (bb.x0 >= x0 + margin and bb.x1 <= x1 - margin and bb.y0 >= y0 + margin
                and bb.y1 <= y1 - margin):
            warns.append(f"WARN [heatmap] {n} does not fit inside its tile with margin")
    path = os.path.join(out_dir, "prithvi_change_f1_heatmap.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return warns, path


# ----------------------------------------------------------------------------- note
def write_note(out_dir, summ, rule, n_cells, n_dedup, n_scored_listed, checks, warns):
    def table(rule_, cols, head):
        L = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|"]
        for r in summ[summ.include_rule == rule_].itertuples():
            L.append("| " + " | ".join(str(f(r)) for f in cols) + " |")
        return L

    d = summ[summ.include_rule == "ref_or_pred"].copy()
    d["share"] = d.n_pred_only / d.n_cells
    heavy = d[d.share > 0.5].sort_values("share", ascending=False)
    top = d.loc[d.share.idxmax()]
    L = ["# prithvi_per_cell_change_f1", "",
         "Per-grid-cell F1 of the three change classes (Harvest, Development, Insect/Disease) for the "
         "four Prithvi models on Landsat, treating each grid cell as one sample. Generated by "
         "`scripts/prithvi_per_cell_change_f1.py`; Landsat/Prithvi counterpart of "
         "`scripts/per_cell_change_f1.py` (Sentinel-2 only).", "",
         "## Method", "",
         "- Scheme and pixels as in `collapsed_5class_confusion.py --prithvi`: label year = reference "
         "target year - 1; Stable, Harvest, Development, Insect/Disease; Beaver, Unknown/Fire and "
         "agricultural change excluded; reference pixels with a Prithvi pixel under their center, minus "
         "the zero-input mask any_year. Stable is kept in each cell's 4x4 matrix but not analyzed.",
         f"- Cells: {n_dedup} de-duplicated reference cells (seeded reviewer selection, seed "
         f"{cc.SEED}); {n_scored_listed} have Prithvi output (scored_cells.csv); **{n_cells}** have at "
         "least one usable pixel and are scored. The per-cell matrices come from the evaluation's own "
         "functions (via `prithvi_class_ci_vs_interpreter.model_cells`). All four models are scored on "
         "the same cells, so there is no separate common cell set.",
         "- Per-cell F1 rule: F1 = 2TP / (reference pixels + predicted pixels) of the class in the cell, "
         "which is the usual 2PR/(P+R) and is 0 when there is no true positive. The evaluation's "
         "`metrics_f1_zero` gives the same value on every cell where the class is in the reference "
         "(checked below), but leaves a prediction-only cell undefined, so it is not used directly: "
         "that would silently drop the commission cells the default rule is meant to include.",
         f"- Include rule used for the figures and the per-cell CSV's main set: **{rule}** (a cell "
         f"counts for a class when {RULE_TEXT[rule]}). The summary CSV and the tables below report "
         "both rules (`include_rule` column).",
         f"- No minimum pixel count: small cells (< {SMALL_PX} px of the class in reference + "
         "prediction) are kept and counted. No cell scores exactly 1 (`n_f1_one` = 0 throughout), so "
         "small cells only add exact zeros.", "",
         "## Commission under the default rule", "",
         f"Under the default rule (reference or prediction) many included cells are prediction-only: "
         f"the model predicts the class in the cell but the reference has none, so the cell scores 0. "
         f"For example {top.model} {top.change_class}: {top.n_pred_only} of {top.n_cells} included "
         "cells are prediction-only. Where this share is above half, the distribution mostly measures "
         "false alarms, not how well the class is mapped where it occurs:", ""]
    L += [f"- {r.model} {r.change_class}: {r.n_pred_only} of {r.n_cells} prediction-only "
          f"({r.share:.0%})" for r in heavy.itertuples()] or ["- none above half."]
    L += ["", "The reference-only rule (table below) drops those cells and shows the per-cell F1 where "
          "the class is actually in the reference.", "",
          "## Default rule (reference or prediction)", ""]
    L += table("ref_or_pred",
               [lambda r: r.change_class, lambda r: r.model, lambda r: r.n_cells, lambda r: r.n_both,
                lambda r: r.n_ref_only, lambda r: r.n_pred_only, lambda r: r.n_small_lt5,
                lambda r: r.n_small_le2, lambda r: r.n_f1_zero, lambda r: f"{r.full_mean_f1:.3f}",
                lambda r: f"{r.full_median_f1:.3f}", lambda r: f"{float(r.pooled_f1):.3f}"],
               ["Class", "Model", "N", "Both", "Ref only", "Pred only", f"< {SMALL_PX} px", "<= 2 px",
                "F1 = 0", "Mean", "Median", "Pooled F1"])
    L += ["", "N = included cells; Both / Ref only / Pred only = where the class is present in the "
          "cell; Pooled F1 = the evaluation's F1 over all cells (eval_class_metrics.csv), for context.",
          "", "## Reference-only rule", ""]
    L += table("ref_only",
               [lambda r: r.change_class, lambda r: r.model, lambda r: r.n_cells,
                lambda r: r.n_small_lt5, lambda r: r.n_f1_zero, lambda r: f"{r.full_mean_f1:.3f}",
                lambda r: f"{r.full_median_f1:.3f}"],
               ["Class", "Model", "N", f"< {SMALL_PX} px", "F1 = 0", "Mean", "Median"])
    L += ["", "## Checks against the evaluation", ""]
    L += [f"- {c}" for c in checks]
    L += ["", "## Figure checks", ""]
    L += [f"- {w}" for w in warns] if warns else [
        "- Fit-check (every text against every other text, the canvas edge and every plot area; heat "
        "tile text inside its tile with margin): 0 WARN."]
    L += ["", "## Outputs", "",
          "- `prithvi_per_cell_change_f1.csv`: one row per (model, change class, cell with the class in "
          "reference or prediction): f1, tp_px, ref_pixels_of_class, pred_pixels_of_class, class_px, "
          "small_cell, present_via, in_ref_only_set (the reference-only subset), and the cell's grid, "
          "sample, target and label year, reviewer and reference file.",
          "- `prithvi_change_f1_summary.csv`: per include rule, change class and model: N, composition, "
          "small cells, mean, median, quartiles, F1 = 0 / 1 counts and shares, pooled F1.",
          f"- `prithvi_change_f1_violins.png`, `prithvi_change_f1_heatmap.png` ({rule} rule).", ""]
    with open(os.path.join(out_dir, "note.md"), "w") as fh:
        fh.write("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", required=True,
                    help="required: eval_class_metrics.csv written by collapsed_5class_confusion.py "
                         "--prithvi; its folder must also hold the masked confusion count CSVs and "
                         "scored_cells.csv")
    ap.add_argument("--include", choices=RULES, default="ref_or_pred",
                    help="include rule for the figures: class in reference or prediction (default) "
                         "or reference only; the summary reports both")
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
    if any(len(per[m]) != n_cells for m in MODELS):
        sys.exit("STOP: models have different scored cell counts")
    if any([k for k, _ in per[m]] != [k for k, _ in per[MODELS[0]]] for m in MODELS):
        sys.exit("STOP: models are scored on different cells")

    n1, bad1 = check_counts(per, eval_dir)
    n2, bad2, pooled_f1 = check_eval(per, args.eval_csv)
    n3, bad3 = check_f1_rule(per)
    n4, bad4, n_listed = check_scored(per, selected, eval_dir)
    bad = bad1 + bad2 + bad3 + bad4
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
        f"3. 2TP/(ref+pred) equals metrics_f1_zero on all {n3} (model, cell, change class) cases "
        "with the class in the reference (to 1e-12).",
        f"4. All {n4} scored cells are listed in scored_cells.csv ({n_listed} rows: cells with "
        "Prithvi output, including those with no usable pixel)."]
    for c in checks:
        print("check " + c)

    df = per_cell_rows(per, selected)
    summ = summary_rows(df, pooled_f1)

    # everything computed and checked: write
    os.makedirs(out_dir, exist_ok=True)
    w1, p1 = violins(out_dir, df, args.include, n_cells)
    w2, p2 = heatmap(out_dir, summ, args.include, n_cells)
    warns = w1 + w2
    for w in warns:
        print(w)
    df.to_csv(os.path.join(out_dir, "prithvi_per_cell_change_f1.csv"), index=False)
    summ.drop(columns=FULL_COLS).to_csv(os.path.join(out_dir, "prithvi_change_f1_summary.csv"),
                                        index=False)
    write_note(out_dir, summ, args.include, n_cells, len(selected), n_listed, checks, warns)

    cols = ["change_class", "model", "n_cells", "n_both", "n_ref_only", "n_pred_only", "n_small_lt5",
            "n_small_le2", "n_f1_zero", "n_f1_one", "mean_f1", "median_f1", "pooled_f1"]
    for rule in RULES:
        print(f"\n===== include rule {rule} =====")
        print(summ[summ.include_rule == rule][cols].to_string(index=False))
    print(f"\n{len(warns)} WARN line(s) from the figure fit-check")
    print(f"outputs -> {out_dir}/")
    for p in (p1, p2):
        print(f"  {p}")


if __name__ == "__main__":
    main()
