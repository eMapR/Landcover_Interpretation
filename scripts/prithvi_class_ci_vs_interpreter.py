#!/usr/bin/env python3
"""Landsat: Prithvi per-class F1 next to the inter-interpreter agreement "ceiling" (four-class).

Landsat counterpart of scripts/model_class_ci_5class.py (Sentinel-2). For each of the four Prithvi
models (scripts/prithvi_loader.py), per-class F1 against the interpreted Landsat reference is drawn
next to the inter-interpreter agreement for the same class, computed here from the Landsat
double-interpreted pairs. The interpreter agreement is a benchmark for reference noise, not a strict
upper bound: the model is scored against one interpretation, so it can in principle score above it.

Scheme and pixels (identical to collapsed_5class_confusion.py --prithvi, zero-input mask any_year):
  - classes: Stable, Harvest, Development, Insect/Disease; Beaver, Unknown/Fire and Prithvi
    agricultural change excluded.
  - pixels: only reference pixels with a Prithvi pixel under their center, minus the zero-input mask
    (any_year), from the same prithvi_loader.load_for_reference call the evaluation uses.

Model side (all cells): the 312 scored cells, rebuilt with the evaluation's own functions
(cc.deduped_cells, PL.load_for_reference, cc.prithvi_cell_tallies, cc.metrics_f1_zero,
cc.bootstrap_cis with seed 42 and 2000 draws, same cell order). Every any_year row of
eval_class_metrics.csv (--eval-csv) must match (UA, PA, F1 and their CIs to 5 decimals, supports and
n_cells exactly), or the script stops before writing anything.

Interpreter side: the Landsat double-interpreted pairs (compare_interpreters.find_pairs), restricted to
pairs with Prithvi output and at least one usable pixel. Both maps are collapsed with
cc.collapse_reference; a pixel counts when it passes the Prithvi footprint and zero-input mask and both
reviewers assigned one of the four classes (Beaver or Unknown/Fire from either reviewer drops it).
F1 = 2*TP / (row + col) (interpreter_class_ci.per_class, symmetric), CI from a pair bootstrap
(seed 42, 2000 draws).

Like-for-like: each model is also scored on ONLY the paired cells (same cells as the interpreter
ceiling; reference = the seeded-selected reviewer, as in the evaluation), and the paired difference
model F1 - interpreter F1 gets a paired bootstrap CI (the same resampled pairs for both, seed 42).
This is reported in the CSVs and note.md, not in the figures.

Outputs -> $LCI_OUT_DIR/reports/prithvi_vs_interpreter/
  - prithvi_vs_interpreter_class_f1.csv   one row per (series, model, scope, class)
  - prithvi_vs_interpreter_gap.csv        per model and class: interpreter, model (all / paired), gaps
  - pairs_used.csv                        all double-interpreted pairs and why each is used or not
  - forest_prithvi_<model>.png            per model: model F1 vs interpreter agreement, per class
  - forest_prithvi_all_models.png         all four models over the interpreter agreement
  - note.md

Refuses to run without --eval-csv, without LCI_OUT_DIR, with LCI_SENSOR other than Landsat, with a zero-input rule other
than any_year or a year rule other than target_minus_1, or when the output folder is inside
<repo>/reports, /vol/v1/HLS_Prithvi, /home/server/hpc/emapr or the Prithvi root (check_out_path).

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs LCI_OUT_DIR=reports_landsat python scripts/prithvi_class_ci_vs_interpreter.py --eval-csv reports_landsat/reports/collapsed_5class_confusion/eval_class_metrics.csv

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collapsed_5class_confusion as cc    # noqa: E402  dedup, collapse, tallies, metrics, bootstrap
import compare_interpreters as CI          # noqa: E402  find_pairs
import interpreter_class_ci as ICI         # noqa: E402  per_class, tier, HIGH, MOD
import prithvi_loader as PL                # noqa: E402

SC = cc.SC
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUT_SUB = "reports/prithvi_vs_interpreter"
MASK = "any_year"
SEED, BOOT = cc.SEED, cc.BOOT              # 42, 2000: must equal the evaluation to reproduce it
CLASSES = cc.P_CLASSES                     # [1, 2, 3, 4]
NAMES = cc.NAMES5
MODELS = PL.PRITHVI_MODELS
SHORT = {m: m.replace("_encoder_q1_w2", "").replace("_q1_w2", "") for m in MODELS}
COLOR = {"fcn_frozen_encoder_q1_w2": "#0072B2", "fcn_random_encoder_q1_w2": "#E69F00",
         "fcn_unfrozen_encoder_q1_w2": "#009E73", "unet3d_q1_w2": "#CC79A7"}   # Okabe-Ito
INTERP_FACE, INTERP_LINE = "#808080", "#BDBDBD"   # gray diamond (white edge), light gray CI bar
NO_AGREE_COLOR = "0.3"                             # "no interpreter agreement" label text
NO_AGREE = "no interpreter agreement"
NBSP = "\u00a0"                              # number-unit glue in captions (never split)
DPI = 300
CI_MODEL = f"cell bootstrap (seed {SEED}, {BOOT} draws), percentile 95% CI, no FPC"
CI_INTERP = f"pair bootstrap (seed {SEED}, {BOOT} draws), percentile 95% CI"


def check_out_path(path):
    """realpath of `path`; ValueError if it is inside <repo>/reports, /vol/v1/HLS_Prithvi,
    /home/server/hpc/emapr or PL.PRITHVI_ROOT. Creates nothing (the path need not exist)."""
    real = os.path.realpath(path)
    for root in (os.path.join(REPO, "reports"), "/vol/v1/HLS_Prithvi", "/home/server/hpc/emapr",
                 PL.PRITHVI_ROOT):
        root = os.path.realpath(root)
        if real == root or real.startswith(root + os.sep):
            raise ValueError(f"output folder {real} is inside {root}")
    return real


# ----------------------------------------------------------------------------- data
def model_cells():
    """{model: [(location_key, 4x4)]} over the scored cells, in the evaluation's order, plus
    {location_key: selected reference path} for every de-duplicated cell."""
    cells = cc.deduped_cells()
    per = {m: [] for m in MODELS}
    selected = {}
    for cell in cells:
        key = cc.location_key(cell)
        selected[key] = cell
        loaded = {m: PL.load_for_reference(cell, m) for m in MODELS}
        if any(v is None for v in loaded.values()):
            if not all(v is None for v in loaded.values()):
                raise RuntimeError(f"{cell}: Prithvi output present for only some models")
            continue
        with rasterio.open(cell) as ds:
            rf = ds.read(1)
        _, zero, _, _ = loaded[MODELS[0]]
        covered = loaded[MODELS[0]][0] != PL.NODATA
        keep = covered & ~zero
        for m, (codes, z, _, _) in loaded.items():
            if not np.array_equal(z, zero) or not np.array_equal(codes != PL.NODATA, covered):
                raise RuntimeError(f"{cell}: footprint differs between models")
            cm, _, _ = cc.prithvi_cell_tallies(rf, codes, keep)
            if cm.sum() > 0:
                per[m].append((key, cm))
    return per, selected


def interpreter_pairs(selected, model_keys):
    """Per-pair 4x4 (reviewer A rows, B cols) over the evaluation pixels, for the pairs with Prithvi
    output and >= 1 usable pixel; and one accounting row per pair."""
    pairs = CI.find_pairs()
    used, rows = [], []
    for key, revs in sorted(pairs.items()):
        if len(revs) != 2:
            raise RuntimeError(f"pair {key}: {len(revs)} reviewers, expected 2")
        (ra, fa), (rb, fb) = revs
        sel = selected[key]
        sel_rev = PL.parse_reference(sel)[2]
        target = int(key[2])
        row = dict(grid=key[0], sample=key[1], target_year=target, reviewer_a=ra, reviewer_b=rb,
                   selected_reviewer=sel_rev, label_year=PL.label_year(target))
        loaded = PL.load_for_reference(sel, MODELS[0])
        if loaded is None:
            ly = PL.label_year(target)
            yrs = PL.available_label_years()
            row.update(prithvi_output=False, zero_input_px="", pair_px=0, in_model_cells=False,
                       status=(f"excluded: label year {ly} outside available outputs "
                               f"({yrs[0]}-{yrs[-1]})" if ly not in yrs
                               else "excluded: no Prithvi output for this cell in label year"))
            rows.append(row)
            continue
        codes, zero, _, _ = loaded
        with rasterio.open(fa) as da, rasterio.open(fb) as db, rasterio.open(sel) as ds:
            if not (da.shape == db.shape == ds.shape and da.transform == db.transform == ds.transform):
                raise RuntimeError(f"pair {key}: reviewer rasters are not on one grid")
            a, b = cc.collapse_reference(da.read(1)), cc.collapse_reference(db.read(1))
        keep = (codes != PL.NODATA) & ~zero
        v = keep & (a >= 1) & (a <= 4) & (b >= 1) & (b <= 4)
        cm = np.zeros((4, 4), np.int64)
        np.add.at(cm, (a[v] - 1, b[v] - 1), 1)
        in_model = all(key in model_keys[m] for m in MODELS)
        row.update(prithvi_output=True, zero_input_px=int(zero.sum()), pair_px=int(cm.sum()),
                   in_model_cells=in_model)
        if cm.sum() == 0:
            row["status"] = "excluded: no usable pixel after the zero-input mask"
        else:
            if not in_model:
                raise RuntimeError(f"pair {key} has usable pixels but is not a scored model cell")
            row["status"] = "used"
            used.append((key, cm))
        rows.append(row)
    return used, rows, len(pairs)


# ----------------------------------------------------------------------------- statistics
def interp_bootstrap(stack):
    """Pair bootstrap of the symmetric per-class F1; (lo, hi, defined draws) per class."""
    rng = np.random.default_rng(SEED)
    n = len(stack)
    bf1 = np.full((BOOT, 4), np.nan)
    for b in range(BOOT):
        idx = rng.integers(0, n, size=n)
        bf1[b] = ICI.per_class(stack[idx].sum(0))[0]
    out = []
    for k in range(4):
        s = bf1[:, k][np.isfinite(bf1[:, k])]
        lo, hi = (np.percentile(s, 2.5), np.percentile(s, 97.5)) if s.size else (np.nan, np.nan)
        out.append((lo, hi, int(s.size)))
    return out


def paired_gap(model_stack, interp_stack):
    """Paired bootstrap of model F1 (metrics_f1_zero) - interpreter F1, same resampled pairs."""
    rng = np.random.default_rng(SEED)
    n = len(interp_stack)
    d = np.full((BOOT, 4), np.nan)
    for b in range(BOOT):
        idx = rng.integers(0, n, size=n)
        mm = cc.metrics_f1_zero(model_stack[idx].sum(0))
        fi = ICI.per_class(interp_stack[idx].sum(0))[0]
        d[b] = [mm[f"F1[{c}]"] - fi[c - 1] for c in CLASSES]
    out = []
    for k in range(4):
        s = d[:, k][np.isfinite(d[:, k])]
        out.append((np.percentile(s, 2.5), np.percentile(s, 97.5), int(s.size)) if s.size
                   else (np.nan, np.nan, 0))
    return out


def model_rows(model, scope, stack):
    """Per-class rows for one model and scope, from the evaluation's own metric and bootstrap."""
    pt = cc.metrics_f1_zero(stack.sum(0))
    dd = {}
    bci = cc.bootstrap_cis(stack, len(stack), metrics_fn=cc.metrics_f1_zero, defined=dd)
    pooled = stack.sum(0)
    rows = []
    for c in CLASSES:
        k = c - 1
        f1 = pt[f"F1[{c}]"]
        lo, hi = bci[f"F1[{c}]"]
        rows.append(dict(
            series="model", model=model, scope=scope, cls=NAMES[c], code=c, unit="cell",
            n_units=len(stack),
            n_units_with_class=int(((stack[:, k, :].sum(1) + stack[:, :, k].sum(1)) > 0).sum()),
            tp_px=int(pooled[k, k]), ref_support_px=int(pooled[k].sum()),
            pred_support_px=int(pooled[:, k].sum()),
            union_px=int(pooled[k].sum() + pooled[:, k].sum() - pooled[k, k]),
            f1=cc._r5(f1), f1_lo=cc._r5(lo), f1_hi=cc._r5(hi), defined_draws=dd[f"F1[{c}]"],
            reliability=ICI.tier(f1) if np.isfinite(f1) else "n/a", mask=MASK, seed=SEED, boot=BOOT,
            ci_method=CI_MODEL,
            _pt=pt, _bci=bci))
    return rows


def interp_rows(stack):
    pooled = stack.sum(0)
    f1, _ = ICI.per_class(pooled)
    ci = interp_bootstrap(stack)
    rows = []
    for c in CLASSES:
        k = c - 1
        rows.append(dict(
            series="interpreter", model="", scope="pairs", cls=NAMES[c], code=c, unit="pair",
            n_units=len(stack),
            n_units_with_class=int(((stack[:, k, :].sum(1) + stack[:, :, k].sum(1)) > 0).sum()),
            tp_px=int(pooled[k, k]), ref_support_px=int(pooled[k].sum()),
            pred_support_px=int(pooled[:, k].sum()),
            union_px=int(pooled[k].sum() + pooled[:, k].sum() - pooled[k, k]),
            f1=cc._r5(f1[k]), f1_lo=cc._r5(ci[k][0]), f1_hi=cc._r5(ci[k][1]),
            defined_draws=ci[k][2], reliability=ICI.tier(f1[k]), mask=MASK, seed=SEED, boot=BOOT,
            ci_method=CI_INTERP + "; ref/pred support = reviewer A / reviewer B pixels"))
    return rows


def check_eval(all_rows, eval_csv):
    """Compare every any_year row of eval_class_metrics.csv with the rebuilt model side.
    Returns (n_rows_checked, list of mismatch strings)."""
    ev = pd.read_csv(eval_csv)
    ev = ev[ev["mask"] == MASK]
    bad, n = [], 0
    for r in all_rows:
        sel = ev[(ev.model == r["model"]) & (ev["class"] == r["cls"])]
        if len(sel) != 1:
            bad.append(f"{r['model']} {r['cls']}: {len(sel)} rows in {eval_csv}")
            continue
        e = sel.iloc[0]
        c = r["code"]
        pt, bci = r["_pt"], r["_bci"]
        mine = {}
        for lab, key in (("UA", "precision"), ("PA", "recall"), ("F1", "F1")):
            mine[lab] = cc._r5(pt[f"{key}[{c}]"])
            mine[f"{lab}_ci_lo"], mine[f"{lab}_ci_hi"] = (cc._r5(x) for x in bci[f"{key}[{c}]"])
        mine.update(ref_support_px=r["ref_support_px"], pred_support_px=r["pred_support_px"],
                    n_cells=r["n_units"])
        for col, v in mine.items():
            ev_v = e[col]
            same = ((v == "" and pd.isna(ev_v)) or
                    (v != "" and not pd.isna(ev_v) and abs(float(v) - float(ev_v)) < 5e-7))
            if not same:
                bad.append(f"{r['model']} {r['cls']} {col}: rebuilt {v!r} vs eval {ev_v!r}")
        n += 1
    if n != len(ev):
        bad.append(f"checked {n} rows but {eval_csv} has {len(ev)} {MASK} rows")
    return n, bad


# ----------------------------------------------------------------------------- figures
def _fmt_f1(v):
    return "" if v == "" else f"{float(v):.3f}"


def _yticklabel(c, n_ref_cells, n_cells, n_pairs_cls, n_pairs):
    return f"{NAMES[c]}\n(reference in {n_ref_cells} of {n_cells} cells;\n{n_pairs_cls} of {n_pairs} pairs)"


def _wrap_measure(fig, text, width_in, fontsize):
    """Greedy wrap of `text` so every line is narrower than width_in (measured with the renderer).
    Returns the wrapped string."""
    r = fig.canvas.get_renderer()
    # split on plain spaces only: str.split() would also break at the non-breaking spaces (NBSP)
    words, lines, cur = [w for w in text.split(" ") if w], [], ""
    probe = fig.text(0, 0, "", fontsize=fontsize)
    for w in words:
        cand = (cur + " " + w).strip()
        probe.set_text(cand)
        if probe.get_window_extent(r).width / fig.dpi <= width_in or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    probe.remove()
    return "\n".join(lines)


class _MarkerBox:
    """Display-space box of one scatter marker: center +- (sqrt(s)/2 + edge/2) points. A
    PathCollection's own window extent is not the marker's box, so overlap checks use this."""

    def __init__(self, ax, x, y, s, edge):
        self.ax, self.x, self.y, self.r_pt = ax, x, y, np.sqrt(s) / 2 + edge / 2

    def get_window_extent(self, renderer=None):
        from matplotlib.transforms import Bbox
        cx, cy = self.ax.transData.transform((self.x, self.y))
        r = self.r_pt * self.ax.figure.dpi / 72.0
        return Bbox([[cx - r, cy - r], [cx + r, cy + r]])


def check_overlaps(fig, texts, data_artists, label):
    """WARN lines for: any text (incl. legend) outside the canvas, any two texts intersecting, any
    text intersecting a data artist (markers, intervals, threshold lines)."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    warns = []
    boxes = [(n, t.get_window_extent(r)) for n, t in texts]
    for n, bb in boxes:
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W or bb.y1 > H:
            warns.append(f"WARN [{label}] {n} extends outside the canvas")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                warns.append(f"WARN [{label}] {boxes[i][0]} overlaps {boxes[j][0]}")
    for n, bb in boxes:
        for dn, a in data_artists:
            if bb.overlaps(a.get_window_extent(r)):
                warns.append(f"WARN [{label}] {n} overlaps {dn}")
    return warns


def draw_forest(series, interp, ylabels, title, caption, legend_items, path, offsets, row_in,
                annotate_y_off, label):
    """series: list of (name, color, {code: (f1, lo, hi)}, y offset); interp: {code: (f1, lo, hi)}.
    Layout fixed in inches; returns WARN lines from check_overlaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    W = 10.0
    L_IN, R_IN = 2.75, 0.35
    AX_W = W - L_IN - R_IN
    AX_H = row_in * len(CLASSES)
    TITLE_IN, GAP_TOP = 0.55, 0.25
    XAX_IN, LEG_IN, CAP_GAP, CAP_PAD = 0.62, 0.42, 0.18, 0.12
    CAP_FS = 9

    fig = plt.figure(figsize=(W, 4.0), dpi=DPI)
    cap_txt = _wrap_measure(fig, caption, W - 0.6, CAP_FS)
    cap = fig.text(0, 0, cap_txt, fontsize=CAP_FS, color="0.3", ha="center", va="bottom")
    cap_h = cap.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    cap.remove()

    H = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XAX_IN + AX_H + GAP_TOP + TITLE_IN
    fig.set_size_inches(W, H)
    ax_y0 = CAP_PAD + cap_h + CAP_GAP + LEG_IN + XAX_IN
    ax = fig.add_axes([L_IN / W, ax_y0 / H, AX_W / W, AX_H / H])
    tr = fig.dpi_scale_trans
    data_artists = []

    for i in range(len(CLASSES)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#eeeef3", edgecolor="none", zorder=0)
    for xv in (ICI.MOD, ICI.HIGH):
        data_artists.append((f"threshold {xv}", ax.axvline(xv, ls="--", lw=0.8, color="gray",
                                                           zorder=1)))

    no_agree = []
    for i, c in enumerate(CLASSES):
        for name, color, vals, off in series:
            if c not in vals or vals[c][0] == "":
                continue
            f1, lo, hi = (float(x) for x in vals[c])
            ln, = ax.plot([lo, hi], [i + off, i + off], color=color, lw=1.8, zorder=3, clip_on=False)
            ax.scatter([f1], [i + off], color=color, marker="o", s=44, zorder=4,
                       edgecolor="white", linewidths=0.5, clip_on=False)
            data_artists += [(f"{name} {NAMES[c]} CI", ln),
                             (f"{name} {NAMES[c]} point", _MarkerBox(ax, f1, i + off, 44, 0.5))]
        f1, lo, hi = (float(x) for x in interp[c])
        iy = i + offsets
        ln, = ax.plot([lo, hi], [iy, iy], color=INTERP_LINE, lw=3.5, zorder=5,
                      solid_capstyle="round", clip_on=False)
        ax.scatter([f1], [iy], color=INTERP_FACE, marker="D", s=110, zorder=6,
                   edgecolor="white", linewidths=1.0, clip_on=False)
        data_artists += [(f"interpreter {NAMES[c]} CI", ln),
                         (f"interpreter {NAMES[c]} point", _MarkerBox(ax, f1, iy, 110, 1.0))]
        if hi == 0.0 and f1 == 0.0:
            no_agree.append((i, c, iy))

    ax.set_xlim(0, 1)
    ax.set_ylim(len(CLASSES) - 0.5, -0.5)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", labelsize=10.5, length=0, pad=10)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("Per-Class F1 (95% CI)", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # "no interpreter agreement" label, placed right of everything drawn on that class band
    texts = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for i, c, iy in no_agree:
        x_right = 0.0
        for name, _, vals, _ in series:
            if c in vals and vals[c][0] != "":
                x_right = max(x_right, float(vals[c][2]))
        x0 = x_right + 0.03
        t = ax.text(x0, iy + annotate_y_off, NO_AGREE, fontsize=10, color=NO_AGREE_COLOR,
                    style="italic", ha="left", va="center", zorder=7)
        texts.append((f"annotation {NAMES[c]}", t))
        bb = t.get_window_extent(r)
        x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
        if x_end >= ICI.MOD - 0.01:
            print(f"WARN [{label}] '{NO_AGREE}' ends at x={x_end:.3f}, reaches the 0.50 line")

    title_t = fig.text(W / 2, H - GAP_TOP / 2 - TITLE_IN / 2, title, transform=tr, ha="center",
                       va="center", fontsize=14, fontweight="bold")
    handles = [Line2D([], [], color=line_col, markerfacecolor=face, marker=mk, ls="-", lw=lw,
                      ms=ms, label=lab, markeredgecolor="white")
               for lab, line_col, face, mk, ms, lw in legend_items]
    leg_y = CAP_PAD + cap_h + CAP_GAP + LEG_IN / 2
    leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(W / 2, leg_y),
                     bbox_transform=tr, ncol=len(handles), fontsize=10, frameon=False,
                     columnspacing=1.8, handletextpad=0.5, handlelength=2.2)
    cap = fig.text(W / 2, CAP_PAD, cap_txt, transform=tr, fontsize=CAP_FS, color="0.3",
                   ha="center", va="bottom")

    fig.canvas.draw()
    texts += [("title", title_t), ("legend", leg), ("caption", cap), ("x label", ax.xaxis.label)]
    texts += [(f"y tick '{t.get_text().splitlines()[0]}'", t) for t in ax.get_yticklabels()]
    texts += [(f"x tick '{t.get_text()}'", t) for t in ax.get_xticklabels() if t.get_text()]
    # every text (tick labels, x label, title, legend, caption, annotation) against every other
    # text and against every marker, interval and threshold line
    warns = check_overlaps(fig, texts, data_artists, label)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return warns


def figures(out_dir, mrows_all, irows, ref_cells_cls, n_cells, n_pairs):
    iv = {r["code"]: (r["f1"], r["f1_lo"], r["f1_hi"]) for r in irows}
    pairs_cls = {r["code"]: r["n_units_with_class"] for r in irows}
    ylabels = [_yticklabel(c, ref_cells_cls[c], n_cells, pairs_cls[c], n_pairs) for c in CLASSES]
    union_ins = next(r["union_px"] for r in irows if r["code"] == 4)
    common = (f"Classes: Stable, Harvest, Development, Insect/Disease (Beaver, Unknown/Fire and "
              f"agricultural change excluded); pixels with a Prithvi pixel under the reference pixel "
              f"center, zero-input mask any_year. Each model is scored against the interpreted Landsat "
              f"reference on {n_cells}{NBSP}cells (cell bootstrap); the inter-interpreter agreement (gray "
              f"diamond) uses the {n_pairs} double-interpreted cells with Prithvi output (pair "
              f"bootstrap); 95% CIs, seed {SEED}, {BOOT}{NBSP}draws. The interpreter agreement is a "
              f"benchmark for reference noise, not a strict upper bound. Insect/Disease: the two "
              f"interpreters never agreed (0 of {union_ins}{NBSP}px), so its agreement is 0 with no "
              f"spread. Dashed lines mark the Moderate (0.50) and High (0.70) thresholds.")
    interp_desc = "(gray diamond, light gray 95% CI bar)"
    warns, paths = [], {}
    for m in MODELS:
        vals = {r["code"]: (r["f1"], r["f1_lo"], r["f1_hi"]) for r in mrows_all[m]}
        p = os.path.join(out_dir, f"forest_prithvi_{SHORT[m]}.png")
        warns += draw_forest(
            series=[(SHORT[m], COLOR[m], vals, 0.18)], interp=iv, ylabels=ylabels,
            title=f"{SHORT[m]}: Per-Class F1 vs the Inter-Interpreter Agreement Ceiling (Landsat)",
            caption=f"Per-class F1 of {SHORT[m]} (colored circle) next to the inter-interpreter "
                    f"agreement for the same class {interp_desc}. " + common,
            legend_items=[(f"{SHORT[m]} vs reference ({n_cells} cells)", COLOR[m], COLOR[m], "o",
                           8, 1.8),
                          (f"interpreter agreement ({n_pairs} pairs)", INTERP_LINE, INTERP_FACE,
                           "D", 9, 3.5)],
            path=p, offsets=-0.18, row_in=0.95, annotate_y_off=0.0, label=SHORT[m])
        paths[m] = p
    offs = [-0.34, -0.17, 0.17, 0.34]
    p = os.path.join(out_dir, "forest_prithvi_all_models.png")
    warns += draw_forest(
        series=[(SHORT[m], COLOR[m], {r["code"]: (r["f1"], r["f1_lo"], r["f1_hi"])
                                      for r in mrows_all[m]}, o) for m, o in zip(MODELS, offs)],
        interp=iv, ylabels=ylabels,
        title="Prithvi Per-Class F1 vs the Inter-Interpreter Agreement Ceiling (Landsat, Four-Class)",
        caption="Per-class F1 of the four Prithvi models (colored circles, offset within each class) "
                f"next to the inter-interpreter agreement for the same class {interp_desc}. " + common,
        legend_items=[(f"interpreter ({n_pairs} pairs)", INTERP_LINE, INTERP_FACE, "D", 9, 3.5)]
        + [(SHORT[m], COLOR[m], COLOR[m], "o", 8, 1.8) for m in MODELS],
        path=p, offsets=0.0, row_in=1.35, annotate_y_off=0.0, label="all_models")
    paths["all"] = p
    return warns, paths


# ----------------------------------------------------------------------------- outputs
def write_note(out_dir, n_pairs_total, pair_rows, irows, mrows_all, mrows_pair, gap_rows, n_cells,
               n_eval, warns):
    used = [r for r in pair_rows if r["status"] == "used"]
    with_out = [r for r in pair_rows if r["prithvi_output"]]
    L = ["# prithvi_vs_interpreter", "",
         "Per-class F1 of the four Prithvi models (Landsat, four-class scheme) next to the "
         "inter-interpreter agreement for the same class, computed from the Landsat "
         "double-interpreted pairs. Generated by `scripts/prithvi_class_ci_vs_interpreter.py`; "
         "Landsat counterpart of `scripts/model_class_ci_5class.py`.", "",
         "## Method", "",
         "- Classes: Stable, Harvest, Development, Insect/Disease; Beaver, Unknown/Fire and Prithvi "
         "agricultural change excluded (as in `collapsed_5class_confusion.py --prithvi`).",
         "- Pixels: reference pixels with a Prithvi pixel under their center, minus the zero-input "
         "mask any_year (main result of the evaluation).",
         f"- Model, all cells: the {n_cells} scored cells rebuilt with the evaluation's own functions "
         f"(metrics_f1_zero, cell bootstrap seed {SEED}, {BOOT} draws). All {n_eval} any_year rows "
         "of eval_class_metrics.csv are reproduced exactly (UA, PA, F1 and CIs to 5 decimals, "
         "supports and n_cells).",
         "- Interpreters: reviewer A vs reviewer B on the same pixels; a pixel counts when both "
         "reviewers assigned one of the four classes (Beaver or Unknown/Fire from either drops it). "
         f"F1 = 2TP/(row+col), symmetric; pair bootstrap seed {SEED}, {BOOT} draws.",
         "- Like-for-like: each model also scored on only the paired cells (reference = the "
         "seeded-selected reviewer, as in the evaluation), and the paired difference model - "
         "interpreter with a paired bootstrap (same resampled pairs for both).", "",
         "## Pairs", "",
         f"{n_pairs_total} double-interpreted pairs; {len(with_out)} have Prithvi output; "
         f"{len(used)} have at least one usable pixel after the zero-input mask and are used. "
         "Details in `pairs_used.csv`:", ""]
    for r in pair_rows:
        if r["status"] != "used":
            L.append(f"- grid {r['grid']} sample {r['sample']} target {r['target_year']}: {r['status']}")
    L += ["", "## Interpreter agreement (ceiling)", "",
          "| Class | F1 (95% CI) | Pairs with class | Union px | TP px | Defined draws | Reliability |",
          "|---|---|---:|---:|---:|---:|---|"]
    for r in irows:
        L.append(f"| {r['cls']} | {_ci(r)} | {r['n_units_with_class']} of {r['n_units']} | "
                 f"{r['union_px']:,} | {r['tp_px']:,} | {r['defined_draws']} | {r['reliability']} |")
    L += ["", "## Model F1 vs interpreter agreement", "",
          "Gap = model F1 - interpreter F1. The all-cells gap has no CI (different cells and "
          "resampling units); the paired gap has a paired bootstrap CI.", "",
          "| Model | Class | Model, all cells | Model, paired cells | Interpreter | Gap (all) | "
          "Gap (paired, 95% CI) |", "|---|---|---|---|---|---:|---|"]
    for g in gap_rows:
        L.append(f"| {g['model']} | {g['cls']} | {_ci3(g, 'model_f1_all')} | "
                 f"{_ci3(g, 'model_f1_paired')} | {_ci3(g, 'interp_f1')} | {_f(g['gap_all'])} | "
                 f"{_f(g['gap_paired'])} ({_f(g['gap_paired_lo'])} to {_f(g['gap_paired_hi'])}) |")
    L += ["", "## Caveats", "",
          "- Insect/Disease: the two interpreters never agreed on a single pixel (only one reviewer "
          "of each pair assigned it), so the agreement is 0 with no spread; labeled \"no "
          "interpreter agreement\" in the figures. A model F1 above it does not mean the model beats "
          "the interpreters.",
          "- Development rests on very few pixels and pairs; its CI is wide and the point value says "
          "little. Harvest is the only change class with usable support on both sides.",
          f"- The models are scored on {n_cells} cells and the interpreters on {len(used)} pairs: "
          "different populations and resampling units, so CI widths are not comparable. The "
          "paired-cells rows compare like with like.",
          "- The interpreter agreement is not a hard bound: the model is scored against one "
          "reviewer, who is also half of the interpreter comparison on the paired cells.",
          "- Grid offset: the Prithvi grid is 9.10 m east of the reference grid, which penalizes "
          "the model at patch edges; the two reviewers share a grid, so the gap is biased slightly "
          "against the model.",
          "- Pixel sets differ slightly between the model and interpreter rows: the model drops a "
          "pixel when the selected reviewer has Beaver/Unknown or the prediction is agricultural "
          "change; the interpreters drop it when either reviewer has Beaver/Unknown.", "",
          "## Figure checks", ""]
    L += [f"- {w}" for w in warns] if warns else ["- overlap check: no WARN."]
    L += ["", "## Outputs", "",
          "- `prithvi_vs_interpreter_class_f1.csv`: one row per series, model, scope and class.",
          "- `prithvi_vs_interpreter_gap.csv`: per model and class, interpreter, model (all cells and "
          "paired cells) and the gaps.",
          "- `pairs_used.csv`: every double-interpreted pair and why it is used or not.",
          "- `forest_prithvi_<model>.png`: per model, model F1 next to the interpreter agreement.",
          "- `forest_prithvi_all_models.png`: all four models over the interpreter agreement.", ""]
    with open(os.path.join(out_dir, "note.md"), "w") as fh:
        fh.write("\n".join(L))


def _f(v, d=3):
    return "-" if v == "" or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.{d}f}"


def _ci(r):
    return f"{_f(r['f1'])} ({_f(r['f1_lo'])} to {_f(r['f1_hi'])})"


def _ci3(g, k):
    return f"{_f(g[k])} ({_f(g[k + '_lo'])} to {_f(g[k + '_hi'])})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", required=True,
                    help="required: eval_class_metrics.csv written by collapsed_5class_confusion.py "
                         "--prithvi; the script checks its own model numbers against it")
    args = ap.parse_args()
    if SC.OUT_DIR is None:
        sys.exit("refusing to run: set LCI_OUT_DIR so outputs cannot overwrite the Sentinel-2 reports/")
    if SC.SENSOR != "Landsat":
        sys.exit(f"refusing to run: LCI_SENSOR is {SC.SENSOR!r}, expected 'Landsat'")
    if PL.PRITHVI_ZERO_INPUT != MASK or PL.PRITHVI_YEAR_RULE != "target_minus_1":
        sys.exit(f"refusing to run: zero-input rule {PL.PRITHVI_ZERO_INPUT!r} / year rule "
                 f"{PL.PRITHVI_YEAR_RULE!r}, expected {MASK!r} / 'target_minus_1'")
    try:
        out_dir = check_out_path(SC.out_path(OUT_SUB))
    except ValueError as e:
        sys.exit(f"refusing to run: {e}")
    if not os.path.isfile(args.eval_csv):
        sys.exit(f"refusing to run: --eval-csv {args.eval_csv} not found")

    per, selected = model_cells()
    model_keys = {m: {k for k, _ in per[m]} for m in MODELS}
    used, pair_rows, n_pairs_total = interpreter_pairs(selected, model_keys)
    n_with = sum(r["prithvi_output"] for r in pair_rows)
    print(f"double-interpreted pairs: {n_pairs_total}; with Prithvi output: {n_with}; "
          f"used (>= 1 usable pixel): {len(used)}")
    pair_keys = [k for k, _ in used]
    istack = np.stack([cm for _, cm in used])

    # model side: all cells (must reproduce the evaluation) and paired cells
    mrows_all, mrows_pair, mstack_pair, eval_rows = {}, {}, {}, []
    for m in MODELS:
        stack = np.stack([cm for _, cm in per[m]])
        mrows_all[m] = model_rows(m, "all_cells", stack)
        eval_rows += mrows_all[m]
        by_key = dict(per[m])
        mstack_pair[m] = np.stack([by_key[k] for k in pair_keys])     # same order as istack
        mrows_pair[m] = model_rows(m, "paired_cells", mstack_pair[m])
    n_cells = len(per[MODELS[0]])
    if any(len(per[m]) != n_cells for m in MODELS):
        sys.exit("STOP: models have different scored cell counts")
    n_eval, bad = check_eval(eval_rows, args.eval_csv)
    if bad:
        print("STOP: rebuilt model side does not reproduce eval_class_metrics.csv:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print(f"eval match: all {n_eval} {MASK} rows of {args.eval_csv} reproduced exactly")

    irows = interp_rows(istack)
    ival = {r["code"]: r for r in irows}
    gap_rows = []
    for m in MODELS:
        gci = paired_gap(mstack_pair[m], istack)
        for ra, rp in zip(mrows_all[m], mrows_pair[m]):
            c = ra["code"]
            ir = ival[c]
            lo, hi, nd = gci[c - 1]
            gap_rows.append(dict(
                model=m, cls=NAMES[c], code=c,
                interp_f1=ir["f1"], interp_f1_lo=ir["f1_lo"], interp_f1_hi=ir["f1_hi"],
                n_pairs=ir["n_units"], n_pairs_with_class=ir["n_units_with_class"],
                model_f1_all=ra["f1"], model_f1_all_lo=ra["f1_lo"], model_f1_all_hi=ra["f1_hi"],
                n_cells_all=ra["n_units"],
                model_f1_paired=rp["f1"], model_f1_paired_lo=rp["f1_lo"],
                model_f1_paired_hi=rp["f1_hi"], n_cells_paired=rp["n_units"],
                gap_all=cc._r5(float(ra["f1"]) - float(ir["f1"])) if ra["f1"] != "" else "",
                gap_paired=(cc._r5(float(rp["f1"]) - float(ir["f1"])) if rp["f1"] != "" else ""),
                gap_paired_lo=cc._r5(lo), gap_paired_hi=cc._r5(hi), gap_paired_defined_draws=nd,
                model_px_paired=int(mstack_pair[m].sum()), interp_px=int(istack.sum()),
                ci_method=f"paired pair bootstrap (seed {SEED}, {BOOT} draws), percentile 95% CI"))

    # pairs table: per-model usable pixels on the paired cells
    for r in pair_rows:
        key = (r["grid"], r["sample"], str(r["target_year"]))
        for m in MODELS:
            cm = dict(per[m]).get(key)
            r[f"model_px_{SHORT[m]}"] = int(cm.sum()) if cm is not None else 0

    # reference-cell counts per class (model independent) for the y labels
    st0 = np.stack([cm for _, cm in per[MODELS[0]]])
    ref_cells_cls = {c: int((st0[:, c - 1, :].sum(1) > 0).sum()) for c in CLASSES}

    # everything computed and checked: write
    os.makedirs(out_dir, exist_ok=True)
    warns, paths = figures(out_dir, mrows_all, irows, ref_cells_cls, n_cells, len(used))
    for w in warns:
        print(w)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in irows]
    for m in MODELS:
        rows += [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in mrows_all[m] + mrows_pair[m]]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "prithvi_vs_interpreter_class_f1.csv"),
                              index=False)
    pd.DataFrame(gap_rows).to_csv(os.path.join(out_dir, "prithvi_vs_interpreter_gap.csv"),
                                  index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(out_dir, "pairs_used.csv"), index=False)
    write_note(out_dir, n_pairs_total, pair_rows, irows, mrows_all, mrows_pair, gap_rows, n_cells,
               n_eval, warns)

    # console tables
    print(f"\n===== interpreter agreement ({len(used)} pairs, pair bootstrap) =====")
    for r in irows:
        print(f"  {r['cls']:<15} {_ci(r):<26} pairs with class {r['n_units_with_class']:>2}  "
              f"union px {r['union_px']:>7,}  TP {r['tp_px']:>7,}  draws {r['defined_draws']}")
    print(f"\n===== model F1: all {n_cells} cells | paired {len(used)} cells | gap vs interpreter =====")
    print(f"  {'model':<13}{'class':<15}{'all cells':<26}{'paired cells':<26}{'gap all':>8}  "
          f"gap paired (95% CI)")
    for g in gap_rows:
        print(f"  {SHORT[g['model']]:<13}{g['cls']:<15}{_ci3(g, 'model_f1_all'):<26}"
              f"{_ci3(g, 'model_f1_paired'):<26}{_f(g['gap_all']):>8}  {_f(g['gap_paired'])} "
              f"({_f(g['gap_paired_lo'])} to {_f(g['gap_paired_hi'])})")
    print(f"\n{len(warns)} WARN line(s) from the figure overlap check")
    print(f"outputs -> {out_dir}/")
    for k, p in paths.items():
        print(f"  {p}")


if __name__ == "__main__":
    main()
