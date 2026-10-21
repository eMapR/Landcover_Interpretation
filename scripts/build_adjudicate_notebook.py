#!/usr/bin/env python3
"""Assemble notebooks/adjudicate_truth.ipynb (an ipywidgets adjudication GUI).

Writes the notebook as nbformat v4 JSON directly, so no nbformat dependency is needed to build it.
Run: python scripts/build_adjudicate_notebook.py
"""

import json
import os

import sensor_config as SC  # RF_DIR, out_path (LCI_RF_DIR / LCI_OUT_DIR)

MD_INTRO = r"""# Interpreter truth adjudication

Select one authoritative CKIT-RF interpretation per location, then export the truth set.

- A **location** is one `grid_id`. Single-interpreted locations pass through automatically; only
  the multi-interpreted ones are stepped through here.
- Each choice is saved to `exports/adjudication_progress.csv` immediately, so you can stop and
  resume. The final union export goes to `exports/truth_selections.csv`.

**Two things this data does not match the original assumption on, reported by the scan cell:**
1. `sample_id` is per **location**, not per reviewer: the two (or three) interpretations of a
   location share one `sample_id` and differ only by reviewer. So the panels are labeled by
   reviewer, and the shared `sample_id` is shown once for NAIP cross-reference.
2. One location (`07630`) has three interpretations, not two. The tool shows all three (A, B, C)
   with a Choose button each, so it is adjudicated without guessing.

Each location also shows an **agreement pane** to the right of the interpretations: green where the
reviewers agree, red where they disagree (for the triple, green means all three agree).

Run the cells top to bottom. Re-running is safe; progress reloads from disk."""

CFG = r'''# config and imports
import os, re, glob
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import ipywidgets as widgets
from IPython.display import display, clear_output

# make matplotlib render inline inside the widget output (no-op outside Jupyter)
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# paths resolve whether the notebook runs from notebooks/ or the repo root
REPO = ".." if os.path.isdir(os.path.join("..", "data", "raw", "rf_class_maps")) else "."
RF_DIR = os.path.join(REPO, "data", "raw", "rf_class_maps")
RF_LEGEND = os.path.join(REPO, "data", "reference", "label_lookup.csv")
EXPORT_DIR = os.path.join(REPO, "exports")
PROGRESS_CSV = os.path.join(EXPORT_DIR, "adjudication_progress.csv")   # incremental working state
TRUTH_CSV = os.path.join(EXPORT_DIR, "truth_selections.csv")           # final union export
os.makedirs(EXPORT_DIR, exist_ok=True)

# everything needed is in the filename
FNAME_RE = re.compile(
    r"reviewer_([A-Za-z]+)_grid_(\d+)_sample_(\d+)_sensor_([A-Za-z0-9-]+)"
    r"_target_(\d+)_opt_(\d{4})_(\d{4})")'''

SCAN = r'''# scan the rasters, group by location, report, and flag anomalies
def scan_rasters():
    locations = defaultdict(list)
    excluded, unparsed = [], []
    for f in sorted(glob.glob(os.path.join(RF_DIR, "**", "rf_class*.tif"), recursive=True)):
        base = os.path.basename(f)
        m = FNAME_RE.search(base)
        if not m:
            unparsed.append(base)
            continue
        rev, gid, samp, sensor, tgt, y1, y2 = m.groups()
        # exclude the out-of-scope 30 m landsat 2003_2005 record if present
        if gid == "17800" or sensor.lower() == "landsat":
            excluded.append(base)
            continue
        gid5 = str(int(gid)).zfill(5)                          # zero-pad the location id to 5 digits
        locations[gid5].append(dict(reviewer=rev.lower(), sample=samp, bracket=f"{y1}_{y2}",
                                    target=tgt, path=f, file=base))
    # order interpretations within a location by reviewer for a stable A, B, C assignment
    locations = {g: sorted(v, key=lambda d: d["reviewer"]) for g, v in sorted(locations.items())}
    return locations, excluded, unparsed


locations, excluded, unparsed = scan_rasters()
single_gids = [g for g, v in locations.items() if len(v) == 1]
multi_gids = [g for g, v in locations.items() if len(v) >= 2]

n_raster = sum(len(v) for v in locations.values())
print("=" * 64)
print("scan summary")
print("=" * 64)
print(f"rasters parsed: {n_raster}   unique locations: {len(locations)}")
print(f"single-interpreted: {len(single_gids)}   multi-interpreted: {len(multi_gids)}")
if excluded:
    print(f"excluded (out-of-scope anomaly): {len(excluded)}")
    for e in excluded:
        print(f"   {e}")
if unparsed:
    print(f"STOP: {len(unparsed)} filename(s) did not parse:")
    for u in unparsed:
        print(f"   {u}")

# flags and sanity checks
over_two = {g: v for g, v in locations.items() if len(v) > 2}
dup_reviewer = {g: v for g, v in locations.items()
                if len({x["reviewer"] for x in v}) < len(v)}
shared_sample = {g: v for g, v in locations.items()
                 if len(v) >= 2 and len({x["sample"] for x in v}) < len(v)}

print("\nflags")
if over_two:
    for g, v in over_two.items():
        print(f"  location {g} has {len(v)} interpretations (not two): "
              f"{[(x['reviewer'], x['sample']) for x in v]} -> shown with a Choose button each")
else:
    print("  no location has more than two interpretations")
if dup_reviewer:
    for g, v in dup_reviewer.items():
        print(f"  STOP: location {g} has the same reviewer twice: {[x['reviewer'] for x in v]}")
else:
    print("  no location has a duplicated reviewer")
# in this data sample_id is shared across the reviewers of a location, so it is per-location, not
# per-reviewer as originally assumed. this is systematic (every multi-interpreted location), so it
# is reported as a note, not treated as a per-location error
print(f"\nnote: {len(shared_sample)} of {len(multi_gids)} multi-interpreted locations share one "
      f"sample_id across their reviewers.")
print("so sample_id is per-location here, not per-reviewer; the interpretations differ by reviewer.")
print("the shared sample_id is shown once per location for NAIP cross-reference.")'''

PALETTE = r'''# palette: fixed class-to-color map so both panels use identical colors. the rasters carry the
# RF interpreter codes, so the palette is loaded from the RF legend. to swap it, edit PALETTE_HEX.
_leg = pd.read_csv(RF_LEGEND)
PALETTE_HEX = {int(r.code): r.color for r in _leg.itertuples()}       # {class_value: color}
CLASS_NAME = {int(r.code): r.display_name for r in _leg.itertuples()}
_PALETTE_RGB = {c: mcolors.to_rgb(h) for c, h in PALETTE_HEX.items()}


def load_class(path):
    # read the single-band classified raster (RF interpreter codes)
    with rasterio.open(path) as ds:
        return ds.read(1)


def rgb_from_class(a):
    # colorize a class array with the fixed palette (white for unmapped values)
    rgb = np.ones(a.shape + (3,), dtype=float)
    for code, col in _PALETTE_RGB.items():
        rgb[a == code] = col
    return rgb


def load_rgb(path):
    return rgb_from_class(load_class(path))'''

STATE = r'''# progress persistence: load any existing choices and notes so work resumes
PROGRESS_COLS = ["grid_id", "chosen_reviewer", "chosen_sample", "chosen_bracket", "note", "decided"]


def load_progress():
    prog = {}
    if os.path.exists(PROGRESS_CSV):
        df = pd.read_csv(PROGRESS_CSV, dtype=str, keep_default_na=False)
        for r in df.itertuples():
            prog[r.grid_id] = dict(chosen_reviewer=r.chosen_reviewer, chosen_sample=r.chosen_sample,
                                   chosen_bracket=r.chosen_bracket, note=r.note,
                                   decided=str(r.decided).lower() == "true")
    return prog


def save_progress():
    rows = [dict(grid_id=g, chosen_reviewer=e.get("chosen_reviewer", ""),
                 chosen_sample=e.get("chosen_sample", ""), chosen_bracket=e.get("chosen_bracket", ""),
                 note=e.get("note", ""), decided=bool(e.get("decided", False)))
            for g, e in sorted(progress.items())]
    pd.DataFrame(rows, columns=PROGRESS_COLS).to_csv(PROGRESS_CSV, index=False)


progress = load_progress()
_n_decided = sum(1 for g in multi_gids if progress.get(g, {}).get("decided"))
print(f"loaded progress: {_n_decided} of {len(multi_gids)} multi-interpreted locations already decided")'''

GUI = r'''# adjudication GUI
state = {"idx": 0}

grid_label = widgets.HTML()
progress_label = widgets.HTML()
choice_label = widgets.HTML()
status_label = widgets.HTML()
img_out = widgets.Output()
note_box = widgets.Textarea(placeholder="reason for this decision (double-interpreted only)...",
                            layout=widgets.Layout(width="70%", height="90px"))
prev_btn = widgets.Button(description="◀ Previous")
next_btn = widgets.Button(description="Next ▶")
choose_box = widgets.HBox([])


def _n_decided():
    return sum(1 for g in multi_gids if progress.get(g, {}).get("decided"))


def render(i):
    state["idx"] = max(0, min(i, len(multi_gids) - 1))
    gid = multi_gids[state["idx"]]
    interps = locations[gid]
    entry = progress.get(gid, {})

    samp = interps[0]["sample"]
    brk = interps[0]["bracket"]
    grid_label.value = (f"<h2 style='margin:2px'>grid_id <span style='color:#1f77b4'>{gid}</span>"
                        f" &nbsp;&nbsp;<span style='font-size:14px;color:#555'>sample_id {samp} "
                        f"(shared), bracket {brk}</span></h2>")
    progress_label.value = (f"location {state['idx'] + 1} of {len(multi_gids)}, "
                            f"{_n_decided()} decided")
    if entry.get("decided"):
        choice_label.value = f"current choice: <b style='color:#2ca02c'>{entry['chosen_reviewer']}</b>"
    else:
        choice_label.value = "current choice: <i>undecided</i>"

    with img_out:
        clear_output(wait=True)
        arrs = [load_class(it["path"]) for it in interps]
        npanel = len(interps) + 1                          # interpretation panels plus agreement
        fig, axes = plt.subplots(1, npanel, figsize=(5.3 * npanel, 5.6))
        for ax, it, arr, letter in zip(axes[:-1], interps, arrs, "ABC"):
            ax.imshow(rgb_from_class(arr), interpolation="nearest")
            chosen = entry.get("decided") and entry.get("chosen_reviewer") == it["reviewer"]
            ax.set_title(f"{letter}: {it['reviewer']}  (sample {it['sample']})"
                         + ("   [CHOSEN]" if chosen else ""),
                         fontsize=12, color="#2ca02c" if chosen else "black")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2ca02c" if chosen else "#bbbbbb")
                s.set_linewidth(3 if chosen else 1)
        # agreement pane: green where all interpretations match, red where they differ
        stack = np.stack(arrs)
        agree = (stack == stack[0]).all(axis=0)
        agree_rgb = np.empty(agree.shape + (3,), dtype=float)
        agree_rgb[agree] = mcolors.to_rgb("#2ca02c")       # green: all reviewers agree
        agree_rgb[~agree] = mcolors.to_rgb("#d62728")      # red: at least one differs
        ax = axes[-1]
        ax.imshow(agree_rgb, interpolation="nearest")
        lab = "agreement" if len(interps) == 2 else "all-reviewer agreement"
        ax.set_title(f"{lab}: green agree, red disagree  ({100 * agree.mean():.0f}% agree)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb"); s.set_linewidth(1)
        # legend below every panel: the classes present in this location, plus the agree colors
        present = sorted(set().union(*[set(np.unique(a).tolist()) for a in arrs]) & set(CLASS_NAME))
        handles = [Patch(facecolor=PALETTE_HEX[c], edgecolor="0.4", label=CLASS_NAME[c])
                   for c in present]
        handles += [Patch(facecolor="#2ca02c", edgecolor="0.4", label="reviewers agree"),
                    Patch(facecolor="#d62728", edgecolor="0.4", label="reviewers disagree")]
        fig.tight_layout(rect=[0, 0.11, 1, 1])
        fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), fontsize=9,
                   frameon=False)
        plt.show()
        plt.close(fig)

    # load the saved note into the box BEFORE any edit, so navigating never blanks a saved note
    note_box.value = entry.get("note", "")

    # one Choose button per interpretation (A, B, and C for the triple)
    btns = []
    for it, letter in zip(interps, "ABC"):
        chosen = entry.get("decided") and entry.get("chosen_reviewer") == it["reviewer"]
        b = widgets.Button(description=f"Choose {letter}: {it['reviewer']}",
                           button_style="success" if chosen else "")
        b.on_click(_make_choose(gid, it))
        btns.append(b)
    choose_box.children = tuple(btns)


def _save_current_note():
    # persist the note for the current location, even without an A/B/C click
    gid = multi_gids[state["idx"]]
    entry = progress.setdefault(gid, {})
    entry["note"] = note_box.value
    entry.setdefault("decided", False)
    save_progress()


def _make_choose(gid, it):
    def cb(_):
        entry = progress.setdefault(gid, {})
        entry["note"] = note_box.value                     # save the note together with the choice
        entry.update(chosen_reviewer=it["reviewer"], chosen_sample=it["sample"],
                     chosen_bracket=it["bracket"], decided=True)
        save_progress()
        status_label.value = f"<span style='color:#2ca02c'>saved: {gid} -> {it['reviewer']}</span>"
        render(_next_undecided(state["idx"]))
    return cb


def _next_undecided(from_idx):
    n = len(multi_gids)
    for step in range(1, n + 1):
        j = (from_idx + step) % n
        if not progress.get(multi_gids[j], {}).get("decided"):
            return j
    return from_idx                                        # all decided: stay put


def _on_prev(_):
    _save_current_note()
    render(state["idx"] - 1)


def _on_next(_):
    _save_current_note()
    render(state["idx"] + 1)


prev_btn.on_click(_on_prev)
next_btn.on_click(_on_next)

ui = widgets.VBox([
    grid_label,
    widgets.HBox([progress_label, widgets.HTML("&nbsp;&nbsp;|&nbsp;&nbsp;"), choice_label]),
    img_out,
    widgets.HTML("<b>Choose the authoritative interpretation:</b>"),
    choose_box,
    widgets.HTML("<b>Decision note</b> (saved on Choose and on Previous/Next):"),
    note_box,
    widgets.HBox([prev_btn, next_btn]),
    status_label,
])
render(0)
display(ui)'''

EXPORT = r'''# export the truth set: union of the automatic singles and the adjudicated multi-locations
def export_truth():
    undecided = [g for g in multi_gids if not progress.get(g, {}).get("decided")]
    if undecided:
        print(f"WARNING: {len(undecided)} multi-interpreted location(s) still undecided; "
              f"not exporting a partial truth set.")
        print("undecided grid_ids:", ", ".join(undecided))
        return
    rows = []
    for gid in sorted(locations):
        interps = locations[gid]
        if len(interps) == 1:                              # single: sole interpretation, no click
            it = interps[0]
            rows.append(dict(grid_id=gid, sample_id=it["sample"], reviewer=it["reviewer"],
                             bracket=it["bracket"], source="single", note=""))
        else:                                              # multi: the adjudicated choice
            e = progress[gid]
            rows.append(dict(grid_id=gid, sample_id=e["chosen_sample"], reviewer=e["chosen_reviewer"],
                             bracket=e["chosen_bracket"], source="adjudicated", note=e.get("note", "")))
    out = pd.DataFrame(rows, columns=["grid_id", "sample_id", "reviewer", "bracket", "source", "note"])
    out.to_csv(TRUTH_CSV, index=False)
    print(f"wrote {TRUTH_CSV}")
    print(f"export rows: {len(out)}   unique locations: {len(locations)}   "
          f"match: {len(out) == len(locations)}")
    print("by source:", out.source.value_counts().to_dict())
    if len(out) != len(locations):
        print("STOP: export row count does not equal the unique-location count.")


export_btn = widgets.Button(description="Export truth_selections.csv", button_style="primary",
                            layout=widgets.Layout(width="260px"))
export_out = widgets.Output()


def _on_export(_):
    with export_out:
        clear_output()
        export_truth()


export_btn.on_click(_on_export)
display(widgets.VBox([export_btn, export_out]))'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def config_cell():
    """CFG with LCI_RF_DIR / LCI_OUT_DIR written in as literal paths when set; unchanged otherwise."""
    cfg = CFG
    subs = []
    if os.environ.get("LCI_RF_DIR"):
        subs.append(('RF_DIR = os.path.join(REPO, "data", "raw", "rf_class_maps")', f"RF_DIR = {SC.RF_DIR!r}"))
    if SC.OUT_DIR:
        subs.append(('EXPORT_DIR = os.path.join(REPO, "exports")', f"EXPORT_DIR = {SC.out_path('exports')!r}"))
    for old, new in subs:
        assert cfg.count(old) == 1, old
        cfg = cfg.replace(old, new)
    return cfg


def main():
    nb = {
        "cells": [md(MD_INTRO), code(config_cell()), code(SCAN), code(PALETTE), code(STATE), code(GUI),
                  code(EXPORT)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = SC.out_path(os.path.join("notebooks", "adjudicate_truth.ipynb"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
