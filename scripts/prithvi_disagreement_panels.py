#!/usr/bin/env python3
"""Map panels of the interpreted Landsat reference next to the four Prithvi model maps (Landsat only).

One PNG per cell: the interpreted reference, the four Prithvi models (fcn_unfrozen, unet3d,
fcn_frozen, fcn_random) and an agree/disagree map for fcn_unfrozen (the best model), in the 4
evaluation classes (Stable, Harvest, Development, Insect/Disease). Pixels the evaluation excludes are
shown, labelled as excluded: Beaver and Unknown/Fire reference pixels, and Prithvi agricultural-change
predictions. Model pixels whose chip input is all-zero (the chips mask water, cloud and shadow to 0;
zero-input mask) are shown as "Masked Input (Water, Cloud or Shadow)" in black, never as a class; the
reference panel is shown unmasked. In the model and agree/disagree panels, pixels with no Prithvi pixel
under the reference pixel centre are white with grey hatching ("No Prithvi Pixel"). Each figure's
legend lists only the classes and special colors that occur in its panels.

Cells and pixels are the same as in collapsed_5class_confusion.py --prithvi: reproducible seeded
selection (seed 42) of one interpretation per double-interpreted location (deduped_cells), Prithvi map
from prithvi_loader.load_for_reference (label year = target year - 1, argmax, centre lookup). Valid
pixels = Prithvi pixel present, not zero-input (unless --mask none), reference in 1..4 and prediction
in 1..4, i.e. the pixels of the 4x4 matrix. Candidate cells are those with at least one valid pixel
for fcn_unfrozen (312 masked / 322 unmasked).

Selection modes (--mode):
  disagreement  top N cells by fcn_unfrozen disagreement fraction over the valid pixels
  change        top N cells by interpreted change pixels (Harvest + Development + Insect/Disease
                reference pixels that are covered and, when masked, not zero-input)
  both          both of the above (default)
  --cells ID[:YEAR],...  plot the given cells instead (in the given order, folder cells/)
Ties are broken by cell id, then target year.

Outputs -> $LCI_OUT_DIR/reports/prithvi_disagreement_panels/<disagreement|change|cells>/
  - rank<NN>_cell<id>_t<target>.png   one panel figure per cell (300 dpi); with --cells the files
                                     are cell<id>_t<target>.png (no ranking)
  - summary.csv                      rank (order with --cells), cell, target year, label year,
                                     reviewer, disagreement fraction, change px, valid px, zero-input
                                     px, reference file. zero_input_px counts the pixels that are
                                     zero-input under the loader's rule (PRITHVI_ZERO_INPUT) and is
                                     reported even with --mask none. If no cell is selected, the CSV
                                     holds only the header.
Refuses to run without LCI_OUT_DIR, with LCI_SENSOR other than Landsat, with --top-n < 1, or when the
output folder resolves inside <repo>/reports, /vol/v1/HLS_Prithvi, /home/server/hpc/emapr or the
Prithvi root (checked before any folder is created). Every figure is checked for text/legend overlap
with maps, other text and the figure edge; failures print WARN.

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs LCI_OUT_DIR=reports_landsat python scripts/prithvi_disagreement_panels.py --mode both

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import rasterio


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cc", "collapsed_5class_confusion.py")          # dedup, collapse, NAMES5, SC
C = _load("C", "compare_interpreted_vs_model.py")          # canonical class legend colors
import prithvi_loader as PL                                 # noqa: E402

SC = cc.SC
OUT_SUB = "reports/prithvi_disagreement_panels"
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CSV_COLS = ["cell_id", "target_year", "label_year", "reviewer", "disagreement_fraction", "change_px",
            "valid_px", "zero_input_px", "reference"]
BEST = "fcn_unfrozen_encoder_q1_w2"
PANEL_MODELS = ["fcn_unfrozen_encoder_q1_w2", "unet3d_q1_w2", "fcn_frozen_encoder_q1_w2",
                "fcn_random_encoder_q1_w2"]
SHORT = {m: m.replace("_encoder_q1_w2", "").replace("_q1_w2", "") for m in PANEL_MODELS}
SCALE_M = 1000.0
DPI = 300

# display codes (one colormap for every class panel)
D_NODATA, D_STABLE, D_HARV, D_DEV, D_INS, D_BEAVER, D_UNK, D_AGRI, D_NOINPUT = range(9)
_C10 = C.load_mappings()[2]
DISPLAY = [  # (code, color, legend label, legend group)
    (D_NODATA, "#ffffff", "No Prithvi Pixel", "other"),
    (D_STABLE, "#cccccc", "Stable", "class"),
    (D_HARV, _C10[1], "Harvest", "class"),
    (D_DEV, _C10[2], "Development", "class"),
    (D_INS, _C10[10], "Insect/Disease", "class"),
    (D_BEAVER, _C10[9], "Beaver (Reference, Excluded)", "other"),
    (D_UNK, "#e6d5f0", "Unknown/Fire (Reference, Excluded)", "other"),
    (D_AGRI, _C10[6], "Agricultural Change (Prithvi, Excluded)", "other"),
    # chips mask water, cloud and shadow to 0 in all bands (check_water/: zero in all 4 years is
    # almost all water, zero in some years mostly cloud/shadow over land)
    (D_NOINPUT, "#000000", "Masked Input (Water, Cloud or Shadow)", "other"),
]
# agree/disagree codes
A_NODATA, A_AGREE, A_DISAGREE, A_EXCL, A_NOINPUT = range(5)
AD = [(A_NODATA, "#ffffff", "No Prithvi Pixel"), (A_AGREE, "#0072B2", "Agree"),
      (A_DISAGREE, "#D55E00", "Disagree"), (A_EXCL, "#009E73", "Excluded Pixel"),
      (A_NOINPUT, "#000000", "Masked Input")]
# "No Prithvi Pixel" is white with light grey hatching (maps and legend), so it reads as no data
NODATA_HATCH, NODATA_HATCH_COLOR = "////", "0.6"

# layout in inches
MAP_IN, GAP_IN, SIDE_IN = 3.0, 0.45, 0.35
TOP_IN, TTL_IN, SB_IN, BOT_IN = 0.85, 0.62, 0.55, 0.45
FIG_W = 2 * SIDE_IN + 4 * MAP_IN + 3 * GAP_IN
FIG_H = TOP_IN + 2 * (TTL_IN + MAP_IN) + SB_IN + BOT_IN


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


def ref_display(rf):
    """Reference raw codes -> display codes (unmasked)."""
    ref = cc.collapse_reference(rf)                         # 0 Unknown/Fire, 1..4, 5 Beaver
    return np.array([D_UNK, D_STABLE, D_HARV, D_DEV, D_INS, D_BEAVER], np.uint8)[ref]


def model_display(codes, zero, apply_zero):
    lut = np.full(PL.AGRICULTURE + 1, D_NODATA, np.uint8)
    lut[[1, 2, 3, 4, PL.AGRICULTURE]] = [D_STABLE, D_HARV, D_DEV, D_INS, D_AGRI]
    d = lut[codes]
    if apply_zero:
        d[zero & (codes != PL.NODATA)] = D_NOINPUT
    return d


def cell_stats(rf, codes, zero, apply_zero):
    ref = cc.collapse_reference(rf)
    covered = codes != PL.NODATA
    keep = covered & ~zero if apply_zero else covered
    valid = keep & (ref >= 1) & (ref <= 4) & (codes >= 1) & (codes <= 4)
    nv = int(valid.sum())
    return dict(valid_px=nv,
                disagreement_fraction=float((ref[valid] != codes[valid]).mean()) if nv else np.nan,
                change_px=int((keep & (ref >= 2) & (ref <= 4)).sum()),
                zero_input_px=int((zero & covered).sum())), ref, keep, valid


def agree_map(ref, codes, keep, valid, zero, apply_zero):
    a = np.full(ref.shape, A_EXCL, np.uint8)
    a[codes == PL.NODATA] = A_NODATA
    if apply_zero:
        a[(codes != PL.NODATA) & zero] = A_NOINPUT
    a[valid & (ref == codes)] = A_AGREE
    a[valid & (ref != codes)] = A_DISAGREE
    return a


# ----------------------------------------------------------------------------- figure
def _frac(left_in, bottom_in, w_in, h_in):
    return [left_in / FIG_W, bottom_in / FIG_H, w_in / FIG_W, h_in / FIG_H]


def check_fit(fig, maps, label):
    """WARN when any text/legend box crosses a map, another text/legend, or the figure edge."""
    from matplotlib.legend import Legend
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    items = []
    for t in fig.texts:
        if t.get_text():
            items.append((f"text {t.get_text()[:30]!r}", t.get_window_extent(r)))
    for ax in fig.axes:
        if ax.get_title():
            items.append((f"title {ax.get_title()[:30]!r}", ax.title.get_window_extent(r)))
        for t in ax.texts:
            items.append((f"text {t.get_text()[:30]!r}", t.get_window_extent(r)))
        legs = {id(a): a for a in ax.get_children() if isinstance(a, Legend)}
        for leg in legs.values():
            items.append((f"legend {leg.get_title().get_text()!r}", leg.get_window_extent(r)))
    for ln in fig.lines:
        items.append(("scale bar", ln.get_window_extent(r)))
    n = 0
    tol = 1.0                                               # px
    for name, b in items:
        if b.x0 < fb.x0 - tol or b.y0 < fb.y0 - tol or b.x1 > fb.x1 + tol or b.y1 > fb.y1 + tol:
            print(f"  WARN {label}: {name} outside the figure"); n += 1
        for ax in maps:
            mb = ax.get_window_extent(r)
            if b.overlaps(mb):
                print(f"  WARN {label}: {name} overlaps map {ax.get_title()[:20]!r}"); n += 1
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].overlaps(items[j][1]):
                print(f"  WARN {label}: {items[i][0]} overlaps {items[j][0]}"); n += 1
    return n


def draw(rec, rf, loaded, apply_zero, title2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle

    cmap = ListedColormap([c for _, c, _, _ in DISPLAY])
    acmap = ListedColormap([c for _, c, _ in AD])
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    row1_b = BOT_IN + MAP_IN + TTL_IN + SB_IN
    row2_b = BOT_IN
    xs = [SIDE_IN + k * (MAP_IN + GAP_IN) for k in range(4)]
    pos = {"ref": (xs[0], row1_b), PANEL_MODELS[0]: (xs[1], row1_b),
           PANEL_MODELS[1]: (xs[2], row1_b), PANEL_MODELS[2]: (xs[3], row1_b),
           PANEL_MODELS[3]: (xs[0], row2_b), "ad": (xs[1], row2_b)}
    maps = {}
    for key, (x, y) in pos.items():
        ax = fig.add_axes(_frac(x, y, MAP_IN, MAP_IN))
        ax.set_xticks([]); ax.set_yticks([])
        maps[key] = ax

    zero = loaded[BEST][1]
    _, ref, keep, valid = cell_stats(rf, loaded[BEST][0], zero, apply_zero)
    shown = {"ref": ref_display(rf)}
    shown.update({m: model_display(loaded[m][0], loaded[m][1], apply_zero) for m in PANEL_MODELS})
    ad = agree_map(ref, loaded[BEST][0], keep, valid, zero, apply_zero)
    present = set(np.unique(np.concatenate([a.ravel() for a in shown.values()])).tolist())
    ad_present = set(np.unique(ad).tolist())

    def hatch_nodata(ax, mask):
        """White pixels with grey hatching where `mask` (no Prithvi pixel), one patch per row run."""
        rects = []
        for r in np.flatnonzero(mask.any(1)):
            row = np.concatenate([[0], mask[r].astype(np.int8), [0]])
            d = np.diff(row)
            for c0, c1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                rects.append(Rectangle((c0 - 0.5, r - 0.5), c1 - c0, 1))
        if rects:
            ax.add_collection(PatchCollection(rects, facecolor="white", edgecolor=NODATA_HATCH_COLOR,
                                              hatch=NODATA_HATCH, linewidth=0))

    maps["ref"].imshow(shown["ref"], cmap=cmap, vmin=-0.5, vmax=len(DISPLAY) - 0.5,
                       interpolation="nearest")
    maps["ref"].set_title("Interpreted Reference", fontsize=11)
    for m in PANEL_MODELS:
        maps[m].imshow(shown[m], cmap=cmap, vmin=-0.5, vmax=len(DISPLAY) - 0.5,
                       interpolation="nearest")
        hatch_nodata(maps[m], shown[m] == D_NODATA)
        maps[m].set_title(SHORT[m], fontsize=11)
    maps["ad"].imshow(ad, cmap=acmap, vmin=-0.5, vmax=len(AD) - 0.5, interpolation="nearest")
    hatch_nodata(maps["ad"], ad == A_NODATA)
    maps["ad"].set_title(f"Agree / Disagree\n({SHORT[BEST]} vs Interpreted)", fontsize=11)

    # legends in the empty bottom-right area (two columns of legend axes); per figure, only the
    # classes and special colors that occur in its panels
    lx = xs[2]
    lax1 = fig.add_axes(_frac(lx, row2_b, MAP_IN * 0.62, MAP_IN + TTL_IN)); lax1.axis("off")
    lax2 = fig.add_axes(_frac(lx + MAP_IN * 0.62 + 0.15, row2_b,
                              FIG_W - SIDE_IN - (lx + MAP_IN * 0.62 + 0.15), MAP_IN + TTL_IN))
    lax2.axis("off")

    def ph(code_is_nodata, c, l):
        if code_is_nodata:
            return Patch(facecolor="white", edgecolor=NODATA_HATCH_COLOR, hatch=NODATA_HATCH,
                         label=l)
        return Patch(facecolor=c, edgecolor="0.4", label=l)

    groups = [
        (lax1, "upper left", "Evaluation Classes",
         [ph(False, c, l) for k, c, l, g in DISPLAY if g == "class" and k in present]),
        (lax1, "lower left", "Agree / Disagree Map",
         [ph(k == A_NODATA, c, l) for k, c, l in AD if k in ad_present]),
        (lax2, "upper left", "Not Evaluated",
         [ph(k == D_NODATA, c, l) for k, c, l, g in DISPLAY if g == "other" and k in present]),
    ]
    for lax, loc, title, handles in groups:
        if not handles:
            continue
        leg = lax.legend(handles=handles, loc=loc, fontsize=9, title=title, title_fontsize=9.5,
                         frameon=False, alignment="left")
        lax.add_artist(leg)                                 # keep several legends per axes

    # suptitle (two lines) and provenance footnote
    fig.text(0.5, 1 - 0.12 / FIG_H,
             f"Cell {rec['cell_id']} · Target Year {rec['target_year']} · "
             f"Prithvi Label Year {rec['label_year']}",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 1 - 0.47 / FIG_H, title2, ha="center", va="top", fontsize=11)
    mask_txt = (f"zero-input mask {PL.PRITHVI_ZERO_INPUT} (model panels)" if apply_zero
                else "no zero-input mask")
    fig.text(0.5, 0.12 / FIG_H,
             f"Reference shown unmasked · {mask_txt} · year rule {PL.PRITHVI_YEAR_RULE} · "
             f"seeded dedup (seed {cc.SEED}) · reviewer {rec['reviewer']} · {rec['reference']}",
             ha="center", va="bottom", fontsize=7.5, color="0.35")

    # scale bar below the reference map, in its reserved band, at the drawn map scale
    fig.canvas.draw()
    ax = maps["ref"]
    h, w = rf.shape
    p0 = ax.transData.transform((-0.5, h - 0.5))
    p1 = ax.transData.transform((-0.5 + SCALE_M / PL.PIXEL_M, h - 0.5))
    inv = fig.transFigure.inverted()
    (fx0, _), (fx1, _) = inv.transform(p0), inv.transform(p1)
    fy = (row1_b - 0.25) / FIG_H
    fig.add_artist(Line2D([fx0, fx1], [fy, fy], transform=fig.transFigure, color="black", lw=2.5))
    for fx in (fx0, fx1):
        fig.add_artist(Line2D([fx, fx], [fy - 0.04 / FIG_H, fy + 0.04 / FIG_H],
                                transform=fig.transFigure, color="black", lw=1.2))
    fig.text(fx1 + 0.08 / FIG_W, fy, f"{SCALE_M / 1000:g} km", ha="left", va="center", fontsize=9)

    fig.canvas.draw()
    nwarn = check_fit(fig, list(maps.values()), os.path.basename(path))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return nwarn


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["disagreement", "change", "both"], default="both")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell ids (optionally ID:YEAR) to plot instead of ranking")
    ap.add_argument("--mask", choices=["any_year", "none"], default="any_year",
                    help="zero-input mask for ranking, valid pixels and the Masked Input color")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--min-valid-px", type=int, default=0,
                    help="drop candidate cells with fewer valid pixels (default 0 = evaluation set)")
    args = ap.parse_args()

    if SC.OUT_DIR is None:
        sys.exit("refusing to run: set LCI_OUT_DIR so outputs cannot overwrite the Sentinel-2 reports/")
    if SC.SENSOR != "Landsat":
        sys.exit(f"refusing to run: LCI_SENSOR is {SC.SENSOR!r}, expected 'Landsat'")
    if args.mask != "none" and args.mask != PL.PRITHVI_ZERO_INPUT:
        sys.exit(f"--mask {args.mask} differs from prithvi_loader.PRITHVI_ZERO_INPUT "
                 f"{PL.PRITHVI_ZERO_INPUT!r}")
    if args.top_n < 1:
        ap.error(f"--top-n must be at least 1, got {args.top_n}")
    apply_zero = args.mask != "none"
    try:                                                    # before any folder is created
        out_root = check_out_path(SC.out_path(OUT_SUB))
    except ValueError as e:
        sys.exit(f"refusing to run: {e}")

    cells = cc.deduped_cells()
    print(f"de-duplicated reference cells (seed {cc.SEED}): {len(cells)}")
    recs, n_output = [], 0
    for path in cells:
        cell_id, target, reviewer = PL.parse_reference(path)
        ly = PL.label_year(target)
        have = [os.path.exists(PL.prithvi_path(m, ly, cell_id)) for m in PL.PRITHVI_MODELS]
        if not any(have):
            continue
        if not all(have):
            raise RuntimeError(f"{path}: Prithvi output present for only some models")
        n_output += 1
        codes, zero, year, _ = PL.load_for_reference(path, BEST)
        with rasterio.open(path) as ds:
            rf = ds.read(1)
        st, *_ = cell_stats(rf, codes, zero, apply_zero)
        recs.append(dict(cell_id=cell_id, target_year=target, label_year=year, reviewer=reviewer,
                         **st, reference=os.path.basename(path), path=path))
    scored = [r for r in recs if r["valid_px"] > 0]
    print(f"cells with Prithvi output: {n_output}")
    print(f"cells with >= 1 valid px ({args.mask}): {len(scored)}")
    print(f"total valid px ({SHORT[BEST]}, {args.mask}): {sum(r['valid_px'] for r in scored):,}")
    cand = [r for r in scored if r["valid_px"] >= args.min_valid_px]
    if args.min_valid_px:
        print(f"candidates with >= {args.min_valid_px} valid px: {len(cand)}")

    tie = lambda r: (r["cell_id"], r["target_year"])  # noqa: E731
    jobs = []
    if args.cells:
        sel = []
        for tok in [t.strip() for t in args.cells.split(",") if t.strip()]:
            cid, _, yr = tok.partition(":")
            cid = cid.zfill(5)
            hits = [r for r in recs if r["cell_id"] == cid and (not yr or r["target_year"] == int(yr))]
            if len(hits) != 1:
                sys.exit(f"--cells {tok}: {len(hits)} matching cells with Prithvi output "
                         f"({[h['target_year'] for h in hits]}); use ID:YEAR")
            sel.append(hits[0])
        jobs.append(("cells", sel, None))
    else:
        if args.mode in ("disagreement", "both"):
            top = sorted(cand, key=lambda r: (-r["disagreement_fraction"], *tie(r)))[:args.top_n]
            jobs.append(("disagreement", top, "disagreement"))
        if args.mode in ("change", "both"):
            top = sorted(cand, key=lambda r: (-r["change_px"], *tie(r)))[:args.top_n]
            jobs.append(("change", top, "change"))

    total_warn = 0
    for folder, sel, kind in jobs:
        out = os.path.join(out_root, folder)
        os.makedirs(out, exist_ok=True)
        order_col = "order" if kind is None else "rank"
        rows = []
        for i, r in enumerate(sel, 1):
            loaded = {}
            for m in PANEL_MODELS:
                codes, zero, _, _ = PL.load_for_reference(r["path"], m)
                loaded[m] = (codes, zero)
            with rasterio.open(r["path"]) as ds:
                rf = ds.read(1)
            dis = (f"{r['disagreement_fraction'] * 100:.0f}%" if r["valid_px"] else "n/a")
            if kind == "disagreement":
                t2 = (f"Rank {i} of {len(sel)} by {SHORT[BEST]} Disagreement ({dis} of "
                      f"{r['valid_px']:,} Valid px)")
            elif kind == "change":
                t2 = (f"Rank {i} of {len(sel)} by Interpreted Change ({r['change_px']:,} px); "
                      f"{SHORT[BEST]} Disagreement {dis} of {r['valid_px']:,} Valid px")
            else:
                t2 = f"{SHORT[BEST]} Disagreement {dis} of {r['valid_px']:,} Valid px"
            stem = f"cell{r['cell_id']}_t{r['target_year']}.png"
            png = os.path.join(out, stem if kind is None else f"rank{i:02d}_{stem}")
            total_warn += draw(r, rf, loaded, apply_zero, t2, png)
            rows.append(dict(**{order_col: i}, **{k: r[k] for k in CSV_COLS}))
        df = pd.DataFrame(rows, columns=[order_col] + CSV_COLS)
        df["disagreement_fraction"] = df["disagreement_fraction"].round(5)
        df.to_csv(os.path.join(out, "summary.csv"), index=False)
        if not rows:
            print(f"{folder}: NOTE no cells selected; wrote a header-only summary.csv -> {out}/")
        else:
            print(f"{folder}: wrote {len(sel)} figures + summary.csv -> {out}/")
    print(f"overlap check: {total_warn} WARN")


if __name__ == "__main__":
    main()
