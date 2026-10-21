#!/usr/bin/env python3
"""Sampling experiment against known truth: approaches A/B/C/D under two designs.

Separate from the exhaustive-tiling runs in window_sampling_by_approach/Case_{A,B,C,D}_window_sampling/ (those are the
census); this measures how fast SAMPLED estimates approach that census, and whether they get
there at all. Draws from designs whose properties we characterize -- NOT accuracy estimates.

Setup: interpreted CKIT-RF cells = reference, model maps v2-v6 = map field; de-duplicated one
interpretation per location (grid+sample+target), default_rng seed 42, all target years. The
cells are a simple random sample from the grid_112_naip_brackets_5_11_26 frame, so the cell is
the primary sampling unit and the pixels within it are a census.

Population of windows: the exhaustive non-overlapping WxW tiling of each cell (positions
i*W,j*W), restricted to windows whose CENTER pixel is jointly valid (reference and model class
both in 1..10). Sampling draws n distinct windows (without replacement) from this pooled,
finite population -- so sampled windows never overlap. Stratum = the center pixel's reference
class (10 strata). W in {1,3,5,7,9}; n in {20,50,100,200,500,1000,2000,5000} (total across the
pooled frame, not per cell); 100 iterations per (n, W, design). Seeds derived from base 42.

Stratum ceiling: the count of population windows per center reference class (per W) -- reported
as a table; it bounds what any design can achieve for a class (once a stratum is exhausted,
more n buys nothing for it).

Design 1 (simple random): n windows uniform from the pool. Self-weighting; unbiased for the
census; expected to FAIL for rare classes (documented, not hidden): per class per (n,W) we
report the fraction of iterations in which the class is entirely absent.

Design 2 (stratified on center reference class): equal allocation n/10 per stratum; if a
stratum has fewer windows than its allocation, take all and record the shortfall (other strata
are not enlarged). Equal allocation oversamples rare classes on purpose. Estimators are
Horvitz-Thompson: window weight = N_h / n_h (population / realized stratum size). We report the
WEIGHTED estimate (weights = true center-class proportions; converges to census) and the
UNWEIGHTED one (won't), so the gap is visible.

Approaches per iteration, per variant:
  A per-pixel: pool all W^2 pixels of sampled windows into a confusion matrix (no aggregation).
  B dominant (map,ref) pair per window; ties = lowest map code then lowest ref code (documented
    substitution for Robert's binary tie rule).
  C plurality class per field independently; ties = lowest class code; also the fraction of
    windows whose plurality was a true majority.
  D per-class prop_map vs prop_ref; CORRELATION is the primary tightness metric (RMSE rewards
    predicting near-zero for rare classes -- corr leads; rmse and bias reported alongside).
    Correlation is undefined for a stratum with no variance -- reported, not dropped.
  W=1 collapses A=B=C to the same per-pixel quantity -- asserted as a correctness check.

Reported (long-format CSVs under reports/Case_ABCD_sampling/):
  stratum_ceiling.csv, census.csv, metrics_by_n.csv (bias/SD vs census, design effect, eff n),
  class_absence.csv, stratum_realized.csv, strat_efficiency.csv, d_correlation.csv
Plots (no gridlines, keep x/y axes): sd_vs_n_OA, bias_vs_n_OA, design_effect_vs_W,
  strat_efficiency, class_absence, d_corr_vs_n.

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreted_vs_model as C
import collapsed_5class_confusion as cc  # canonical 5-class collapse
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = "reports/Case_ABCD_sampling"
N = 10
WS = [1, 3, 5, 7, 9]
NSAMP = [20, 50, 100, 200, 500, 1000, 2000, 5000]
VERSIONS = ["v2", "v3", "v4", "v5", "v6"]
ITERS = 100
SEED = 42

# ---- optional 5-class collapse (see --collapse): merge all stable classes into one, keep the
# change classes distinct, exclude Unknown (observed-but-unattributed change, no model equivalent).
COLLAPSE = False
NAMES_5 = {1: "Stable", 2: "Harvest", 3: "Development", 4: "Insect/Disease", 5: "Beaver"}
# reference is in RF codes; Stable = urban/ag/grass-shrub/forest/water/wetland/other; Unknown(10)
# and Fire(40, absent) -> 0 (excluded).
# canonical 5-class collapse, shared from collapsed_5class_confusion (Other -> Stable, Unknown drop)
_REF_COLLAPSE = cc._REF_COLLAPSE
_MODEL_COLLAPSE = cc._MODEL_COLLAPSE


# ---- metrics from a (float-weighted) confusion matrix (rows = reference, cols = map). Returns
# scalars plus per-class F1, recall (producer's accuracy) and precision (user's accuracy). -------
def select_by_truth(cells, truth_csv):
    """Keep one raster per location using the adjudicated reviewer from the truth CSV.

    Returns (kept_cells, missing_grids, mismatches). Groups the rasters by grid_id and picks the
    one whose reviewer matches the adjudicated choice.
    """
    df = pd.read_csv(truth_csv, dtype=str, keep_default_na=False)
    truth = {str(int(g)).zfill(5): rev.strip().lower() for g, rev in zip(df.grid_id, df.reviewer)}
    rx = re.compile(r"reviewer_([A-Za-z]+)_grid_(\d+)_", re.I)
    by_grid = defaultdict(list)
    for c in cells:
        m = rx.search(os.path.basename(c))
        if m:
            by_grid[str(int(m.group(2))).zfill(5)].append((m.group(1).lower(), c))
    kept, missing, mismatch = [], [], []
    for gid, revpaths in by_grid.items():
        want = truth.get(gid)
        if want is None:
            missing.append(gid)
            continue
        match = [c for r, c in revpaths if r == want]
        if not match:
            mismatch.append((gid, want, [r for r, _ in revpaths]))
            continue
        kept.append(match[0])
    return sorted(kept), missing, mismatch


def cm_metrics(cm):
    tp = np.diag(cm).astype(float)
    row = cm.sum(1); col = cm.sum(0); tot = cm.sum()
    if tot <= 0:
        nanv = np.full(N, np.nan)
        return np.nan, np.nan, np.nan, nanv, nanv, nanv
    oa = tp.sum() / tot
    pe = (row * col).sum() / (tot * tot)
    kappa = (oa - pe) / (1 - pe) if (1 - pe) != 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(col > 0, tp / col, np.nan)    # user's accuracy (TP / map total)
        rec = np.where(row > 0, tp / row, np.nan)      # producer's accuracy (TP / ref total)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), np.nan)
    present = (row + col) > 0
    macro = np.nanmean(f1[present]) if present.any() else np.nan
    return oa, kappa, macro, f1, rec, prec


PRED_BAND = {"v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5}   # band order in pred_<bracket>_cell*.tif


def _cell_bracket_gid(path):
    # the cell's NAIP bracket (opt_<y1>_<y2>) and zero-padded grid id, from the filename
    m = re.search(r"grid_(\d+)_.*opt_(\d{4}_\d{4})", os.path.basename(path))
    return str(int(m.group(1))).zfill(5), m.group(2)


def build_model_cache(cells, version, preds_dir, tiles, ref_cache):
    """Per-cell model field for one variant (common codes 1..10, collapsed to 1..5 if COLLAPSE).

    With preds_dir, the map is the cell's temporally-matched per-bracket prediction band, aligned
    pixel-for-pixel with the reference. Otherwise it is the static v2-v6 mosaic stitched to the cell.
    """
    band = PRED_BAND[version]
    cache, bad = {}, []
    for f in cells:
        if preds_dir:
            gid, bracket = _cell_bracket_gid(f)
            pp = os.path.join(preds_dir, bracket, f"pred_{bracket}_cell{gid}.tif")
            if not os.path.exists(pp):
                bad.append((os.path.basename(f), f"no prediction: {pp}")); continue
            with rasterio.open(pp) as ds:
                mdl = ds.read(band)
        else:
            with rasterio.open(f) as ds:
                mdl = C.stitch_model_to_cell(ds, tiles)
        if COLLAPSE:                                  # common 1..10 -> collapsed 1..5
            mdl = _MODEL_COLLAPSE[np.where(mdl <= 10, mdl, 0)]
        if mdl.shape != ref_cache[f].shape:
            bad.append((os.path.basename(f), f"shape {mdl.shape} vs ref {ref_cache[f].shape}")); continue
        cache[f] = mdl
    if bad:
        print(f"STOP: {len(bad)} model/prediction problem(s):")
        for b in bad[:10]:
            print("  ", b)
        raise SystemExit(1)
    return cache


def precompute_population(cells, ref_cache, model_cache, W):
    """Build per-window population arrays for one variant and window size W.

    Returns dict with, over windows whose center pixel is jointly valid:
      cent (uint8 center ref class 1..10), codes (Npop, W*W) uint8 pair code or 255,
      domB, plmap, plref (uint8), majmap, majref (bool), pmap/pref (Npop,10) f32, nvalid (int).
    """
    cent_l, codes_l, domB_l, plmap_l, plref_l = [], [], [], [], []
    majmap_l, majref_l, pmap_l, pref_l, nval_l = [], [], [], [], []
    half = W // 2
    for f in cells:
        ref = ref_cache[f]
        mdl = model_cache[f]                          # common 1..10 (or collapsed 1..5), per cell
        H, Wd = ref.shape
        nH, nW = H // W, Wd // W
        if nH == 0 or nW == 0:
            continue
        r = ref[:nH * W, :nW * W].astype(np.int16)
        m = mdl[:nH * W, :nW * W].astype(np.int16)
        v = (r >= 1) & (r <= N) & (m >= 1) & (m <= N)
        # window blocks: (nH, W, nW, W) -> (nH, nW, W, W)
        rb = r.reshape(nH, W, nW, W).transpose(0, 2, 1, 3).reshape(nH * nW, W * W)
        mb = m.reshape(nH, W, nW, W).transpose(0, 2, 1, 3).reshape(nH * nW, W * W)
        vb = v.reshape(nH, W, nW, W).transpose(0, 2, 1, 3).reshape(nH * nW, W * W)
        # center pixel of each window
        rc = r[half::W, half::W][:nH, :nW].reshape(-1)
        mc = m[half::W, half::W][:nH, :nW].reshape(-1)
        keep = (rc >= 1) & (rc <= N) & (mc >= 1) & (mc <= N)
        if not keep.any():
            continue
        rb, mb, vb = rb[keep], mb[keep], vb[keep]
        nvalid = vb.sum(1)
        codes = np.where(vb, (rb - 1) * N + (mb - 1), 255).astype(np.uint8)
        # per-window class counts (map, ref) over valid pixels -> plurality, majority, props
        mc_counts = np.zeros((len(rb), N), np.int32)
        rc_counts = np.zeros((len(rb), N), np.int32)
        for c in range(N):
            mc_counts[:, c] = ((mb == c + 1) & vb).sum(1)
            rc_counts[:, c] = ((rb == c + 1) & vb).sum(1)
        plmap = mc_counts.argmax(1).astype(np.uint8)          # ties -> lowest class code
        plref = rc_counts.argmax(1).astype(np.uint8)
        majmap = mc_counts.max(1) * 2 > nvalid
        majref = rc_counts.max(1) * 2 > nvalid
        nv = np.maximum(nvalid, 1)[:, None]
        pmap = (mc_counts / nv).astype(np.float32)
        pref = (rc_counts / nv).astype(np.float32)
        # dominant joint pair per window (ties -> lowest pair code == lowest map then ref)
        pair_counts = np.zeros((len(rb), N * N), np.int32)
        for p in range(N * N):
            pair_counts[:, p] = (codes == p).sum(1)
        domB = pair_counts.argmax(1).astype(np.uint8)

        cent_l.append(rc[keep].astype(np.uint8)); codes_l.append(codes)
        domB_l.append(domB); plmap_l.append(plmap); plref_l.append(plref)
        majmap_l.append(majmap); majref_l.append(majref)
        pmap_l.append(pmap); pref_l.append(pref); nval_l.append(nvalid.astype(np.int32))

    return dict(cent=np.concatenate(cent_l), codes=np.concatenate(codes_l),
                domB=np.concatenate(domB_l), plmap=np.concatenate(plmap_l),
                plref=np.concatenate(plref_l), majmap=np.concatenate(majmap_l),
                majref=np.concatenate(majref_l), pmap=np.concatenate(pmap_l),
                pref=np.concatenate(pref_l), nvalid=np.concatenate(nval_l))


def confusion_from_windows(pop, idx, weights, W):
    """Weighted A/B/C confusion matrices for the sampled window indices."""
    wB = weights
    cmB = np.bincount(pop["domB"][idx], weights=wB, minlength=N * N).reshape(N, N)
    cmC = np.bincount(pop["plref"][idx] * N + pop["plmap"][idx], weights=wB,
                      minlength=N * N).reshape(N, N)
    # A: pixel codes of sampled windows, each pixel weighted by its window weight
    codes = pop["codes"][idx].reshape(-1)
    wpix = np.repeat(weights, W * W)
    good = codes != 255
    cmA = np.bincount(codes[good], weights=wpix[good], minlength=N * N).reshape(N, N)
    return cmA, cmB, cmC


def d_correlation(pop, idx):
    """Per-class Pearson corr of prop_map vs prop_ref over sampled windows (+ rmse, bias)."""
    pm = pop["pmap"][idx]; pr = pop["pref"][idx]
    corr = np.full(N, np.nan); rmse = np.full(N, np.nan); bias = np.full(N, np.nan)
    for c in range(N):
        x, y = pm[:, c], pr[:, c]
        rmse[c] = np.sqrt(np.mean((x - y) ** 2))
        bias[c] = float(x.mean() - y.mean())
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            corr[c] = float(np.corrcoef(x, y)[0, 1])
    return corr, rmse, bias


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--ws", nargs="+", type=int, default=WS)
    ap.add_argument("--ns", nargs="+", type=int, default=NSAMP)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--plots-only", action="store_true",
                    help="regenerate plots from the existing CSVs without re-running the experiment")
    ap.add_argument("--collapse", action="store_true",
                    help="5-class collapse: merge all stable classes into Stable, keep the change "
                         "classes distinct, exclude Unknown; writes to Case_ABCD_sampling_5class/")
    ap.add_argument("--truth", default=None,
                    help="use the adjudicated reviewer per location from this CSV instead of the "
                         "seed-42 random dedup (the sampling randomness stays seed 42)")
    ap.add_argument("--preds", default=None,
                    help="use the temporally-matched per-bracket prediction rasters in this dir "
                         "(pred_<bracket>_cell<id>.tif) as the map field instead of the static mosaics")
    args = ap.parse_args()

    global N, OUT, COLLAPSE
    rf2common, names, colors = C.load_mappings()
    if args.collapse:
        N, OUT, COLLAPSE, names = 5, "reports/Case_ABCD_sampling_5class", True, NAMES_5

    if args.plots_only:
        _make_plots(names, args.versions)
        print(f"regenerated plots -> {OUT}/")
        return
    cells = sorted(glob.glob(os.path.join(C.RF_DIR, "**", SC.RF_GLOB), recursive=True))
    if args.truth:
        cells, missing, mismatch = select_by_truth(cells, args.truth)
        if mismatch:
            print("STOP: truth reviewer with no matching raster:", mismatch)
            return
        if missing:
            print(f"note: {len(missing)} grid(s) not in the truth set, dropped: {missing[:10]}")
        ref_src = f"adjudicated reviewer per location ({os.path.basename(args.truth)})"
        print(f"adjudicated reference: {len(cells)} cells from {args.truth}")
    else:
        cells, ndup = C.dedupe_cells(cells, SEED)
        ref_src = f"one interpretation per location, random dedup, seed {SEED}"
        print(f"de-duplicated: {ndup} location(s); {len(cells)} cells (all target years, seed {SEED})")
    map_src = (f"temporally-matched per-bracket predictions ({args.preds})" if args.preds
               else "static v2-v6 mosaics (single 2018/2020 classification)")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reference.txt"), "w") as fh:
        fh.write(f"reference points: {ref_src}\n"
                 f"map field: {map_src}\n"
                 f"sampling randomness: numpy.SeedSequence({SEED}) (unchanged)\n")

    ref_cache = {}
    if COLLAPSE:
        unk = tot = 0
        for f in cells:
            with rasterio.open(f) as ds:
                rf = ds.read(1)
            safe = np.where((rf >= 0) & (rf <= 62), rf, 0)
            ref_cache[f] = _REF_COLLAPSE[safe]         # RF codes -> collapsed 1..5 (Unknown/Fire -> 0)
            unk += int((rf == 10).sum()); tot += rf.size
        with open(os.path.join(OUT, "exclusion.txt"), "w") as fh:
            fh.write("5-class collapse: Stable = urban/agriculture/grass-shrub/forest/water/wetland/other;\n"
                     "change classes kept distinct = Harvest, Development, Insect/Disease, Beaver.\n\n"
                     "Unknown (unattributed change, no model equivalent) is EXCLUDED — a substantive\n"
                     "exclusion, not a technicality: dropping observed-but-unattributed disturbance makes\n"
                     "the Stable stratum marginally purer than the landscape.\n\n"
                     f"excluded Unknown pixels: {unk:,}\n"
                     f"total reference pixels:  {tot:,}\n"
                     f"share of frame:          {unk / tot:.4%}\n")
        print(f"[collapse] excluded Unknown pixels: {unk:,} ({unk / tot:.3%} of the frame)")
    else:
        for f in cells:
            with rasterio.open(f) as ds:
                ref_cache[f] = C.to_common_rf(ds.read(1), rf2common)

    ss = np.random.SeedSequence(SEED)
    ceiling_rows, census_rows, metric_rows = [], [], []
    absence_rows, realized_rows, deff_rows, dcorr_rows, effrows, pcm_rows = [], [], [], [], [], []

    for v in args.versions:
        tiles = None if args.preds else [rasterio.open(t) for t in C.model_tiles(v)]
        model_cache = build_model_cache(cells, v, args.preds, tiles, ref_cache)
        for W in args.ws:
            pop = precompute_population(cells, ref_cache, model_cache, W)
            Npop = pop["cent"].size
            strata = {h: np.where(pop["cent"] == h + 1)[0] for h in range(N)}
            Nh = np.array([strata[h].size for h in range(N)])
            wclass = Nh / Nh.sum()                     # true center-class proportions
            # stratum ceiling (report once per W, version-independent center classes -> report per version anyway)
            if v == args.versions[0]:
                for h in range(N):
                    ceiling_rows.append(dict(W=W, cls=names[h + 1], n_windows=int(Nh[h]),
                                             proportion=round(float(wclass[h]), 5)))
            # census (all windows, unit weights)
            wall = np.ones(Npop)
            cmA, cmB, cmC = confusion_from_windows(pop, np.arange(Npop), wall, W)
            cenA, cenB, cenC = cm_metrics(cmA), cm_metrics(cmB), cm_metrics(cmC)
            corr_all, _, _ = d_correlation(pop, np.arange(Npop))
            for ap_name, cen in [("A", cenA), ("B", cenB), ("C", cenC)]:
                census_rows.append(dict(version=v, W=W, approach=ap_name,
                                        oa=round(cen[0], 4), kappa=round(cen[1], 4),
                                        macro_f1=round(cen[2], 4)))
            if W == 1:  # correctness check
                assert np.allclose(cenA[0], cenB[0]) and np.allclose(cenA[0], cenC[0]), \
                    f"W=1 A/B/C mismatch for {v}"

            rng = np.random.default_rng(ss.spawn(1)[0])
            for n in args.ns:
                # storage across iterations for each design
                store = {d: dict(oaA=[], kaA=[], mfA=[], oaB=[], oaC=[], oaAw=[], mfAw=[],
                                 f1A=[], f1Aw=[], recAw=[], precAw=[], major=[], npix=[])
                         for d in ("simple", "strat")}
                absA = {d: np.zeros(N) for d in ("simple", "strat")}
                dcorr = {d: [] for d in ("simple", "strat")}
                realized = np.zeros(N); shortfall = np.zeros(N)
                per = n // N
                for it in range(args.iters):
                    # --- design 1: simple random (self-weighting) ---
                    k = min(n, Npop)
                    idx = rng.choice(Npop, size=k, replace=False)
                    w = np.ones(k)
                    _run(pop, idx, w, W, store["simple"], absA["simple"], dcorr["simple"], unweighted=True)
                    # --- design 2: stratified on center ref class, equal allocation ---
                    sidx, sw = [], []
                    for h in range(N):
                        g = strata[h]
                        take = min(per, g.size)
                        realized[h] += take; shortfall[h] += max(0, per - g.size)
                        if take == 0:
                            continue
                        pick = g[rng.choice(g.size, size=take, replace=False)]
                        sidx.append(pick)
                        sw.append(np.full(take, Nh[h] / take))          # HT weight N_h/n_h
                    sidx = np.concatenate(sidx); sw = np.concatenate(sw)
                    _run(pop, sidx, sw, W, store["strat"], absA["strat"], dcorr["strat"],
                         unweighted=True)
                # summarize this (n, W, version)
                _summarize(v, W, n, Npop, Nh, store, absA, dcorr, realized, shortfall, per,
                           names, wclass, cenA, cenB, cenC, corr_all,
                           metric_rows, absence_rows, realized_rows, deff_rows, dcorr_rows,
                           effrows, pcm_rows)
            print(f"  {v} W={W} done", flush=True)
        if tiles:
            for t in tiles:
                t.close()

    _write_and_plot(names, ceiling_rows, census_rows, metric_rows, absence_rows,
                    realized_rows, deff_rows, dcorr_rows, effrows, pcm_rows, args.versions)
    print(f"\noutputs -> {OUT}/  (draws from designs, NOT accuracy estimates)")


def _run(pop, idx, w, W, st, absA, dcorr, unweighted):
    cmA, cmB, cmC = confusion_from_windows(pop, idx, w, W)
    cmA_u, _, _ = confusion_from_windows(pop, idx, np.ones(len(idx)), W)  # unweighted for absence/simple
    oaA, kaA, mfA, f1A, _, _ = cm_metrics(cmA_u)    # unweighted (self-weighting for simple; contrast for strat)
    oaAw, _, mfAw, f1Aw, recAw, precAw = cm_metrics(cmA)   # weighted (design-consistent)
    oaB = cm_metrics(cmB)[0]
    oaC = cm_metrics(cmC)[0]
    st["oaA"].append(oaA); st["kaA"].append(kaA); st["mfA"].append(mfA)
    st["oaAw"].append(oaAw); st["mfAw"].append(mfAw)
    st["oaB"].append(oaB); st["oaC"].append(oaC)
    st["f1A"].append(f1A); st["f1Aw"].append(f1Aw)
    st["recAw"].append(recAw); st["precAw"].append(precAw)
    st["major"].append(float(pop["majmap"][idx].mean()))
    st["npix"].append(int((pop["codes"][idx] != 255).sum()))
    # class absence: no sampled pixel labels the class in ref or map (unweighted A support)
    row = cmA_u.sum(1); col = cmA_u.sum(0)
    absA += ((row + col) == 0).astype(float)
    corr, rmse, bias = d_correlation(pop, idx)
    dcorr.append((corr, rmse, bias))


def _summarize(v, W, n, Npop, Nh, store, absA, dcorr, realized, shortfall, per, names, wclass,
               cenA, cenB, cenC, corr_all, metric_rows, absence_rows, realized_rows,
               deff_rows, dcorr_rows, effrows, pcm_rows):
    iters = len(store["simple"]["oaA"])
    census = dict(A=cenA[0], B=cenB[0], C=cenC[0], mfA=cenA[2], kaA=cenA[1])
    for design in ("simple", "strat"):
        s = store[design]
        for metric, arr, cval in [("A_oa", s["oaA"], census["A"]), ("A_kappa", s["kaA"], census["kaA"]),
                                  ("A_macrof1", s["mfA"], census["mfA"]), ("B_oa", s["oaB"], census["B"]),
                                  ("C_oa", s["oaC"], census["C"]),
                                  ("A_oa_wtd", s["oaAw"], census["A"]),
                                  ("A_macrof1_wtd", s["mfAw"], census["mfA"])]:
            a = np.array(arr, float)
            metric_rows.append(dict(design=design, version=v, W=W, n=n, metric=metric,
                                    census=round(float(cval), 4), mean=round(float(np.nanmean(a)), 4),
                                    sd=round(float(np.nanstd(a, ddof=1)), 5),
                                    bias=round(float(np.nanmean(a) - cval), 5),
                                    lo=round(float(np.nanpercentile(a, 2.5)), 4),
                                    hi=round(float(np.nanpercentile(a, 97.5)), 4)))
        # majority fraction (C)
        metric_rows.append(dict(design=design, version=v, W=W, n=n, metric="C_frac_majority",
                                census=np.nan, mean=round(float(np.mean(s["major"])), 4),
                                sd=round(float(np.std(s["major"], ddof=1)), 5), bias=np.nan,
                                lo=np.nan, hi=np.nan))
    # design effect (approach A OA, simple design)
    meanN = float(np.mean(store["simple"]["npix"]))
    p = census["A"]
    sd_obs = float(np.nanstd(store["simple"]["oaA"], ddof=1))
    sd_bin = np.sqrt(p * (1 - p) / meanN) if 0 < p < 1 and meanN else np.nan
    deff = (sd_obs / sd_bin) ** 2 if sd_bin and sd_bin > 0 else np.nan
    deff_rows.append(dict(version=v, W=W, n=n, mean_pixels=round(meanN, 1),
                          sd_obs=round(sd_obs, 5), sd_binom=round(float(sd_bin), 6),
                          design_effect=round(float(deff), 3),
                          eff_sample_size=round(meanN / deff, 1) if deff == deff and deff else np.nan))
    # class absence (both designs)
    for design in ("simple", "strat"):
        for h in range(N):
            absence_rows.append(dict(design=design, version=v, W=W, n=n, cls=names[h + 1],
                                     frac_absent=round(absA[design][h] / iters, 4)))
    # realized stratified allocation + shortfall + finite-population correction magnitude.
    # the reported SDs are empirical Monte Carlo from WITHOUT-replacement draws, so the FPC is
    # applied implicitly (a fully-sampled stratum is identical every iteration -> zero variance);
    # the sampling fraction and FPC factor make its size visible.
    for h in range(N):
        realized_h = realized[h] / iters
        Nh_h = int(Nh[h])
        frac = realized_h / Nh_h if Nh_h > 0 else np.nan
        fpc = np.sqrt(max(0.0, (Nh_h - realized_h) / (Nh_h - 1))) if Nh_h > 1 else np.nan
        realized_rows.append(dict(version=v, W=W, n=n, cls=names[h + 1], stratum_ceiling=Nh_h,
                                  target_alloc=per, mean_realized=round(realized_h, 2),
                                  mean_shortfall=round(shortfall[h] / iters, 2),
                                  sampling_fraction=round(float(frac), 5),
                                  fpc_sd_factor=round(float(fpc), 4)))
    # per-class recall (producer's), precision (user's) and F1, design-consistent weighted, vs census
    cen_f1, cen_rec, cen_prec = cenA[3], cenA[4], cenA[5]
    for design in ("simple", "strat"):
        s = store[design]
        for mname, arrkey, cen in [("recall", "recAw", cen_rec), ("precision", "precAw", cen_prec),
                                   ("f1", "f1Aw", cen_f1)]:
            A = np.array(s[arrkey], float)          # (iters, N)
            for h in range(N):
                col = A[:, h]
                cv = float(cen[h]) if cen[h] == cen[h] else np.nan
                pcm_rows.append(dict(design=design, version=v, W=W, n=n, cls=names[h + 1],
                                     metric=mname, census=round(cv, 4) if cv == cv else np.nan,
                                     mean=round(float(np.nanmean(col)), 4),
                                     sd=round(float(np.nanstd(col, ddof=1)), 5),
                                     bias=round(float(np.nanmean(col) - cv), 5) if cv == cv else np.nan,
                                     frac_undefined=round(float(np.isnan(col).mean()), 4)))
    # per-class F1 SD, stratification efficiency = SD_strat / SD_simple.
    # use the design-consistent estimand for each arm: simple is self-weighting (f1A);
    # stratified equal-allocation must use the WEIGHTED per-class F1 (f1Aw), else the ratio
    # compares different estimands and rare classes look wrongly inflated.
    f1s_simple = np.array(store["simple"]["f1A"])   # (iters, N)
    f1s_strat = np.array(store["strat"]["f1Aw"])
    sd_simple = np.nanstd(f1s_simple, axis=0, ddof=1)
    sd_strat = np.nanstd(f1s_strat, axis=0, ddof=1)
    for h in range(N):
        eff = sd_strat[h] / sd_simple[h] if sd_simple[h] > 0 else np.nan
        effrows.append(dict(version=v, W=W, n=n, cls=names[h + 1],
                            sd_simple=round(float(sd_simple[h]), 5), sd_strat=round(float(sd_strat[h]), 5),
                            strat_efficiency=round(float(eff), 3) if eff == eff else np.nan))
    # D correlation per class per design
    for design in ("simple", "strat"):
        arr = dcorr[design]
        corrs = np.array([a[0] for a in arr]); rmses = np.array([a[1] for a in arr])
        biases = np.array([a[2] for a in arr])
        for h in range(N):
            cvals = corrs[:, h]
            dcorr_rows.append(dict(design=design, version=v, W=W, n=n, cls=names[h + 1],
                                   census_corr=round(float(corr_all[h]), 4) if corr_all[h] == corr_all[h] else np.nan,
                                   mean_corr=round(float(np.nanmean(cvals)), 4),
                                   frac_undefined=round(float(np.isnan(cvals).mean()), 4),
                                   mean_rmse=round(float(np.nanmean(rmses[:, h])), 4),
                                   mean_bias=round(float(np.nanmean(biases[:, h])), 4)))


def _classic(ax):
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _logscale(ax, axes="x"):
    """Set log scale only where the plotted data is present and strictly positive."""
    for a in axes:
        vals = np.concatenate([(ln.get_xdata() if a == "x" else ln.get_ydata())
                               for ln in ax.lines]) if ax.lines else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size and (vals > 0).all():
            (ax.set_xscale if a == "x" else ax.set_yscale)("log")


LABELED_N = (20, 100, 500, 2000, 5000)   # label a readable subset; rest are unlabeled ticks


def _caption(fig, text, width=120):
    """Add a wrapped descriptive caption below the figure, reserving space for it. Call this in place
    of fig.tight_layout(); the caption is captured by bbox_inches='tight' at save time."""
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.035 * nlines, 1, 1])   # reserve bottom room for the caption
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def _nticks(ax, n_values, label_all=False, labeled=None):
    """Put ticks at the actual n values. Label only a subset by default (no sci-notation / minor
    clutter); pass label_all=True to label every n value, or labeled=... to override the subset."""
    labeled = LABELED_N if labeled is None else labeled
    nv = sorted(n_values)
    ax.set_xticks(nv)
    ax.set_xticklabels([str(n) if (label_all or n in labeled) else "" for n in nv])
    ax.minorticks_off()


def _write_and_plot(names, ceiling_rows, census_rows, metric_rows, absence_rows, realized_rows,
                    deff_rows, dcorr_rows, effrows, pcm_rows, versions):
    pd.DataFrame(ceiling_rows).to_csv(os.path.join(OUT, "stratum_ceiling.csv"), index=False)
    pd.DataFrame(census_rows).to_csv(os.path.join(OUT, "census.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(OUT, "metrics_by_n.csv"), index=False)
    pd.DataFrame(absence_rows).to_csv(os.path.join(OUT, "class_absence.csv"), index=False)
    pd.DataFrame(realized_rows).to_csv(os.path.join(OUT, "stratum_realized.csv"), index=False)
    pd.DataFrame(deff_rows).to_csv(os.path.join(OUT, "design_effect.csv"), index=False)
    pd.DataFrame(effrows).to_csv(os.path.join(OUT, "strat_efficiency.csv"), index=False)
    pd.DataFrame(dcorr_rows).to_csv(os.path.join(OUT, "d_correlation.csv"), index=False)
    pd.DataFrame(pcm_rows).to_csv(os.path.join(OUT, "per_class_metrics.csv"), index=False)
    _make_plots(names, versions)


def _make_plots(names, versions):
    """Regenerate all plots from the CSVs in OUT (so plotting can be redone without re-sampling)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md = pd.read_csv(os.path.join(OUT, "metrics_by_n.csv"))
    ab = pd.read_csv(os.path.join(OUT, "class_absence.csv"))
    de = pd.read_csv(os.path.join(OUT, "design_effect.csv"))
    ef = pd.read_csv(os.path.join(OUT, "strat_efficiency.csv"))
    dc = pd.read_csv(os.path.join(OUT, "d_correlation.csv"))
    n_values = sorted(md.n.unique())
    wpal = {1: "#1f77b4", 3: "#2ca02c", 5: "#9467bd", 7: "#ff7f0e", 9: "#d62728"}
    vpal = {"v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd", "v5": "#ff7f0e", "v6": "#d62728"}
    # per-class colors from the canonical legend; collapsed Stable is grey and the change classes
    # reuse the 10-class legend colors (collapsed code -> 10-class code)
    class_colors_10 = C.load_mappings()[2]
    if COLLAPSE:
        _to10 = {2: 1, 3: 2, 4: 10, 5: 9}
        ccol = {nm: ("#cccccc" if c == 1 else class_colors_10[_to10[c]]) for c, nm in names.items()}
    else:
        ccol = {nm: class_colors_10[c] for c, nm in names.items()}

    # 1) SD of A_oa vs n (log-log), line per W, panel per version + 1/sqrt(n) reference.
    # 2x3 layout: two variants on top, three on the bottom, one shared legend in the empty top cell.
    # rendered twice: the default log-decade y labels, and a version 2 with plain-number y ticks.
    import matplotlib.ticker as mticker

    def _sd_vs_n(out_name, plain_y):
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharey=True)
        cells = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]       # panel cells; (0, 2) holds the legend
        handles_labels = None
        for (rr, cc), v in zip(cells, versions):
            ax = axes[rr, cc]
            s = md[(md.metric == "A_oa") & (md.design == "simple") & (md.version == v)]
            for W in sorted(s.W.unique()):
                sw = s[s.W == W].sort_values("n")
                ax.plot(sw.n, sw.sd, "o-", color=wpal[W], label=f"W={W}", ms=4)
            # 1/sqrt(n) reference anchored at the W=1 line's smallest-n SD (slope -0.5)
            w1 = s[s.W == 1].sort_values("n")
            if len(w1):
                n0, sd0 = float(w1.n.iloc[0]), float(w1.sd.iloc[0])
                nn = np.array(n_values, float)
                ax.plot(nn, sd0 * np.sqrt(n0 / nn), "k--", lw=1, zorder=1,
                        label="independent (slope −0.5)")
            ax.set_xlabel("n (windows)", fontsize=12); ax.set_title(v, fontsize=13)
            _logscale(ax, "xy"); _nticks(ax, n_values)
            if plain_y:                                        # plain-number y ticks, slightly more frequent
                yt = [0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
                ax.yaxis.set_major_locator(mticker.FixedLocator(yt))
                ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in yt]))
                ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(labelsize=10)
            if cc == 0:
                ax.set_ylabel("SD of OA (approach A)", fontsize=12)
            _classic(ax)
            if handles_labels is None:
                handles_labels = ax.get_legend_handles_labels()
        # one shared legend in the empty top-right cell
        axes[0, 2].axis("off")
        axes[0, 2].legend(*handles_labels, loc="center", fontsize=11, frameon=False, title="window size")
        fig.suptitle("Sampling Precision vs Sample Size", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(OUT, out_name), dpi=200, bbox_inches="tight")
        plt.close(fig)

    _sd_vs_n("sd_vs_n_OA.png", plain_y=False)
    if not COLLAPSE:                                          # the plain-tick v2 is only for the 10-class figure 3.7
        _sd_vs_n("sd_vs_n_OA_v2.png", plain_y=True)

    # 2) bias vs n (v2, W=3): simple vs stratified weighted vs stratified unweighted
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, design, metric, c in [("simple (unwtd)", "simple", "A_oa", "#1f77b4"),
                                   ("stratified, weighted", "strat", "A_oa_wtd", "#2ca02c"),
                                   ("stratified, unweighted", "strat", "A_oa", "#d62728")]:
        s = md[(md.design == design) & (md.metric == metric) & (md.version == "v2") & (md.W == 3)].sort_values("n")
        ax.plot(s.n, s.bias, "o-", color=c, label=lab)
    ax.axhline(0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("n (windows)", fontsize=11); ax.set_ylabel("mean sampled OA − census OA", fontsize=11)
    _logscale(ax, "x")
    _nticks(ax, n_values, labeled=(20, 200, 500, 1000, 2000, 5000))   # drop 100, label 200 and 1000
    ax.tick_params(labelsize=9)
    fig.suptitle("Estimator Bias vs Sample Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, frameon=False); _classic(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bias_vs_n_OA.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3) design effect vs W (mean over n), line per version
    fig, ax = plt.subplots(figsize=(8, 5))
    g = de.groupby(["version", "W"]).design_effect.mean().reset_index()
    for v in versions:
        s = g[g.version == v]
        ax.plot(s.W, s.design_effect, "o-", color=vpal[v], label=v)
    ax.axhline(1, ls="--", color="k", lw=0.8); ax.set_xticks(WS)
    ax.set_xlabel("window size W", fontsize=12)
    ax.set_ylabel("design effect (observed variance /\nindependent-sampling variance)", fontsize=12)
    ax.tick_params(labelsize=9)
    fig.suptitle("Design Effect vs Window Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, frameon=False); _classic(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "design_effect_vs_W.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 4) stratification efficiency per class (v2, W=1, largest n)
    fig, ax = plt.subplots(figsize=(10, 5))
    nmax = max(ef.n)
    s = ef[(ef.version == "v2") & (ef.W == 1) & (ef.n == nmax)].dropna(subset=["strat_efficiency"])
    order = s.sort_values("strat_efficiency")
    ax.bar(order.cls, order.strat_efficiency,
           color=["#2ca02c" if x < 1 else "#d62728" for x in order.strat_efficiency])
    ax.axhline(1, ls="--", color="k", lw=0.8)
    ax.set_ylabel("SD_stratified / SD_simple  (<1 = stratification helps)", fontsize=11)
    fig.suptitle("Stratification Efficiency by Class", fontsize=15, fontweight="bold")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order.cls, rotation=45, ha="right")
    ax.tick_params(labelsize=9); _classic(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "strat_efficiency.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 5) class absence vs n (simple, v2, W=1), line per class
    fig, ax = plt.subplots(figsize=(9, 5))
    s = ab[(ab.design == "simple") & (ab.version == "v2") & (ab.W == 1)]
    for cls in s.cls.unique():
        sc = s[s.cls == cls].sort_values("n")
        ax.plot(sc.n, sc.frac_absent, "o-", label=cls, color=ccol.get(cls), ms=4)
    ax.set_xlabel("n (windows, log scale)", fontsize=12)
    ax.set_ylabel("fraction of iterations where class is entirely absent", fontsize=12); _logscale(ax, "x")
    _nticks(ax, n_values, label_all=True)                     # label every tick; the log axis is noted in the x label
    ax.tick_params(labelsize=10)
    fig.suptitle("Class Absence vs Sample Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2); _classic(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_absence.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 6) D correlation vs n, per class (v2, W=5, simple)
    fig, ax = plt.subplots(figsize=(9, 5))
    s = dc[(dc.design == "simple") & (dc.version == "v2") & (dc.W == 5)]
    for cls in s.cls.unique():
        sc = s[s.cls == cls].sort_values("n")
        ax.plot(sc.n, sc.mean_corr, "o-", label=cls, color=ccol.get(cls), ms=4)
    ax.set_xlabel("n (windows)", fontsize=11)
    ax.set_ylabel("mean per-class corr(prop_map, prop_ref)", fontsize=11); _logscale(ax, "x")
    _nticks(ax, n_values, label_all=True)
    ax.tick_params(labelsize=9)
    fig.suptitle("Per-Class Proportion Correlation vs Sample Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2); _classic(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "d_corr_vs_n.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
