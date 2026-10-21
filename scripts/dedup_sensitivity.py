#!/usr/bin/env python3
"""Selection-sensitivity test for the interpreted-vs-model comparison.

Many locations were interpreted by more than one reviewer. The main comparison keeps
one interpretation per location; this script asks how much the pooled metrics depend on
*which* one. It repeats the "pick one interpretation per location" draw N times (default
100) with different random selections and reports the distribution of overall accuracy,
macro-F1, mean IoU, and Cohen's kappa per model version.

Efficiency: the model stitch for a cell footprint is identical across runs, so each
interpreted raster's confusion matrix (vs. each model version) is computed ONCE up front;
each run is then just choosing one raster per location and summing precomputed matrices.

Outputs (reports/model_comparison/):
  - dedup_sensitivity_runs.csv      one row per (run, version): the four metrics
  - dedup_sensitivity_summary.csv   per version: mean / std / min / max / p2.5 / p97.5
  - dedup_sensitivity_box.png       distribution of each metric across runs, by version

Usage:
    python scripts/dedup_sensitivity.py
    python scripts/dedup_sensitivity.py --runs 200 --targets 2019
    python scripts/dedup_sensitivity.py --versions v2 v5

Requires: rasterio, numpy, pandas, matplotlib
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_interpreted_vs_model as C  # stitch / crosswalk / confusion / metrics
import sensor_config as SC  # SENSOR, RF_GLOB, pixel area from transform

OUT = "reports/model_comparison"
METRICS = ["overall_accuracy", "macro_f1", "mean_iou", "kappa"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=100, help="number of random selections (default: 100)")
    ap.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5", "v6"])
    ap.add_argument("--targets", nargs="+", default=None, help="restrict to these target years")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed (default: 0)")
    args = ap.parse_args()

    rf2common, names, colors = C.load_mappings()
    cells = sorted(glob.glob(os.path.join(C.RF_DIR, "**", SC.RF_GLOB), recursive=True))
    suffix = ""
    if args.targets:
        keep = set(args.targets)
        cells = [c for c in cells if C.target_year(c) in keep]
        suffix = "_target" + "-".join(args.targets)

    # group rasters by location (grid + sample + target)
    locations = defaultdict(list)
    for f in cells:
        locations[C.location_key(f)].append(f)
    loc_items = [(k, sorted(v)) for k, v in sorted(locations.items())]
    n_dup = sum(1 for _, v in loc_items if len(v) > 1)
    print(f"cells: {len(cells)}   locations: {len(loc_items)}   "
          f"multi-interpretation: {n_dup}   runs: {args.runs}")

    # ---- precompute confusion matrix per (version, raster) ----
    # stitch the model once per (location, version) and reuse for each reviewer there.
    cms = {v: {} for v in args.versions}
    for v in args.versions:
        tiles = [rasterio.open(t) for t in C.model_tiles(v)]
        for k, files in loc_items:
            with rasterio.open(files[0]) as ds0:      # identical footprint within a location
                model = C.stitch_model_to_cell(ds0, tiles)
            for f in files:
                with rasterio.open(f) as ds:
                    rf_common = C.to_common_rf(ds.read(1), rf2common)
                cm, _ = C.confusion(rf_common, model)
                cms[v][f] = cm
        for t in tiles:
            t.close()
        print(f"  precomputed {v}: {len(cms[v])} matrices", flush=True)

    # ---- N random selections; each run sums the chosen matrices ----
    rng = np.random.default_rng(args.seed)
    records = []
    for r in range(args.runs):
        picks = [files[rng.integers(len(files))] for _, files in loc_items]
        for v in args.versions:
            pooled = np.sum([cms[v][f] for f in picks], axis=0)
            m = C.metrics_from_cm(pooled)
            records.append(dict(run=r, version=v,
                                overall_accuracy=m["overall_accuracy"],
                                macro_f1=m["macro_f1"], mean_iou=m["mean_iou"], kappa=m["kappa"]))
    runs_df = pd.DataFrame(records)
    os.makedirs(OUT, exist_ok=True)
    runs_df.round(4).to_csv(os.path.join(OUT, f"dedup_sensitivity_runs{suffix}.csv"), index=False)

    # ---- summary per version ----
    rows = []
    for v in args.versions:
        sub = runs_df[runs_df.version == v]
        for metric in METRICS:
            x = sub[metric].to_numpy()
            rows.append(dict(version=v, metric=metric,
                             mean=round(float(x.mean()), 4), std=round(float(x.std()), 4),
                             min=round(float(x.min()), 4), max=round(float(x.max()), 4),
                             p2_5=round(float(np.percentile(x, 2.5)), 4),
                             p97_5=round(float(np.percentile(x, 97.5)), 4),
                             range=round(float(x.max() - x.min()), 4)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, f"dedup_sensitivity_summary{suffix}.csv"), index=False)

    make_plot(runs_df, args.versions, os.path.join(OUT, f"dedup_sensitivity_box{suffix}.png"), args.runs)

    print("\n" + "=" * 68)
    print(f"selection sensitivity across {args.runs} runs (mean ± std [min, max]):")
    for v in args.versions:
        s = summ[summ.version == v].set_index("metric")
        oa, kp = s.loc["overall_accuracy"], s.loc["kappa"]
        print(f"  {v}:  OA {oa['mean']:.3f} ± {oa['std']:.3f} [{oa['min']:.3f}, {oa['max']:.3f}]   "
              f"kappa {kp['mean']:.3f} ± {kp['std']:.3f} [{kp['min']:.3f}, {kp['max']:.3f}]")
    print(f"\noutputs -> {OUT}/dedup_sensitivity{suffix}_*.csv/png")


def _caption(fig, text, top=1.0, width=110):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width))
    nlines = wrapped.count("\n") + 1
    fig.tight_layout(rect=[0, 0.02 + 0.04 * nlines, 1, top])
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8, color="0.35")


def make_plot(runs_df, versions, path, n_runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"overall_accuracy": "Overall accuracy", "macro_f1": "Macro F1",
              "mean_iou": "Mean IoU", "kappa": "Cohen's kappa"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.ravel(), METRICS):
        data = [runs_df[runs_df.version == v][metric].to_numpy() for v in versions]
        ax.boxplot(data, tick_labels=versions, showfliers=True)
        ax.set_title(titles[metric]); ax.set_ylabel(metric); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Sensitivity of model-comparison metrics to the choice of interpretation\n"
                 f"({n_runs} random one-per-location selections)", fontsize=12)
    _caption(fig, "Four boxplots, one per metric (overall accuracy, macro-F1, mean IoU, and Cohen's "
             f"kappa), show the distribution of that metric across {n_runs} random draws that pick one "
             "interpretation per multi-interpreted location, with one box per model version. The "
             "spread of each box is how much the pooled metric depends on which interpretation is "
             "kept rather than on the model itself. Read a narrow box as a result that is robust to "
             "the interpretation choice, and compare boxes across versions to rank the models while "
             "accounting for that selection uncertainty.", top=0.9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
