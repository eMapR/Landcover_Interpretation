# Landsat Migration Log

## 2026-10-21: Pixel Size From Raster Metadata

Change: scripts read pixel area from each raster's transform instead of a hard-coded
10 m value (PIX_HA = 0.01). New shared settings in scripts/sensor_config.py:
LCI_SENSOR, LCI_RF_DIR, LCI_OUT_DIR (defaults reproduce the Sentinel-2 setup).

Test script: scripts/interpreter_change_stable_conflicts.py

- 10 m Sentinel-2: committed vs. edited version produced byte-identical outputs
  (5 files, 72 interpreter pairs).
- 30 m Landsat (/vol/v1/HLS_Prithvi/HLS_prithvi_outputs): ran successfully,
  29 interpreter pairs, 0.09 ha per pixel; area_ha = pixels x 0.09 on all rows.

Landsat run command:

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs \
    LCI_OUT_DIR=reports_landsat python scripts/interpreter_change_stable_conflicts.py

Open items:
- Landsat results are provisional (not committed) pending two questions to the data
  provider: whether all-zero edge rows/columns (code 0) are padding, and whether the
  June batch used an older label set.
- Some scripts (e.g. spatial_structure, sampling_experiment_ABCD, window_sampling_approach*,
  build_spectral_confusion, model_class_ci_5class) follow LCI_RF_DIR but not LCI_OUT_DIR.
  Do not run them with LCI_RF_DIR set until they use out_path, or they will overwrite reports/.
- Scripts that need data/raw/model_maps/ were only checked for clean imports and identical
  file patterns (model maps not on Islay); full output comparison still to do.

## 2026-10-21: beaver_example_figure.py Scale Bar, Zoom, and Overlap Fix

Change: the scale bar (length, height, position), patch-box padding, and zoom margin are now set
in metres and converted with each raster's resolution, instead of fixed pixel counts that assumed
10 m. At 30 m the old bar was 100 px (3 km) but labelled "1 km", and the zoom margin (45 px)
covered nearly the whole 113-px cell. The 1 km scale bar now moves to the bottom-right when its
bar or label would overlap the padded patch box (prints a WARN if both corners collide).

- 10 m Sentinel-2 (variables unset, outputs to scratch): bekka 36142 and mina 13691 are
  byte-identical to the previous version. peter 07373 changed on purpose: its patch is in the
  bottom-left corner, so the old bar ran through the patch box. The bar is now bottom-right.
- 30 m Landsat: ran successfully, outputs in reports_landsat/reports/beaver_examples/
  (bekka 50601: 81 px, ~7.3 ha; myscon 07373: 20 px, ~1.8 ha; 0.09 ha per pixel). Bar is
  1000/30 = 33.3 px, matching its "1 km" label; zoom margin is 15 px. Both PNGs inspected:
  no text or bar overlaps the patch, the box, or the panel edge.

Landsat run command:

    LCI_SENSOR=Landsat LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs \
    LCI_OUT_DIR=reports_landsat python scripts/beaver_example_figure.py

Open items:
- Grid 10333 sample 41 is skipped (ALREADY_SHOWN, from the Sentinel-2 work); decide whether to
  keep skipping it for Landsat.

## 2026-10-22: plot_change_stable_examples.py Cell Width in the Title

Change: the cell width in each figure title was `a.shape[1] * 10 / 1000` km, which assumed 10 m
pixels. It now uses the raster's pixel width (`abs(transform.a)` times the CRS linear unit). At
30 m the old title said 1.1 km for a 113-px Landsat cell; it now says 3.4 km (113 x 30 m).

- 10 m Sentinel-2 (only LCI_OUT_DIR=check_sentinel): reran interpreter_change_stable_conflicts.py
  and plot_change_stable_examples.py. All 10 example PNGs and all CSV/text outputs are
  byte-identical to the copy saved before the fix.
- 30 m Landsat (LCI_SENSOR=Landsat, LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs,
  LCI_OUT_DIR=check_landsat): the CSV/text outputs are unchanged, and only the 10 PNGs changed.
  Every Landsat rf_class raster is 113 px wide at 30 m, so all 10 titles now say 3.4 km
  (checked visually on rank01 and rank07).
- No other 10 m assumptions were found: the figure has no scale bar, and hectares in the title
  come from the CSV, which uses SC.uniform_pixel_area_m2.

Open item (fixed 2026-10-22, see below): the legend (`bbox_to_anchor=(0.5, 0.20)`) was drawn
over the bottom strip of both map panels, hiding part of the map and some outlines.

## 2026-10-22: plot_change_stable_examples.py Legend Overlap Fixed

Change: the legend now sits in its own strip between the maps and the caption, so it no longer
covers the map panels, the outlines, or any text. The script measures the legend height with the
renderer, makes the figure taller by that height plus a 0.15 in gap above and below (rounded up to
whole 150 dpi pixels), and anchors the legend 0.15 in under the drawn map bottoms. The maps, title,
and caption keep the same size and spacing as before. A fit-check prints the legend, map, and
caption positions for each figure and a WARN if the legend overlaps a map column or the caption, or
is wider than the figure.

- 10 m Sentinel-2 (only LCI_OUT_DIR=check_sentinel): the CSV/text outputs are byte-identical. The
  10 example PNGs changed only in legend position. Pixel comparison with the copy saved before the
  fix: the title band and caption are pixel-identical (the caption is shifted down by the new
  strip, 76 or 102 px). Above the old legend, the map band differs only by 1-2/255 anti-aliasing
  on contour-line edges, which is not visible. The only real change is the old legend band: the
  map pixels it used to hide are now shown, and the legend is drawn below the maps. Figures are
  76 or 102 px taller (one or two legend rows).
- 30 m Landsat (LCI_SENSOR=Landsat, LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs,
  LCI_OUT_DIR=check_landsat): the CSV/text outputs are unchanged. No WARNs in any of the 10
  figures. The legend strip of all 10 was inspected visually: the legend is clear of the maps,
  the outlines, and the caption.
- Python environment for these runs: /vol/v1/bekka/venvs/lci (matplotlib 3.11.2).

## 2026-10-22: change_class_examples.py Scale Bar, Zoom, and Overlap Fix

Change: the same 10 m assumptions that beaver_example_figure.py had (fixed 2026-10-21) are now
gone from change_class_examples.py. The 1 km scale bar was a fixed 100 px, which is right at 10 m
but almost the whole 113-px cell at 30 m. The bar length, bar height and offset, the patch-box
padding (6 px = 60 m) and the zoom margin (45 px = 450 m) are now set in metres and converted with
the raster's pixel size (`abs(transform.a)`). The script reuses `_overlaps` from
beaver_example_figure.py. If the bar or its label would overlap the padded patch box, the bar
moves to the bottom-right corner, as in the beaver script. If that corner is also blocked, it
tries top-right and then top-left, with the label below the bar. It prints which corner it used,
or a WARN if every corner collides.

- 10 m Sentinel-2 (only LCI_OUT_DIR=check_sentinel, all four classes): a fresh run of the
  unchanged script was byte-identical to the existing check_sentinel PNGs and was saved as the
  "before" copy. After the fix, 16 of 20 PNGs are byte-identical. The other 4 changed on purpose,
  because the old bar crossed the patch box: harvest peter 48800, development robert 01993,
  insect_disease peter 51183, and beaver peter 07373. In each of these the bar is now
  bottom-right. Image sizes are unchanged, and the only pixel differences are in the bar and
  "1 km" label rows, at the old bottom-left spot and the new bottom-right spot. All four were
  inspected: the bar no longer touches the box.
- 30 m Landsat (LCI_SENSOR=Landsat, LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs,
  LCI_OUT_DIR=check_landsat, all four classes): every cell is 113 x 113 px at 30 m. The bar is
  1000/30 = 33.3 px, which matches its "1 km" label. Box padding is 2 px and the zoom margin is
  15 px. The zoom panels now show 5-68% of the cell; with the old 45 px margin they showed most
  of it. Bar moves: insect_disease bekka 31832 and beaver myscon 07373 went bottom-right.
  harvest bekka 37243 went top-right, because its 130 ha patch box reaches both bottom corners
  (the first version, with only the bottom-right fallback, warned and the bar crossed the box).
  No WARNs. All 20 PNGs were inspected: no bar or label overlaps the patch box, a box edge, or
  the panel edge.
- Python environment: /vol/v1/bekka/venvs/lci.

Open items:
- With --class beaver, this script writes to reports/beaver_examples/ with the same file names
  as beaver_example_figure.py (for example beaver_example_bekka_cell36142_sample567.png). Whichever
  runs last overwrites the other's figure under the same LCI_OUT_DIR.
- beaver_example_figure.py still has only the bottom-right fallback. None of its current figures
  need more, but it would warn on a cell like harvest 37243.

## 2026-10-22: Scripts Passing on Both Sensors Without Code Changes

- interpreter_change_change_conflicts.py: Sentinel-2 output byte-identical to the committed
  reports/; Landsat 29 pairs, 45 conflict px (4.0 ha) at 0.09 ha/px. Cosmetic: the
  "a few thousand pixels" note (printed when < 5000 px) is inaccurate for Landsat.
- inspect_and_plot.py (--stats-all and --montage 9, run separately; the script does one task
  per run): Sentinel-2 class distribution identical to an earlier laptop run (272 rasters);
  Landsat read all 391 rasters, codes 13/40/50/62 present, Fire only in 1 raster (315 px).
  Montages look correct for both sensors.

Open items:
- Urban (code 0) is present in nearly every raster for both sensors (Sentinel-2 271/272,
  Landsat 388/391) and is about twice as common in Landsat (5.9% vs 3.0%). Check whether
  all-zero edge rows/columns are padding in the Sentinel-2 rasters as well as Landsat.

## 2026-10-22: compare_interpreters.py Passes on Both Sensors

- No code changes needed. Sentinel-2 (72 pairs) and Landsat (29 pairs) both ran without
  errors; agreement metrics in per_pair_metrics.csv are in range and the figures have no
  overlapping text. Outputs go under outputs/, which is gitignored, so there is no committed
  copy to diff against; the edit itself was verified byte-identical on Sentinel-2 earlier.

All seven scripts that need only the interpreter rasters now run on both sensors:
interpreter_change_stable_conflicts, interpreter_change_change_conflicts, compare_interpreters,
plot_change_stable_examples, change_class_examples, beaver_example_figure, inspect_and_plot.

## 2026-10-26: disagreement_geometry.py

Changes:
- Output folder now goes through SC.out_path. Before, it was a fixed
  "reports/interpreter_agreement/geometry", so LCI_OUT_DIR was ignored and any run, including a
  Landsat run, overwrote the committed reports/.
- The width_ecdf.png caption said "the resolution limit at 10 m". It now uses the raster's pixel
  side (square root of the pixel area), so it reads "10 m" on Sentinel-2 and "30 m" on Landsat.
  The script now stops with an error if the cells have different pixel sizes.
- Docstring and comments no longer say "69 cells", "1 px = 0.01 ha" or "0.1 ha = 10 px at 10 m";
  pixel size is described in general terms. No thresholds changed.

- 10 m Sentinel-2 (only LCI_OUT_DIR=check_sentinel): 72 pairs. All 6 outputs are byte-identical
  before and after the caption/docstring edit. Against the committed reports/: the 3 CSVs are
  byte-identical; area_ecdf_focus.png and shape_index_ecdf_focus.png are pixel-identical (only
  the embedded matplotlib version differs, 3.11.0 vs 3.11.2); the committed width_ecdf.png was
  later post-processed by annotate_figures.py (extra text panel), so it is not this script's
  direct output. Nothing differs because of the newer data.
- 30 m Landsat (LCI_SENSOR=Landsat, LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs,
  LCI_OUT_DIR=check_landsat): 29 pairs, 29,301 patches (21,341 disagreement, 7,960 agreement),
  0.09 ha per pixel on every row. No "10 m" or "0.01 ha" text in the CSVs or figures. All 3
  figures inspected: no overlapping text. Development->Urban has only 1 patch, so its curve is
  not drawn (curves need >= 3 patches) and that panel shows only Urban->Development (n=11).

Open items (thresholds for advisor; not changed):
- AREA_THRESH_HA = 0.1 ha (frac_area_under/over_0p1ha, console "<0.1ha"): 10 px at 10 m, about
  1.1 px at 30 m. The smallest Landsat patch is 0.09 ha, so "< 0.1 ha" now means exactly the
  single-pixel patches.
- Width cut-offs of 1 and 2 px (frac_le_1px, frac_le_2px, frac_gt_2px and the dashed lines on
  width_ecdf.png) are in pixels: 10/20 m on Sentinel-2, 30/60 m on Landsat.
- width_ecdf.png x-axis limit of 8 px: 80 m vs 240 m. The Landsat curve is flat by ~3.5 px, so
  nothing is cut off.
- annotate_figures.py (separate script) still says "10 m each" in its width_ecdf.png text; it
  would be wrong if run on the Landsat figures.

## 2026-10-26: disagreement_geometry_bysize.py

Changes:
- Added os.makedirs(OUT, exist_ok=True) in main(). All 4 outputs already went through
  SC.out_path (OUT = DG.OUT), but the script never created the folder. It only ever ran against
  the existing reports/ folder, so with LCI_OUT_DIR set to a new folder the first to_csv would
  have failed. Without LCI_OUT_DIR this does nothing.
- Importing disagreement_geometry (and compare_interpreters, sensor_config) writes nothing:
  imported with LCI_OUT_DIR set to an empty scratch folder, which was still empty afterwards.

- 10 m Sentinel-2 (only LCI_OUT_DIR=check_sentinel): 72 pairs, 255,985 disagreement patches.
  size_conditional_summary.csv and gs_wetland_top10.csv are byte-identical to the committed
  reports/. Nothing differs because of the newer data.
- 30 m Landsat (LCI_SENSOR=Landsat, LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs,
  LCI_OUT_DIR=check_landsat): 29 pairs, 21,341 disagreement patches, 0.09 ha per pixel on every
  row of gs_wetland_top10.csv. No "10 m" or "0.01 ha" text in the CSVs or figures (the figures
  give no resolution). Development->Urban has only 1 patch, so that ECDF panel shows only
  Urban->Development.
- git status after each run: nothing under reports/, outputs/ or exports/ modified.

PNG differences against the committed reports/ (Sentinel-2):
- gs_wetland_top10.png: the committed copy was later post-processed by annotate_figures.py
  (extra caption strip, 4533 vs 4350 px tall), so it is not this script's direct output. In
  the shared 4350 rows, 363 pixels differ by at most 2/255 (antialiasing, matplotlib 3.11.0 vs
  3.11.2).
- shape_index_area_weighted_ecdf.png: same curves and title, but the committed copy has larger
  axis-label, tick and legend fonts than the committed code produces. The plotting code is
  unchanged since fd99751, the commit that last wrote this PNG, and there is no matplotlibrc, so
  the committed PNG was not made by the committed code. Rerunning this script to regenerate
  figure_3_4c would shrink the fonts.

Open items (not changed):
1. gs_wetland_top10.png: the legend lists only the stable classes (codes 0-5), so the change
   classes shown in the maps (yellow and others) are missing from it.
2. gs_wetland_top10.png: the last word of the bottom caption ("patch.") overlaps the top
   border of the legend box, on both sensors.
3. For advisor: THRESH_HA = 0.1 ha (the lt_0.1ha / ge_0.1ha bins) is 10 px at 10 m but about
   1.1 px at 30 m. On Landsat, lt_0.1ha holds only the single-pixel patches (0.09 ha) and
   ge_0.1ha everything 2 px (0.18 ha) or larger.

## 2026-10-26: reviewer_directional_asymmetry.py

Changes:
- OUT is now SC.out_path("reports/interpreter_agreement"); it was the hard-coded
  "reports/interpreter_agreement", so LCI_OUT_DIR was ignored and a check run would have
  overwritten the committed CSVs and heatmap. All 3 outputs are built from OUT.
- Added os.makedirs(OUT, exist_ok=True) at the start of main(). Without LCI_OUT_DIR both
  changes leave the output location as before.
- Docstring made sensor-neutral: "all 69 double-interpreted pairs" -> "all double-interpreted
  pairs of the selected sensor", and the Outputs line notes LCI_OUT_DIR. No thresholds changed.
- The imported helpers (CI.load_legend, CI.find_pairs, CI.confusion) only read files.

- 10 m Sentinel-2 (only LCI_OUT_DIR=check_sentinel): 72 pairs, 5 reviewers.
  reviewer_class_overassignment.csv and reviewer_directed_classpairs.csv are byte-identical to
  the committed reports/. reviewer_overassignment_heatmap.png is pixel-identical; only the
  embedded matplotlib version differs (3.11.0 vs 3.11.2). Nothing differs because of the newer
  data (the committed outputs already used 72 pairs; the old docstring's 69 was stale).
- 30 m Landsat (LCI_SENSOR=Landsat, LCI_RF_DIR=/vol/v1/HLS_Prithvi/HLS_prithvi_outputs,
  LCI_OUT_DIR=check_landsat): 29 pairs, 3 reviewers (bekka 24, myscon 23, robert 11 pairs),
  0.09 ha per pixel from the raster transform. No "10 m", "10m" or "0.01 ha" text in the CSVs or
  figure. Heatmap inspected: no overlapping text. Significant over-assignments: robert
  Insect/Disease (+5.81) and myscon Unknown (+4.17), both on about 5 ha.
- git status after each run: nothing under reports/, outputs/ or exports/ modified.

Hard-coded values (for advisor; not changed):

| Value | Where | 10 m | 30 m |
|---|---|---|---|
| self + partner < 50 px: class skipped for that reviewer | main(), class loop | 0.5 ha | 4.5 ha |
| pair < 200 px: directed class pair skipped | main(), class-pair loop | 2 ha | 18 ha |
| +1 px pseudo-count in log2((a + 1) / (b + 1)) | index, bootstrap, class pairs | 0.01 ha | 0.09 ha (pulls small classes harder toward 0) |

Bootstrap count (2000), seed (42) and dpi (140) do not depend on resolution.

Skipped by the minimum-pixel rules:

| | Class rule (< 50 px) | Pair rule (< 200 px) |
|---|---|---|
| Sentinel-2 | 5 of 65 reviewer x class cells; all Fire (0 px), one per reviewer | 442 of 780 directed pairs skipped (206 with > 0 px); 338 kept (43%) |
| Landsat | 4 of 39 cells: myscon Fire (0 px); robert Other (0 px), Unknown (31 px, ~2.8 ha), Development (23 px, ~2.1 ha) | 378 of 468 skipped (168 with > 0 px); 90 kept (19%) |

On Landsat the class rule drops classes that do have disagreement, which never happens on
Sentinel-2.

Open items (not changed):
1. Fire is missing from the Landsat heatmap: the column list in heatmap() is fixed and has no
   "Fire", so the column is dropped silently although the CSV has values (bekka +8.30, robert
   -8.30). The fix should show Fire only when it has data, so the Sentinel-2 figure (no Fire
   pixels) stays unchanged.
2. Reviewer letters are assigned alphabetically per sensor, so they mean different people on
   the two figures (Landsat A/B/C = bekka/myscon/robert; on Sentinel-2 bekka is B and robert
   is E). The code comment says they match reviewer_letter_key.csv (figure 3.5), which holds
   only for Sentinel-2.
3. Cells removed by the 50 px rule show as blank white with no explanation in the figure
   (Landsat: robert Other, Development, Unknown).
