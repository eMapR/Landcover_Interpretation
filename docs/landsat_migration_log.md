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
