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
