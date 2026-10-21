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
