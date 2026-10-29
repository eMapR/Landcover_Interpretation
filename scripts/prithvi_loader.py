"""Shared loader for the Prithvi change-detection outputs (Landsat, 30 m), for comparison with the
interpreted Landsat reference rasters.

Source (read-only): PRITHVI_ROOT/outputs/<label_year>/<model>/<cell_id>.tif, one per grid_112 cell,
label year and model. Only the per-year folders are read; the four model folders directly under
outputs/ are never used. Each output is 112 x 112 px at 30 m, EPSG:5070, 5 float32 softmax bands
(agriculture, development, harvest, insect_disease_defo, background) with no argmax applied and no
nodata marker. The model input chips are PRITHVI_ROOT/chips/<label_year>/<cell_id>.tif (224 x 224 px,
the output cell plus 56 px of context, 4 years x 6 bands).

read_probs() refuses any file that is not exactly that 5-band probability stack, so the bands can
never be read as class codes. collapsed_argmax() takes the argmax and maps it to the collapsed codes
of collapsed_5class_confusion.py (1 Stable, 2 Harvest, 3 Development, 4 Insect/Disease); agriculture
gets its own code AGRICULTURE (6), since its meaning is to be confirmed, and Beaver (5) is never
produced. align_to_reference() then puts the codes on a reference raster's own grid by centre-based
lookup: each reference pixel takes the Prithvi pixel under its centre, or NODATA (0) when there is
none. The Prithvi grid sits ~9.1 m east of the reference grid, so no resampling choice is involved.

Settings (TO BE CONFIRMED / PROVISIONAL, see docs/landsat_migration_log.md):
  PRITHVI_YEAR_RULE      "target": the Prithvi label year is the reference target year.
  PRITHVI_ZERO_INPUT     which pixels to mask because their input is all-zero (all 6 bands) in the
                         chip: "any_year" (any of the 4 input years, the default), "label_year", or
                         "none". The chips store missing input as 0, not NaN; whether inference
                         turned those 0s into NaN/-1 is unknown.

    LCI_PRITHVI_ROOT=/path/to/prithvi_change_results  overrides PRITHVI_ROOT.
"""

import functools
import os
import re

import numpy as np
import rasterio

_root_env = os.environ.get("LCI_PRITHVI_ROOT")
PRITHVI_ROOT = (os.path.abspath(_root_env) if _root_env
                else "/home/server/hpc/emapr/prithvi_change_results")
OUTPUTS_DIR = os.path.join(PRITHVI_ROOT, "outputs")
CHIPS_DIR = os.path.join(PRITHVI_ROOT, "chips")

PRITHVI_MODELS = ["fcn_frozen_encoder_q1_w2", "fcn_random_encoder_q1_w2",
                  "fcn_unfrozen_encoder_q1_w2", "unet3d_q1_w2"]

PRITHVI_YEAR_RULE = "target"            # TO BE CONFIRMED: label year = reference target year
PRITHVI_ZERO_INPUT = "any_year"         # PROVISIONAL: "any_year", "label_year" or "none"
ZERO_INPUT_RULES = ("any_year", "label_year", "none")

PRITHVI_BANDS = ("agriculture", "development", "harvest", "insect_disease_defo", "background")
NODATA = 0
AGRICULTURE = 6                         # TO BE CONFIRMED: kept out of the 5-class scheme
# band index (argmax) -> collapsed code: agriculture, development, harvest, insect, background
PRITHVI_COLLAPSE = np.array([AGRICULTURE, 3, 2, 4, 1], np.uint8)

SUM_TOL = 1e-4                          # max |sum of bands - 1| per pixel
PIXEL_M = 30.0
EPSG = 5070
CELL_PX = 112
CHIP_MARGIN = 56                        # chip = cell + 56 px on every side
CHIP_YEARS, CHIP_BANDS = 4, 6           # chip bands ordered year by year: Y-1, Y, Y+1, Y+2
LABEL_YEAR_INDEX = 1                    # Y sits at index 1 of the 4 chip years


def label_year(target_year):
    """Prithvi label year for a reference target year, under PRITHVI_YEAR_RULE."""
    if PRITHVI_YEAR_RULE == "target":
        return int(target_year)
    raise ValueError(f"unknown PRITHVI_YEAR_RULE {PRITHVI_YEAR_RULE!r}")


def prithvi_path(model, year, cell_id):
    if model not in PRITHVI_MODELS:
        raise ValueError(f"unknown Prithvi model {model!r}")
    return os.path.join(OUTPUTS_DIR, str(int(year)), model, f"{cell_id}.tif")


def chip_path(year, cell_id):
    return os.path.join(CHIPS_DIR, str(int(year)), f"{cell_id}.tif")


def _check_grid(ds, what):
    t = ds.transform
    if ds.crs is None or ds.crs.to_epsg() != EPSG or t.b or t.d or t.a != PIXEL_M or t.e != -PIXEL_M:
        raise ValueError(f"{what} {ds.name}: expected EPSG:{EPSG} at {PIXEL_M:g} m north-up, "
                         f"got {ds.crs} {tuple(t)[:6]}")


def read_probs(path):
    """(probs (5, 112, 112) float32, transform) for one Prithvi output; ValueError unless it is the
    documented probability stack: 5 float32 bands named PRITHVI_BANDS in order, all finite, in
    [0, 1], summing to 1 within SUM_TOL."""
    with rasterio.open(path) as ds:
        _check_grid(ds, "Prithvi output")
        if ds.count != len(PRITHVI_BANDS) or tuple(ds.descriptions) != PRITHVI_BANDS:
            raise ValueError(f"{path}: bands {ds.descriptions}, expected {PRITHVI_BANDS}")
        if set(ds.dtypes) != {"float32"}:
            raise ValueError(f"{path}: dtype {ds.dtypes}, expected float32 probabilities")
        if (ds.height, ds.width) != (CELL_PX, CELL_PX):
            raise ValueError(f"{path}: shape {ds.height}x{ds.width}, expected {CELL_PX}x{CELL_PX}")
        probs, transform = ds.read(), ds.transform
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1 + SUM_TOL).any():
        raise ValueError(f"{path}: non-finite or out-of-range probabilities")
    dev = np.abs(probs.sum(0, dtype=np.float64) - 1).max()
    if dev > SUM_TOL:
        raise ValueError(f"{path}: bands sum to 1 +/- {dev:.2g}, above {SUM_TOL:g}")
    return probs, transform


def collapsed_argmax(probs):
    """Argmax over the 5 bands -> collapsed codes (uint8): 1 Stable, 2 Harvest, 3 Development,
    4 Insect/Disease, AGRICULTURE for agriculture. Ties go to the lower band index."""
    return PRITHVI_COLLAPSE[np.argmax(probs, axis=0)]


@functools.lru_cache(maxsize=64)
def _zero_input(year, cell_id):
    """(per-year all-zero input (4, 112, 112) bool, transform of the centre crop) from the chip."""
    with rasterio.open(chip_path(year, cell_id)) as ds:
        _check_grid(ds, "Prithvi chip")
        if ds.count != CHIP_YEARS * CHIP_BANDS:
            raise ValueError(f"{ds.name}: {ds.count} bands, expected {CHIP_YEARS * CHIP_BANDS}")
        a = ds.read()
        transform = ds.transform * ds.transform.translation(CHIP_MARGIN, CHIP_MARGIN)
    crop = a.reshape(CHIP_YEARS, CHIP_BANDS, *a.shape[1:])[
        :, :, CHIP_MARGIN:CHIP_MARGIN + CELL_PX, CHIP_MARGIN:CHIP_MARGIN + CELL_PX]
    return (crop == 0).all(1), transform


def zero_input_mask(year, cell_id, transform, rule=None):
    """Bool (112, 112), True where the chip input is all-zero under `rule` (default
    PRITHVI_ZERO_INPUT). `transform` is the output's; the chip's centre crop must sit on it."""
    rule = PRITHVI_ZERO_INPUT if rule is None else rule
    if rule not in ZERO_INPUT_RULES:
        raise ValueError(f"unknown zero-input rule {rule!r}, expected one of {ZERO_INPUT_RULES}")
    if rule == "none":
        return np.zeros((CELL_PX, CELL_PX), bool)
    per_year, crop_transform = _zero_input(int(year), cell_id)
    if not crop_transform.almost_equals(transform, precision=1e-6):
        raise ValueError(f"chip {year}/{cell_id} centre crop is not on the output grid")
    return per_year.any(0) if rule == "any_year" else per_year[LABEL_YEAR_INDEX].copy()


def align_to_reference(arr, src_transform, ref_ds, fill=NODATA):
    """Centre-based lookup of a 2-D Prithvi-grid array onto the reference raster's own grid: each
    reference pixel takes the value under its centre, or `fill` where no Prithvi pixel is."""
    _check_grid(ref_ds, "reference")
    t = ref_ds.transform
    xs = t.c + (np.arange(ref_ds.width) + 0.5) * t.a
    ys = t.f + (np.arange(ref_ds.height) + 0.5) * t.e
    cols = np.floor((xs - src_transform.c) / src_transform.a).astype(int)
    rows = np.floor((ys - src_transform.f) / src_transform.e).astype(int)
    vc = (cols >= 0) & (cols < arr.shape[1])
    vr = (rows >= 0) & (rows < arr.shape[0])
    out = np.full((ref_ds.height, ref_ds.width), fill, dtype=arr.dtype)
    out[np.ix_(vr, vc)] = arr[np.ix_(rows[vr], cols[vc])]
    return out


_REF_RX = re.compile(r"reviewer_([A-Za-z]+)_grid_(\d+)_sample_(\d+)_sensor_[A-Za-z0-9_-]+?_target_(\d{4})")


def parse_reference(path):
    """(cell_id, target_year, reviewer) from a reference raster filename."""
    m = _REF_RX.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse reviewer/grid/target from {path}")
    return m.group(2).zfill(5), int(m.group(4)), m.group(1).lower()


def load_for_reference(ref_path, model):
    """Prithvi map field for one reference raster and model, on the reference grid, or None when no
    output exists for its cell and label year. Returns (codes, zero_input, label_year, path):
    codes are collapsed_argmax codes with NODATA where no Prithvi pixel is under the reference pixel
    centre; zero_input is the PRITHVI_ZERO_INPUT mask on the same grid (False where no Prithvi
    pixel). Codes are not masked here, so callers can report masked and unmasked results."""
    cell_id, target, _ = parse_reference(ref_path)
    year = label_year(target)
    path = prithvi_path(model, year, cell_id)
    if not os.path.exists(path):
        return None
    probs, transform = read_probs(path)
    codes = collapsed_argmax(probs)
    zero = zero_input_mask(year, cell_id, transform)
    with rasterio.open(ref_path) as ref_ds:
        return (align_to_reference(codes, transform, ref_ds),
                align_to_reference(zero, transform, ref_ds, fill=False), year, path)
