"""Shared sensor setting and pixel-area helpers for the interpreted reference rasters.

SENSOR is the token in the reference filenames (rf_class_..._sensor_<SENSOR>_target_...tif) that
the globs and regexes select on. Default "Sentinel-2"; override per run without editing code:

    LCI_SENSOR=Landsat-8 python scripts/compare_interpreters.py

SENSOR is inserted into regexes and globs as-is (not re.escape'd, so the default patterns stay
byte-identical to the old hard-coded ones), so it is restricted to letters, digits, '_' and '-'.

RF_DIR is the root searched for the reference rasters, and out_path() places a script's outputs.
Both keep the old hard-coded paths unless overridden:

    LCI_RF_DIR=/path/to/rf_rasters LCI_OUT_DIR=/path/to/out python scripts/compare_interpreters.py

LCI_RF_DIR replaces data/raw/rf_class_maps. LCI_OUT_DIR, when set, moves everything a script writes
under that folder, keeping the script's usual repo-relative path below it (reports/..., outputs/...,
exports/...), so a run on another reference set cannot overwrite the default results. Relative
values are resolved against the current directory once, at import.

Pixel area comes from each raster's affine transform and CRS units instead of a hard-coded
10 m pixel (0.01 ha, 1e-4 km^2). For a 10 m EPSG:5070 raster the values are bit-identical to the
old constants. Pooled pixel counts are only convertible to area when every raster shares one pixel
size, so uniform_pixel_area_m2 raises if the rasters disagree.
"""

import os
import re

import rasterio

SENSOR = os.environ.get("LCI_SENSOR", "Sentinel-2")
if not re.fullmatch(r"[A-Za-z0-9_-]+", SENSOR):
    raise ValueError(f"SENSOR {SENSOR!r} must contain only letters, digits, '_' or '-'")

RF_GLOB = f"rf_class*{SENSOR}*.tif"                    # reference rasters for this sensor

_rf_env = os.environ.get("LCI_RF_DIR")
RF_DIR = os.path.abspath(_rf_env) if _rf_env else "data/raw/rf_class_maps"
_out_env = os.environ.get("LCI_OUT_DIR")
OUT_DIR = os.path.abspath(_out_env) if _out_env else None   # None = scripts keep their own paths


def out_path(path, root=None):
    """Where a script writes `path`, its usual repo-relative output path (e.g. "reports/x").

    LCI_OUT_DIR unset: `path` unchanged, joined to `root` for scripts that anchor on the repo root.
    LCI_OUT_DIR set: the same relative path under LCI_OUT_DIR.
    """
    if OUT_DIR is None:
        return os.path.join(root, path) if root else path
    return os.path.join(OUT_DIR, path)

M2_PER_HA = 10_000.0
M2_PER_KM2 = 1e6


def pixel_area_m2(transform, crs):
    """Area of one pixel in m^2 from an affine transform (|det|) and the CRS linear unit."""
    if crs is None or not crs.is_projected:
        raise ValueError(f"pixel area needs a projected CRS, got {crs}")
    unit_m = crs.linear_units_factor[1]
    return abs(transform.a * transform.e - transform.b * transform.d) * unit_m ** 2


def uniform_pixel_area_m2(paths):
    """The single pixel area (m^2) shared by every raster in `paths`; ValueError if they differ."""
    areas = {}
    for p in paths:
        with rasterio.open(p) as ds:
            areas.setdefault(pixel_area_m2(ds.transform, ds.crs), p)
    if not areas:
        raise ValueError("no rasters to take a pixel area from")
    if len(areas) > 1:
        raise ValueError("rasters have different pixel areas, pooled pixel counts would mix units: "
                         + ", ".join(f"{a:g} m^2 ({os.path.basename(p)})" for a, p in areas.items()))
    return next(iter(areas))
