"""
# temporal_utils.py
# ---------------------------------------------------------
# Helpers for processing PRISM temporal (tmean) rasters:
# - Robust date parsing from PRISM filenames
# - Clip to California boundary (in-memory)
# - Fast tiling using a precomputed tile_id lookup grid
#
# Output is designed to match the modern precipitation pipeline.
"""

from pathlib import Path
import re
import datetime as dt
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd


# ==========================================================
# 1. Filename → date parser
# ==========================================================
def parsedate_tmean(path: str | Path) -> dt.date:
    """
    Parse YYYYMM from PRISM tmean filenames.

    Handles names like:
      PRISM_tmean_stable_4kmM3_198101.bil
      PRISM_tmean_stable_4kmM3_198101_bil.bil
      PRISM_tmean_stable_4kmM3_198101_bil_CA.tif
      PRISM_tmean_stable_4kmM3_198101_bil_CA_1km.tif

    Returns:
        datetime.date(YYYY, MM, 1)
    """
    fname = Path(path).name
    m = re.search(r"_(\d{6})(?:_bil)?(?:_CA)?(?:_1km)?\.(?:bil|tif)$", fname)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {fname}")
    ym = m.group(1)
    year = int(ym[:4])
    month = int(ym[4:6])
    return dt.date(year, month, 1)


# ==========================================================
# 2. Clip PRISM raster to California
# ==========================================================
def clip_to_ca(src: rasterio.io.DatasetReader,
               ca_shp_path: str | Path,
               return_transform: bool = True) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Clip an open PRISM raster to the California boundary.

    Parameters
    ----------
    src : rasterio DatasetReader
        Open PRISM raster (tmean) in its native CRS.
    ca_shp_path : path-like
        Path to CA_State.shp
    return_transform : bool
        If True, return (array, transform); if False, only array.

    Returns
    -------
    arr : 2D np.ndarray
        Clipped raster values.
    transform : Affine
        Affine transform of the clipped raster (if requested).
    """
    ca = gpd.read_file(ca_shp_path)
    ca = ca.to_crs(src.crs)

    geoms = [g.__geo_interface__ for g in ca.geometry]
    out_img, out_transform = mask(src, geoms, crop=True)
    arr = out_img[0]

    # Normalize nodata → NaN
    if src.nodata is not None:
        arr = arr.astype("float32", copy=False)
        arr[arr == src.nodata] = np.nan

    return (arr, out_transform) if return_transform else arr


# ==========================================================
# 3. Fast tile aggregation using precomputed lookup
# ==========================================================
def tile_tmean_fast(arr: np.ndarray, tile_lookup: np.ndarray) -> pd.DataFrame:
    """
    Aggregate a clipped + resampled monthly tmean raster to tiles.

    Parameters
    ----------
    arr : 2D np.ndarray (float)
        Monthly tmean raster, already clipped to CA and resampled to the
        same grid as `tile_lookup`.
    tile_lookup : 2D np.ndarray (int)
        tile_id grid where each pixel holds its tile_id, or 0 for "no tile".

    Returns
    -------
    DataFrame with columns:
        tile_id, tmean
    (year/month are added later in the driver script).
    """
    arr = arr.astype("float32", copy=False)

    # Extra safety: PRISM nodata is usually -9999
    arr[arr <= -9990] = np.nan

    # Only consider real tiles (tile_id > 0)
    mask_valid = (tile_lookup > 0)

    if not np.any(mask_valid):
        return pd.DataFrame(columns=["tile_id", "tmean"])

    df = pd.DataFrame({
        "tile_id": tile_lookup[mask_valid].astype("int32"),
        "tmean": arr[mask_valid]
    })

    df = df.groupby("tile_id", as_index=False)["tmean"].mean()
    return df
