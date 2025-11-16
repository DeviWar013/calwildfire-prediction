# precip_utils.py
# ---------------------------------------------------------
# Shared helper functions for PRISM precipitation processing
# (modernized + aligned with legacy clipped raster logic)

import re
import datetime as dt
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio import features
from shapely.geometry import mapping
import pandas as pd


# ==========================================================
# 1. Parse PRISM date from filename
# ==========================================================
def parsedate_ppt(path: str) -> dt.date:
    """Extract YYYYMM from PRISM file name (robust across .bil/.tif variants)."""
    fname = str(path).split("/")[-1]
    m = re.search(r"_(\d{6})(?:_bil)?(?:_CA)?(?:_1km)?\.(?:bil|tif)$", fname)
    if not m:
        raise ValueError(f"Cannot parse date from {fname}")
    y, mth = int(m.group(1)[:4]), int(m.group(1)[4:6])
    return dt.date(y, mth, 1)


# ==========================================================
# 2. Clip raster to California
# ==========================================================
def clip_to_ca(src, ca_path: str):
    """Clip an open rasterio dataset to California boundary (returns array, transform)."""
    ca = gpd.read_file(ca_path).to_crs(src.crs)
    geoms = [mapping(geom) for geom in ca.geometry]
    out_image, out_transform = mask(src, geoms, crop=True)
    return out_image[0], out_transform


# ==========================================================
# 3. Build tile lookup from a CLIPPED raster (legacy-aligned)
# ==========================================================
def build_lookup_from_clipped(clipped_sample_path, tiles_shp_path, out_npy_path):
    """
    Build tile_id lookup grid *using the clipped raster* as spatial reference.

    This reproduces legacy pipeline behavior
    (shape typically ~1056 × 1158 depending on PRISM month).
    """
    import numpy as np

    tiles = gpd.read_file(tiles_shp_path)

    with rasterio.open(clipped_sample_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform

        tile_shapes = [(geom, tid)
                       for geom, tid in zip(tiles.geometry, tiles.tile_id)]

        lookup = features.rasterize(
            tile_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="int32"
        )

    np.save(out_npy_path, lookup)
    print(f"[Lookup] Saved tile lookup grid → {out_npy_path}")
    return lookup


# ==========================================================
# 4. Fast per-tile precipitation (matches legacy)
# ==========================================================
def tile_precip_fast(arr, tile_lookup):
    import pandas as pd

    arr = arr.astype("float32")
    arr[arr <= -9990] = np.nan  # convert nodata to NaN

    # all real tiles
    all_tiles = pd.DataFrame({"tile_id": np.unique(tile_lookup[tile_lookup > 0])})

    # include all pixels, even NaN-containing tiles
    mask_valid = (tile_lookup > 0)

    df = pd.DataFrame({
        "tile_id": tile_lookup[mask_valid].astype("int32"),
        "ppt_mm": arr[mask_valid]
    }).groupby("tile_id", as_index=False)["ppt_mm"].mean()

    # ensure tiles with all-NaN become NaN rows
    df = all_tiles.merge(df, on="tile_id", how="left")

    return df

