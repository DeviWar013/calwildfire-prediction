"""
# precip_process.py
# ---------------------------------------------------------
# Modernized precipitation pipeline
# Output: ppt_long.parquet
#
# Logic (aligned with legacy):
#   1. Take FIRST clipped raster as spatial reference → build lookup
#   2. Loop through raw PRISM data
#   3. Clip → resample (1 km) → aggregate per tile
#   4. Save unified long-format DataFrame
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling

from precip_utils import (
    parsedate_ppt,
    clip_to_ca,
    tile_precip_fast,
    build_lookup_from_clipped
)

# ==========================================================
# Paths
# ==========================================================
BASE = Path(__file__).resolve().parents[2]

RAW_DIR = BASE / "data/raw/precipitation_data/PRISM_ppt_stable_4kmM3_198101_202409_bil"
CA_SHP = BASE / "data/raw/ca_state/CA_State.shp"
TILES_SHP = BASE / "data/processed/tiled_ca/ca_tiles_1km.shp"

CLIPPED_REF_DIR = BASE / "data/processed/precipitation_data/clipped_reference"
LOOKUP_PATH = BASE / "data/processed/precipitation_data/tile_id_grid.npy"

OUT_PATH = BASE / "data/processed/precipitation_data/ppt_long.parquet"

TARGET_RES = 0.009   # approx 1 km
START_YEAR, END_YEAR = 1981, 2024


# ==========================================================
# Build lookup from FIRST clipped raster
# ==========================================================
def ensure_lookup_ready():
    """
    Build tile lookup from *upsampled* clipped raster (1 km),
    ensuring alignment with the upsampled precipitation arrays.
    """
    if LOOKUP_PATH.exists():
        print("[Lookup] Existing lookup found.")
        return np.load(LOOKUP_PATH)

    # Prepare folder for reference rasters
    CLIPPED_REF_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pick the FIRST raw PRISM .bil raster
    first_raw = sorted(RAW_DIR.glob("*.bil"))[0]

    # 2. Clip it to California
    with rasterio.open(first_raw) as src:
        arr_clip, tr_clip = clip_to_ca(src, CA_SHP)

    # 3. Upsample clipped raster to 1 km grid (same as main pipeline)
    transform, width, height = calculate_default_transform(
        src.crs, src.crs,
        arr_clip.shape[1], arr_clip.shape[0],
        *rasterio.transform.array_bounds(arr_clip.shape[0], arr_clip.shape[1], tr_clip),
        resolution=(TARGET_RES, TARGET_RES)
    )

    upsampled = np.empty((height, width), dtype=np.float32)
    reproject(
        source=arr_clip,
        destination=upsampled,
        src_transform=tr_clip,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=src.crs,
        resampling=Resampling.nearest
    )

    # Save reference upsampled raster
    ref_tif_path = CLIPPED_REF_DIR / "ref_clip_1km.tif"
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": upsampled.dtype,
        "crs": src.crs,
        "transform": transform
    }
    with rasterio.open(ref_tif_path, "w", **meta) as dst:
        dst.write(upsampled, 1)

    # 4. Build lookup from the *upsampled* reference raster
    return build_lookup_from_clipped(
        clipped_sample_path=str(ref_tif_path),
        tiles_shp_path=str(TILES_SHP),
        out_npy_path=str(LOOKUP_PATH)
    )



# ==========================================================
# MAIN PROCESSING
# ==========================================================
def run_precip_pipeline():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Ensure lookup exists (or build it)
    tile_lookup = ensure_lookup_ready()

    records = []
    FULL_TILE_IDS = gpd.read_file(TILES_SHP)["tile_id"].unique()

    # 2. Process each raw .bil file
    for fname in sorted(RAW_DIR.glob("*.bil")):
        try:
            date_value = parsedate_ppt(fname)
        except ValueError:
            continue

        if not (START_YEAR <= date_value.year <= END_YEAR):
            continue

        with rasterio.open(fname) as src:
            # 2.1 Cli

            arr_clip, tr_clip = clip_to_ca(src, CA_SHP)

            # 2.2 Resample to 1 km grid
            transform, width, height = calculate_default_transform(
                src.crs, src.crs,
                arr_clip.shape[1], arr_clip.shape[0],
                *rasterio.transform.array_bounds(arr_clip.shape[0], arr_clip.shape[1], tr_clip),
                resolution=(TARGET_RES, TARGET_RES)
            )

            upsampled = np.empty((height, width), dtype=np.float32)
            reproject(
                source=arr_clip,
                destination=upsampled,
                src_transform=tr_clip,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )

            # 2.3 Tile aggregation
            df = tile_precip_fast(upsampled, tile_lookup)

            # FULL TILE SET from SHAPEFILE (426,644 tiles)
            all_tiles = pd.DataFrame({"tile_id": FULL_TILE_IDS})

            # Build month skeleton
            skeleton = all_tiles.copy()
            skeleton["year"] = date_value.year
            skeleton["month"] = date_value.month

            # Merge precipitation data onto the skeleton
            df = skeleton.merge(df, on="tile_id", how="left")

            records.append(df)

    # 3. Combine + save
    if not records:
        print("No PRISM rasters processed.")
        return

    df_all = pd.concat(records, ignore_index=True)
    df_all.to_parquet(OUT_PATH, index=False)
    print(f"[Done] Saved → {OUT_PATH} ({len(df_all):,} rows)")


if __name__ == "__main__":
    run_precip_pipeline()
