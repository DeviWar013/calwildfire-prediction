"""
temporal_process.py
---------------------------------------------------------
Modernized PRISM tmean pipeline (GitHub-ready).

Behavior mirrors precip_process.py:

  1. Assume a precomputed tile_id lookup grid (tile_id_grid.npy)
     that is aligned with the 1 km PRISM grid
     (produced by the precipitation pipeline).

  2. Loop through raw PRISM tmean .bil rasters:
       - Clip to California
       - Resample to ~1 km grid (same TARGET_RES as precipitation)
       - Aggregate to tiles using tile_id lookup
       - Restore full tile × month skeleton using all tile_ids
         from the 1 km tiles shapefile

  3. Save unified long-format DataFrame:
       columns = [tile_id, tmean, year, month]
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from temporal_utils import parsedate_tmean, clip_to_ca, tile_tmean_fast


# ==========================================================
# Paths & constants
# ==========================================================
BASE = Path(__file__).resolve().parents[2]

RAW_DIR = BASE / "data/raw/temporal_data/PRISM_tmean_stable_4kmM3_198101_202409_bil"
CA_SHP = BASE / "data/raw/ca_state/CA_State.shp"
TILES_SHP = BASE / "data/processed/tiled_ca/ca_tiles_1km.shp"

# Reuse the SAME lookup grid that precipitation built
LOOKUP_PATH = BASE / "data/processed/precipitation_data/tile_id_grid.npy"

OUT_PATH = BASE / "data/processed/temporal_data/tmean_long.parquet"

# Same target resolution as precipitation (approx 1km in degrees)
TARGET_RES = 0.009
START_YEAR, END_YEAR = 1981, 2024

# Load full tile universe once (426,644 tiles; no tile_id 0)
FULL_TILE_IDS = gpd.read_file(TILES_SHP)["tile_id"].unique()


# ==========================================================
# Helpers
# ==========================================================
def load_lookup() -> np.ndarray:
    """
    Load the precomputed tile lookup grid.

    The lookup is produced by precip_process.py and saved to:
        data/processed/precipitation_data/tile_id_grid.npy

    If it doesn't exist, the user should run the precipitation
    pipeline first.
    """
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Lookup grid not found at {LOOKUP_PATH}.\n"
            "Run the precipitation pipeline first to generate tile_id_grid.npy."
        )
    print("[Lookup] Loaded existing tile lookup grid.")
    return np.load(LOOKUP_PATH)


# ==========================================================
# Main processing
# ==========================================================
def run_temporal_pipeline():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load lookup grid
    tile_lookup = load_lookup()

    records: list[pd.DataFrame] = []

    # 2. Process each raw tmean .bil file
    for fname in sorted(RAW_DIR.glob("*.bil")):
        try:
            date_value = parsedate_tmean(fname)
        except ValueError:
            # Skip non-monthly files
            continue

        if not (START_YEAR <= date_value.year <= END_YEAR):
            continue

        with rasterio.open(fname) as src:
            # 2.1 Clip to California
            arr_clip, tr_clip = clip_to_ca(src, CA_SHP)

            # 2.2 Resample to TARGET_RES (~1 km) in the same CRS
            transform, width, height = calculate_default_transform(
                src.crs,
                src.crs,
                arr_clip.shape[1],
                arr_clip.shape[0],
                *rasterio.transform.array_bounds(
                    arr_clip.shape[0], arr_clip.shape[1], tr_clip
                ),
                resolution=(TARGET_RES, TARGET_RES),
            )

            upsampled = np.empty((height, width), dtype=np.float32)
            reproject(
                source=arr_clip,
                destination=upsampled,
                src_transform=tr_clip,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest,
            )

        # 2.3 Tile aggregation to tmean per tile
        df = tile_tmean_fast(upsampled, tile_lookup)

        # 2.4 Restore FULL tile × month skeleton (like precipitation)
        all_tiles = pd.DataFrame({"tile_id": FULL_TILE_IDS})
        skeleton = all_tiles.copy()
        skeleton["year"] = date_value.year
        skeleton["month"] = date_value.month

        df = skeleton.merge(df, on="tile_id", how="left")

        records.append(df)

    # 3. Combine + save
    if not records:
        print("No PRISM tmean rasters processed.")
        return

    df_all = pd.concat(records, ignore_index=True)
    df_all.to_parquet(OUT_PATH, index=False)
    print(f"[Done] Saved → {OUT_PATH} ({len(df_all):,} rows)")


if __name__ == "__main__":
    run_temporal_pipeline()
