"""
process_topography.py
-------------------------------------
Rebuilds the topographic features.md for the California wildfire ML dataset.

Pipeline:
  1. Merge raw DEM tiles (USGS) → merged_dem.tif
  2. Clip merged DEM to California boundary → dem_clipped_ca.tif
  3. Reproject to EPSG:3310 (meters) → dem_clipped_ca_meters.tif
  4. Compute slope (degrees, 30 m resolution) → dem_slope_ca_meters.tif
  5. Aggregate elevation & slope to 1 km tiles → tile_topography_summary.parquet

Inputs:
  data/raw/topographic_data/*.tif       (raw DEM tiles)
  data/processed/tiled_ca/ca_tiles_1km.shp
  data/processed/boundaries/ca_boundary.shp

Outputs:
  data/processed/topographic_data/dem_clipped_ca_meters.tif
  data/processed/topographic_data/dem_slope_ca_meters.tif
  data/processed/topographic_data/tile_topography_summary.parquet

Author: Desmond Wang
"""

import os
import time
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DEM_DIR = BASE_DIR / "data" / "raw" / "topographic_data"
PROC_DIR = BASE_DIR / "data" / "processed" / "topographic_data"
CA_BOUNDARY = BASE_DIR / "data" / "processed" / "boundaries" / "ca_boundary.shp"
TILES_SHP = BASE_DIR / "data" / "processed" / "tiled_ca" / "ca_tiles_1km.shp"

DEM_MERGED = PROC_DIR / "merged_dem.tif"
DEM_CLIPPED = PROC_DIR / "dem_clipped_ca.tif"
DEM_METERS = PROC_DIR / "dem_clipped_ca_meters.tif"
SLOPE_TIF = PROC_DIR / "dem_slope_ca_meters.tif"
SUMMARY_PARQ = PROC_DIR / "tile_topography_summary.parquet"

os.makedirs(PROC_DIR, exist_ok=True)

# ============================================================
# STEP 1. Merge and clip DEM
# ============================================================

def merge_and_clip_dem(raw_dir, boundary_path, merged_out, clipped_out):
    print("Merging DEM tiles ...")
    src_files = [rasterio.open(p) for p in Path(raw_dir).glob("*.tif")]
    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    with rasterio.open(merged_out, "w", **out_meta) as dst:
        dst.write(mosaic)
    [src.close() for src in src_files]
    print(f"Merged DEM saved: {merged_out}")

    # Clip to California
    print("Clipping to California boundary ...")
    boundary = gpd.read_file(boundary_path)
    with rasterio.open(merged_out) as src:
        out_img, out_transform = mask(src, boundary.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform
        })
        with rasterio.open(clipped_out, "w", **out_meta) as dst:
            dst.write(out_img)
    print(f"✅ Clipped DEM saved: {clipped_out}")
    return clipped_out


# ============================================================
# STEP 2. Reproject DEM to meters
# ============================================================

def reproject_to_meters(src_tif, dst_tif, dst_crs="EPSG:3310"):
    with rasterio.open(src_tif) as src:
        if src.crs.to_string() == dst_crs:
            print("DEM already in EPSG:3310.")
            return src_tif
        print("Reprojecting DEM to EPSG:3310 ...")
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": src.nodata
        })
        with rasterio.open(dst_tif, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear)
    print(f"Saved reprojected DEM: {dst_tif}")
    return dst_tif


# ============================================================
# STEP 3. Compute slope raster (chunked, safe memory)
# ============================================================

def compute_slope(dem_path, slope_out, chunk_size=2000):
    with rasterio.open(dem_path) as src:
        profile = src.profile
        profile.update(dtype="float32", count=1)
        nodata = src.nodata
        dx, dy = src.transform.a, abs(src.transform.e)
        height, width = src.height, src.width

        with rasterio.open(slope_out, "w", **profile) as dst:
            print(f"⚙️  Computing slope in chunks ({chunk_size}×{chunk_size}) ...")
            for row_off in range(0, height, chunk_size):
                for col_off in range(0, width, chunk_size):
                    win = Window(
                        col_off=col_off, row_off=row_off,
                        width=min(chunk_size, width - col_off),
                        height=min(chunk_size, height - row_off))
                    arr = src.read(1, window=win, masked=True).astype("float32")

                    # Clean NoData
                    if np.ma.is_masked(arr):
                        arr = arr.filled(np.nan)
                    if nodata is not None:
                        arr[arr == nodata] = np.nan
                    arr[arr < -10000] = np.nan

                    dzdy, dzdx = np.gradient(arr, dy, dx)
                    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
                    slope_deg = np.degrees(slope_rad).astype("float32")
                    dst.write(np.clip(slope_deg, 0, 90), 1, window=win)
            print(f"Slope raster created: {slope_out}")
    return slope_out


# ============================================================
# STEP 4. Summarize elevation & slope per 1 km tile
# ============================================================

def summarize_to_tiles(dem_path, slope_path, tile_path, out_parq):
    print("Summarizing elevation and slope per tile ...")
    tiles = gpd.read_file(tile_path).to_crs("EPSG:3310")
    centroids = tiles.geometry.centroid
    results = []

    with rasterio.open(dem_path) as elev_src, rasterio.open(slope_path) as slope_src:
        elev_arr = elev_src.read(1)
        slope_arr = slope_src.read(1)
        px_size = elev_src.res[0]
        half_win = int(500 / px_size)

        for i, (pt, tid) in enumerate(zip(centroids, tiles["tile_id"])):
            r, c = elev_src.index(pt.x, pt.y)
            row_min, row_max = max(r - half_win, 0), min(r + half_win, elev_arr.shape[0])
            col_min, col_max = max(c - half_win, 0), min(c + half_win, elev_arr.shape[1])

            elev_win = elev_arr[row_min:row_max, col_min:col_max].astype("float32")
            slope_win = slope_arr[row_min:row_max, col_min:col_max].astype("float32")

            elev_win[elev_win < -10000] = np.nan
            slope_win[slope_win < -10000] = np.nan

            elev_mean = float(np.nanmean(elev_win))
            slope_mean = float(np.nanmean(slope_win))
            results.append((tid, elev_mean, slope_mean))

            if i % 5000 == 0 and i > 0:
                print(f"   → Processed {i:,} tiles")

    df = pd.DataFrame(results, columns=["tile_id", "elev_mean", "slope_mean"])
    df.to_parquet(out_parq, index=False)
    print(f"Saved summary parquet: {out_parq}")
    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    t0 = time.time()
    print("=== Topography  processing started ===")

    # Step 1: Merge + clip (skip if already done)
    if not DEM_CLIPPED.exists():
        merge_and_clip_dem(RAW_DEM_DIR, CA_BOUNDARY, DEM_MERGED, DEM_CLIPPED)

    # Step 2: Reproject
    dem_proj = reproject_to_meters(DEM_CLIPPED, DEM_METERS)

    # Step 3: Compute slope
    slope_tif = compute_slope(dem_proj, SLOPE_TIF)

    # Step 4: Summarize to 1 km tiles
    summarize_to_tiles(dem_proj, slope_tif, TILES_SHP, SUMMARY_PARQ)

    print(f"\ Done! Total time: {(time.time()-t0)/60:.1f} min")
