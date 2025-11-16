"""
==============================================================================
NDVI → TILE-MONTH PARQUET CONVERSION
==============================================================================

This script converts processed yearly NDVI rasters into a tile-level dataset.
Each tile centroid (from the 1 km California grid) is sampled against the
monthly NDVI raster stack to produce a long-format table:

    tile_id | year | month | ndvi

PRIMARY OUTPUT:
    data/processed/NDVI/ndvi_tile_month.parquet

WORKFLOW SUMMARY:
1. Load tile grid (1 km).
2. Load yearly NDVI NetCDF files.
3. Extract monthly NDVI values at each tile centroid via nearest-neighbor.
4. Build a unified long-format DataFrame.
5. Save as a parquet file for downstream model integration.

NOTE:
This script does *not* merge NDVI into training datasets.
Joining is handled exclusively in the modeling module for consistency.

==============================================================================
"""

import xarray as xr
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ==============================
# ndvi → tile×month integration
# ==============================
BASE = Path(__file__).resolve().parents[2]
NDVI_DIR = BASE/"data/processed/NDVI"
TILES_FP = BASE/"data/processed/tiled_ca/ca_tiles_1km.shp"
OUT_PARQ = BASE/"data/processed/NDVI/ndvi_tile_month.parquet"

# --- load tiles & prepare centroids ---
tiles = gpd.read_file(TILES_FP).to_crs(4326)
tiles["lat"] = tiles.geometry.centroid.y
tiles["lon"] = tiles.geometry.centroid.x
tiles = tiles[["tile_id", "lat", "lon"]]

dfs = []

# --- iterate through yearly ndvi files ---
for fp in sorted(NDVI_DIR.glob("NDVI_CA_*.nc")):
    print(f"Processing {fp.name} ...")
    ds = xr.open_dataset(fp)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    var_name = [v for v in ds.data_vars if "ndvi" in v.upper()][0]

    # build temporary Dataset for vectorized sampling
    pts = xr.Dataset({
        lat_name: (("points",), tiles["lat"].values),
        lon_name: (("points",), tiles["lon"].values),
    })

    # sample ndvi at tile centroids (nearest pixel)
    sample = ds[var_name].interp({lat_name: pts[lat_name], lon_name: pts[lon_name]}, method="nearest")

    # convert to DataFrame
    df = (pd.DataFrame(sample.values.T, columns=pd.to_datetime(ds.time.values))
            .assign(tile_id=tiles["tile_id"].values))
    df = df.melt(id_vars="tile_id", var_name="time", value_name="ndvi")
    dfs.append(df)

# --- combine all years ---
ndvi_tbl = pd.concat(dfs, ignore_index=True).sort_values(["tile_id", "time"]).reset_index(drop=True)
ndvi_tbl["time"] = pd.to_datetime(ndvi_tbl["time"], errors="coerce")

# normalize to year–month level
ndvi_tbl["year"] = ndvi_tbl["time"].dt.year
ndvi_tbl["month"] = ndvi_tbl["time"].dt.month
ndvi_tbl = ndvi_tbl[["tile_id", "year", "month", "ndvi"]]

# --- save as Parquet ---
OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
ndvi_tbl.to_parquet(OUT_PARQ, index=False)
print(f" ndvi tile×month table saved to: {OUT_PARQ}")
print(f"Rows: {len(ndvi_tbl):,} | Tiles: {ndvi_tbl.tile_id.nunique():,} | Months: {ndvi_tbl.month.nunique():,}")
