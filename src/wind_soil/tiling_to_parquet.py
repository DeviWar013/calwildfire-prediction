# tiling_to_parquet.py
import xarray as xr, geopandas as gpd, pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from pathlib import Path
import os, time

print("=== ERA5 → Tile nearest-grid assignment ===")

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tile_path = os.path.join(BASE_DIR, "pdata/processed/tiled_ca/ca_tiles_1km.shp")
era5_dir = os.path.join(BASE_DIR, "pdata/processed/era5/clipped")
out_dir = os.path.join(BASE_DIR, "pdata/processed/era5/tiled")
os.makedirs(out_dir, exist_ok=True)

# === Load tile centroids ===
print("Loading tiles...")
tiles = gpd.read_file(tile_path).to_crs("EPSG:3310")
centroids = np.column_stack((tiles.geometry.centroid.x, tiles.geometry.centroid.y))
tile_ids = tiles["tile_id"].values
print(f"Tiles loaded: {len(tile_ids)}")

# === Helper: nearest-neighbor extraction ===
def assign_nearest(ds: xr.Dataset, var: str):
    print(f"Assigning variable: {var}")
    t0 = time.time()

    # ERA5 grid coordinates
    xv, yv = np.meshgrid(ds.x.values, ds.y.values)
    points = np.column_stack((xv.ravel(), yv.ravel()))
    tree = cKDTree(points)

    # Match tiles → nearest ERA5 grid cell
    dist, idx = tree.query(centroids)
    ny, nx = len(ds.y), len(ds.x)
    yy, xx = np.unravel_index(idx, (ny, nx))

    # Extract all time steps
    values = ds[var].values[:, yy, xx]  # shape: (time, n_tiles)

    # Build DataFrame
    df = pd.DataFrame(values.T, columns=pd.to_datetime(ds.time.values))
    df.insert(0, "tile_id", tile_ids)
    print(f"  Done {var}: {time.time()-t0:.2f}s")
    return df

# === Soil layers ===
for vfile, var in [("era5_swvl1_clipped.nc", "swvl1"),
                   ("era5_swvl2_clipped.nc", "swvl2"),
                   ("era5_src_clipped.nc",   "src")]:
    path = os.path.join(era5_dir, vfile)
    if not os.path.exists(path):
        print(f"Skipping missing file: {vfile}")
        continue
    ds = xr.open_dataset(path)
    df = assign_nearest(ds, var)
    out_path = os.path.join(out_dir, f"era5_{var}_tiled.parquet")
    df.to_parquet(out_path)
    print(f"Saved {var} → {out_path}")

# === Wind components ===
wind_path = os.path.join(era5_dir, "era5_wind_clipped.nc")
if os.path.exists(wind_path):
    ds = xr.open_dataset(wind_path)
    print("Assigning wind components...")

    df_u = assign_nearest(ds, "u10")
    df_v = assign_nearest(ds, "v10")

    # Compute wind speed magnitude per cell/time
    print("Computing wind speed magnitude...")
    arr_u = df_u.drop(columns="tile_id").values
    arr_v = df_v.drop(columns="tile_id").values
    arr_speed = np.sqrt(arr_u**2 + arr_v**2)
    df_speed = pd.DataFrame(arr_speed, columns=df_u.columns[1:])
    df_speed.insert(0, "tile_id", tile_ids)

    # Save all
    df_u.to_parquet(os.path.join(out_dir, "era5_u10_tiled.parquet"))
    df_v.to_parquet(os.path.join(out_dir, "era5_v10_tiled.parquet"))
    df_speed.to_parquet(os.path.join(out_dir, "era5_windspeed_tiled.parquet"))
    print("Saved wind components & speed.")
else:
    print("No wind dataset found — skipping.")

print("\n=== All ERA5 variables assigned to tiles successfully ===")
