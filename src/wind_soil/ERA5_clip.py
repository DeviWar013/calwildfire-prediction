# ERA5_clip_to_CA_v2.py
import xarray as xr
import geopandas as gpd
from pathlib import Path
import rioxarray

# === Directories ===
BASE = Path(__file__).resolve().parents[2]
proc_dir = BASE/"data/processed/era5"
clip_dir = proc_dir / "clipped"
clip_dir.mkdir(parents=True, exist_ok=True)

# === Load California boundary ===
ca_path = BASE/"data/raw/ca_state/CA_State.shp"
ca = gpd.read_file(ca_path).to_crs("EPSG:3310")

# === Helper function ===
def clip_dataset(filename: str):
    fpath = proc_dir / filename
    print(f"\n--- Clipping {filename} ---")

    ds = xr.open_dataset(fpath)

    # Drop non-spatial coordinates that confuse rioxarray
    drop_vars = [v for v in ["number", "step", "depthBelowLandLayer"] if v in ds.coords]
    if drop_vars:
        print(f"  Dropping coords: {drop_vars}")
        ds = ds.drop_vars(drop_vars)

    # Rename valid_time → time if needed
    if "valid_time" in ds.coords and "time" not in ds.coords:
        print("  Renaming 'valid_time' → 'time'")
        ds = ds.rename({"valid_time": "time"})
    # Standardize coordinate names so rioxarray sees spatial grid
    if {"longitude", "latitude"} <= set(ds.coords):
        ds = ds.rename({"longitude": "x", "latitude": "y"})
    elif {"lon", "lat"} <= set(ds.coords):
        ds = ds.rename({"lon": "x", "lat": "y"})

    ds = xr.open_dataset(fpath)

    # --- Clean metadata ---
    for bad in ["number", "step", "surface", "depthBelowLandLayer"]:
        if bad in ds.coords:
            ds = ds.drop_vars(bad)

    # --- Ensure spatial axes exist ---
    if {"longitude", "latitude"} <= set(ds.coords):
        ds = ds.rename({"longitude": "x", "latitude": "y"})
    elif {"lon", "lat"} <= set(ds.coords):
        ds = ds.rename({"lon": "x", "lat": "y"})

    # --- Attach CRS ---
    ds = ds.rio.write_crs("EPSG:4326")

    # --- Optional reprojection to match CA polygon ---
    ds_proj = ds.rio.reproject("EPSG:3310")

    # --- Clip ---
    ca = gpd.read_file(ca_path).to_crs("EPSG:3310")
    ds_clip = ds_proj.rio.clip(ca.geometry, ca.crs)

    # Save to new file
    out_name = filename.replace("_raw", "_clipped")
    out_path = clip_dir / out_name
    ds_clip.to_netcdf(out_path, engine="netcdf4")
    print(f"  ✔ Saved clipped dataset to: {out_path}")
    print(ds_clip.dims)

# === Apply to all ERA5 files ===
file_list = [
    "era5_wind_raw.nc",
    "era5_swvl1_raw.nc",
    "era5_swvl2_raw.nc",
    "era5_src_raw.nc",
]

for f in file_list:
    clip_dataset(f)

print("\nAll datasets clipped successfully.")

