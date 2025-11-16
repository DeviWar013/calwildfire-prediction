# ERA5_dataload.py
import cfgrib
import xarray as xr
from pprint import pprint
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
path = BASE/"data/raw/era5/wind_soil_data.grib"

print("=== Listing available groups in the GRIB file ===")
groups = cfgrib.open_datasets(path)
pprint(groups)
print(f"\nTotal groups found: {len(groups)}\n")

datasets = {}

# === Try opening each group separately ===
for i, g in enumerate(groups):
    try:
        name = g["ids"]["typeOfLevel"]
    except KeyError:
        name = f"group_{i}"

    print(f"--- Loading group: {name} ---")
    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": g["filter_by_keys"]}
        )
        datasets[name] = ds
        print(f"Variables: {list(ds.data_vars.keys())}")
        print(f"Dimensions: {dict(ds.dims)}")
        print("-" * 50)
    except Exception as e:
        print(f"Skipped group {name}: {e}")

print(f"\nSuccessfully loaded {len(datasets)} groups:")
print(list(datasets.keys()))

# Optional: show a quick summary of the main groups
for name, ds in datasets.items():
    print(f"\nSummary for '{name}':")
    print(ds)
