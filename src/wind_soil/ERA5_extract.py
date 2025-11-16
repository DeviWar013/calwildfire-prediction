import cfgrib
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
path = BASE/"data/raw/era5/wind_soil_data.grib"
out_dir = BASE/"data/processed/era5"
out_dir.mkdir(parents=True, exist_ok=True)

ds_list = cfgrib.open_datasets(path)
print(f"{len(ds_list)} groups found")

names = ["swvl1", "swvl2", "wind", "src"]
for i, ds in enumerate(ds_list):
    out_file = out_dir / f"era5_{names[i]}_raw.nc"
    ds.to_netcdf(out_file, engine="netcdf4")
    print(f"✔ Saved group {i} ({names[i]}) → {out_file}")