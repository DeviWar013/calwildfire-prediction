"""
==============================================================================
NDVI DOWNLOAD & PREPROCESSING PIPELINE
==============================================================================

This script downloads raw NDVI data (NetCDF format) from NOAA’s CDR archive,
clips the raster to California’s geographic extent, scales NDVI values to the
expected [-1, 1] range, aggregates daily NDVI into monthly composites, and
saves one clipped-and-processed NetCDF file per year.

PRIMARY OUTPUT:
    data/processed/NDVI/NDVI_CA_<YEAR>.nc

WORKFLOW SUMMARY:
1. Fetch daily NDVI rasters from NOAA.
2. Clip each raster to the California boundary.
3. Convert raw NDVI values (0–255) to standardized NDVI (-1 to 1).
4. Aggregate daily rasters into monthly mean NDVI.
5. Save processed NDVI for yearly integration.

This module produces *raster NDVI* only.
Sampling NDVI onto the tile grid is handled in `integrate_parquet_NDVI.py`.

==============================================================================
"""
import os
import requests
import xarray as xr
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from tempfile import TemporaryDirectory

# NOAA ndvi Version 5 – direct download access
BASE_URL = "https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access"
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE/"data/processed/NDVI"

# California bounding box
LAT_NORTH, LAT_SOUTH = 42.0, 32.0
LON_WEST, LON_EAST = -125.0, -113.0

YEARS = range(1981, 2024)  # test range


def list_noaa_files(year: int):
    """Scrape all .nc filenames from NOAA index page for the given year."""
    url = f"{BASE_URL}/{year}/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)]
    nc_files = [lnk for lnk in links if lnk.endswith(".nc")]
    return [f"{url}{f}" for f in nc_files]


def download_file(url: str, out_path: str):
    """Download file with stream and progress info."""
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        raise Exception(f"Failed to download {url}")
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def process_year(year: int):
    urls = list_noaa_files(year)
    if not urls:
        print(f"No files found for {year}")
        return

    print(f"Found {len(urls)} daily files for {year}. Starting aggregation...")

    ndvi_monthly = []

    # Create temp cache
    with TemporaryDirectory() as tmpdir:
        for i, file_url in enumerate(urls):
            local_path = os.path.join(tmpdir, os.path.basename(file_url))
            try:
                download_file(file_url, local_path)
                with xr.open_dataset(local_path) as ds:  # auto-close
                    var_name = [v for v in ds.data_vars if "ndvi" in v.upper()]
                    if not var_name:
                        continue
                    ndvi = ds[var_name[0]]

                    # scaling
                    if ndvi.max() > 2:
                        ndvi = ndvi * 0.0001

                    # handle coord naming
                    lat_name = "lat" if "lat" in ndvi.coords else "latitude"
                    lon_name = "lon" if "lon" in ndvi.coords else "longitude"

                    ndvi = ndvi.sel({lat_name: slice(LAT_NORTH, LAT_SOUTH),
                                     lon_name: slice(LON_WEST, LON_EAST)})

                    ndvi_monthly.append(ndvi)

                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(urls)} files...")
            except Exception as e:
                print(f"Skipping {file_url} ({e})")
                continue

    if not ndvi_monthly:
        print(f"No valid ndvi pdata for {year}")
        return

    # Concatenate daily → monthly mean
    ds_concat = xr.concat(ndvi_monthly, dim="time")
    ds_monthly = ds_concat.resample(time="1M").mean()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"NDVI_CA_{year}.nc")
    ds_monthly.to_netcdf(out_path)
    print(f"Saved {out_path}")


def main():
    for year in YEARS:
        process_year(year)
    print("\nAll done.")


if __name__ == "__main__":
    main()
