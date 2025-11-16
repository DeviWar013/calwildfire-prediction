"""
joindata_yearly.py
---------------------------------------------------
Yearly Feature Assembly Pipeline

Description:
    Merges all feature modules—ERA5 climate, precipitation, temporal wind,
    ndvi, topography, vegetation, roads, landcover—with wildfire labels to
    build training data for a single year.

Inputs:
    • Per-tile climate and temporal parquet files
    • Static features.md: topo, ndvi, vegetation, landcover, roads
    • Fire events parquet (label source)

Outputs:
    • A complete yearly_train_YYYY.parquet file

Notes:
    This is one of the most critical modules; logic is preserved entirely.
"""


import pandas as pd, geopandas as gpd, gc
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "src" / "modeling" / "data" / "processed" / "training_yearly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["fire_index", "fire_occurred"]

# Data sources
SRC = {
    "fire":      BASE / "data" / "processed" / "fire" / "fire_events_tile_year.parquet",
    "tmean":     BASE / "data" / "processed" / "temporal_data" / "tmean_long.parquet",
    "ppt":       BASE / "data" / "processed" / "precipitation_data" / "ppt_long.parquet",
    "ndvi":      BASE / "data" / "processed" / "ndvi" / "ndvi_tile_month.parquet",
    "topo":      BASE / "data" / "processed" / "topographic_data" / "tile_topography_summary.parquet",
    "landcover": BASE / "data" / "processed" / "landcover_data" / "ca_tiles_landcover.shp",
    "roads":     BASE / "data" / "processed" / "roaddensity_data" / "tiled_roaddensity.shp",
}
ERA5_DIR = BASE / "data" / "processed" / "era5" / "tiled"
ERA5_FEATURES = ["src", "swvl1", "swvl2", "windspeed"]

# ============================================================
# HELPERS
# ============================================================
NLCD_CLASSES = {
    11: "Water", 12: "Ice", 21: "DevOpen", 22: "DevLow", 23: "DevMed",
    24: "DevHigh", 31: "Barren", 41: "DeForest", 42: "EvForest",
    43: "MixForest", 52: "Shrub", 71: "Grass", 81: "Pasture",
    82: "Crop", 90: "WoodyWet", 95: "HerbWet",
}

def read_table(path: Path):
    if path.suffix in [".shp", ".gpkg"]:
        gdf = gpd.read_file(path)
        return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

def load_era5(name: str, year: int) -> pd.DataFrame:
    """
    Read a wide ERA5 parquet and reshape only the 12 month columns
    for the requested year into long format (tile_id, year, month, value).
    NaNs are filled with 0.
    """
    import pandas as pd, pyarrow.parquet as pq, gc

    fp = ERA5_DIR / f"era5_{name}_tiled.parquet"
    print(f"Loading ERA5 feature for {year}: {fp.name}")

    # read schema
    pf = pq.ParquetFile(str(fp))
    all_cols = [c for c in pf.schema.names if c != "tile_id"]

    # keep only columns for this year (e.g., '1981-01-01 00:00:00')
    year_cols = [c for c in all_cols if c.startswith(str(year))]
    subcols = ["tile_id"] + year_cols

    df = pd.read_parquet(fp, columns=subcols)

    # ---- fill NaNs immediately ----
    df = df.fillna(0)

    # reshape wide → long
    long_ = df.melt(id_vars="tile_id", value_vars=year_cols,
                    var_name="time", value_name=name)
    del df; gc.collect()

    # extract year/month from timestamps
    long_["time"] = pd.to_datetime(long_["time"], errors="coerce")
    long_["year"] = long_["time"].dt.year.astype("int16")
    long_["month"] = long_["time"].dt.month.astype("int8")
    long_ = long_.drop(columns="time")

    # keep months 1–12 only
    long_ = long_[long_["month"].between(1, 12)]

    # ---- fill NaN again for safety ----
    long_[name] = long_[name].fillna(0).astype("float32")
    long_["tile_id"] = long_["tile_id"].astype("int32")

    # drop duplicates just in case
    long_ = long_.drop_duplicates(subset=["tile_id", "year", "month"], keep="first")

    print(f"  → {len(long_):,} rows for {year}, months={sorted(long_['month'].unique())}")
    return long_


def encode_domcover(df: pd.DataFrame) -> pd.DataFrame:
    if "DomCoverID" not in df.columns:
        return df
    df["DomCoverName"] = df["DomCoverID"].map(NLCD_CLASSES).fillna("Unknown")
    dummies = pd.get_dummies(df["DomCoverName"], prefix="DomCover")
    df = pd.concat([df.drop(columns=["DomCoverID","DomCoverName"]), dummies], axis=1)
    return df

def clean_static(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["lifeform","cwhr_type","cwhr_density",
                 "cover_type","structure","calveg_zone",
                 "DomPct","region","area_km2","length_m"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ============================================================
# MAIN JOIN FUNCTION
# ============================================================
def build_year(year: int):
    print(f"\n=== Building {year} ===")
    fire = pd.read_parquet(SRC["fire"])
    fire = fire[fire["year"]==year].copy()
    base = fire[["tile_id","year","month"] + LABEL_COLS].copy()

    # Climate
    for name in ["tmean","ppt"]:
        df = pd.read_parquet(SRC[name])
        df = df[df["year"]==year].copy()
        base = base.merge(df, on=["tile_id","year","month"], how="left")

    # ndvi
    ndvi = pd.read_parquet(SRC["ndvi"])
    ndvi = ndvi[ndvi["year"]==year].copy()
    base = base.merge(ndvi, on=["tile_id","year","month"], how="left")
    base["ndvi"] = base["ndvi"].fillna(0)

    # ERA5
    # ----- ERA5 integration -----
    for name in ["src", "swvl1", "swvl2", "windspeed"]:
        feat = load_era5(name, year).fillna(0)
        base = base.merge(feat, on=["tile_id", "year", "month"], how="left")
        base[name] = base[name].fillna(0).astype("float32")
    # Compute combined soil-water mean from layers 1 & 2
    if {"swvl1", "swvl2"} <= set(base.columns):
        base["swvl_mean"] = ((base["swvl1"] + base["swvl2"]) / 2).astype("float32")
        base = base.drop(columns=["swvl1", "swvl2"])

    # Static layers
    for key in ["topo","landcover","roads"]:
        layer = read_table(SRC[key])
        layer = layer.drop_duplicates(subset="tile_id")
        layer = clean_static(layer)
        if key == "landcover":
            layer = encode_domcover(layer)
        base = base.merge(layer, on="tile_id", how="left")

    # Fill and downcast
    base = base.fillna(0)
    for c in base.select_dtypes("float64"):
        base[c] = base[c].astype("float32")
    for c in base.select_dtypes("int64"):
        base[c] = base[c].astype("int32")
    base = base[base["month"].between(1, 12)]
    base = base.drop_duplicates(subset=["tile_id", "year", "month"], keep="first")

    # Save
    out_fp = OUT_DIR / f"training_{year}.parquet"
    base.to_parquet(out_fp, index=False)
    print(f"Saved {out_fp.name} — {len(base):,} rows, {len(base.columns)} cols")
    del base; gc.collect()

# ============================================================
# RUNNER
# ============================================================
def main():
    for y in range(1981, 2024):
        build_year(y)
    print("\nAll yearly datasets generated and schema standardized.")

if __name__ == "__main__":
    main()
