# ----------------------------------------------------------
# fire_utils.py
# ----------------------------------------------------------
"""
Wildfire Utility Functions (LEGACY — logic preserved)

This module contains the **core logic** used by both fire_clean.py and
fire_process.py. Logic is intentionally left unchanged to maintain perfect
reproducibility with the original wildfire module.

Notes
-----
• tile_fire_perimeters uses overlay(), not sjoin(), because overlay is
  geometrically robust and required for correct polygon slicing.
• Saving negative geometry to GPKG is NOT supported (huge memory).
• All data transformations follow the legacy behavior exactly.
"""
import pandas as pd
import geopandas as gpd
from shapely import make_valid
from matplotlib import pyplot as plt
import os

def parse_date_cur(val):
    """Normalize DATE_CUR strings of various lengths into datetime objects."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if len(s) == 11: s = s[:8] + "0" + s[8:]
    if len(s) == 10: s = s + "00"
    if len(s) == 9:  s = s + "000"
    try:
        if len(s) == 12:
            return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")
        elif len(s) == 8:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        elif len(s) == 4:
            return pd.to_datetime(s, format="%Y", errors="coerce")
    except Exception:
        return pd.NaT
    return pd.NaT


def normalize_fire_dates(gdf):
    """Apply parse_date_cur to DATE_CUR and add a DATE_CUR_parsed column."""
    gdf["DATE_CUR_parsed"] = gdf["DATE_CUR"].apply(parse_date_cur)
    valid_pct = gdf["DATE_CUR_parsed"].notna().mean().round(3) * 100
    print(f"Parsed DATE_CUR coverage: {valid_pct}% valid")
    return gdf


def clip_fire_to_california(fire_fp, ca_fp):
    """Load, reproject, and spatially clip fire perimeters to California."""
    fire = gpd.read_file(fire_fp)
    ca = gpd.read_file(ca_fp)

    # Reproject
    if fire.crs.to_epsg() != 4326:
        fire = fire.to_crs(4326)
    ca = ca.to_crs(4326)

    # Repair invalid geometries
    print("Repairing invalid fire geometries...")
    fire["geometry"] = fire["geometry"].apply(make_valid)

    # Some geometries might still fail; drop empties
    fire = fire[~fire.geometry.is_empty & fire.geometry.notna()]

    # Clip to California
    print("Clipping to California boundary...")
    fire_ca = gpd.clip(fire, ca)

    print(f"Records within California: {len(fire_ca):,}")
    return fire_ca

def verify_date_column(gdf, date_col="DATE_CUR_parsed"):
    """Verify datetime parsing quality and consistency."""
    print("\n--- Date Column Verification ---")
    if date_col not in gdf.columns:
        print(f"Column {date_col} not found.")
        return

    total = len(gdf)
    valid = gdf[date_col].notna().sum()
    invalid = total - valid
    min_date = gdf[date_col].min()
    max_date = gdf[date_col].max()
    year_counts = gdf[date_col].dt.year.value_counts().sort_index().tail(10)

    print(f"Total records: {total:,}")
    print(f"Valid dates: {valid:,} ({(valid/total*100):.2f}%)")
    print(f"Invalid/missing dates: {invalid:,}")
    print(f"Earliest date: {min_date}")
    print(f"Latest date: {max_date}")
    print("\nRecent year counts:")
    print(year_counts)


def filter_fire_years(gdf, year_threshold=1980, year_col="FIRE_YEAR"):
    """Keep records with FIRE_YEAR >= year_threshold."""
    if year_col not in gdf.columns:
        print(f"Column {year_col} not found.")
        return gdf
    gdf[year_col] = pd.to_numeric(gdf[year_col], errors="coerce")
    before = len(gdf)
    gdf = gdf[gdf[year_col] >= year_threshold].copy()
    after = len(gdf)
    print(f"Filtered by year >= {year_threshold}: {after:,} records kept ({after/before:.1%})")
    return gdf


def plot_fire_perimeters(gdf, sample=20000, color="orangered"):
    """Quick map to visualize perimeters (sampled for speed)."""
    print(f"\nPlotting {min(sample, len(gdf))} random perimeters for preview...")
    sample_gdf = gdf.sample(min(sample, len(gdf)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 8))
    sample_gdf.plot(ax=ax, color=color, alpha=0.4, linewidth=0.5)
    plt.title("Sample of California Fire Perimeters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def tile_fire_perimeters(fire_path, tiles_path, start_year=1981, end_year=2023, out_csv=None):
    """
    Summarize fire perimeters into tile–month fire labels.

    Produces a complete table of (tile_id, year, month) with:
        burned_area_ha : summed burned area within tile (ha)
        fire_index     : normalized burned area (0–1)
        fire_occurred  : binary flag (1 if any burn >0)

    Parameters
    ----------
    fire_path : str
        Path to clipped fire perimeter shapefile/GPKG.
    tiles_path : str
        Path to tile grid (1 km × 1 km) with 'tile_id' field.
    start_year, end_year : int
        Temporal coverage to include.
    out_csv : str, optional
        If given, save the output table.

    Returns
    -------
    pd.DataFrame
    """
    print("Loading data ...")
    fires = gpd.read_file(fire_path)
    tiles = gpd.read_file(tiles_path)

    tiles = tiles.to_crs("EPSG:3310")
    fires = fires.to_crs("EPSG:3310")

    # ---- Clean and extract temporal fields ----
    fires = fires.to_crs(tiles.crs)
    fires["year"] = pd.to_datetime(fires["DATE_CUR_parsed"], errors="coerce").dt.year
    fires["month"] = pd.to_datetime(fires["DATE_CUR_parsed"], errors="coerce").dt.month
    fires = fires[(fires["year"] >= start_year) & (fires["year"] <= end_year)]

    # ---- Spatial intersection ----
    print("Intersecting fire perimeters with tiles ... (this may take a while)")
    join = gpd.overlay(tiles, fires, how="intersection")

    # Compute burned area (ha)
    join["burned_area_ha"] = join.geometry.area / 10_000.0

    # ---- Aggregate per tile–month ----
    summary = (
        join.groupby(["tile_id", "year", "month"])
        .agg(burned_area_ha=("burned_area_ha", "sum"))
        .reset_index()
    )

    # ---- Build full index of all tile–month combinations ----
    print("Expanding to include non-fire months ...")
    tile_ids = tiles["tile_id"].unique()
    years = range(start_year, end_year + 1)
    months = range(1, 13)

    full_index = pd.MultiIndex.from_product(
        [tile_ids, years, months],
        names=["tile_id", "year", "month"]
    ).to_frame(index=False)

    # Merge and fill missing with zeros
    df = full_index.merge(summary, on=["tile_id", "year", "month"], how="left")
    df["burned_area_ha"] = df["burned_area_ha"].fillna(0)
    df["fire_index"] = (df["burned_area_ha"] / 100).clip(0, 1)
    df["fire_occurred"] = (df["burned_area_ha"] > 0).astype(int)

    print(
        f"Final coverage: {len(df):,} tile-month records "
        f"({df['fire_occurred'].sum():,} positives, "
        f"{len(df)-df['fire_occurred'].sum():,} negatives)"
    )

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")

    return df



def save_fire_events(summary, tiles, out_fp, include_negatives=False):
    #save tiled results to file. If include_negative is true, all records will be stored in the output file.

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    # --- positive subset with geometry ---
    summary_pos = summary[summary["fire_occurred"] == 1].copy()
    fire_events_pos = summary_pos.merge(tiles[["tile_id", "geometry"]], on="tile_id", how="left")
    fire_events_pos = gpd.GeoDataFrame(fire_events_pos, geometry="geometry", crs=tiles.crs)
    fire_events_pos = fire_events_pos.to_crs(4326)

    # --- negatives only (no geometry) ---
    if include_negatives:
        summary_neg = summary[summary["fire_occurred"] == 0].copy()
        summary_neg["geometry"] = None
        fire_events = pd.concat([fire_events_pos, summary_neg], ignore_index=True)
        fire_events = gpd.GeoDataFrame(fire_events, geometry="geometry", crs="EPSG:4326")
    else:
        fire_events = fire_events_pos

    fire_events.to_file(out_fp, driver="GPKG")
    print(f"Saved fire events to: {out_fp}")
    print(f"Total records: {len(fire_events):,} "
          f"({(fire_events['fire_occurred'] == 1).sum():,} positives)")

    return fire_events