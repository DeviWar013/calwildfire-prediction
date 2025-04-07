import geopandas as gpd
import os
import pandas as pd
import numpy as np


def clip_to_california(wildfires_gdf, ca_gdf):
    """

    :param wildfires_gdf: Input full wildfires shapefiles
    :param ca_gdf: Input shapefiles depicting the boundary of California
    :return: the processed shapefiles with all original attributes
    """

    #clipping data
    print("Clipping wildfires to California boundary")
    clipped = gpd.clip(wildfires_gdf, ca_gdf)

    #round high-precision numeric fields to prevent shapefile overflow issues
    for field in ['Shape__Are', 'Shape__Len']:
        if field in clipped.columns:
            print(f"Dropping {field} because it causes problems")
            clipped= clipped.drop(columns=[field])


    print(f"Clipped dataset attributes: {clipped.columns}")
    return clipped


def clean_fire_attributes(gdf):
    """
    Cleans and standardizes wildfire attributes:
    - Drops irrelevant or redundant columns
    - Filters to valid fire types
    - Retains both IRWINID and UNQE_FIRE_ for flexibility
    - Creates composite fire_id

    Returns:
        GeoDataFrame with cleaner, more focused attributes
    """
    gdf = gdf.copy()

    # Drop unnecessary columns
    drop_cols = [
        'OBJECTID', 'COMMENTS', 'GEO_ID', 'LOCAL_NUM', 'USER_NAME',
        'FIRE_YEAR_', 'OTHERID'
    ]
    print("Dropping unnecessary columns")
    gdf = gdf.drop(columns=[col for col in drop_cols if col in gdf.columns])

    # Create unified fire_id
    gdf['fire_id'] = gdf['IRWINID'].fillna(gdf['UNQE_FIRE_'])

    print(f"Retained dataset attributes: {gdf.columns}")
    return gdf


def normalize_date_cur(gdf, column='DATE_CUR', seed=42):
    """
    Parses the DATE_CUR column (which includes YYYY, YYYYMMDD, YYYYMMDDHHMM)
    and extracts consistent monthly date fields.

    If the record lacks a full date, it assigns a randomized month from the
    typical fire season [May–Sep] and flags it as imputed.
    So that removing imputed values in later training sessions to assess bias is possible.

    Parameters:
        gdf (GeoDataFrame): Input fire dataset
        column (str): Column name containing date strings
        seed (int): Random seed for reproducibility. Default 42

    Returns:
        GeoDataFrame with:
            - parsed_date (best-effort datetime)
            - year, month
            - alarm_date (normalized to 1st of month)
            - time_index (e.g., "2021-08")
            - date_imputed (True if month was imputed)
    """
    gdf = gdf.copy()
    np.random.seed(seed)

    def parse_date(val):
        if pd.isna(val):
            return None
        val_str = str(int(val))  # remove trailing .0 if float
        if len(val_str) == 4:
            return pd.to_datetime(val_str, format="%Y", errors='coerce')
        elif len(val_str) == 8:
            return pd.to_datetime(val_str, format="%Y%m%d", errors='coerce')
        elif len(val_str) == 12:
            return pd.to_datetime(val_str, format="%Y%m%d%H%M", errors='coerce')
        else:
            return None

    # Parse base date
    gdf['P_DATE'] = gdf[column].apply(parse_date)
    #gdf['YEAR'] = gdf['P_DATE'].dt.year

    # Month: use from date if available, else assign random fire season month
    def assign_month(row):
        if pd.notna(row['P_DATE']):
            return row['P_DATE'].month, False
        else:
            return np.random.choice([5, 6, 7, 8, 9]), True

    # Apply to each row
    month_info = gdf.apply(assign_month, axis=1, result_type='expand')
    gdf['MONTH'] = month_info[0].astype(int)
    gdf['IMPUTE'] = month_info[1]

    # Final fire date string
    #gdf['FIRE_DATE'] = gdf['YEAR'].astype(str) + "-" + gdf['MONTH'].astype(str).str.zfill(2)

    return gdf

def wildfire_process(wildfires_gdf, ca_gdf, output_path="data/processed/clipped_data/wildfires_clipped.shp"):
    #Detecting if file already processed to avoid redundancy
    if os.path.exists(output_path):
        print("Clipped wildfire data already exists. Loading from file.")
        return gpd.read_file(output_path)

    #Clipping file
    clipped_gdf = clip_to_california(wildfires_gdf, ca_gdf)

    #Building timeline
    normalized_gdf = normalize_date_cur(clipped_gdf)

    #Dropping unnecessary columns
    cleaned_gdf = clean_fire_attributes(normalized_gdf)

    #Saving file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_gdf.to_file(output_path)
    print(f"✅ Saved processed fire data to: {output_path}. "
          f"Final attributes: {cleaned_gdf.columns}")

    return cleaned_gdf