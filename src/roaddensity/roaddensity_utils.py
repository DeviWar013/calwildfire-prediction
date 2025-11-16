"""
Road Density Utilities
----------------------

This module provides the core geospatial operations required to compute
road density for each 1 km tile in the California grid. It includes:

Functions:
    - check_roads(): Load and inspect raw road shapefiles.
    - plot_roads(): Visualize statewide or clipped road networks.
    - tile_roads(): Intersect road geometries with the tile grid.
    - calc_road_density(): Compute road length per tile area (km/km²).

Workflow Supported:
1. Read raw TIGER/Line primary & secondary road centerlines.
2. Reproject and clip road geometries to the tile grid.
3. Calculate total road length within each tile in meters.
4. Convert tile area to km² and compute density as (km of road) / (km² tile).

Outputs (produced via `roaddensity_process.py`):
    - Tiled road geometries
    - Tile-level road density shapefile
"""

import os
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_roads(road_path):
    gdf = gpd.read_file(road_path)
    print("Road shapefile loaded.")
    print("CRS:", gdf.crs)
    print("Total features.md:", len(gdf))
    print("Columns:", list(gdf.columns))
    print("Geometry type:", gdf.geom_type.unique())
    print(gdf.head(3))
    return gdf


def plot_roads(gdf, boundary_gdf=None, title="California Roads"):
    fig, ax = plt.subplots(figsize=(10, 10))
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)
    gdf.plot(ax=ax, linewidth=0.3, color='dimgray')
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def tile_roads(road_gdf, tile_gdf):
    # Ensure same CRS
    if road_gdf.crs != tile_gdf.crs:
        road_gdf = road_gdf.to_crs(tile_gdf.crs)

    # Drop invalid or empty geometries
    road_gdf = road_gdf[road_gdf.is_valid & road_gdf.geometry.notnull()]
    tile_gdf = tile_gdf[tile_gdf.is_valid & tile_gdf.geometry.notnull()]

    # Intersect roads with tiles — retain non-matching geometry types
    clipped_roads = gpd.overlay(tile_gdf, road_gdf, how='intersection', keep_geom_type=False)
    print("Roads clipped to tile grid.")
    return clipped_roads


def calc_road_density(tiled_roads: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates road density (km/km²) for each tile.

    Parameters:
    - tiled_roads: GeoDataFrame with roads clipped to tiles (must have 'tile_id')
    - tiles: GeoDataFrame of original tiles with 'tile_id' and geometry

    Returns:
    - GeoDataFrame with 'tile_id', 'length_m', 'area_km2', and 'density_km_per_km2'
    """

    # Project to California Albers (EPSG:3310) for metric calculations
    roads_proj = tiled_roads.to_crs(epsg=3310)
    tiles_proj = tiles.to_crs(epsg=3310)

    # Calculate total road length per tile (in meters)
    roads_proj["length_m"] = roads_proj.geometry.length
    length_sum = roads_proj.groupby("tile_id")["length_m"].sum().reset_index()

    # Calculate tile area in km²
    tiles_proj["area_km2"] = tiles_proj.geometry.area / 1e6

    # Merge and compute road density (km per km²)
    result = tiles_proj.merge(length_sum, on="tile_id", how="left")
    result["length_m"] = result["length_m"].fillna(0)
    result["RoadDens"] = result["length_m"] / 1000 / result["area_km2"]

    return result

