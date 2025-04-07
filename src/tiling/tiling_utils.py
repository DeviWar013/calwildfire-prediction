import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import os

#Constants
#Dynamically locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Goes from src/tiling/

#Join path together
CALIFORNIA_BOUNDARY_PATH = os.path.join(BASE_DIR, 'data/raw/ca_state/CA_State.shp')
OUTPUT_TILE_PATH = os.path.join(BASE_DIR, "data/processed/tiled_ca/ca_tiles_1km.shp")

#Other constants
PROJECTED_CRS = "EPSG:3310"  # California Albers
GEOGRAPHIC_CRS = "EPSG:4326"
TILE_SIZE_METERS = 1000


def generate_tiles(gdf, tile_size=TILE_SIZE_METERS):
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, x + tile_size, y + tile_size)
            tiles.append(tile)
            y += tile_size
        x += tile_size

    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
    return tiles_gdf

def clip_tiles_to_boundary(tiles_gdf, boundary_gdf):
    return gpd.overlay(tiles_gdf, boundary_gdf, how='intersection')

def visualize_tiles(boundary_gdf, tiles_gdf, show_labels=True, max_labels=300):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(30, 30))
    boundary_gdf.boundary.plot(ax=ax, color='blue', linewidth=1)
    tiles_gdf.boundary.plot(ax=ax, color='black', linewidth=0.1)

    if show_labels:
        # Avoid clutter: only label the first N tiled_ca
        sample = tiles_gdf.head(max_labels)
        for _, row in sample.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row.tile_id),
                fontsize=6,
                ha='center',
                va='center',
                color='blue',
                alpha=0.6
            )

    ax.set_title("California 1km x 1km Tiled Grid with Tile IDs")
    plt.axis('equal')
    plt.show()