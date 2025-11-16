"""
generate_tiles.py
---------------------------------------------------------
Generate a 1 km × 1 km California tile grid clipped to the state boundary.

Outputs:
  • ca_tiles_1km_new.shp
  • output/California_Tiled_new.png

"""

import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
BOUNDARY_PATH = BASE_DIR / "data/raw/ca_state/CA_State.shp"
TILE_OUT = BASE_DIR / "data/processed/tiled_ca/ca_tiles_1km.shp"
LOOKUP_OUT = BASE_DIR / "data/processed/tiled_ca/tile_id_grid.npy"
FIG_OUT = BASE_DIR / "output/California_Tiled_new.png"

PROJECTED_CRS = "EPSG:3310"
GEOGRAPHIC_CRS = "EPSG:4326"
TILE_SIZE = 1000  # meters


# ==========================================================
# Core Functions
# ==========================================================
def generate_tiles(boundary_gdf: gpd.GeoDataFrame, tile_size: int = TILE_SIZE) -> gpd.GeoDataFrame:
    """Generate square tiles covering the boundary extent."""
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    xs = np.arange(minx, maxx, tile_size)
    ys = np.arange(miny, maxy, tile_size)
    tiles = [box(x, y, x + tile_size, y + tile_size) for x in xs for y in ys]
    return gpd.GeoDataFrame(geometry=tiles, crs=boundary_gdf.crs)


def clip_tiles_to_boundary(tiles_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip tiles to the California boundary."""
    return gpd.overlay(tiles_gdf, boundary_gdf, how="intersection")



def visualize_tiles(boundary_gdf: gpd.GeoDataFrame, coarse_factor: int = 10, save_path: Path = None):
    """Render coarse-grid preview and save."""
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    xs = np.linspace(minx, maxx, int((maxx - minx) / (TILE_SIZE * coarse_factor)))
    ys = np.linspace(miny, maxy, int((maxy - miny) / (TILE_SIZE * coarse_factor)))
    coarse_tiles = [box(x, y, x + TILE_SIZE * coarse_factor, y + TILE_SIZE * coarse_factor)
                    for x in xs for y in ys]
    coarse_gdf = gpd.GeoDataFrame(geometry=coarse_tiles, crs=boundary_gdf.crs)

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary_gdf.boundary.plot(ax=ax, color="blue", linewidth=0.8)
    coarse_gdf.boundary.plot(ax=ax, color="gray", linewidth=0.3)
    ax.set_title(f"California Tile Grid (coarse {coarse_factor} km preview)")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization → {save_path}")
    plt.close(fig)


# ==========================================================
# Runner
# ==========================================================
def run_tiling(save_output=True, visualize=True):
    """Run tiling workflow; outputs match legacy structure with _new suffix."""
    ca = gpd.read_file(BOUNDARY_PATH)[["geometry"]].to_crs(PROJECTED_CRS)

    tiles = generate_tiles(ca, TILE_SIZE)
    clipped = clip_tiles_to_boundary(tiles, ca)
    clipped["tile_id"] = np.arange(len(clipped))
    result = clipped[["tile_id", "geometry"]].to_crs(GEOGRAPHIC_CRS)

    if save_output:
        TILE_OUT.parent.mkdir(parents=True, exist_ok=True)
        result.to_file(TILE_OUT)
        print(f"Saved tiles → {TILE_OUT} ({len(result):,})")

    if visualize:
        try:
            visualize_tiles(ca.to_crs(GEOGRAPHIC_CRS), save_path=FIG_OUT)
        except Exception as e:
            print(f"Visualization skipped due to error: {e}")
    return result


if __name__ == "__main__":
    run_tiling(save_output=True, visualize=True)
