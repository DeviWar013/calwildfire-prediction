import tiling_utils
import geopandas as gpd
import os


def tiling_process():
    # Load California boundary shapefile
    ca_gdf = gpd.read_file(tiling_utils.CALIFORNIA_BOUNDARY_PATH)

    # Drop all non-geometry columns ASAP
    ca_boundary = ca_gdf[["geometry"]]

    # Reproject to projected CRS
    ca_projected = ca_boundary.to_crs(tiling_utils.PROJECTED_CRS)

    # Generate tiled_ca in projected CRS
    tiles = tiling_utils.generate_tiles(ca_projected)

    # Clip tiled_ca to California boundary
    clipped_tiles = tiling_utils.clip_tiles_to_boundary(tiles, ca_projected)

    # Assign unique tile ID
    clipped_tiles["tile_id"] = range(len(clipped_tiles))

    # Reproject tiled_ca back to EPSG:4326
    # Keep only tile ID and geometry
    final_tiles = clipped_tiles[["tile_id", "geometry"]].to_crs(tiling_utils.GEOGRAPHIC_CRS)

    # Save as shapefile
    os.makedirs(os.path.dirname(tiling_utils.OUTPUT_TILE_PATH), exist_ok=True)
    final_tiles.to_file(tiling_utils.OUTPUT_TILE_PATH)

    # Visualize
    ca_boundary_4326 = ca_projected.to_crs(tiling_utils.GEOGRAPHIC_CRS)
    tiling_utils.visualize_tiles(ca_boundary_4326, final_tiles)

    # To configure labels' amount
    #visualize_tiles(ca_boundary_4326, final_tiles, show_labels=True, max_labels=1000)

tiling_process()