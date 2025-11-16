import os
import numpy as np
from rasterstats import zonal_stats
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import box


def clip_raster_to_shape(raster_path, shapefile_path, out_path):
    """
    Clip a raster (GeoTIFF) to the boundary of a shapefile (e.g., California).

    Parameters
    ----------
    raster_path : str
        Path to input GeoTIFF.
    shapefile_path : str
        Path to shapefile (must match CRS or be reprojectable).
    out_path : str
        Path to save clipped GeoTIFF.
    """
    # check if output already exists
    if os.path.exists(out_path):
        print("Clipped raster already exists. Skipping clipping.")
        return out_path

    #load boundary
    shape = gpd.read_file(shapefile_path)

    #load land cover dataset
    with rasterio.open(raster_path) as src:
        #reproject the boundary's CRS to raster's CRS if different
        if shape.crs != src.crs:
            shape = shape.to_crs(src.crs)

        geoms = [feature["geometry"] for feature in shape.__geo_interface__["features.md"]]

        # mask (clip)
        out_img, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

        # update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform
        })

        # write clipped raster
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_img)

    print(f"Clipped raster saved at: {out_path}")


def plot_tiles_zoomed(tile_shp, xmin, ymin, xmax, ymax, column=None, cmap="tab20"):
    gdf = gpd.read_file(tile_shp)

    # define bounding box
    bbox = box(xmin, ymin, xmax, ymax)

    # subset tiles
    subset = gdf[gdf.intersects(bbox)]

    # plot
    ax = subset.plot(column=column, cmap=cmap, legend=True, figsize=(8, 8))
    ax.set_title("Zoomed-in Tiles")
    plt.show()

    return subset


def verify_LC_raster(raster_path, band=1, title=None, cmap="tab20",max_size=2000):
    """
    Verify a raster by printing metadata and plotting.

    Parameters
    ----------
    raster_path : str
        Path to GeoTIFF raster file.
    band : int, optional
        Raster band to plot (default: 1).
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Colormap for plotting (default: 'tab20' works well for categorical rasters like NLCD).
    """
    with rasterio.open(raster_path) as src:
        print("Metadata:")
        print(src.meta)

        # compute scale factor
        scale = max(src.width / max_size, src.height / max_size, 1)

        # read at reduced resolution
        arr = src.read(
            band,
            out_shape=(
                1,
                int(src.height / scale),
                int(src.width / scale)
            )
        )

# NLCD urban codes
URBAN_CODES = {21, 22, 23, 24}

def landcover_features(tile_shp, nlcd_tif, out_shp=None):
    """
    Calculate land cover features.md for each tile:
    - dominant_cover_id
    - dominant_cover_pct
    - urban_pct

    Parameters
    ----------
    tile_shp : str
        Path to input tile shapefile (1 km tiles).
    nlcd_tif : str
        Path to clipped NLCD raster (GeoTIFF).
    out_shp : str, optional
        Path to save output shapefile. If None, only returns GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Tiles with added land cover columns.
    """
    # check if output already exists
    if os.path.exists(out_shp):
        print("Clipped raster already exists. Skipping clipping.")
        return out_shp

    # read tiles
    tiles = gpd.read_file(tile_shp)

    with rasterio.open(nlcd_tif) as src:
        raster_crs = src.crs

    tiles = gpd.read_file(tile_shp)

    # reproject tiles to raster CRS
    if tiles.crs != raster_crs:
        tiles = tiles.to_crs(raster_crs)

    # run zonal histogram
    zs = zonal_stats(
        tiles,
        nlcd_tif,
        categorical=True,
        nodata=250,
        all_touched=True  # count pixels touching tile edges
    )

    # process results
    dominant_ids = []
    dominant_pcts = []
    urban_pcts = []

    for h in zs:
        if not h:
            # empty tile (e.g., ocean or missing pdata)
            dominant_ids.append(None)
            dominant_pcts.append(0.0)
            urban_pcts.append(0.0)
            continue

        total = sum(h.values())
        if total == 0:
            dominant_ids.append(None)
            dominant_pcts.append(0.0)
            urban_pcts.append(0.0)
            continue

        # dominant cover
        dom_id = max(h, key=h.get)
        dom_pct = h[dom_id] / total

        # urban cover
        urb_count = sum(h.get(code, 0) for code in URBAN_CODES)
        urb_pct = urb_count / total

        dominant_ids.append(dom_id)
        dominant_pcts.append(dom_pct)
        urban_pcts.append(urb_pct)

    # add new columns
    tiles["DomCoverID"] = dominant_ids
    tiles["DomPct"] = np.round(dominant_pcts, 3)
    tiles["UrbanPct"] = np.round(urban_pcts, 3)

    # save if requested
    if out_shp:
        tiles.to_file(out_shp)

    return tiles


def verify_landcover_tiles(tile_shp, n=5, plot=True):
    """
    Verify land cover features.md in the tile shapefile.

    Parameters
    ----------
    tile_shp : str
        Path to shapefile with land cover attributes.
    n : int
        Number of rows to preview (default: 5).
    plot : bool
        If True, plots dominant cover categories.
    """
    gdf = gpd.read_file(tile_shp)

    print("\nPreview of attributes:")
    print(gdf[["tile_id", "DomCoverID", "DomPct", "UrbanPct"]].head(n))

    print("\nSummary statistics:")
    print(gdf[["DomPct", "UrbanPct"]].describe())

    if plot:
        # quick plot, coloring by dominant cover id
        gdf.plot(column="DomCoverID", legend=True, cmap="tab20", figsize=(8, 8))
        plt.title("Tiles Colored by Dominant Cover ID")
        plt.axis("off")
        plt.show()

    return gdf


# Appendix: NLCD 2019/2021 class codes and names in case needed
NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}

