import geopandas as gpd

def inspect_shapefile(path, save_report=False, report_path="output/column_report.txt"):
    """
    Load and inspect a shapefile or other GIS dataset.

    Parameters:
        path (str): Path to the shapefile (.shp) or other supported GIS file.
        save_report (bool): Whether to save the column list to a text file.
        report_path (str): Path to the output report file (if save_report=True).

    Returns:
        gdf (GeoDataFrame): The loaded dataset.
    """
    print(f"Loading file: {path}")
    gdf = gpd.read_file(path)

    print("\nColumns in the dataset:")
    print(list(gdf.columns))

    print("\nColumn data types:")
    print(gdf.dtypes)

    print("\nSample records:")
    print(gdf.head(5).T)  # Transposed for easier scanning

    if save_report:
        with open(report_path, "w") as f:
            f.write("Column Names:\n")
            for col in gdf.columns:
                f.write(f"{col} ({gdf[col].dtype})\n")
        print(f"\n Column report saved to: {report_path}")

    return gdf