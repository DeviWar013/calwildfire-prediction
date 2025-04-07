from src.wildfires.inspect_gis import inspect_shapefile

inspect_shapefile(
    'data/processed/clipped_data/wildfires_clipped.shp',
    save_report=False,
    report_path="output/wildfire_column_report.txt"
)
