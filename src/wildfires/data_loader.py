import geopandas as gpd

#Initially read original datasets
def load_california_boundary(path="data/raw/ca_state/CA_State.shp"):
    return gpd.read_file(path)


def load_wildfires(path="data/raw/InterAgencyFirePerimeterHistory_All_Years_View/InterAgencyFirePerimeterHistory_All_Years_View.shp"):
    wildfires = gpd.read_file(path)

    # Fix invalid geometries
    if not wildfires.is_valid.all():
        wildfires['geometry'] = wildfires.buffer(0)

    return wildfires