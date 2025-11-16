import cdsapi
# YOU SHOULD MANUALLY MOVE THE DOWNLOADED DATA TO: data/raw/era5
 
dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
            "variable": [ "skin_reservoir_content", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "10m_u_component_of_wind", "10m_v_component_of_wind" ],
            "year": [ "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023" ],
            "month": [ "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12" ],
            "time": ["00:00"],
            "data_format": "grib",
            "download_format": "zip",
            "area": [43, -126, 31, -113]
            }
client = cdsapi.Client()
client.retrieve(dataset, request).download()
