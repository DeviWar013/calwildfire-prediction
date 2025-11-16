# Data Acquisition Guide

This project uses multiple public datasets across climate, land, vegetation, topography, wildfire perimeters, and infrastructure.  
Below is a complete guide to download each dataset manually (when required) and notes on modules that provide automated download tools.

---

## **1. Fire Perimeter Data (Raw Wildfire History)**  
**Source:** NIFC – Wildland Fire Open Data  
We use the **WFIGS Interagency Fire Perimeters** dataset (full historic shapefile).

Download (complete shapefile):  
https://data-nifc.opendata.arcgis.com/search?tags=historic_wildlandfire_opendata%2CCategory  

---

## **2. California State Boundary**  
**Source:** California Open Data Portal  
Used for clipping and constraining all spatial datasets.

Download:  
https://data.ca.gov/dataset/ca-geographic-boundaries  

---

## **3. PRISM Climate Data (Temperature & Precipitation)**  
**Source:** PRISM Climate Group, Oregon State University  
We use **monthly mean temperature (tmean)** and **monthly precipitation (ppt)**:

- Monthly normals (pre-1981)  
- Monthly recent data (1981–2024)

Download:  
https://prism.oregonstate.edu/normals/

Files should be downloaded manually into your preferred raw directory structure.  
The model expects **.bil** raster files.

---

## **4. Topography (Elevation) Data**  
**Source:** USGS TNM Download v2  
We use the “Elevation Products” digital elevation model.

Download:  
https://apps.nationalmap.gov/downloader/

---

## **5. Road Network (TIGER/Line 2024)**  
**Source:** U.S. Census TIGER/Line  
We use **Primary & Secondary Roads (2024)** for California.

Download:  
https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2024&layergroup=Roads  

---

## **6. Land Cover Data (NLCD Land Cover)**  
**Source:** MRLC  
We use **NLCD 2024 Land Cover (CONUS)**.

Download:  
https://www.mrlc.gov/data?f%5B0%5D=category%3ALand%20Cover  

---

## **7. Vegetation Data (USFS CALVEG)**   
### THIS PART IS DEPRECATED. DO NOT USE. 
**Source:** USDA Forest Service  
We use the **Existing Vegetation – Region 5 (Central Coast)** dataset.  
Files will appear with names like:



Download:  
https://data.fs.usda.gov/geodata/edw/datasets.php?dsetCategory=biota  

---

## **8. NDVI (Normalized Difference Vegetation Index)**  
**Source:** NOAA NCEI  
This project includes an **automatic downloader**:  
src/ndvi/fetch_process_ndvi.py
If downloading manually:  
https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access  

---

## **9. ERA5 Soil Moisture & Wind Speed (Wind/Soil)**  
**Source:** Copernicus Climate Data Store (CDS)  
Requires user-configured CDS API.

Automatic download script:  
src/wind_soil/fetchdata_api.py
CDS API setup:  
https://cds.climate.copernicus.eu/how-to-api  

---

## **10. Fire Facility Locations (CAL FIRE Facility Map)**  
**Source:** CAL FIRE  
We use the **2025 Facilities** dataset (GDB format).

Download:  
https://34c031f8-c9fd-4018-8c5a-4159cdff6b0d-cdn-endpoint.azureedge.net/-/media/calfire-website/what-we-do/fire-resource-assessment-program---frap/gis-data/facility251gdb.zip?rev=19b8f0db2c86496c8107d9cfd09191c8&hash=7E973BBAC49E254D7D5E3F7ED0B14D61

---

If any dataset is updated in your environment, ensure your preprocessing scripts still align with the file structure and 
projections expected by the pipeline.

👉To exactly match the raw data directory structure we used after you've acquired the dataset, go here: 
[Raw Data Directory Structure](data_structure.md)
