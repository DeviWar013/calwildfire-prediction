| Feature        | Description                                                                                                                                                           |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ppt**        | Monthly mean precipitation (mm). Aggregated from monthly PRISM precipitation rasters. Represents moisture availability and fuel wetness.                              |
| **tmean**      | Monthly mean air temperature (°C). Higher temperatures increase fuel drying and fire likelihood.                                                                      |
| **windspeed**  | Monthly mean 10-m wind speed (m/s) from ERA5-Land. Wind supports fire spread and intensity.                                                                           |
| **swvl_mean**  | Monthly mean volumetric soil moisture (0–1) from ERA5 soil layer. Indicates fuel dryness; lower values → drier fuels.                                                 |
| **src**        | Surface soil moisture (ERA5, 0–7 cm). Similar to swvl_mean but more sensitive to short-term drought patterns.                                                         |
| **ndvi**       | Monthly mean NDVI (Normalized Difference Vegetation Index). Proxy for vegetation vigor, greenness, and live fuel load. Higher NDVI → denser/more vigorous vegetation. |
| **elev_mean**  | Mean elevation (meters) within each 1-km tile. Influences temperature, vegetation, and wind behavior.                                                                 |
| **slope_mean** | Mean slope (degrees). Steeper slopes tend to promote faster fire spread.                                                                                              |
| **UrbanPct**   | Percentage of tile classified as developed (urban). Represents human presence, ignition likelihood, and fire suppression pressures.                                   |
| **RoadDens**   | Road density (km of roadway per km²). Roads influence accessibility, human ignition risk, and suppression dynamics.                                                   |



| Dummy                                            | Meaning                                                                        |
|--------------------------------------------------|--------------------------------------------------------------------------------|
| **DomCover_Grass**                               | Tile dominated by grassland/herbaceous cover.                                  |
| **DomCover_Shrub**                               | Tile dominated by shrub/scrub vegetation.                                      |
| **DomCover_EvForest**                            | Evergreen forest dominance.                                                    |
| **DomCover_MixForest**                           | Mixed forest.                                                                  |
| **DomCover_Crop**                                | Agricultural cropland.                                                         |
| **DomCover_HerbWet**                             | Emergent herbaceous wetlands.                                                  |
| **DomCover_WoodyWet**                            | Woody wetlands.                                                                |
| **DomCover_Pasture**                             | Pasture/hay.                                                                   |
| **DomCover_Barren**                              | Barren land / sparse cover.                                                    |
| **DomCover_Water**                               | Open water.                                                                    |
| **DomCover_DevLow / DevMed / DevHigh / DevOpen** | Low/medium/high-density developed areas (urban), plus open space development.  |
| **DomCover_Unknown**                             | Tiles where NLCD does not provide a stable dominant class.                     |
| **DomCover_Ice**                                 | Perennial ice/snow (rare or absent in California).                             |




