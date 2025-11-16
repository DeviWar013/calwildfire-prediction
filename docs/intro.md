### Stacked Ensemble Modeling of California Wildfire Susceptibility with Resource–Risk Alignment Diagnostics

This repository presents a complete and modular system for constructing a 1-km–resolution wildfire susceptibility model 
for California covering the years 1981 through 2024. The project processes diverse environmental, climatic, infrastructural, 
and historical wildfire data sources into a unified spatiotemporal dataset, and trains a stacked ensemble model to 
estimate tile-level susceptibility. The system also includes a diagnostic component that evaluates the alignment between 
predicted susceptibility and fire-related infrastructure distribution, enabling a data-driven assessment of resource coverage.

The design of the pipeline follows several core principles. First, wildfire behavior in California is influenced by both 
short-term climate variability and long-term environmental structure. For this reason, the system integrates monthly 
climate features (precipitation, mean temperature, soil moisture, wind speed) together with static ecological and 
geographic variables, including land cover composition, NDVI, road density, and 
topographic attributes derived from DEM data. Each dataset is processed independently, aligned to the same 1-km spatial grid, 
and aggregated to annual tile-level summaries. This modularization ensures that large datasets can be handled efficiently, 
updated individually, and reused without reprocessing the entire pipeline.

Historical wildfire perimeters are cleaned, clipped to the state boundary, and intersected with the tile grid to produce 
annual fire event records that indicate whether a tile burned and the proportional area affected. These records form the 
empirical basis for supervised learning. The modeling component uses a stacked ensemble, combining gradient boosted tree 
models with a linear baseline to leverage both nonlinear relationships and stable global trends. This design choice 
balances predictive accuracy, interpretability, and robustness when working with multi-decadal environmental datasets of 
varying resolution.

Once susceptibility predictions are generated, the project includes an analytical module that compares 
predicted risk with the distribution of fire-related facilities. This diagnostic step is meant to identify 
potential gaps between areas of elevated susceptibility and the placement of available response resources. 
The workflow supports both visualization and quantitative summary of these discrepancies, offering insights 
into whether resource allocation patterns align with long-term spatial fire risk.

Across evaluations, the ensemble model achieves stable multi-decadal performance, with RMSE around 0.229, MAE around 0.155, 
and an R² of approximately 0.715. These results indicate effective capture of broad spatiotemporal patterns in wildfire 
occurrence and produce a susceptibility estimate suitable for subsequent diagnostic and interpretive analysis.

More detailed information on dataset acquisition, pipeline architecture, modeling methodology, reproducibility 
considerations, and troubleshooting can be found in the accompanying documents within the docs/ directory. 
These documents describe the necessary raw data sources, the logic of each processing module, 
the construction of annual modeling tables, the structure of the ensemble model, recommended environment configuration, 
and common issues encountered when working with large geospatial datasets.