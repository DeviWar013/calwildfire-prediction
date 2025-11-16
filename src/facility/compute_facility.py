"""
facility_index_final.py
---------------------------------
Computes facility accessibility (facility index) for each 1-km tile
using an exponential distance-decay gravity model.

Outputs:
- facility_index_fire
- facility_index_air
- facility_index_other
- facility_index

Computation Function:
#   A_j = Σ_i ( w_t * exp(-d_ij / λ) )
# where:
#   A_j      = accessibility (facility index) at tile j
#   w_t      = facility-type weight (fire=1, air=2, other=0.5)
#   d_ij     = distance between tile j and facility i (meters)
#   λ (LAMBDA_M) = decay constant controlling influence radius
#
# Facilities beyond RADIUS_M (50 km) are excluded.

Refs:
Wang (2020) IJGIS • Wu et al. (2021) Appl. Geogr. •
Zhou et al. (2022) Sust. Cities & Soc. • Tang et al. (2019) Fire Safety J.
"""
import numpy as np
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
facility_gdb = BASE_DIR / "pdata" / "raw" / "facility_data" / "facility251gdb" / "facility25_1.gdb"
tiles_path   = BASE_DIR / "pdata" / "processed" / "tiled_ca" / "ca_tiles_1km.shp"
out_path     = BASE_DIR / "pdata" / "processed" / "facility_index" / "facility_index_final.parquet"

RADIUS_M = 50_000     # 50 km search radius
LAMBDA_M = 20_000     # exponential decay constant (~30-min response)
WEIGHTS  = {"fire": 1.0, "air": 2.0, "other": 0.5}
CRS_CA   = 3310       # California Albers (meters)


# ============================================================
# LOAD DATA
# ============================================================

fac = gpd.read_file(facility_gdb)
fac = fac[fac["FACILITY_STATUS"] == "Active"].copy()
tiles = gpd.read_file(tiles_path)

fac = fac.to_crs(epsg=CRS_CA)
tiles = tiles.to_crs(epsg=CRS_CA)
_tile_centroids = tiles.geometry.centroid  # points in EPSG:3310
tile_pts = np.column_stack([_tile_centroids.x.to_numpy(),
                            _tile_centroids.y.to_numpy()])

# ============================================================
# FACILITY TYPE CLASSIFICATION
# ============================================================

def classify_type(v: str) -> str:
    if not isinstance(v, str):
        return "other"
    v = v.upper()
    if any(x in v for x in ["AAB", "AIR", "HB"]):
        return "air"
    elif any(x in v for x in ["FS", "FSA", "FSB", "FSL", "FSO", "FSCC", "HQ"]):
        return "fire"
    return "other"

fac["main_type"] = fac["TYPE"].apply(classify_type)
fac_groups = {t: fac[fac["main_type"] == t].copy() for t in WEIGHTS}


# ============================================================
# ACCESSIBILITY COMPUTATION
# ============================================================
# Exponential-decay gravity model (after Wang, 2020; Wu et al., 2021):
#   A_j = Σ_i ( w_t * exp(-d_ij / λ) )
# where:
#   A_j      = accessibility (facility index) at tile j
#   w_t      = facility-type weight (fire=1, air=2, other=0.5)
#   d_ij     = distance between tile j and facility i (meters)
#   λ (LAMBDA_M) = decay constant controlling influence radius
# Facilities beyond RADIUS_M (50 km) are excluded.

def compute_access_and_counts(tile_points, fac_df, weight, lam, radius):
    """Return both exponential-decay accessibility and raw facility count."""
    if fac_df.empty:
        return np.zeros(len(tile_points)), np.zeros(len(tile_points))
    acc = np.zeros(len(tile_points))
    count = np.zeros(len(tile_points))
    fpts = np.array([[g.x, g.y] for g in fac_df.geometry])

    for fx, fy in tqdm(fpts, desc=f"{fac_df.iloc[0]['main_type']:<5}", ncols=80):
        dx = tile_points[:, 0] - fx
        dy = tile_points[:, 1] - fy
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist <= radius
        acc[mask] += weight * np.exp(-dist[mask] / lam)
        count[mask] += 1
    return acc, count


for t, w in WEIGHTS.items():
    acc, cnt = compute_access_and_counts(tile_pts, fac_groups[t], w, LAMBDA_M, RADIUS_M)
    tiles[f"facility_index_{t}"] = acc
    tiles[f"facility_count_{t}"] = cnt


# ============================================================
# AGGREGATE + RESCALE
# ============================================================

def rescale(series: np.ndarray) -> np.ndarray:
    """Min-max rescale to [0, 1]."""
    cmin, cmax = np.nanmin(series), np.nanmax(series)
    return (series - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(series)


# aggregated index (sum of all types)
tiles["facility_index"] = (
    tiles["facility_index_fire"] +
    tiles["facility_index_air"] +
    tiles["facility_index_other"]
)

# aggregated raw count
tiles["facility_count_total"] = (
    tiles["facility_count_fire"] +
    tiles["facility_count_air"] +
    tiles["facility_count_other"]
)

# rescale index columns only
for col in ["facility_index_fire", "facility_index_air", "facility_index_other", "facility_index"]:
    tiles[col] = rescale(tiles[col])


# ============================================================
# SAVE RESULTS
# ============================================================

out_path.parent.mkdir(parents=True, exist_ok=True)
tiles.drop(columns="geometry").to_parquet(out_path, index=False)
print(f"Saved facility accessibility index to: {out_path}")
