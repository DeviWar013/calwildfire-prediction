# alignment.py
# -----------------------------------------------------------
# Purpose:
#   Evaluate predicted wildfire susceptibility alignment with existing facility distribution.
#
#   For each 1-km tile, we compute:
#       - Multi-year mean predicted fire index (California, 2013–2023)
#       - Facility index (density-based coverage metric)
#       - Discrepancy metrics (gap, standardized z-gap, ratio)
#       - Under/over-protection flags (quantile-based)
#
#   We then quantify alignment using four classes of metrics:
#
#   1. Global monotonic association:
#          - Spearman correlation (nonparametric)
#
#   2. Directional proportionality:
#          - Linear calibration regression (β, R², CI)
#
#   3. Nonlinear dependence:
#          - Mutual Information (MI)
#          - Normalized Mutual Information (NMI)
#
#   4. Spatial co-location (cross-correlation in space):
#          - Spatial Pearson (Lee’s L equivalent)
#            using Queen contiguity spatial weights
#
# Outputs:
#   parquet:
#       - tile_discrepancy.parquet
#
#   csv (alignment metrics):
#       - alignment_correlation.csv
#       - alignment_regression.csv
#       - alignment_mutual_info.csv
#       - alignment_leesL.csv
#       - coverage_alignment_metrics.csv
#
# Notes:
#   - This module is the canonical discrepancy/alignment
#     analysis for the CalFire ML project.
#   - Spatial weights exclude self-neighbors (W_ii = 0)
#     to measure neighborhood-based co-location.
# -----------------------------------------------------------


import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from esda.lee import Spatial_Pearson
from libpysal.weights import Queen

# ----------------------------
# Config
# ----------------------------
BASE = Path(__file__).resolve().parents[2]

PRED_MULTIYEAR = BASE / r"src\modeling\data\predictions\raw_predictions_2013_2023.parquet"
FACILITY_FP    = BASE / r"data\processed\facility_index\facility_index_final.parquet"
TILES_FP = BASE/ r"data\processed\tiled_ca\ca_tiles_1km.shp"
OUT_PARQUET    = BASE / r"output\analysis\discrepancy\tile_discrepancy.parquet"
COVERAGE_CSV   = BASE / r"output\analysis\discrepancy\coverage_alignment_metrics.csv"
CORR_CSV       = BASE / r"output\analysis\discrepancy\alignment_correlation.csv"
REG_CSV        = BASE / r"output\analysis\discrepancy\alignment_regression.csv"
COVERAGE_FIG   = BASE / r"output\analysis\discrepancy\coverage_alignment_trend.png"
MI_CSV  = BASE / r"output\analysis\discrepancy\alignment_mutual_info.csv"
LEE_CSV = BASE / r"output\analysis\discrepancy\alignment_leesL.csv"

HIGH_FIRE_PCTL = 0.85
LOW_FAC_PCTL   = 0.35


# ----------------------------
# Existing functions (unchanged)
# ----------------------------
def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)
    need_cols = {"tile_id", "year", "pred_fire_index"}
    if missing := (need_cols - set(df.columns)):
        raise ValueError(f"Predictions missing columns: {missing}")
    return df.fillna(0)


def load_facility(fp: Path) -> pd.DataFrame:
    f = pd.read_parquet(fp)
    need_cols = {"tile_id", "facility_index"}
    if missing := (need_cols - set(f.columns)):
        raise ValueError(f"Facility file missing columns: {missing}")
    return f[["tile_id", "facility_index"]].fillna(0)


def build_pred_fire_index(pred: pd.DataFrame, year_start=2013, year_end=2023) -> pd.DataFrame:
    scope = pred[(pred["year"] >= year_start) & (pred["year"] <= year_end)].copy()
    agg = scope.groupby("tile_id")["pred_fire_index"].mean().reset_index()
    return agg


def compute_standardized_gap(df: pd.DataFrame) -> pd.DataFrame:
    df["gap"]     = df["pred_fire_index"] - df["facility_index"]
    df["abs_gap"] = df["gap"].abs()
    df["ratio"]   = df["pred_fire_index"] / (df["facility_index"] + 1e-6)

    z_mf  = (df["pred_fire_index"] - df["pred_fire_index"].mean()) / (df["pred_fire_index"].std(ddof=0) + 1e-12)
    z_fac = (df["facility_index"] - df["facility_index"].mean()) / (df["facility_index"].std(ddof=0) + 1e-12)
    df["z_gap"] = z_mf - z_fac

    hi_fire = df["pred_fire_index"] >= df["pred_fire_index"].quantile(HIGH_FIRE_PCTL)
    lo_fac  = df["facility_index"] <= df["facility_index"].quantile(LOW_FAC_PCTL)
    df["under_protected"] = (hi_fire & lo_fac).astype("int8")

    lo_fire = df["pred_fire_index"] <= df["pred_fire_index"].quantile(1 - HIGH_FIRE_PCTL)
    hi_fac  = df["facility_index"] >= df["facility_index"].quantile(1 - LOW_FAC_PCTL)
    df["over_protected"]  = (lo_fire & hi_fac).astype("int8")
    return df


# ----------------------------
# Extended Alignment Functions
# ----------------------------
def compute_correlation(df: pd.DataFrame, out_csv: Path):
    rho, pval = stats.spearmanr(df["pred_fire_index"], df["facility_index"])
    pd.DataFrame({"Spearman_rho": [rho], "p_value": [pval]}).to_csv(out_csv, index=False)
    print(f"Spearman correlation → ρ={rho:.4f}, p={pval:.6f}")


def compute_regression(df: pd.DataFrame, out_csv: Path):
    X = sm.add_constant(df["pred_fire_index"])
    model = sm.OLS(df["facility_index"], X).fit()
    beta = model.params["pred_fire_index"]
    r2 = model.rsquared
    ci = model.conf_int().loc["pred_fire_index"].tolist()
    pd.DataFrame({
        "beta": [beta],
        "R_squared": [r2],
        "conf_int_low": [ci[0]],
        "conf_int_high": [ci[1]]
    }).to_csv(out_csv, index=False)
    print(f"Calibration regression → β={beta:.4f}, R²={r2:.4f}, CI={ci}")


def compute_mutual_information(df: pd.DataFrame, out_csv: Path, bins: int = 20):
    """Compute mutual information and normalized MI between fire and facility indices."""
    df["fire_bin"] = pd.cut(df["pred_fire_index"], bins=bins, labels=False)
    df["fac_bin"]  = pd.cut(df["facility_index"], bins=bins, labels=False)

    mi = mutual_info_score(df["fire_bin"], df["fac_bin"])
    h_x = mutual_info_score(df["fire_bin"], df["fire_bin"])
    h_y = mutual_info_score(df["fac_bin"], df["fac_bin"])
    nmi = mi / np.sqrt(h_x * h_y)

    pd.DataFrame({"MI": [mi], "NMI": [nmi], "bins": [bins]}).to_csv(out_csv, index=False)
    print(f"Mutual Information → MI={mi:.4f}, NMI={nmi:.4f}")


def compute_spatial_pearson(df, tiles_fp, out_csv):
    """Compute global bivariate spatial Pearson statistic using PySAL’s Spatial_Pearson."""
    gdf = gpd.read_file(tiles_fp)[["tile_id", "geometry"]].merge(df, on="tile_id", how="left")
    gdf = gdf.dropna(subset=["pred_fire_index", "facility_index"]).reset_index(drop=True)

    w_obj = Queen.from_dataframe(gdf)
    w_obj.transform = "r"
    w = w_obj.sparse

    sp = Spatial_Pearson(connectivity=w, permutations=999)
    sp.fit(
        gdf["pred_fire_index"].values.reshape(-1, 1),
        gdf["facility_index"].values.reshape(-1, 1)
    )

    stat = sp.association_[0, 1]
    p_val = sp.significance_[0, 1]
    pd.DataFrame({"Spatial_Pearson_stat": [stat], "p_value": [p_val]}).to_csv(out_csv, index=False)

    print(f"Spatial Pearson (Lee’s L equivalent): stat={stat:.4f}, p={p_val:.4f}")



# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    preds = load_predictions(PRED_MULTIYEAR)
    fac   = load_facility(FACILITY_FP)

    pred_fire_index = build_pred_fire_index(preds)
    df = pred_fire_index.merge(fac, on="tile_id", how="left").fillna(0)

    df = compute_standardized_gap(df)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved discrepancy data → {OUT_PARQUET} ({len(df):,} rows)")

    # Alignment assessments
    compute_correlation(df, CORR_CSV)
    compute_regression(df, REG_CSV)
    print(f"All alignment metrics saved under → {BASE / 'output/analysis/discrepancy'}")

    # Mutual information and Spatial Person
    compute_mutual_information(df, MI_CSV)
    compute_spatial_pearson(df, TILES_FP, LEE_CSV)

if __name__ == "__main__":
    main()
