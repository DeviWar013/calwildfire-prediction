"""
final_prediction.py
----------------------------------------------------------
Evaluate regression-based fire susceptibility predictions
(2013–2023) using both percentile-based classification and
continuous ranking metrics (ROC-AUC, calibration).

This script assumes:
  • training_yearly/*.parquet exist for 2013–2023
  • trained stack_reg_final_*.joblib model available
Outputs:
  - raw_predictions_2013_2023.parquet
  - metrics_rankeval_2013_2023_multi.csv
  - metrics_auc_2013_2023.csv
  - calibration_deciles_2013_2023.csv
  - summary_overall_2013_2023.csv
"""

import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ============================================================
# CONFIGURATION
# ============================================================
BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "src/modeling/models/stack_reg_final_20251104_111614.joblib"
DATA_DIR  = BASE / "src/modeling/data/processed/training_yearly"
OUT_DIR   = BASE / "src/modeling/data/predictions"
YEARS = range(2013, 2024)
PERCENTILES = [10, 20]     # "top p%" thresholds
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent file names
RUN_NAME   = f"{YEARS.start}_{YEARS.stop-1}"
RAW_PATH   = OUT_DIR / f"raw_predictions_{RUN_NAME}.parquet"
RANK_CSV   = OUT_DIR / f"metrics_rankeval_{RUN_NAME}_multi.csv"
AUC_CSV    = OUT_DIR / f"metrics_auc_{RUN_NAME}.csv"
CALIB_CSV  = OUT_DIR / f"calibration_deciles_{RUN_NAME}.csv"
SUMMARY_CSV= OUT_DIR / f"summary_overall_{RUN_NAME}.csv"

# ============================================================
# STEP 1 — LOAD / GENERATE RAW MONTHLY PREDICTIONS
# ============================================================
if RAW_PATH.exists():
    print(f"Found existing predictions → {RAW_PATH}\nSkipping re-prediction.")
    df_all = pd.read_parquet(RAW_PATH)
else:
    print("Generating new monthly predictions ...")
    model = load(MODEL_PATH)
    preds = []
    for year in YEARS:
        fp = DATA_DIR / f"training_{year}.parquet"
        if not fp.exists():
            print(f"Missing {fp.name}, skipping.")
            continue
        df = pd.read_parquet(fp)
        feats = [c for c in df.columns if c not in ["tile_id","year","month","fire_index","fire_occurred"]]
        df["pred_fire_index"] = model.predict(df[feats]).clip(0,1)
        preds.append(df[["tile_id","year","month","pred_fire_index","fire_occurred"]])
    df_all = pd.concat(preds, ignore_index=True)
    df_all.to_parquet(RAW_PATH, index=False)
    print(f"Saved monthly predictions → {RAW_PATH}\nRows: {len(df_all):,}")

# ============================================================
# STEP 2 — PERCENTILE-BASED CLASSIFICATION METRICS
# ============================================================
print("\n=== Evaluating percentile-based ranking ===")
results = []

for year, df_y in df_all.groupby("year"):
    # Aggregate to tile-year maxima
    agg = (
        df_y.groupby("tile_id")["pred_fire_index"]
        .max().reset_index().rename(columns={"pred_fire_index":"pred_fire_index_max"})
    )
    truth = (
        df_y.groupby("tile_id")["fire_occurred"]
        .max().reset_index().rename(columns={"fire_occurred":"fire_occurred_any"})
    )
    merged = agg.merge(truth,on="tile_id",how="left")

    for p in PERCENTILES:
        # 100-p → top p% of highest predictions
        thr = np.percentile(merged["pred_fire_index_max"], 100 - p)
        merged[f"pred_above_p{p}"] = (merged["pred_fire_index_max"] >= thr).astype(int)

        y_true = merged["fire_occurred_any"]
        y_pred = merged[f"pred_above_p{p}"]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        burned    = merged[y_true == 1]
        success   = burned[f"pred_above_p{p}"].mean() if len(burned) > 0 else np.nan

        results.append({
            "year": year, "percentile": p, "threshold_val": thr,
            "burned_tiles": len(burned),
            "success_rate": success,
            "precision": precision, "recall": recall, "f1": f1
        })

eval_df = pd.DataFrame(results)
eval_df.to_csv(RANK_CSV, index=False)

# Print quick summary
pivot_sr = eval_df.pivot(index="year", columns="percentile", values="success_rate")
pivot_f1 = eval_df.pivot(index="year", columns="percentile", values="f1")
print("\n--- Success Rate (recall on burned tiles) ---")
print(pivot_sr.round(3).fillna("-"))
print("\n--- F1 Scores ---")
print(pivot_f1.round(3).fillna("-"))
overall_rank = (
    eval_df.groupby("percentile")[["success_rate","precision","recall","f1"]]
    .mean().round(3)
)
overall_rank.reset_index().to_csv(SUMMARY_CSV,index=False)
print("\n--- Overall Mean Metrics ---")
print(overall_rank)
print(f"Saved: {RANK_CSV}, {SUMMARY_CSV}")

# ============================================================
# STEP 3 — ROC-AUC
# ============================================================
print("\n=== Calculating ROC-AUC ===")
agg_all = (
    df_all.groupby(["tile_id","year"])
    .agg(pred_fire_index_max=("pred_fire_index","max"),
         fire_occurred_any=("fire_occurred","max"))
    .reset_index()
)

auc_rows = []
for year, g in agg_all.groupby("year"):
    if g["fire_occurred_any"].nunique() < 2:
        continue
    auc = roc_auc_score(g["fire_occurred_any"], g["pred_fire_index_max"])
    auc_rows.append({"year": year, "roc_auc": auc})
auc_df = pd.DataFrame(auc_rows).sort_values("year")
auc_df.to_csv(AUC_CSV, index=False)
mean_auc = auc_df["roc_auc"].mean() if not auc_df.empty else np.nan

print("\n--- ROC-AUC by Year ---")
print(auc_df.round(3).to_string(index=False))
print(f"\nOverall mean ROC-AUC: {mean_auc:.3f}")
print(f"Saved: {AUC_CSV}")

# ============================================================
# STEP 4 — CALIBRATION BY DECILES
# ============================================================
print("\n=== Computing calibration (predicted vs observed) ===")
calib_rows = []
for year, g in agg_all.groupby("year"):
    if g["pred_fire_index_max"].nunique() <= 1:
        continue
    g = g.copy()
    g["decile"] = pd.qcut(g["pred_fire_index_max"], q=10, labels=False, duplicates="drop")
    grp = g.groupby("decile", as_index=False).agg(
        pred_mean=("pred_fire_index_max","mean"),
        obs_rate=("fire_occurred_any","mean"),
        n=("tile_id","size")
    )
    grp["year"] = year
    calib_rows.append(grp)

if calib_rows:
    calib_df = pd.concat(calib_rows, ignore_index=True)
    calib_df.to_csv(CALIB_CSV, index=False)
    print(f"Saved calibration deciles → {CALIB_CSV}")
    print("\n--- Calibration sample (latest year) ---")
    y_latest = calib_df["year"].max()
    print(calib_df.query("year == @y_latest")[["decile", "pred_mean", "obs_rate"]].round(4))
else:
    print("No calibration pdata generated (constant predictions).")

