"""
Model Hyperparameter Optimization (Optuna)

Description:
    Performs Optuna-based hyperparameter search for LightGBM and XGBoost
    regressors using cross-validated evaluation on historical training data.

Inputs:
    • Yearly merged training data
    • Optuna study settings

Outputs:
    • Best hyperparameter settings (joblib/json)
    • Optimization history and performance plots

Notes:
    Search spaces and objective definitions remain unchanged for consistency.
"""


import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import gc
import glob
import os

# Assuming X and y are already prepared (you can load from saved Parquet)
# ------------------------------------------------------------
# X, y = ...
BASE_DIR = Path(__file__).resolve().parents[2]
parquet_dir = BASE_DIR/"src"/"modeling"/"pdata"/"processed"/"training_balanced"
all_parquets = glob.glob(os.path.join(parquet_dir, "training_*.parquet"))
df = pd.concat([pd.read_parquet(fp) for fp in all_parquets], ignore_index=True)
max_rows = 40000
if len(df) > max_rows:
    df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {len(df):,} rows for tuning.")


# Sanitize NaNs & dtypes
df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# ---------------------------
# Features / target
# ---------------------------
exclude = ["tile_id", "year", "month", "fire_index","fire_occurred"]
features = [c for c in df.columns if c not in exclude]
X = df[features]
y = df["fire_index"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------------------------------------

def objective(trial):
    # --- LightGBM params ---
    lgbm_params = {
        "learning_rate": trial.suggest_float("lgb_lr", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("lgb_leaves", 15, 255),
        "n_estimators": trial.suggest_int("lgb_estimators", 100, 800),
        "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("lgb_min_child", 10, 60),
        "reg_alpha": trial.suggest_float("lgb_alpha", 1e-5, 1e-1, log=True),
        "reg_lambda": trial.suggest_float("lgb_lambda", 1e-5, 1e-1, log=True),
        "n_jobs": 1,
        "verbose": -1,
    }
    lgbm = LGBMRegressor(**lgbm_params)

    # --- CatBoost params ---
    cat_params = {
        "learning_rate": trial.suggest_float("cat_lr", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("cat_depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 1.0, 10.0, log=True),
        "n_estimators": trial.suggest_int("cat_estimators", 100, 800),
        "subsample": trial.suggest_float("cat_subsample", 0.5, 1.0),
        "thread_count": 4,
        "verbose": 0
    }
    cat = CatBoostRegressor(**cat_params)

    # --- XGBoost params ---
    xgb_params = {
        "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("xgb_depth", 3, 10),
        "n_estimators": trial.suggest_int("xgb_estimators", 150, 800),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("xgb_alpha", 1e-5, 1e-1, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1e-5, 1e-1, log=True),
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0
    }
    xgb = XGBRegressor(**xgb_params)

    # --- Efficient MLP (scikit-learn) ---
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(
                trial.suggest_int("mlp_units", 32, 128),
            ),
            learning_rate_init=trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
            alpha=trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=trial.suggest_int("mlp_patience", 8, 15),
            max_iter=trial.suggest_int("mlp_maxit", 150, 300),
            random_state=42
        )
    )

    # --- Ridge meta ---
    ridge_alpha = trial.suggest_float("ridge_alpha", 1e-3, 10.0, log=True)

    # --- Stack ---
    stack = StackingRegressor(
        estimators=[
            ("lgbm", lgbm),
            ("cat", cat),
            ("xgb", xgb),
            ("mlp", mlp)
        ],
        final_estimator=Ridge(alpha=ridge_alpha),
        cv=2,
        n_jobs=1
    )

    stack.fit(X_train, y_train)
    preds = stack.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    del stack, lgbm, cat, xgb, mlp
    gc.collect()
    return rmse


# -------------------------------
# Run Optuna study
# -------------------------------
sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=80, n_jobs = 1)  # 80 trials

print("\nBest RMSE:", study.best_value)
print("Best params:", study.best_params)

# Optionally save study
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(BASE_DIR) / "src" / "modeling" / "optuna_results" / f"stack_reg_{timestamp}"
result_dir.mkdir(parents=True, exist_ok=True)

# --- Save trial results to CSV ---
results = []
for t in study.trials:
    trial_dict = t.params.copy()
    trial_dict["value_rmse"] = t.value
    # optional secondary metrics if available
    trial_dict["number"] = t.number
    results.append(trial_dict)
pd.DataFrame(results).to_csv(result_dir / "optuna_trials.csv", index=False)
print(f"Trial results saved to {result_dir/'optuna_trials.csv'}")

# --- Save best parameters as JSON ---
with open(result_dir / "best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
print(f"Best parameters saved to {result_dir/'best_params.json'}")

# --- Optional evaluation on validation for R² trend ---
rmse_list = [t.value for t in study.trials if t.value is not None]
r2_list = []
for t in study.trials:
    # quick recompute of R² if stored in user_attrs (optional)
    r2_list.append(t.user_attrs.get("r2", np.nan) if hasattr(t, "user_attrs") else np.nan)

# --- Plot RMSE trend ---
plt.figure(figsize=(7,5))
plt.plot(range(1, len(rmse_list)+1), rmse_list, color="tab:blue", lw=2)
plt.title("Optuna RMSE Trend per Trial")
plt.xlabel("Trial Number")
plt.ylabel("RMSE")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(result_dir / "optuna_rmse_trend.png", dpi=300)
plt.close()

# --- Plot R² trend (if available) ---
if not all(np.isnan(r2_list)):
    plt.figure(figsize=(7,5))
    plt.plot(range(1, len(r2_list)+1), r2_list, color="tab:green", lw=2)
    plt.title("Optuna R² Trend per Trial")
    plt.xlabel("Trial Number")
    plt.ylabel("R²")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(result_dir / "optuna_r2_trend.png", dpi=300)
    plt.close()

print(f"Plots saved to {result_dir}")