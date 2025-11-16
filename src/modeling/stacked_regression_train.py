"""
Stacked Regression Training Pipeline

Description:
    Trains the base models (RF, XGB, LGBM, CatBoost, optional SVR) and fits a
    meta-learner on out-of-fold predictions to produce the final stacked model.

Inputs:
    • Yearly merged training data
    • Best hyperparameters from Optuna

Outputs:
    • Trained base models
    • Meta-learner model
    • Complete stacked regression artifact (joblib)

Notes:
    Model structure and training order are preserved exactly as in the
    validated version.
"""


import os, glob, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import ast
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

# ---------------------------
# Setup. Run Optuna first before you run this.
# ---------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(42)

parquet_dir = "data/processed/training_balanced"
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------
# Load
# ---------------------------
all_parquets = glob.glob(os.path.join(parquet_dir, "training_*.parquet"))
df = pd.concat([pd.read_parquet(fp) for fp in all_parquets], ignore_index=True)



# ---------------------------
# Features / target
# ---------------------------
exclude = ["tile_id", "year", "month", "fire_index","fire_occurred"]
features = [c for c in df.columns if c not in exclude]
X = df[features]
y = df["fire_index"]

# Train/val split (for reporting only)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

# ---------------------------
# Models, Load best Optuna parameters
# ---------------------------

BASE = Path(__file__).resolve().parents[2]
optuna_dir = BASE /"src/modeling/optuna_results"

# Try to find latest Optuna results automatically
try:
    results = sorted(optuna_dir.glob("stack_reg_*/best_params.json"))
    if results:
        latest_result = results[-1]
    else:
        # fallback: manually specify if directory structure not matched. Edit your optuna results here.
        latest_result = Path(r"C:\Users\desmo\PycharmProjects\CalFire_MachineLearning\src\modeling\optuna_results\stack_reg_20251104_022625\best_params.json")
    print(f"📂 Loading best Optuna params from: {latest_result}")
except Exception as e:
    print("⚠️ Could not find Optuna results automatically. Please specify manually.")
    latest_result = Path(r"C:\Users\desmo\PycharmProjects\CalFire_MachineLearning\src\modeling\optuna_results\stack_reg_20251104_022625\best_params.json")

with open(latest_result, "r", encoding="utf-8") as f:
    best_params = json.load(f)

## Helper for cleaner access
get = lambda k, default=None: best_params.get(k, default)

# LightGBM
lgbm = LGBMRegressor(
    learning_rate=get("lgb_lr"),
    num_leaves=int(get("lgb_leaves")),
    n_estimators=int(get("lgb_estimators")),
    subsample=get("lgb_subsample"),
    colsample_bytree=get("lgb_colsample"),
    min_child_samples=int(get("lgb_min_child")),
    reg_alpha=get("lgb_alpha"),
    reg_lambda=get("lgb_lambda"),
    n_jobs=4,
    verbose=-1
)

# CatBoost
cat = CatBoostRegressor(
    learning_rate=get("cat_lr"),
    depth=int(get("cat_depth")),
    l2_leaf_reg=get("cat_l2"),
    n_estimators=int(get("cat_estimators")),
    subsample=get("cat_subsample"),
    thread_count=4,
    verbose=0
)

# XGBoost
xgb = XGBRegressor(
    learning_rate=get("xgb_lr"),
    max_depth=int(get("xgb_depth")),
    n_estimators=int(get("xgb_estimators")),
    subsample=get("xgb_subsample"),
    colsample_bytree=get("xgb_colsample"),
    reg_alpha=get("xgb_alpha"),
    reg_lambda=get("xgb_lambda"),
    n_jobs=4,
    tree_method="hist",
    verbosity=0
)

# MLP
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(int(get("mlp_units")),),
        learning_rate_init=get("mlp_lr"),
        alpha=get("mlp_alpha"),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=int(get("mlp_patience")),
        max_iter=int(get("mlp_maxit")),
        random_state=42
    )
)

# Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ("lgbm", lgbm),
        ("cat", cat),
        ("xgb", xgb),
        ("mlp", mlp)
    ],
    final_estimator=Ridge(alpha=get("ridge_alpha")),
    cv=3,
    n_jobs=1
)

# ---------------------------
# Train with timing
# ---------------------------
print("🚀 Training final tuned stacked ensemble...")
t0 = time.time()
stack.fit(X_train, y_train)
train_secs = time.time() - t0
print(f"⏱️  Training time: {train_secs:.1f}s")

# ---------------------------
# Evaluate
# ---------------------------
preds = stack.predict(X_val)
mse = mean_squared_error(y_val, preds)
rmse = float(np.sqrt(mse))
mae = float(np.mean(np.abs(y_val - preds)))
r2 = float(r2_score(y_val, preds))

print(f"\n✅ Final Training Complete.")
print(f"RMSE: {rmse:.5f}  MAE: {mae:.5f}  R²: {r2:.5f}")

# ---------------------------
# Persist models & metadata
# ---------------------------
meta = {
    "timestamp": ts,
    "features.md": features,
    "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    "models": ["stack", "lgbm", "cat", "xgb", "mlp"],
    "notes": "Final tuned CalFire_ML stacked regression (LGBM, CatBoost, XGBoost, MLP) + Ridge meta."
}

stack_path = model_dir / f"stack_reg_final_{ts}.joblib"
joblib.dump(stack, stack_path)

for name, est in stack.named_estimators_.items():
    joblib.dump(est, model_dir / f"{name}_final_{ts}.joblib")

with open(model_dir / f"stack_reg_final_{ts}_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\n💾 Saved:")
print(f" - Stack: {stack_path}")
for name in ["lgbm", "cat", "xgb", "mlp"]:
    print(f" - {name}: {model_dir / f'{name}_final_{ts}.joblib'}")
print(f" - Metadata: {model_dir / f'stack_reg_final_{ts}_meta.json'}")
