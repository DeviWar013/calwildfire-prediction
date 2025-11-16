"""
multicollinearity.py
-------------------------------------
Evaluates feature multicollinearity for wildfire ML datasets.
Works with both balanced and complete yearly Parquet files.

Outputs:
  - correlation heatmap (optional)
  - VIF table sorted by severity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
train_dir = BASE_DIR / "src" / "modeling" /"pdata" / "processed" / "training_yearly"
balanced_dir = BASE_DIR / "src" / "modeling" / "pdata" / "processed" / "training_balanced"

YEAR = 2010          # ← choose any year to inspect
BALANCED = True      # ← set to False for complete dataset
SAVE_PLOTS = True

# ============================================================
# LOAD DATA
# ============================================================

def load_data(year, balanced=True):
    if balanced:
        path = balanced_dir / f"training_{year}_bal.parquet"
    else:
        path = train_dir / f"training_{year}.parquet"
    df = pd.read_parquet(path)
    print(f"📂 Loaded {path.name} | {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

df = load_data(YEAR, BALANCED)

# ============================================================
# SELECT NUMERIC FEATURES
# ============================================================

exclude = ["tile_id", "year", "month", "fire_occurred", "fire_index"]
num_df = df.drop(columns=[c for c in exclude if c in df.columns])
num_df = num_df.select_dtypes(include=["number"]).dropna()

print(f"Analyzing {num_df.shape[1]} numeric features after cleanup.")

# ============================================================
# CORRELATION MATRIX
# ============================================================

corr = num_df.corr().round(2)
print("\n--- Top correlated feature pairs (> |0.8|) ---")
high_corr = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr.columns = ["feature_1", "feature_2", "corr"]
print(high_corr[high_corr["corr"].abs() > 0.8].sort_values("corr", ascending=False))

if SAVE_PLOTS:
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(f"Feature Correlation Matrix ({YEAR}, {'balanced' if BALANCED else 'full'})")
    plt.tight_layout()
    plt.show()

# ============================================================
# VARIANCE INFLATION FACTOR (VIF)
# ============================================================

print("\n--- Variance Inflation Factor (VIF) ---")
X = num_df.assign(const=1)
vif_data = pd.DataFrame({
    "feature": num_df.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1]-1)]
})
vif_data.sort_values("VIF", ascending=False, inplace=True)
print(vif_data.head(15))

# Flag potential multicollinearity issues
high_vif = vif_data[vif_data["VIF"] > 5]
if not high_vif.empty:
    print(f"\n⚠️  {len(high_vif)} features with VIF > 5:")
    print(high_vif)
else:
    print("\n✅ No severe multicollinearity detected (VIF ≤ 5).")
