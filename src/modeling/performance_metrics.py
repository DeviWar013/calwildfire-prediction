"""
Model Performance & Interpretability Tools

Description:
    Generates model evaluation metrics (RMSE, MAE, R²), feature contributions,
    partial dependence plots, and SHAP summaries for interpretability.

Inputs:
    • Trained stacked regression model
    • Evaluation datasets

Outputs:
    • Metrics tables
    • Permutation importance
    • PDP/ALE visualizations

Notes:
    Evaluation logic follows the exact methodology used in model development.
"""


import os, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from PyALE import ale
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parents[2]
data_dir = BASE_DIR / "src" / "modeling" / "pdata" / "processed" / "training_balanced"
model_dir = BASE_DIR / "src" / "modeling" / "models"
out_dir = BASE_DIR / "output" / "discrepancy" / "performance_outputs"
out_dir.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
TEST_SIZE = 0.30


def file_exists(filename):
    """Check if a file already exists in output directory."""
    return (out_dir / filename).exists()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    nrmse = rmse / (y_true.max() - y_true.min())
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MBE": mbe, "NRMSE": nrmse}


# ============================================================
# LOAD MODEL & DATA
# ============================================================
stack_path = sorted(model_dir.glob("stack_reg_final_*.joblib"), key=os.path.getmtime)[-1]
meta_path = sorted(model_dir.glob("stack_reg_final_*_meta.json"), key=os.path.getmtime)[-1]
stack = joblib.load(stack_path)
meta = json.load(open(meta_path, "r", encoding="utf-8"))
features = meta["features.md"]
print(f"Loaded model: {stack_path.name}")

df = pd.concat([pd.read_parquet(fp) for fp in data_dir.glob("training_*.parquet")], ignore_index=True)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X = df[features]; y = df["fire_index"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
print(f"Validation subset: {X_val.shape[0]:,} samples")

# ============================================================
# PERFORMANCE METRICS & HEXBIN
# ============================================================
if not file_exists("metrics_summary.json"):
    print("Computing metrics...")
    y_pred = stack.predict(X_val)
    metrics = evaluate_metrics(y_val, y_pred)
    json.dump(metrics, open(out_dir / "metrics_summary.json", "w", encoding="utf-8"), indent=2)
    print(", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(y_val, y_pred, gridsize=60, cmap="inferno", bins="log")
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    plt.xlabel("Actual Fire Index"); plt.ylabel("Predicted Fire Index")
    plt.title("Actual vs. Predicted Fire Index (Validation Set)")
    cb = plt.colorbar(hb); cb.set_label("Sample density (log scale)")
    plt.text(0.05, 1.05, "Bright = denser, well-predicted samples",
             transform=plt.gca().transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=300)
    plt.close()
else:
    print("Skipped metrics (already exist).")

# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================
if not file_exists("feature_importance_permutation.csv"):
    print("Computing permutation importance...")
    perm = permutation_importance(stack, X_val, y_val, scoring="r2", n_repeats=5, random_state=RANDOM_SEED)
    importances = pd.DataFrame({
        "feature": X_val.columns,
        "importance": perm.importances_mean,
        "std": perm.importances_std
    }).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "feature_importance_permutation.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.barh(importances["feature"].iloc[:15][::-1],
              importances["importance"].iloc[:15][::-1],
              color="#4C92C3", alpha=0.9)
    plt.errorbar(importances["importance"].iloc[:15][::-1],
                 importances["feature"].iloc[:15][::-1],
                 xerr=importances["std"].iloc[:15][::-1],
                 fmt='none', ecolor='black', elinewidth=1.2, capsize=3)
    plt.xlabel("Permutation Importance (ΔR²)")
    plt.title("Top Feature Importance (Permutation)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance_permutation.png", dpi=300)
    plt.close()
else:
    print("Skipped permutation importance (already exist).")
    importances = pd.read_csv(out_dir / "feature_importance_permutation.csv")

# ============================================================
# PDP CURVES
# ============================================================
if not file_exists("pdp_effect_curves.png"):
    print("Generating PDP curves...")
    top_features = importances["feature"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, f in enumerate(top_features):
        try:
            PartialDependenceDisplay.from_estimator(stack, X_val, [f], ax=axes[i], kind="average")
            axes[i].set_title(f"PDP: {f}", fontsize=10)
        except Exception as e:
            axes[i].text(0.5, 0.5, "Error", ha="center", va="center")
            print(f"⚠ PDP failed for {f}: {e}")
    plt.tight_layout()
    plt.savefig(out_dir / "pdp_effect_curves.png", dpi=300)
    plt.close()
else:
    print("Skipped PDP curves (already exist).")

# ============================================================
# ALE CURVES
# ============================================================
ale_outputs = []
for f in importances["feature"].head(6):
    ale_file = out_dir / f"ale_{f}.png"
    ale_outputs.append(ale_file)
    if not ale_file.exists():
        try:
            _ = ale(X=X_val, model=stack, feature=[f], grid_size=20, include_CI=False)
            fig = plt.gcf()
            fig.savefig(ale_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"   • Saved ALE curve for {f}")
        except Exception as e:
            print(f"⚠ ALE failed for {f}: {e}")
    else:
        print(f"Skipped ALE for {f} (already exists).")

# --- Merge ALE plots ---
valid_ales = [img for img in ale_outputs if img.exists()]
if valid_ales:
    imgs = [Image.open(img) for img in valid_ales]
    w, h = imgs[0].size
    grid_w, grid_h = 3, 2  # 6 features.md max
    merged = Image.new("RGB", (w * grid_w, h * grid_h), "white")
    for i, im in enumerate(imgs):
        merged.paste(im, box=((i % grid_w) * w, (i // grid_w) * h))
    merged.save(out_dir / "ale_all_features.png")
    print(f"Combined ALE figure saved to {out_dir/'ale_all_features.png'}")
else:
    print("No ALE plots found to merge.")

print("\nAll discrepancy steps complete.")
