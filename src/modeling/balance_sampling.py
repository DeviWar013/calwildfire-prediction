"""
Balanced Sampling Utility

Description:
    Provides class-balanced sampling for wildfire modeling by downsampling
    dominant negative observations and ensuring equal representation of fire
    and non-fire months during model training.

Inputs:
    Any merged training dataframe passed into the sampler.

Outputs:
    Returns a balanced dataframe suitable for fitting tree-based and stacked models.

Notes:
    Sampling ratios and random seed are intentionally kept identical to the
    original pipeline for reproducibility.
"""


import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[2]
yearly_dir = BASE_DIR / "src" / "modeling" / "pdata" / "processed" / "training_yearly"
balanced_dir = BASE_DIR / "src" / "modeling" / "pdata" / "processed" / "training_balanced"
balanced_dir.mkdir(parents=True, exist_ok=True)


def generate_balanced_training_set(
    yearly_dir: Path,
    output_dir: Path,
    label_col: str = "fire_occurred",
    pos_value: int = 1,
    id_col: str = "tile_id",
    tile_partitions: int = 50,
) -> None:
    """Balance positive and negative samples by month."""
    yearly_files = sorted(yearly_dir.glob("training_*.parquet"))

    for file in tqdm(yearly_files, desc="Balancing yearly datasets"):
        print(f"\n⏳ Processing {file.name} ...")
        df = pd.read_parquet(file)

        # label check
        if label_col not in df.columns:
            print(f"⚠️ Label column '{label_col}' missing in {file.name}; skipping.")
            continue

        # Split positive / negative
        positives = df[df[label_col] == pos_value]
        negatives = df[df[label_col] != pos_value]

        if positives.empty or negatives.empty:
            print(f"⚠️ Skipping {file.name} (no positive or negative samples).")
            continue

        n_pos = len(positives)
        tiles = sorted(negatives[id_col].unique())
        tile_chunks = np.array_split(tiles, tile_partitions)

        sampled = []
        for chunk in tile_chunks:
            chunk_df = negatives[negatives[id_col].isin(chunk)]
            for m in range(1, 13):
                month_df = chunk_df[chunk_df["month"] == m]
                n_sample = n_pos // tile_partitions // 12
                if n_sample > 0 and len(month_df) > 0:
                    sampled.append(month_df.sample(n=min(n_sample, len(month_df)), random_state=42))

        negatives_sampled = pd.concat(sampled, axis=0) if sampled else pd.DataFrame()
        balanced = pd.concat([positives, negatives_sampled], axis=0).sample(frac=1.0, random_state=42)

        # Fill NaNs just in case
        balanced = balanced.fillna(0)

        out_path = output_dir / f"{file.stem}_bal.parquet"
        balanced.to_parquet(out_path, index=False)
        print(f"✅ Saved {out_path.name} — {len(balanced):,} rows")


if __name__ == "__main__":
    generate_balanced_training_set(yearly_dir, balanced_dir, tile_partitions=50)
