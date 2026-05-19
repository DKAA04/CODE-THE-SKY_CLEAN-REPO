import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_DIR
from src.train import evaluate_all_splits

df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
print(f"Loaded {len(df)} rows, {len(df.columns)} cols")
print(f"Labels: {df['label'].value_counts().to_dict()}\n")

results = evaluate_all_splits(df, model_name="gbm")
print("\n=== MULTI-SPLIT ACCURACY TABLE ===")
print(results.to_string(index=False))

results.to_csv(PROCESSED_DIR / "split_accuracy_table.csv", index=False)
print(f"\nSaved to {PROCESSED_DIR / 'split_accuracy_table.csv'}")
