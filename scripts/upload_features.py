import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_DIR
from src.upload_to_bq import upload_features
from src.splitter import split

df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
print(f"Loaded {len(df)} rows")

df = split(df, strategy="by_board", seed=42)
print(f"After split: {df['split'].value_counts().to_dict()}")

df = df.drop(columns=["wave_type_1", "wave_type_2"], errors="ignore")

table_ref = upload_features(df)
print(f"\nDone. Table: {table_ref}")
