import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from src.data_loader import discover_captures, load_capture_array
from src.features import extract_features_batch
from src.config import PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Discovering captures...")
captures = discover_captures()
print(f"Extracting features for {len(captures)} captures...")

t0 = time.time()
features_df = extract_features_batch(captures, load_capture_array)
print(f"Done in {time.time()-t0:.1f}s. Shape: {features_df.shape}")

out_path = PROCESSED_DIR / "features.parquet"
features_df.to_parquet(out_path)
print(f"Saved: {out_path}")
print(f"\nLabel distribution after extraction:")
print(features_df['label'].value_counts())
print(f"\nFeature columns (first 10): {[c for c in features_df.columns if c not in {'path','run','test_id','board','ts','label','replicate','wave_type_1','start_freq_1','end_freq_1','sweep_time_1','amp_1','wave_type_2','start_freq_2','end_freq_2'}][:10]}")
