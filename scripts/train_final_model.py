import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from src.config import PROCESSED_DIR
from src.train import feature_columns

df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
cols = feature_columns(df)
X = df[cols].fillna(0.0).to_numpy()
y = df["label"].to_numpy()

print(f"Training on {len(df)} captures, {len(cols)} features, classes={sorted(set(y))}")
clf = GradientBoostingClassifier(random_state=42)
clf.fit(X, y)

out_path = PROCESSED_DIR / "model_gbm.joblib"
joblib.dump({"model": clf, "feature_cols": cols, "classes": clf.classes_.tolist()}, out_path)
print(f"Saved model to {out_path}")
print(f"Train accuracy: {clf.score(X, y):.3f}")
