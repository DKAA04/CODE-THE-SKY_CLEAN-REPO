"""Copy the reviewed derived feature table into the local training workspace."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "data" / "features.parquet"
expected = json.loads((root / "data" / "feature_manifest.json").read_text())["sha256"]
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise SystemExit("Feature-table checksum mismatch; do not train from this file.")
destination = root / "data" / "processed" / "features.parquet"
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise SystemExit("A local feature table already exists; keep it or move it before preparing the demo.")
shutil.copyfile(source, destination)
print("Prepared the reviewed 8,221-capture feature table. Run scripts/train_final_model.py next.")
