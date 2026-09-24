"""Record a fixed, untuned multi-split benchmark and the data needed to inspect it."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import DATA_ROOT, PROJECT_ROOT, PROCESSED_DIR
from src.data_loader import normalize_label
from src.splitter import split
from src.train import feature_columns


def main() -> None:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df["label"] = df["label"].map(normalize_label)
    # Keep the reusable feature table portable and free of local machine paths.
    df["path"] = df["path"].map(lambda p: Path(p).relative_to(DATA_ROOT).as_posix() if Path(p).is_absolute() else p)
    df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    cols = feature_columns(df)
    labels = ["30NM", "Loose", "Mix-45"]
    raw_files = sorted(DATA_ROOT.rglob("*.csv"))
    manifest_path = reports / "dataset_manifest.csv"
    if raw_files:
        manifest = [{"path": p.relative_to(DATA_ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw_files]
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    else:
        previous = json.loads((reports / "evaluation.json").read_text())
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != previous["dataset_manifest_sha256"]:
            raise ValueError("Published raw-data manifest checksum mismatch")
        print("Using the published raw-data manifest; source CSVs are not reverified in this run.")
    source_hashes = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ["src", "scripts"] for p in sorted((PROJECT_ROOT / folder).glob("*.py"))}
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Code the Sky 2026 ADB Safegate structural integrity lab captures",
        "capture_count": len(df), "feature_count": len(cols), "class_counts": df["label"].value_counts().to_dict(),
        "label_normalization": "Mix- 45 Deg and Mix-45 Deg both map to Mix-45",
        "raw_csvs_verified_this_run": bool(raw_files),
        "features_sha256": hashlib.sha256((PROCESSED_DIR / "features.parquet").read_bytes()).hexdigest(),
        "seed": 42, "model": "GradientBoostingClassifier(random_state=42); all other sklearn defaults",
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow"]},
        "source_sha256": source_hashes,
        "dataset_manifest_sha256": hashlib.sha256((reports / "dataset_manifest.csv").read_bytes()).hexdigest(),
        "splits": [],
    }
    assignments = df[["path", "board", "replicate", "label"]].copy()
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    for axis, strategy in zip(axes, ["random", "by_replicate", "by_board", "by_excitation", "combined"]):
        part = split(df, strategy=strategy, seed=42)
        assignments[strategy] = part["split"]
        train, test = part[part["split"] == "train"], part[part["split"] == "test"]
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(train[cols], train["label"])
        pred = clf.predict(test[cols])
        cm = confusion_matrix(test["label"], pred, labels=labels)
        record = {"strategy": strategy, "train_count": len(train), "test_count": len(test),
                  "train_accuracy": accuracy_score(train["label"], clf.predict(train[cols])),
                  "test_accuracy": accuracy_score(test["label"], pred),
                  "train_boards": sorted(train["board"].astype(str).unique().tolist()),
                  "test_boards": sorted(test["board"].astype(str).unique().tolist()),
                  "classification_report": classification_report(test["label"], pred, labels=labels, output_dict=True, zero_division=0),
                  "confusion_matrix": cm.tolist(), "class_order": labels}
        result["splits"].append(record)
        axis.imshow(cm, cmap="Blues")
        axis.set(title=f'{strategy}\naccuracy {record["test_accuracy"]:.1%}', xlabel="Predicted", ylabel="Actual")
        axis.set_xticks(range(3), labels, rotation=45, ha="right"); axis.set_yticks(range(3), labels)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
        (reports / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(strategy, record["test_accuracy"], flush=True)
    assignments.to_csv(reports / "split_assignments.csv", index=False)
    fig.tight_layout(); fig.savefig(reports / "confusion_matrices.png", dpi=160); plt.close(fig)
    lines = ["# Reproduced evaluation", "", "Recorded locally on the authorized hackathon dataset. No cloud services were called.", "",
             f"{len(df):,} captures; {len(cols)} signal features; three normalized condition labels; seed 42.", "",
             "| Split | Train captures | Test captures | Test accuracy |", "| --- | ---: | ---: | ---: |"]
    for r in result["splits"]:
        lines.append(f'| {r["strategy"]} | {r["train_count"]} | {r["test_count"]} | {r["test_accuracy"]:.2%} |')
    lines += ["", "![Confusion matrices](confusion_matrices.png)", "", "## Interpretation", "",
              "The board split holds out whole sensor boards. The combined split holds out only the intersection of selected boards and replicates; it is not an unseen-board evaluation. Random splits share experimental conditions between training and test data. These are fixed single-seed holdouts, not cross-validation or independent field validation.", "",
              "The original metadata uses two spellings of the partially loosened condition. Both are normalized to Mix-45 here, giving the three classes specified by the challenge. These results therefore supersede the unverified historical local score; they do not reproduce the original label handling. BigQuery ML was not rerun.", "",
              "The lab runs correspond to bolt conditions, so run-specific effects can be confounded with condition. Generalization to other fixtures, mounting environments, unseen faults or operational airports is unproven. Class probabilities are uncalibrated.", "",
              "## Reproduction", "", "For a fresh checkout, run `python scripts/prepare_demo.py` then `python scripts/evaluate_reproducible.py`. To also verify feature extraction, obtain the authorized raw archive and run `python scripts/extract_all_features.py` before evaluation. Without raw CSVs, the script preserves the published manifest and explicitly records that it did not reverify the source files. See [data provenance](../docs/DATA.md).", "",
              "[Machine-readable results and environment](evaluation.json) · [Exact split assignments](split_assignments.csv) · [Dataset checksums](dataset_manifest.csv)", "",
              "This portfolio maintenance and evaluation pass was performed with Codex assistance on 24 September 2026; it is separate from the original hackathon work."]
    (reports / "evaluation.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
