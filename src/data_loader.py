"""Walks the Run/TestID/SensorBoardID/Timestamp.csv tree, joins with Test.csv metadata."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import DATA_ROOT, TEST_CSV, N_SAMPLES

CAPTURE_RE = re.compile(r".*/(?P<run>[ABC])/(?P<test_id>\d+)/(?P<board>\d+)/(?P<ts>\d+)\.csv$")

EXCITATION_COLS = [
    "Wave Type 1", "Start Freq 1", "End Freq 1", "Sweep Time 1", "Amp 1",
    "Wave Type 2", "Start Freq 2", "End Freq 2",
]
EXCITATION_RENAME = {
    "Wave Type 1": "wave_type_1", "Start Freq 1": "start_freq_1",
    "End Freq 1": "end_freq_1", "Sweep Time 1": "sweep_time_1", "Amp 1": "amp_1",
    "Wave Type 2": "wave_type_2", "Start Freq 2": "start_freq_2", "End Freq 2": "end_freq_2",
}


def _replicate(test_id: int) -> int:
    return test_id % 10


def load_test_metadata(path: Path = TEST_CSV) -> pd.DataFrame:
    """Load the master Test.csv index. Be permissive about column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_label(value: str) -> str:
    """Collapse spelling/spacing variants of the three documented lab conditions."""
    key = re.sub(r"[\s-]+", "", str(value)).lower()
    labels = {"30nm": "30NM", "loose": "Loose", "mix45deg": "Mix-45", "mix45": "Mix-45"}
    if key not in labels:
        raise ValueError(f"Unknown bolt-condition label: {value!r}")
    return labels[key]


def discover_captures(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Walk the tree and join with Test.csv; returns one row per capture CSV."""
    rows = []
    root = Path(root)
    for csv_path in tqdm(sorted(root.rglob("*.csv")), desc="Discovering CSVs"):
        if csv_path.name == "Test.csv":
            continue
        m = CAPTURE_RE.match(csv_path.as_posix())
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "path": str(csv_path),
            "run": d["run"],
            "test_id": d["test_id"],
            "board": d["board"],
            "ts": d["ts"],
        })

    if not rows:
        raise ValueError(f"No capture CSVs found under {root}. See docs/DATA.md.")
    captures = pd.DataFrame(rows)
    captures["test_id_int"] = captures["test_id"].astype(int)

    meta = load_test_metadata(root / "Test.csv")
    required = {"Test", "Run", "Bolt Status", *EXCITATION_COLS}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Test.csv is missing columns: {sorted(missing)}")
    meta["test_id_int"] = meta["Test"].astype(int)
    meta_slim = meta[["Run", "test_id_int", "Bolt Status"] + EXCITATION_COLS].copy()

    merged = captures.merge(
        meta_slim,
        left_on=["run", "test_id_int"],
        right_on=["Run", "test_id_int"],
        how="left",
        validate="many_to_one",
    )

    unmatched = merged["Bolt Status"].isna().sum()
    if unmatched:
        raise ValueError(f"{unmatched} captures have no Test.csv label; extraction stopped.")

    merged = merged.rename(columns={"Bolt Status": "label", **EXCITATION_RENAME})
    merged["label"] = merged["label"].map(normalize_label)
    merged["replicate"] = merged["test_id_int"].apply(_replicate)
    merged = merged.drop(columns=["Run", "test_id_int"])

    col_order = ["path", "run", "test_id", "board", "ts", "label", "replicate",
                 "wave_type_1", "start_freq_1", "end_freq_1", "sweep_time_1", "amp_1",
                 "wave_type_2", "start_freq_2", "end_freq_2"]
    return merged[col_order]


def load_capture(path) -> pd.DataFrame:
    """Load one CSV. Returns DataFrame with columns X-axis, Y-Axis, Z-Axis (8,192 rows)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Be tolerant to column-name variants
    rename_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("-", "")
        if cl in ("xaxis", "x"):
            rename_map[c] = "X-axis"
        elif cl in ("yaxis", "y"):
            rename_map[c] = "Y-Axis"
        elif cl in ("zaxis", "z"):
            rename_map[c] = "Z-Axis"
    df = df.rename(columns=rename_map)
    axes = ["X-axis", "Y-Axis", "Z-Axis"]
    if not df.columns.is_unique:
        raise ValueError("Capture has duplicate axis columns.")
    missing = set(axes) - set(df.columns)
    if missing:
        raise ValueError(f"Capture is missing axes: {sorted(missing)}")
    if len(df) < N_SAMPLES:
        raise ValueError(f"Capture needs at least {N_SAMPLES} samples; received {len(df)}.")
    arr = df[axes].iloc[:N_SAMPLES].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Capture contains missing or non-finite samples.")
    return df


def load_capture_array(path):
    """Return the first N_SAMPLES validated samples; never pad missing observations."""
    df = load_capture(path)
    arr = df[["X-axis", "Y-Axis", "Z-Axis"]].to_numpy(dtype=float)
    return arr[:N_SAMPLES]


if __name__ == "__main__":
    captures = discover_captures()
    print(f"Found {len(captures)} captures")
    print(captures.head())
    print("\nLabel distribution:")
    print(captures["label"].value_counts())
