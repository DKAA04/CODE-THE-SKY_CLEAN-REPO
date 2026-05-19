"""Walks the Run/TestID/SensorBoardID/Timestamp.csv tree, joins with Test.csv metadata."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
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


def discover_captures(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Walk the tree and join with Test.csv; returns one row per capture CSV."""
    rows = []
    for csv_path in tqdm(list(root.rglob("*.csv")), desc="Discovering CSVs"):
        if csv_path.name == "Test.csv":
            continue
        m = CAPTURE_RE.match(str(csv_path))
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

    captures = pd.DataFrame(rows)
    captures["test_id_int"] = captures["test_id"].astype(int)

    meta = load_test_metadata()
    meta["test_id_int"] = meta["Test"].astype(int)
    meta_slim = meta[["Run", "test_id_int", "Bolt Status"] + EXCITATION_COLS].copy()

    merged = captures.merge(
        meta_slim,
        left_on=["run", "test_id_int"],
        right_on=["Run", "test_id_int"],
        how="left",
    )

    unmatched = merged["Bolt Status"].isna().sum()
    if unmatched:
        print(f"WARNING: {unmatched} captures had no Test.csv match")

    merged = merged.rename(columns={"Bolt Status": "label", **EXCITATION_RENAME})
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
    return df


def load_capture_array(path):
    """Returns (N, 3) numpy array of [X, Y, Z]. Trims/pads to N_SAMPLES."""
    import numpy as np
    df = load_capture(path)
    arr = df[["X-axis", "Y-Axis", "Z-Axis"]].to_numpy(dtype=float)
    if arr.shape[0] >= N_SAMPLES:
        return arr[:N_SAMPLES]
    pad = np.zeros((N_SAMPLES - arr.shape[0], 3))
    return np.vstack([arr, pad])


if __name__ == "__main__":
    captures = discover_captures()
    print(f"Found {len(captures)} captures")
    print(captures.head())
    print("\nLabel distribution:")
    print(captures["label"].value_counts())
