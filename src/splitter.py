"""Leakage-aware train/test splits per student manual 5.2."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

SplitStrategy = Literal["random", "by_replicate", "by_board", "by_excitation", "combined"]


def split(df: pd.DataFrame, strategy: SplitStrategy = "by_replicate", seed: int = 42):
    """Returns df with new column `split` in {'train','test'}."""
    df = df.copy()
    rng = np.random.default_rng(seed)

    if strategy == "random":
        df["split"] = rng.choice(["train", "test"], size=len(df), p=[0.8, 0.2])

    elif strategy == "by_replicate":
        df["split"] = "train"
        mask_test = df["replicate"].isin([4, 5])
        df.loc[mask_test, "split"] = "test"

    elif strategy == "by_board":
        boards = sorted(df["board"].dropna().unique())
        if len(boards) < 2:
            raise ValueError("A board holdout requires at least two distinct boards.")
        n_test = min(len(boards) - 1, max(2, len(boards) // 7))
        test_boards = rng.choice(boards, size=n_test, replace=False)
        df["split"] = np.where(df["board"].isin(test_boards), "test", "train")
        print(f"  test boards: {sorted(test_boards.tolist())}")

    elif strategy == "by_excitation":
        exc_cols = ["start_freq_1", "end_freq_1", "start_freq_2", "end_freq_2"]
        filled = df[exc_cols].fillna(-1)
        df["excitation_pair"] = list(zip(filled["start_freq_1"], filled["end_freq_1"],
                                         filled["start_freq_2"], filled["end_freq_2"]))
        pairs = sorted(df["excitation_pair"].unique())
        n_test = max(1, len(pairs) // 8)
        test_pairs = [tuple(p) for p in rng.choice(np.array(pairs, dtype=object), size=n_test, replace=False)]
        df["split"] = np.where(df["excitation_pair"].isin(test_pairs), "test", "train")
        df = df.drop(columns=["excitation_pair"])
        print(f"  test excitation pairs: {test_pairs}")

    elif strategy == "combined":
        boards = sorted(df["board"].dropna().unique())
        if len(boards) < 2:
            raise ValueError("A combined split requires at least two distinct boards.")
        test_boards = set(rng.choice(boards, size=min(len(boards) - 1, max(2, len(boards) // 7)), replace=False))
        df["split"] = "train"
        df.loc[
            df["board"].isin(test_boards) & df["replicate"].isin([4, 5]),
            "split",
        ] = "test"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    counts = df["split"].value_counts().to_dict()
    if not counts.get("train") or not counts.get("test"):
        raise ValueError(f"{strategy} produced an empty train or test partition.")
    print(f"[split={strategy}] {counts}")
    return df


def all_splits_summary(df: pd.DataFrame) -> dict:
    """Returns a dict of {strategy: split_df} for all 5 strategies."""
    return {
        s: split(df, strategy=s)
        for s in ["random", "by_replicate", "by_board", "by_excitation", "combined"]
    }
