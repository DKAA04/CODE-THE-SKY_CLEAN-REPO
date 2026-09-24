import io
import unittest

import numpy as np
import pandas as pd

from src.data_loader import load_capture_array, normalize_label
from src.features import extract_features
from src.splitter import split


class PipelineTests(unittest.TestCase):
    def test_all_partial_looseness_spellings_are_one_class(self):
        for value in ["Mix- 45 Deg", "Mix-45 Deg", "Mix-45"]:
            self.assertEqual(normalize_label(value), "Mix-45")
        with self.assertRaises(ValueError):
            normalize_label("unknown")

    def test_short_recordings_are_rejected_not_padded(self):
        with self.assertRaisesRegex(ValueError, "8192"):
            load_capture_array(io.StringIO("X,Y,Z\n1,2,3\n"))

    def test_axis_aliases_share_the_same_loader(self):
        csv = io.StringIO(pd.DataFrame({"x": np.zeros(8192), "Y Axis": np.ones(8192), "z-axis": np.ones(8192)}).to_csv(index=False))
        self.assertEqual(load_capture_array(csv).shape, (8192, 3))

    def test_missing_axes_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing axes"):
            load_capture_array(io.StringIO("x,y\n1,2\n"))

    def test_nonfinite_samples_rejected(self):
        values = np.zeros((8192, 3)); values[5, 0] = np.inf
        csv = io.StringIO(pd.DataFrame(values, columns=["x", "y", "z"]).to_csv(index=False))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_capture_array(csv)

    def test_constant_signal_has_finite_features(self):
        features = extract_features(np.ones((8192, 3)))
        self.assertEqual(len(features), 104)
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertEqual(features["x_rms"], 0)

    def test_board_holdout_is_disjoint_and_reproducible(self):
        df = pd.DataFrame({"board": np.repeat(["a", "b", "c", "d"], 10)})
        a = split(df, "by_board"); b = split(df, "by_board")
        self.assertTrue(a.equals(b))
        self.assertFalse(set(a[a.split == "train"].board) & set(a[a.split == "test"].board))

    def test_single_board_cannot_be_a_board_holdout(self):
        with self.assertRaisesRegex(ValueError, "two distinct"):
            split(pd.DataFrame({"board": ["one", "one"]}), "by_board")

    def test_two_board_split_keeps_training_data(self):
        result = split(pd.DataFrame({"board": ["a", "b"]}), "by_board")
        self.assertEqual(set(result.split), {"train", "test"})

    def test_empty_replicate_holdout_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            split(pd.DataFrame({"replicate": [1, 2, 3]}), "by_replicate")


if __name__ == "__main__":
    unittest.main()
