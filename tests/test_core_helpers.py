from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from data_preprocessing import parse_attribute_names  # noqa: E402
from evaluation import apply_response_transform, inverse_response_transform  # noqa: E402
from pipeline_utils import compute_kfold_cv_metrics, compute_vif, fit_ols_model, safe_to_csv  # noqa: E402


class CoreHelperTests(unittest.TestCase):
    def test_parse_attribute_names_reads_arff_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            names_path = Path(temp_dir) / "sample.names"
            names_path.write_text(
                "@relation example\n"
                "@attribute state numeric\n"
                "@attribute communityname string\n"
                "@attribute ViolentCrimesPerPop numeric\n",
                encoding="utf-8",
            )

            self.assertEqual(
                parse_attribute_names(names_path),
                ["state", "communityname", "ViolentCrimesPerPop"],
            )

    def test_response_transform_round_trips_log_model(self) -> None:
        y = pd.Series([0.0, 0.2, 1.0])
        transformed = apply_response_transform(y, "log(y + 0.035)")
        recovered = inverse_response_transform(transformed.to_numpy(), "log(y + 0.035)")

        np.testing.assert_allclose(recovered, y.to_numpy(), rtol=1e-12, atol=1e-12)

    def test_kfold_metrics_are_reproducible_with_fixed_seed(self) -> None:
        X = pd.DataFrame({"x1": np.linspace(0, 1, 20), "x2": np.linspace(1, 0, 20)})
        y = pd.Series(1.0 + 2.0 * X["x1"] - X["x2"])

        first = compute_kfold_cv_metrics(X, y, k=5, random_state=42)
        second = compute_kfold_cv_metrics(X, y, k=5, random_state=42)

        self.assertEqual(first["random_state"], 42)
        self.assertEqual(first["press_kfold"], second["press_kfold"])
        self.assertTrue(math.isfinite(first["r2_pred_kfold"]))

    def test_compute_vif_returns_finite_values_for_independent_features(self) -> None:
        X = pd.DataFrame(
            {
                "x1": [0.0, 1.0, 0.0, 1.0, 0.5],
                "x2": [0.0, 0.0, 1.0, 1.0, 0.25],
            }
        )

        vif_df = compute_vif(X)

        self.assertEqual(set(vif_df["variable"]), {"x1", "x2"})
        self.assertTrue(np.isfinite(vif_df["VIF"]).all())

    def test_safe_to_csv_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "table.csv"
            returned_path = safe_to_csv(pd.DataFrame({"a": [1]}), output_path, index=False)

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
