from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_DIR))

from crime_regression_pipeline import main  # noqa: E402
from pipeline_config import PipelineSettings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Communities and Crime regression pipeline.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used by holdout and CV checks.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout fraction for prediction checks.")
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Keep observations that would otherwise be removed by the influence rule.",
    )
    parser.add_argument(
        "--ignore-warnings",
        action="store_true",
        help="Suppress warnings during the full analysis run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        PipelineSettings(
            random_state=args.random_state,
            holdout_test_size=args.test_size,
            remove_outliers=not args.keep_outliers,
            ignore_warnings=args.ignore_warnings,
        )
    )
