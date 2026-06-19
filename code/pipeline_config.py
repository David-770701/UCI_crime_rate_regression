from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib
import pandas as pd


matplotlib.use("Agg")

# Central configuration for paths, thresholds, and key output names.
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

RAW_DATA_FILE = PROJECT_ROOT / "communities.data"
RAW_NAMES_FILE = PROJECT_ROOT / "communities.names"
DATA_FILE = PROJECT_ROOT / "data" / "communities.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
IMG_DIR = OUTPUT_DIR / "images"

IMPORTANT_CSV_FILES = [
    "communities.csv",
    "communities_excluded_police.csv",
    "target_transform_comparison.csv",
    "stepwise_f_test_summary.csv",
    "vif_removal_log.csv",
    "feature_count_summary.csv",
    "quadratic_term_screening.csv",
    "final_model_selection_comparison.csv",
    "final_model_coefficients.csv",
    "final_model_cross_validation.csv",
    "final_model_kfold_cv_100.csv",
    "partial_r2_ranking.csv",
    "overall_model_f_test.csv",
    "final_engineered_vif.csv",
]

TARGET = "ViolentCrimesPerPop"
ID_COLS = ["state", "county", "community", "communityname", "fold"]
HIGH_MISSING_THRESHOLD = 0.80
CORR_THRESHOLD = 0.70
VIF_THRESHOLD = 10.0
RANDOM_STATE = 42
HOLDOUT_TEST_SIZE = 0.20
REMOVE_OUTLIERS = True
USE_STEPWISE = True
STEPWISE_EXCLUDE_COLS = ["LemasPctOfficDrugUn"]
# Weak adjustment variables removed before the final engineered model is refit.
FINAL_DROP_VARIABLES = ["NumStreet", "PctPopUnderPov"]
QUADRATIC_SEQUENCE = ["racepctblack", "PctIlleg", "PctPersDenseHous"]
REPRESENTATIVE_CANDIDATE_IDS = ["M0", "M1", "M4", "M6", "M10"]
LOG_C_VALUES = [
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.045,
    0.05,
    0.055,
    0.06,
    0.065,
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.15,
]
BOXCOX_C_VALUES = LOG_C_VALUES.copy()
LOGIT_EPS = 1e-3


@dataclass(frozen=True)
class PipelineSettings:
    random_state: int = RANDOM_STATE
    holdout_test_size: float = HOLDOUT_TEST_SIZE
    remove_outliers: bool = REMOVE_OUTLIERS
    use_stepwise: bool = USE_STEPWISE
    ignore_warnings: bool = False


def configure_runtime(settings: PipelineSettings | None = None) -> PipelineSettings:
    settings = settings or PipelineSettings()
    if settings.ignore_warnings:
        warnings.filterwarnings("ignore")
    return settings


def ensure_output_dirs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
