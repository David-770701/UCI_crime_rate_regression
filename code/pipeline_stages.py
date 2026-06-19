from data_preprocessing import (
    stage_0_output_policy,
    stage_1_data_overview,
    stage_2_clean_data,
    stage_3_detect_outliers,
)
from feature_selection import (
    stage_4_stepwise,
    stage_5_correlation_review,
    stage_6_vif_filtering,
)
from model_diagnostics import (
    stage_7_pre_engineering_diagnostics,
    stage_7_transform_and_diagnostics,
    stage_8_quadratic_engineering,
)

# Compatibility re-export module.
# Older imports can still access stage functions from a single place.
