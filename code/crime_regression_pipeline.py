from __future__ import annotations

from pipeline_config import PipelineSettings, configure_runtime, ensure_output_dirs
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


def main(settings: PipelineSettings | None = None) -> None:
    settings = configure_runtime(settings)
    ensure_output_dirs()
    stage_0_output_policy()
    raw_df, _ = stage_1_data_overview()
    clean_df, _, _ = stage_2_clean_data(raw_df)
    if settings.remove_outliers:
        df_no_outliers, _ = stage_3_detect_outliers(raw_df, clean_df)
    else:
        df_no_outliers = clean_df.reset_index(drop=True)
    transform_result = stage_7_transform_and_diagnostics(df_no_outliers)
    stepwise_df, X_step, y_step, y_raw_step, selected_vars = stage_4_stepwise(
        df_no_outliers,
        y_step=transform_result["final_response_series"],
        response_label=transform_result["final_model_name"],
    )
    stage_5_correlation_review(
        stepwise_df,
        y_corr=y_step,
        response_label=transform_result["final_model_name"],
    )
    _, X_final, y_final, y_raw_final, _ = stage_6_vif_filtering(
        stepwise_df,
        X_step,
        selected_vars,
        y_vif=y_step,
        y_raw_reference=y_raw_step,
        response_label=transform_result["final_model_name"],
    )
    pre_engineering_result = stage_7_pre_engineering_diagnostics(
        X_final=X_final,
        y_raw_final=y_raw_final,
        transform_result=transform_result,
    )
    stage_8_quadratic_engineering(
        X_final=X_final,
        final_model_name=pre_engineering_result["final_model_name"],
        final_response_series=y_final,
        y_raw_final=y_raw_final,
        raw_df=raw_df,
        holdout_test_size=settings.holdout_test_size,
        random_state=settings.random_state,
    )


if __name__ == "__main__":
    main()
