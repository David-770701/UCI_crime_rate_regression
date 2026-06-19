from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pipeline_config import CORR_THRESHOLD, CSV_DIR, STEPWISE_EXCLUDE_COLS, TARGET, VIF_THRESHOLD
from pipeline_utils import (
    compute_feature_f_tests,
    compute_high_corr_pairs,
    compute_vif,
    fit_ols_model,
    iterative_vif_filter,
    print_dataframe,
    print_section,
    safe_to_csv,
    save_current_figure,
    stepwise_selection,
)


def stage_4_stepwise(
    df_no_outliers: pd.DataFrame,
    y_step: pd.Series | None = None,
    response_label: str = "raw y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    y_raw = df_no_outliers[TARGET].copy()
    if y_step is None:
        y_step = y_raw.copy()
    else:
        y_step = pd.Series(y_step, index=df_no_outliers.index, name=getattr(y_step, "name", TARGET))

    X_step = df_no_outliers.drop(columns=[TARGET]).copy()
    excluded_before_stepwise = [col for col in STEPWISE_EXCLUDE_COLS if col in X_step.columns]
    X_step = X_step.drop(columns=excluded_before_stepwise, errors="ignore")

    print_section("Run Stepwise Variable Selection")
    print(f"Response used for Stepwise: {response_label}")
    if excluded_before_stepwise:
        print(f"Excluded before stepwise: {excluded_before_stepwise}")
    # Stepwise selection uses partial F-tests rather than p-values.
    selected_vars = stepwise_selection(X_step, y_step, alpha_in=0.05, alpha_out=0.05, verbose=True)

    selected_X = X_step[selected_vars].copy()
    stepwise_df = pd.concat([selected_X, y_raw.rename(TARGET)], axis=1)
    stepwise_model = fit_ols_model(selected_X, y_step)
    stepwise_f_tests = compute_feature_f_tests(X_step, y_step, selected_vars)

    print_section("Stepwise Selection Result")
    print(f"Number of retained variables: {len(selected_vars)}")
    for i, var in enumerate(selected_vars, 1):
        print(f"{i:2d}. {var}")

    print_section("Stepwise Model Summary")
    print(f"Adjusted R^2: {stepwise_model.rsquared_adj:.4f}")
    print(f"AIC: {stepwise_model.aic:.2f}")
    print(f"BIC: {stepwise_model.bic:.2f}")
    print("The Stepwise variable list is shown in the console and used in later stages.")

    print_section("Stepwise Feature Partial F Tests")
    print_dataframe(stepwise_f_tests.round(6))
    safe_to_csv(stepwise_f_tests, CSV_DIR / "stepwise_f_test_summary.csv", index=False, encoding="utf-8-sig")

    return stepwise_df, X_step, y_step, y_raw, selected_vars


def stage_5_correlation_review(
    stepwise_df: pd.DataFrame,
    y_corr: pd.Series,
    response_label: str = "raw y",
) -> pd.DataFrame:
    X_corr = stepwise_df.drop(columns=[TARGET]).copy()
    y_corr = pd.Series(y_corr, index=stepwise_df.index, name=getattr(y_corr, "name", TARGET))
    # Correlation review is descriptive only; actual deletion happens in the VIF stage.
    high_corr_pairs_df = compute_high_corr_pairs(X_corr, y_corr, CORR_THRESHOLD)

    print_section(f"Highly Correlated Variable Pairs (|r| > {CORR_THRESHOLD})")
    print(f"Response used for correlation review: {response_label}")
    if not high_corr_pairs_df.empty:
        print_dataframe(high_corr_pairs_df.drop(columns=["abs_correlation"]))
    else:
        print("No variable pairs in the Stepwise result exceed the correlation threshold.")

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        X_corr.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap After Stepwise")
    save_current_figure("02_correlation_heatmap_after_stepwise.png")
    safe_to_csv(high_corr_pairs_df, CSV_DIR / "high_corr_pairs_summary.csv", index=False, encoding="utf-8-sig")
    return high_corr_pairs_df


def stage_6_vif_filtering(
    stepwise_df: pd.DataFrame,
    X_step: pd.DataFrame,
    selected_vars: list[str],
    y_vif: pd.Series,
    y_raw_reference: pd.Series,
    response_label: str = "raw y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    X_vif_start = stepwise_df.drop(columns=[TARGET]).copy()
    y_vif = pd.Series(y_vif, index=stepwise_df.index, name=getattr(y_vif, "name", TARGET))
    y_raw_reference = pd.Series(y_raw_reference, index=stepwise_df.index, name=TARGET)

    print_section("Before VIF Filtering")
    print(f"Response used for VIF filtering: {response_label}")
    vif_before = compute_vif(X_vif_start)
    print_dataframe(vif_before.round(6))

    # Remove variables iteratively by comparing candidate trial drops,
    # rather than deleting the current max-VIF variable mechanically.
    X_vif_final, removed_by_vif, trial_history, vif_history = iterative_vif_filter(
        X_vif_start,
        y_vif,
        threshold=VIF_THRESHOLD,
        corr_threshold=CORR_THRESHOLD,
    )

    final_df = pd.concat([X_vif_final, y_raw_reference.rename(TARGET)], axis=1)
    X_final = final_df.drop(columns=[TARGET]).copy()
    y_raw_final = final_df[TARGET].copy()
    filtered_model = fit_ols_model(X_final, y_vif)

    print_section("VIF Iterative Removal Log")
    if not removed_by_vif.empty:
        print_dataframe(removed_by_vif)
    else:
        print("No variables were removed due to excessive VIF.")

    for i, history_item in enumerate(vif_history, 1):
        print_section(f"VIF Table {i}: {history_item['stage']}")
        print_dataframe(history_item["vif_table"].round(6))

    print_section("After VIF Filtering")
    vif_after = compute_vif(X_final)
    print_dataframe(vif_after.round(6))

    feature_count_summary = pd.DataFrame(
        [
            {
                "Step": "Available Features After Outlier Removal",
                "Feature Count": int(X_step.shape[1]),
                "Description": "All explanatory variables before variable selection",
            },
            {
                "Step": "After Stepwise",
                "Feature Count": int(len(selected_vars)),
                "Description": "Number of features retained by Stepwise",
            },
            {
                "Step": "After VIF Removal",
                "Feature Count": int(X_final.shape[1]),
                "Description": "Number of features in the final model",
            },
        ]
    )

    print_section("Baseline OLS Summary After VIF Filtering")
    print(f"Final number of variables: {X_final.shape[1]}")
    print(f"Adjusted R^2: {filtered_model.rsquared_adj:.4f}")

    print_section("Feature Count Summary")
    print_dataframe(feature_count_summary)

    print_section("Retained Features After VIF Screening")
    for i, var in enumerate(X_final.columns, 1):
        print(f"{i:2d}. {var}")

    safe_to_csv(feature_count_summary, CSV_DIR / "feature_count_summary.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(removed_by_vif, CSV_DIR / "vif_removal_log.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(final_df, CSV_DIR / "communities_variable_screening.csv", index=False, encoding="utf-8-sig")
    return final_df, X_final, y_vif.loc[X_final.index], y_raw_final, removed_by_vif
