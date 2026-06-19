from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from pipeline_config import (
    CSV_DIR,
    DATA_FILE,
    HIGH_MISSING_THRESHOLD,
    ID_COLS,
    IMG_DIR,
    IMPORTANT_CSV_FILES,
    RAW_DATA_FILE,
    RAW_NAMES_FILE,
    TARGET,
)
from pipeline_utils import print_dataframe, print_section, safe_to_csv, save_current_figure


def parse_attribute_names(names_file: str | Path) -> list[str]:
    attribute_names: list[str] = []
    with open(names_file, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.lower().startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attribute_names.append(parts[1])
    return attribute_names


def load_communities_dataset() -> pd.DataFrame:
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    if not RAW_DATA_FILE.exists() or not RAW_NAMES_FILE.exists():
        raise FileNotFoundError(
            "Could not find prepared data/communities.csv or the raw "
            "communities.data / communities.names files."
        )

    columns = parse_attribute_names(RAW_NAMES_FILE)
    raw_df = pd.read_csv(RAW_DATA_FILE, header=None, names=columns, na_values="?")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    print(f"Prepared CSV data file: {DATA_FILE}")
    return raw_df


def stage_0_output_policy() -> None:
    print_section("CSV Output Policy")
    print(f"Keep only these core CSV files: {IMPORTANT_CSV_FILES}")
    print(f"CSV output directory: {CSV_DIR}")
    print(f"Image output directory: {IMG_DIR}")


def stage_1_data_overview() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_communities_dataset()
    raw_df = raw_df.replace("?", np.nan)

    # Keep the community name as text and coerce the remaining columns to numeric.
    for col in raw_df.columns:
        if col != "communityname":
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    print_section("Raw Data Overview")
    print(f"Data file: {DATA_FILE.resolve()}")
    print(f"Number of observations: {raw_df.shape[0]}")
    print(f"Number of columns: {raw_df.shape[1]}")
    print(f"Target variable: {TARGET}")

    print_section("Target Distribution Overview")
    print(raw_df[TARGET].describe(percentiles=[0.25, 0.5, 0.75]).round(4))

    missing_summary = pd.DataFrame(
        {
            "Missing Count": raw_df.isna().sum(),
            "Missing Percentage": (raw_df.isna().mean() * 100).round(2),
        }
    ).sort_values(["Missing Count", "Missing Percentage"], ascending=False)

    print_section("Top 20 Variables with the Most Missing Values")
    print_dataframe(missing_summary.head(20))

    missing_plot_df = (
        missing_summary.head(20)
        .reset_index()
        .rename(columns={"index": "variable"})
        .sort_values("Missing Percentage", ascending=True)
    )
    plt.figure(figsize=(9, 7))
    sns.barplot(
        data=missing_plot_df,
        x="Missing Percentage",
        y="variable",
        color="#4C78A8",
    )
    plt.title("Top 20 Variables by Missing Percentage")
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Variable")
    save_current_figure("00_missingness_top20_before_cleaning.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(raw_df[TARGET].dropna(), bins=30, kde=True)
    plt.title("Distribution of ViolentCrimesPerPop")
    plt.xlabel(TARGET)
    plt.ylabel("Frequency")
    save_current_figure("01_target_distribution.png")

    return raw_df, missing_summary


def stage_2_clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    working_df = raw_df.copy()
    missing_ratio = working_df.isna().mean()

    # Drop identifiers and variables with excessive missingness before imputation.
    high_missing_cols = missing_ratio[missing_ratio > HIGH_MISSING_THRESHOLD].index.tolist()
    clean_df = working_df.drop(columns=ID_COLS + high_missing_cols, errors="ignore").copy()

    remaining_missing = clean_df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)

    print_section("Cleaning Rules Summary")
    print(f"Dropped identifier columns: {ID_COLS}")
    print(f"Number of variables with missing ratio > {HIGH_MISSING_THRESHOLD:.0%}: {len(high_missing_cols)}")
    print(high_missing_cols)

    if not remaining_missing.empty:
        print_section("Variables Still Containing Missing Values After Dropping High-Missing Columns")
        remaining_missing_df = pd.DataFrame(
            {
                "Missing Count": remaining_missing,
                "Missing Percentage": (clean_df[remaining_missing.index].isna().mean() * 100).round(4),
            }
        )
        print_dataframe(remaining_missing_df)
    else:
        print_section("No Missing Values Remain After Dropping High-Missing Columns")

    imputed_cols: list[str] = []
    for col in clean_df.columns:
        if clean_df[col].isna().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
            imputed_cols.append(col)

    print_section("Mean Imputation Summary")
    print(f"Variables imputed with mean values: {imputed_cols if imputed_cols else 'None'}")
    print(f"Cleaned data shape: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
    print(f"Remaining missing entries after cleaning: {int(clean_df.isna().sum().sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    if not remaining_missing.empty:
        remaining_missing_plot_df = (
            pd.DataFrame(
                {
                    "variable": remaining_missing.head(15).index,
                    "Missing Percentage": (working_df[remaining_missing.head(15).index].isna().mean() * 100).values,
                }
            )
            .sort_values("Missing Percentage", ascending=True)
        )
        sns.barplot(
            data=remaining_missing_plot_df,
            x="Missing Percentage",
            y="variable",
            color="#F58518",
            ax=axes[0],
        )
        axes[0].set_title("Remaining Missingness Before Mean Imputation")
        axes[0].set_xlabel("Missing Percentage (%)")
        axes[0].set_ylabel("Variable")
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No missing values remain\nbefore imputation.",
            ha="center",
            va="center",
            fontsize=12,
        )

    cleaning_summary_df = pd.DataFrame(
        {
            "Stage": ["Raw data", "After dropping columns", "After mean imputation"],
            "Missing cells": [
                int(working_df.isna().sum().sum()),
                int(working_df.drop(columns=ID_COLS + high_missing_cols, errors="ignore").isna().sum().sum()),
                int(clean_df.isna().sum().sum()),
            ],
        }
    )
    sns.barplot(data=cleaning_summary_df, x="Stage", y="Missing cells", color="#54A24B", ax=axes[1])
    axes[1].set_title("Missing Cells Before and After Cleaning")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Missing cell count")
    axes[1].tick_params(axis="x", rotation=15)
    save_current_figure("03_cleaning_missingness_summary.png")

    safe_to_csv(clean_df, CSV_DIR / "communities_excluded_police.csv", index=False, encoding="utf-8-sig")
    return clean_df, high_missing_cols, imputed_cols


def stage_3_detect_outliers(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    y_clean = clean_df[TARGET].copy()
    X_clean = clean_df.drop(columns=[TARGET]).copy()

    # Use externally studentized residuals together with Cook's distance
    # so that flagged points are both unusual and influential.
    baseline_model = sm.OLS(y_clean, sm.add_constant(X_clean, has_constant="add")).fit()
    influence = baseline_model.get_influence()
    r_student = influence.resid_studentized_external
    cook_d = influence.cooks_distance[0]
    cook_threshold = 4 / len(clean_df)

    outlier_mask = (np.abs(r_student) > 3) & (cook_d > cook_threshold)
    outlier_rows = clean_df.index[outlier_mask].tolist()

    print_section("Outlier Detection Summary")
    print(f"Total observations: {len(clean_df)}")
    print(f"Cook threshold: {cook_threshold:.6f}")
    print(f"Number of detected outliers: {len(outlier_rows)}")
    print(f"Outlier row indices: {outlier_rows}")

    if outlier_rows:
        outlier_info = raw_df.loc[outlier_rows, ["communityname", TARGET]].copy()
        outlier_info["R_student"] = r_student[outlier_rows]
        outlier_info["Cook_distance"] = cook_d[outlier_rows]
        print_dataframe(outlier_info.head(15).round(6))

    outliers_df = clean_df.loc[outlier_rows]
    normal_df = clean_df.drop(index=outlier_rows)

    if len(outliers_df) > 0:
        mean_diff = (
            outliers_df.mean(numeric_only=True) - normal_df.mean(numeric_only=True)
        ).abs().sort_values(ascending=False)
        print_section("Top 15 Variables with the Largest Mean Differences Between Outliers and Non-Outliers")
        print_dataframe(mean_diff.head(15).to_frame("Absolute Mean Difference").round(6))

    obs_index = np.arange(len(clean_df))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].scatter(obs_index, r_student, alpha=0.55, s=18, color="#4C78A8", label="Observations")
    if outlier_rows:
        axes[0].scatter(np.array(outlier_rows), np.array(r_student)[outlier_rows], color="crimson", s=28, label="Flagged outliers")
    axes[0].axhline(3, color="crimson", linestyle="--", linewidth=1)
    axes[0].axhline(-3, color="crimson", linestyle="--", linewidth=1)
    axes[0].set_title("Externally Studentized Residuals")
    axes[0].set_ylabel("R-student")
    axes[0].legend()

    axes[1].scatter(obs_index, cook_d, alpha=0.55, s=18, color="#72B7B2", label="Observations")
    if outlier_rows:
        axes[1].scatter(np.array(outlier_rows), np.array(cook_d)[outlier_rows], color="crimson", s=28, label="Flagged outliers")
    axes[1].axhline(cook_threshold, color="crimson", linestyle="--", linewidth=1, label=f"Cook threshold = {cook_threshold:.4f}")
    axes[1].set_title("Cook's Distance")
    axes[1].set_xlabel("Observation index")
    axes[1].set_ylabel("Cook's distance")
    axes[1].legend()
    save_current_figure("04_outlier_diagnostics.png")

    plt.figure(figsize=(8.5, 6))
    abs_r_student = np.abs(r_student)
    plt.scatter(abs_r_student, cook_d, alpha=0.5, s=18, color="#4C78A8")
    if outlier_rows:
        plt.scatter(abs_r_student[outlier_rows], np.array(cook_d)[outlier_rows], color="crimson", s=30, label="Flagged outliers")
    plt.axvline(3, color="crimson", linestyle="--", linewidth=1, label="|R-student| = 3")
    plt.axhline(cook_threshold, color="darkorange", linestyle="--", linewidth=1, label=f"Cook threshold = {cook_threshold:.4f}")
    plt.title("Outlier Identification Rule")
    plt.xlabel("|Externally studentized residual|")
    plt.ylabel("Cook's distance")
    plt.legend()
    save_current_figure("05_outlier_rule_scatter.png")

    df_no_outliers = clean_df.drop(index=outlier_rows).reset_index(drop=True)
    safe_to_csv(df_no_outliers, CSV_DIR / "communities_without_outliers.csv", index=False, encoding="utf-8-sig")
    print(f"Remaining observations after removing outliers: {len(df_no_outliers)}")
    return df_no_outliers, outlier_rows
