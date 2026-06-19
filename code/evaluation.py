from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special, stats
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_config import CSV_DIR, HIGH_MISSING_THRESHOLD, ID_COLS, OUTPUT_DIR, TARGET
from pipeline_utils import print_dataframe, print_section, safe_to_csv, save_current_figure


def inverse_response_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    log_match = re.match(r"log\(y \+ ([0-9.]+)\)", transform_name)
    if log_match:
        c_value = float(log_match.group(1))
        return np.exp(values) - c_value
    boxcox_match = re.match(r"Box-Cox\(y \+ ([0-9.]+)\), lambda=([-0-9.]+)", transform_name)
    if boxcox_match:
        c_value = float(boxcox_match.group(1))
        lambda_value = float(boxcox_match.group(2))
        return special.inv_boxcox(values, lambda_value) - c_value
    if transform_name.startswith("arcsin(sqrt(y))"):
        return np.square(np.sin(values))
    if transform_name.startswith("logit(y)"):
        return 1.0 / (1.0 + np.exp(-values))
    return values


def apply_response_transform(values: pd.Series | np.ndarray, transform_name: str) -> pd.Series:
    series = pd.Series(values, dtype=float)
    log_match = re.match(r"log\(y \+ ([0-9.]+)\)", transform_name)
    if log_match:
        return np.log(series + float(log_match.group(1)))

    boxcox_match = re.match(r"Box-Cox\(y \+ ([0-9.]+)\), lambda=([-0-9.]+)", transform_name)
    if boxcox_match:
        return pd.Series(
            stats.boxcox(series + float(boxcox_match.group(1)), lmbda=float(boxcox_match.group(2))),
            index=series.index,
        )

    if transform_name.startswith("arcsin(sqrt(y))"):
        return np.arcsin(np.sqrt(np.clip(series, 0, 1)))
    if transform_name.startswith("logit(y)"):
        clipped = np.clip(series, 1e-3, 1 - 1e-3)
        return np.log(clipped / (1 - clipped))
    return series


def build_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "scale": scale,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_holdout_performance(
    X_design: pd.DataFrame,
    y_transformed: pd.Series,
    y_raw: pd.Series,
    response_transform: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    split = train_test_split(
        X_design,
        y_transformed,
        y_raw,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = split

    model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    train_pred_transformed = np.asarray(model.predict(sm.add_constant(X_train, has_constant="add")), dtype=float)
    test_pred_transformed = np.asarray(model.predict(sm.add_constant(X_test, has_constant="add")), dtype=float)

    train_pred_raw = np.clip(inverse_response_transform(train_pred_transformed, response_transform), 0.0, 1.0)
    test_pred_raw = np.clip(inverse_response_transform(test_pred_transformed, response_transform), 0.0, 1.0)

    metrics_df = pd.DataFrame(
        [
            {
                "split": "train",
                **build_prediction_metrics(np.asarray(y_train, dtype=float), train_pred_transformed, "transformed_response"),
            },
            {
                "split": "test",
                **build_prediction_metrics(np.asarray(y_test, dtype=float), test_pred_transformed, "transformed_response"),
            },
            {
                "split": "train",
                **build_prediction_metrics(np.asarray(y_raw_train, dtype=float), train_pred_raw, "raw_response"),
            },
            {
                "split": "test",
                **build_prediction_metrics(np.asarray(y_raw_test, dtype=float), test_pred_raw, "raw_response"),
            },
        ]
    )

    prediction_df = pd.DataFrame(
        {
            "actual_raw": np.asarray(y_raw_test, dtype=float),
            "predicted_raw": test_pred_raw,
            "actual_transformed": np.asarray(y_test, dtype=float),
            "predicted_transformed": test_pred_transformed,
            "residual_raw": np.asarray(y_raw_test, dtype=float) - test_pred_raw,
            "residual_transformed": np.asarray(y_test, dtype=float) - test_pred_transformed,
        },
        index=X_test.index,
    ).sort_index()

    safe_to_csv(
        metrics_df,
        CSV_DIR / "final_model_holdout_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )
    safe_to_csv(
        prediction_df,
        CSV_DIR / "final_model_holdout_predictions.csv",
        index=True,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(prediction_df["actual_raw"], prediction_df["predicted_raw"], alpha=0.55, s=22)
    axes[0].plot([0, 1], [0, 1], color="crimson", linestyle="--", linewidth=1.2)
    axes[0].set_title("Holdout Actual vs Predicted")
    axes[0].set_xlabel("Actual violent crime rate")
    axes[0].set_ylabel("Predicted violent crime rate")

    axes[1].scatter(prediction_df["predicted_raw"], prediction_df["residual_raw"], alpha=0.55, s=22)
    axes[1].axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    axes[1].set_title("Holdout Residuals")
    axes[1].set_xlabel("Predicted violent crime rate")
    axes[1].set_ylabel("Residual")
    save_current_figure("14_holdout_prediction_diagnostics.png")

    print_section("Final Model Holdout Evaluation")
    print_dataframe(metrics_df.round(6))
    return metrics_df


def coerce_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    coerced = coerced.replace("?", np.nan)
    for col in coerced.columns:
        if col != "communityname":
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
    return coerced


def prepare_train_test_clean_frames(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    high_missing_threshold: float = HIGH_MISSING_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    train_df = coerce_modeling_frame(train_raw)
    test_df = coerce_modeling_frame(test_raw)

    high_missing_cols = train_df.columns[train_df.isna().mean() > high_missing_threshold].tolist()
    drop_cols = list(dict.fromkeys(ID_COLS + high_missing_cols))
    train_clean = train_df.drop(columns=drop_cols, errors="ignore").copy()
    test_clean = test_df.drop(columns=drop_cols, errors="ignore").copy()

    impute_means = train_clean.mean(numeric_only=True)
    train_clean = train_clean.fillna(impute_means)
    for col in train_clean.columns:
        if col not in test_clean.columns:
            test_clean[col] = np.nan
    test_clean = test_clean[train_clean.columns].fillna(impute_means)
    return train_clean, test_clean, high_missing_cols, impute_means


def detect_training_outliers(clean_train_df: pd.DataFrame) -> list[int]:
    y_train = clean_train_df[TARGET].copy()
    X_train = clean_train_df.drop(columns=[TARGET]).copy()
    model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    influence = model.get_influence()
    r_student = influence.resid_studentized_external
    cook_d = influence.cooks_distance[0]
    cook_threshold = 4 / len(clean_train_df)
    outlier_mask = (np.abs(r_student) > 3) & (cook_d > cook_threshold)
    return clean_train_df.index[outlier_mask].tolist()


def build_prespecified_design(
    clean_df: pd.DataFrame,
    final_columns: list[str],
    quadratic_centers: dict[str, float],
) -> pd.DataFrame:
    design = pd.DataFrame(index=clean_df.index)
    missing_terms: list[str] = []
    for col in final_columns:
        if col.endswith("_sq_centered"):
            base_col = col[: -len("_sq_centered")]
            if base_col not in clean_df.columns:
                missing_terms.append(col)
                continue
            center = quadratic_centers.get(base_col, float(clean_df[base_col].mean()))
            design[col] = (clean_df[base_col] - center) ** 2
        elif col in clean_df.columns:
            design[col] = clean_df[col]
        else:
            missing_terms.append(col)

    if missing_terms:
        raise KeyError(f"Missing terms needed for the prespecified holdout design: {missing_terms}")
    return design[final_columns]


def evaluate_prespecified_holdout_performance(
    raw_df: pd.DataFrame,
    final_columns: list[str],
    response_transform: str,
    test_size: float = 0.20,
    random_state: int = 42,
    remove_training_outliers: bool = True,
) -> pd.DataFrame:
    train_raw, test_raw = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
    )
    train_clean, test_clean, high_missing_cols, _ = prepare_train_test_clean_frames(train_raw, test_raw)

    outlier_rows: list[int] = []
    train_model_df = train_clean.copy()
    if remove_training_outliers:
        outlier_rows = detect_training_outliers(train_clean)
        train_model_df = train_clean.drop(index=outlier_rows)

    quadratic_bases = {
        col[: -len("_sq_centered")]
        for col in final_columns
        if col.endswith("_sq_centered")
    }
    quadratic_centers = {base: float(train_model_df[base].mean()) for base in quadratic_bases if base in train_model_df.columns}

    X_train = build_prespecified_design(train_model_df, final_columns, quadratic_centers)
    X_test = build_prespecified_design(test_clean, final_columns, quadratic_centers)
    y_raw_train = train_model_df[TARGET]
    y_raw_test = test_clean[TARGET]
    y_train = apply_response_transform(y_raw_train, response_transform)
    y_test = apply_response_transform(y_raw_test, response_transform)

    model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    train_pred_transformed = np.asarray(model.predict(sm.add_constant(X_train, has_constant="add")), dtype=float)
    test_pred_transformed = np.asarray(model.predict(sm.add_constant(X_test, has_constant="add")), dtype=float)
    train_pred_raw = np.clip(inverse_response_transform(train_pred_transformed, response_transform), 0.0, 1.0)
    test_pred_raw = np.clip(inverse_response_transform(test_pred_transformed, response_transform), 0.0, 1.0)

    metrics_df = pd.DataFrame(
        [
            {
                "split": "train",
                "evaluation": "pre_split_prespecified_model",
                **build_prediction_metrics(np.asarray(y_train, dtype=float), train_pred_transformed, "transformed_response"),
            },
            {
                "split": "test",
                "evaluation": "pre_split_prespecified_model",
                **build_prediction_metrics(np.asarray(y_test, dtype=float), test_pred_transformed, "transformed_response"),
            },
            {
                "split": "train",
                "evaluation": "pre_split_prespecified_model",
                **build_prediction_metrics(np.asarray(y_raw_train, dtype=float), train_pred_raw, "raw_response"),
            },
            {
                "split": "test",
                "evaluation": "pre_split_prespecified_model",
                **build_prediction_metrics(np.asarray(y_raw_test, dtype=float), test_pred_raw, "raw_response"),
            },
        ]
    )
    metrics_df["test_size"] = float(test_size)
    metrics_df["random_state"] = int(random_state)
    metrics_df["train_high_missing_columns"] = len(high_missing_cols)
    metrics_df["train_outliers_removed"] = len(outlier_rows)

    prediction_df = pd.DataFrame(
        {
            "actual_raw": np.asarray(y_raw_test, dtype=float),
            "predicted_raw": test_pred_raw,
            "actual_transformed": np.asarray(y_test, dtype=float),
            "predicted_transformed": test_pred_transformed,
            "residual_raw": np.asarray(y_raw_test, dtype=float) - test_pred_raw,
            "residual_transformed": np.asarray(y_test, dtype=float) - test_pred_transformed,
        },
        index=X_test.index,
    ).sort_index()

    safe_to_csv(
        metrics_df,
        CSV_DIR / "leakage_aware_holdout_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )
    safe_to_csv(
        prediction_df,
        CSV_DIR / "leakage_aware_holdout_predictions.csv",
        index=True,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(prediction_df["actual_raw"], prediction_df["predicted_raw"], alpha=0.55, s=22)
    axes[0].plot([0, 1], [0, 1], color="crimson", linestyle="--", linewidth=1.2)
    axes[0].set_title("Pre-Split Holdout Actual vs Predicted")
    axes[0].set_xlabel("Actual violent crime rate")
    axes[0].set_ylabel("Predicted violent crime rate")
    axes[1].scatter(prediction_df["predicted_raw"], prediction_df["residual_raw"], alpha=0.55, s=22)
    axes[1].axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    axes[1].set_title("Pre-Split Holdout Residuals")
    axes[1].set_xlabel("Predicted violent crime rate")
    axes[1].set_ylabel("Residual")
    save_current_figure("16_leakage_aware_holdout_diagnostics.png")

    print_section("Pre-Split Prespecified Holdout Evaluation")
    print("The raw data are split before imputation and outlier screening; the final model specification is reused from the main explanatory pipeline.")
    print_dataframe(metrics_df.round(6))
    return metrics_df


def evaluate_lasso_stability(
    X_design: pd.DataFrame,
    y_transformed: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    selected_counts = pd.Series(0, index=X_design.columns, dtype=int)
    coefficient_sums = pd.Series(0.0, index=X_design.columns, dtype=float)
    total_runs = 0

    for train_idx, _ in cv.split(X_design):
        X_train = X_design.iloc[train_idx]
        y_train = y_transformed.iloc[train_idx]
        model = make_pipeline(
            StandardScaler(),
            LassoCV(cv=5, random_state=random_state, max_iter=20000),
        )
        model.fit(X_train, y_train)
        lasso = model.named_steps["lassocv"]
        coef = pd.Series(lasso.coef_, index=X_design.columns)
        selected = coef.abs() > 1e-8
        selected_counts.loc[selected] += 1
        coefficient_sums += coef
        total_runs += 1

    stability_df = pd.DataFrame(
        {
            "variable": X_design.columns,
            "lasso_selection_frequency": selected_counts.values / total_runs,
            "mean_standardized_lasso_coef": coefficient_sums.values / total_runs,
        }
    ).sort_values(
        ["lasso_selection_frequency", "mean_standardized_lasso_coef"],
        ascending=[False, False],
    ).reset_index(drop=True)

    safe_to_csv(
        stability_df,
        CSV_DIR / "lasso_feature_stability.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    plot_df = stability_df.head(15).sort_values("lasso_selection_frequency", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["variable"], plot_df["lasso_selection_frequency"], color="#4C78A8")
    plt.xlabel("Selection frequency")
    plt.ylabel("Variable")
    plt.title("Lasso Feature Stability Across Repeated CV")
    save_current_figure("15_lasso_feature_stability.png")

    print_section("Lasso Feature Stability Check")
    print_dataframe(stability_df.head(15).round(6))
    return stability_df


def build_model_report(
    final_model_name: str,
    selected_model_label: str,
    final_design_df: pd.DataFrame,
    holdout_metrics_df: pd.DataFrame,
    lasso_stability_df: pd.DataFrame,
    leakage_aware_metrics_df: pd.DataFrame | None = None,
) -> Path:
    selection_path = CSV_DIR / "final_model_selection_comparison.csv"
    coefficients_path = CSV_DIR / "final_model_coefficients.csv"
    partial_r2_path = CSV_DIR / "partial_r2_ranking.csv"
    kfold_path = CSV_DIR / "final_model_kfold_cv_100.csv"

    selected_summary = pd.read_csv(selection_path).iloc[0] if selection_path.exists() else pd.Series(dtype=object)
    coefficients_df = pd.read_csv(coefficients_path).head(8) if coefficients_path.exists() else pd.DataFrame()
    partial_r2_df = pd.read_csv(partial_r2_path).head(8) if partial_r2_path.exists() else pd.DataFrame()
    kfold_df = pd.read_csv(kfold_path) if kfold_path.exists() else pd.DataFrame()

    test_raw = holdout_metrics_df[
        (holdout_metrics_df["split"] == "test") & (holdout_metrics_df["scale"] == "raw_response")
    ].iloc[0]
    leakage_test_raw = None
    if leakage_aware_metrics_df is not None and not leakage_aware_metrics_df.empty:
        leakage_test_raw = leakage_aware_metrics_df[
            (leakage_aware_metrics_df["split"] == "test") & (leakage_aware_metrics_df["scale"] == "raw_response")
        ].iloc[0]

    lines = [
        "# Model Report",
        "",
        "## Final Model",
        "",
        f"- Response transformation: `{final_model_name}`",
        f"- Selected candidate: `{selected_model_label}`",
        f"- Final predictor count: {final_design_df.shape[1]}",
        "- Holdout metrics are computed after model specification selection, so they are a final-model sanity check rather than a fully nested model-selection estimate.",
        "- The pre-split prespecified holdout splits raw data before imputation and training-only outlier screening, then reuses the selected final model specification.",
    ]
    if not selected_summary.empty:
        lines.extend(
            [
                f"- Adjusted R-squared: {float(selected_summary['adjusted_r2']):.4f}",
                f"- LOOCV predicted R-squared: {float(selected_summary['r2_pred']):.4f}",
            ]
        )
    if not kfold_df.empty:
        lines.append(f"- 100-fold CV predicted R-squared: {float(kfold_df.loc[0, 'r2_pred_kfold']):.4f}")

    lines.extend(
        [
            f"- Holdout raw-scale R-squared: {float(test_raw['r2']):.4f}",
            f"- Holdout raw-scale RMSE: {float(test_raw['rmse']):.4f}",
            f"- Holdout raw-scale MAE: {float(test_raw['mae']):.4f}",
            "",
            "## Pre-Split Prespecified Holdout",
            "",
        ]
    )
    if leakage_test_raw is not None:
        lines.extend(
            [
                f"- Raw-scale R-squared: {float(leakage_test_raw['r2']):.4f}",
                f"- Raw-scale RMSE: {float(leakage_test_raw['rmse']):.4f}",
                f"- Raw-scale MAE: {float(leakage_test_raw['mae']):.4f}",
                f"- Random state: {int(leakage_test_raw['random_state'])}",
                f"- Training outliers removed: {int(leakage_test_raw['train_outliers_removed'])}",
            ]
        )
    else:
        lines.append("- Not available for this run.")

    lines.extend(
        [
            "",
            "## Most Influential Terms",
            "",
        ]
    )

    if not coefficients_df.empty:
        for _, row in coefficients_df.iterrows():
            lines.append(
                f"- `{row['variable']}`: coef={float(row['coef']):.4f}, "
                f"HC3 p-value={float(row['hc3_p_value']):.3g}"
            )

    lines.extend(["", "## Partial R-squared Ranking", ""])
    if not partial_r2_df.empty:
        for _, row in partial_r2_df.iterrows():
            lines.append(f"- `{row['variable_group']}`: partial R-squared={float(row['partial_r2']):.4f}")

    lines.extend(["", "## Lasso Stability Cross-check", ""])
    for _, row in lasso_stability_df.head(8).iterrows():
        lines.append(f"- `{row['variable']}`: selected in {float(row['lasso_selection_frequency']):.0%} of repeated CV fits")

    lines.extend(
        [
            "",
            "## Key Artifacts",
            "",
            "- `outputs/csv/final_model_coefficients.csv`",
            "- `outputs/csv/final_model_holdout_metrics.csv`",
            "- `outputs/csv/leakage_aware_holdout_metrics.csv`",
            "- `outputs/csv/lasso_feature_stability.csv`",
            "- `outputs/images/14_holdout_prediction_diagnostics.png`",
            "- `outputs/images/16_leakage_aware_holdout_diagnostics.png`",
            "- `outputs/images/15_lasso_feature_stability.png`",
        ]
    )

    report_path = OUTPUT_DIR / "model_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved model report: {report_path}")
    return report_path


def run_project_evaluation(
    final_model_name: str,
    selected_model_label: str,
    final_design_df: pd.DataFrame,
    final_response_series: pd.Series,
    y_raw_final: pd.Series,
    raw_df: pd.DataFrame | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> None:
    y_raw_aligned = pd.Series(y_raw_final, index=final_design_df.index)
    y_transformed_aligned = pd.Series(final_response_series, index=final_design_df.index)

    holdout_metrics_df = evaluate_holdout_performance(
        X_design=final_design_df,
        y_transformed=y_transformed_aligned,
        y_raw=y_raw_aligned,
        response_transform=final_model_name,
        test_size=test_size,
        random_state=random_state,
    )
    lasso_stability_df = evaluate_lasso_stability(
        X_design=final_design_df,
        y_transformed=y_transformed_aligned,
        random_state=random_state,
    )
    leakage_aware_metrics_df = None
    if raw_df is not None:
        leakage_aware_metrics_df = evaluate_prespecified_holdout_performance(
            raw_df=raw_df,
            final_columns=final_design_df.columns.tolist(),
            response_transform=final_model_name,
            test_size=test_size,
            random_state=random_state,
        )
    build_model_report(
        final_model_name=final_model_name,
        selected_model_label=selected_model_label,
        final_design_df=final_design_df,
        holdout_metrics_df=holdout_metrics_df,
        lasso_stability_df=lasso_stability_df,
        leakage_aware_metrics_df=leakage_aware_metrics_df,
    )
