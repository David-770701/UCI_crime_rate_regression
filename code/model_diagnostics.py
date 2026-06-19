from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats as scipy_stats
from scipy.stats import boxcox
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import linear_reset

from pipeline_config import (
    BOXCOX_C_VALUES,
    CORR_THRESHOLD,
    CSV_DIR,
    FINAL_DROP_VARIABLES,
    HOLDOUT_TEST_SIZE,
    LOGIT_EPS,
    LOG_C_VALUES,
    QUADRATIC_SEQUENCE,
    RANDOM_STATE,
    REPRESENTATIVE_CANDIDATE_IDS,
    STEPWISE_EXCLUDE_COLS,
    TARGET,
    VIF_THRESHOLD,
)
from pipeline_utils import (
    build_adjustment_variable_dropone_table,
    build_inference_table,
    build_overall_f_test_summary,
    collect_model_diagnostics,
    compute_high_corr_pairs,
    compute_kfold_cv_metrics,
    compute_loocv_press_metrics,
    compute_partial_r2_ranking,
    compute_vif,
    evaluate_engineered_model,
    fit_ols_model,
    get_studentized_residuals,
    get_final_response_series,
    print_dataframe,
    print_section,
    safe_to_csv,
    save_current_figure,
)


def stage_7_transform_and_diagnostics(df_no_outliers: pd.DataFrame) -> dict:
    X_transform = df_no_outliers.drop(columns=[TARGET]).drop(columns=STEPWISE_EXCLUDE_COLS, errors="ignore").copy()
    y_raw_final = df_no_outliers[TARGET].copy()

    print_section("Compare Raw y, log(y+c), Box-Cox(y+c), Yeo-Johnson, arcsin(sqrt(y)), and logit(y)")

    raw_model, raw_metrics = collect_model_diagnostics(
        label="raw_y",
        transform_name="raw_y",
        X=X_transform,
        y_transformed=y_raw_final,
    )

    log_results = []
    for c in LOG_C_VALUES:
        y_log = np.log(y_raw_final + c)
        log_model, log_metrics = collect_model_diagnostics(
            label=f"log_y_plus_{c:g}",
            transform_name="log(y+c)",
            X=X_transform,
            y_transformed=y_log,
            extra_info={"c_value": float(c)},
        )
        log_results.append((c, log_model, log_metrics))

    log_compare_df = pd.DataFrame([item[2] for item in log_results])
    log_compare_df["jb_rank"] = log_compare_df["jb_stat"].rank(method="min", ascending=True)
    log_compare_df["qq_rank"] = log_compare_df["qq_rmse"].rank(method="min", ascending=True)
    log_compare_df["bp_rank"] = log_compare_df["bp_f_stat"].rank(method="min", ascending=True)
    log_compare_df["score"] = log_compare_df["jb_rank"] + log_compare_df["qq_rank"] + log_compare_df["bp_rank"]
    log_compare_df = log_compare_df.sort_values(
        ["score", "jb_stat", "qq_rmse", "bp_f_stat"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    best_log_c = float(log_compare_df.loc[0, "c_value"])
    best_log_tuple = next(item for item in log_results if abs(item[0] - best_log_c) < 1e-12)
    best_log_model = best_log_tuple[1]
    best_log_metrics = best_log_tuple[2]

    boxcox_results = []
    for c in BOXCOX_C_VALUES:
        y_shifted = y_raw_final + c
        y_boxcox, lam = boxcox(y_shifted)
        bc_model, bc_metrics = collect_model_diagnostics(
            label=f"boxcox_y_plus_{c:g}",
            transform_name="boxcox(y+c)",
            X=X_transform,
            y_transformed=y_boxcox,
            extra_info={"c_value": float(c), "lambda_value": float(lam)},
        )
        boxcox_results.append((c, lam, bc_model, bc_metrics))

    boxcox_compare_df = pd.DataFrame([item[3] for item in boxcox_results])
    boxcox_compare_df["jb_rank"] = boxcox_compare_df["jb_stat"].rank(method="min", ascending=True)
    boxcox_compare_df["qq_rank"] = boxcox_compare_df["qq_rmse"].rank(method="min", ascending=True)
    boxcox_compare_df["bp_rank"] = boxcox_compare_df["bp_f_stat"].rank(method="min", ascending=True)
    boxcox_compare_df["score"] = boxcox_compare_df["jb_rank"] + boxcox_compare_df["qq_rank"] + boxcox_compare_df["bp_rank"]
    boxcox_compare_df = boxcox_compare_df.sort_values(
        ["score", "jb_stat", "qq_rmse", "bp_f_stat"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    best_boxcox_c = float(boxcox_compare_df.loc[0, "c_value"])
    best_boxcox_lambda = float(boxcox_compare_df.loc[0, "lambda_value"])
    best_boxcox_tuple = next(item for item in boxcox_results if abs(item[0] - best_boxcox_c) < 1e-12)
    best_boxcox_model = best_boxcox_tuple[2]
    best_boxcox_metrics = best_boxcox_tuple[3]

    y_yeojohnson, best_yeojohnson_lambda = scipy_stats.yeojohnson(y_raw_final)
    yeojohnson_model, yeojohnson_metrics = collect_model_diagnostics(
        label="yeojohnson_y",
        transform_name="yeojohnson(y)",
        X=X_transform,
        y_transformed=y_yeojohnson,
        extra_info={"lambda_value": float(best_yeojohnson_lambda)},
    )

    y_arcsin = np.arcsin(np.sqrt(np.clip(y_raw_final, 0, 1)))
    arcsin_model, arcsin_metrics = collect_model_diagnostics(
        label="arcsin_sqrt_y",
        transform_name="arcsin(sqrt(y))",
        X=X_transform,
        y_transformed=y_arcsin,
    )

    y_logit = np.log(
        np.clip(y_raw_final, LOGIT_EPS, 1 - LOGIT_EPS)
        / (1 - np.clip(y_raw_final, LOGIT_EPS, 1 - LOGIT_EPS))
    )
    logit_model, logit_metrics = collect_model_diagnostics(
        label="logit_y",
        transform_name="logit(y)",
        X=X_transform,
        y_transformed=y_logit,
        extra_info={"logit_eps": float(LOGIT_EPS)},
    )

    print_section("Refined log(y+c) Comparison")
    print_dataframe(
        log_compare_df[["c_value", "adjusted_r2", "jb_stat", "bp_f_stat", "resid_skew_abs", "qq_rmse", "score"]],
        round_digits=6,
    )
    print(f"Best c value for log(y+c): {best_log_c:g}")

    print_section("Box-Cox(y+c) Comparison")
    print_dataframe(
        boxcox_compare_df[
            ["c_value", "lambda_value", "adjusted_r2", "jb_stat", "bp_f_stat", "resid_skew_abs", "qq_rmse", "score"]
        ],
        round_digits=6,
    )
    print(f"Best c value for Box-Cox(y+c): {best_boxcox_c:g}")
    print(f"Corresponding best lambda: {best_boxcox_lambda:.6f}")

    final_compare_df = pd.DataFrame(
        [
            raw_metrics,
            {**best_log_metrics, "c_value": best_log_c},
            {**best_boxcox_metrics, "c_value": best_boxcox_c, "lambda_value": best_boxcox_lambda},
            {**yeojohnson_metrics, "lambda_value": best_yeojohnson_lambda},
            arcsin_metrics,
            logit_metrics,
        ]
    )[
        [
            "label",
            "transform",
            "c_value",
            "lambda_value",
            "adjusted_r2",
            "jb_stat",
            "bp_f_stat",
            "resid_skew_abs",
            "qq_rmse",
            "resid_fitted_corr_abs",
        ]
    ]

    print_section("Comparison of Raw y, Best log(y+c), Best Box-Cox(y+c), and logit(y)")
    print_dataframe(final_compare_df, round_digits=6)
    safe_to_csv(final_compare_df, CSV_DIR / "target_transform_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.16g")

    candidate_models = [
        ("raw y", raw_model, raw_metrics),
        (f"log(y + {best_log_c:g})", best_log_model, best_log_metrics),
        (f"Box-Cox(y + {best_boxcox_c:g}), lambda={best_boxcox_lambda:.4f}", best_boxcox_model, best_boxcox_metrics),
        (f"Yeo-Johnson(y), lambda={best_yeojohnson_lambda:.4f}", yeojohnson_model, yeojohnson_metrics),
        ("arcsin(sqrt(y))", arcsin_model, arcsin_metrics),
        ("logit(y)", logit_model, logit_metrics),
    ]
    best_name, _, _ = min(
        candidate_models,
        key=lambda item: (item[2]["jb_stat"], item[2]["qq_rmse"], item[2]["bp_f_stat"]),
    )

    print_section("Recommended Response Transformation for Final Fitting")
    print(f"Recommended model: {best_name}")
    print("Selection rule: prioritize smaller JB statistic and QQ RMSE, then use smaller BP F-statistic as a secondary criterion.")

    plot_models = [
        ("Raw y", raw_model),
        (f"log(y + {best_log_c:g})", best_log_model),
        (f"Box-Cox(y + {best_boxcox_c:g})", best_boxcox_model),
        ("logit(y)", logit_model),
    ]
    fig, axes = plt.subplots(len(plot_models), 2, figsize=(12, 19))
    for row_idx, (title, model) in enumerate(plot_models):
        resid = model.resid
        fitted = model.fittedvalues
        axes[row_idx, 0].scatter(fitted, resid, alpha=0.5)
        axes[row_idx, 0].axhline(0, color="red", linestyle="--")
        axes[row_idx, 0].set_title(f"{title}: Residuals vs Fitted")
        axes[row_idx, 0].set_xlabel("Fitted values")
        axes[row_idx, 0].set_ylabel("Residuals")
        sm.qqplot(get_studentized_residuals(model), line="q", ax=axes[row_idx, 1])
        axes[row_idx, 1].set_title(f"{title}: Studentized QQ Plot")
    save_current_figure("06_target_transform_diagnostics_en.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    raw_studentized = get_studentized_residuals(raw_model)
    axes[0].scatter(raw_model.fittedvalues, raw_model.resid, alpha=0.5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Raw y: Residuals vs Fitted")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    sm.qqplot(raw_studentized, line="q", ax=axes[1])
    axes[1].set_title("Raw y: Studentized QQ Plot")
    save_current_figure("06_raw_y_diagnostics_only.png")

    transform_only_models = [
        (f"log(y + {best_log_c:g})", best_log_model),
        (f"Box-Cox(y + {best_boxcox_c:g})", best_boxcox_model),
        ("logit(y)", logit_model),
    ]
    fig, axes = plt.subplots(len(transform_only_models), 2, figsize=(12, 14.5))
    for row_idx, (title, model) in enumerate(transform_only_models):
        axes[row_idx, 0].scatter(model.fittedvalues, model.resid, alpha=0.5)
        axes[row_idx, 0].axhline(0, color="red", linestyle="--")
        axes[row_idx, 0].set_title(f"{title}: Residuals vs Fitted")
        axes[row_idx, 0].set_xlabel("Fitted values")
        axes[row_idx, 0].set_ylabel("Residuals")
        sm.qqplot(get_studentized_residuals(model), line="q", ax=axes[row_idx, 1])
        axes[row_idx, 1].set_title(f"{title}: Studentized QQ Plot")
    save_current_figure("06_three_transformations_diagnostics.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(log_compare_df["c_value"], log_compare_df["score"], marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("c value (log scale)")
    axes[0].set_ylabel("Composite ranking score (lower is better)")
    axes[0].set_title("Comparison of c Values for log(y+c)")

    axes[1].plot(boxcox_compare_df["c_value"], boxcox_compare_df["score"], marker="o")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("c value (log scale)")
    axes[1].set_ylabel("Composite ranking score (lower is better)")
    axes[1].set_title("Comparison of c Values for Box-Cox(y+c)")
    save_current_figure("07_transform_score_curves_en.png")

    final_response_series = get_final_response_series(
        final_model_name=best_name,
        y_raw_final=y_raw_final,
        best_log_c=best_log_c,
        best_boxcox_c=best_boxcox_c,
        best_boxcox_lambda=best_boxcox_lambda,
        best_yeojohnson_lambda=float(best_yeojohnson_lambda),
    )
    final_transform_model = next(model for name, model, _ in candidate_models if name == best_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    residual_distribution_specs = [
        ("Before Transformation", raw_model.resid),
        (f"After Transformation: {best_name}", final_transform_model.resid),
    ]
    for ax, (title, resid) in zip(axes, residual_distribution_specs):
        resid = np.asarray(resid, dtype=float)
        sns.histplot(resid, bins=30, stat="density", kde=True, ax=ax, color="#4C78A8", alpha=0.6)
        x_grid = np.linspace(resid.min(), resid.max(), 300)
        normal_pdf = scipy_stats.norm.pdf(x_grid, loc=float(np.mean(resid)), scale=float(np.std(resid, ddof=1)))
        ax.plot(x_grid, normal_pdf, color="crimson", linewidth=2, label="Normal fit")
        ax.set_title(title)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.legend()
    save_current_figure("12_residual_distribution_before_after_transformation.png")

    if best_name.startswith("Box-Cox("):
        plt.figure(figsize=(8, 4.8))
        plt.hist(final_response_series, bins=30, alpha=0.75, color="teal", edgecolor="white")
        final_response_series.plot(kind="kde", color="darkcyan", linewidth=2)
        plt.title(f"Distribution of {final_response_series.name} After Box-Cox Transformation")
        plt.xlabel(final_response_series.name)
        plt.ylabel("Frequency")
        save_current_figure("06_boxcox_transformed_target_distribution.png")

    return {
        "final_model_name": best_name,
        "best_log_c": best_log_c,
        "best_boxcox_c": best_boxcox_c,
        "best_boxcox_lambda": best_boxcox_lambda,
        "best_yeojohnson_lambda": float(best_yeojohnson_lambda),
        "final_response_series": final_response_series,
    }


def stage_7_pre_engineering_diagnostics(
    X_final: pd.DataFrame,
    y_raw_final: pd.Series,
    transform_result: dict,
) -> dict:
    final_model_name = transform_result["final_model_name"]
    final_response_series = get_final_response_series(
        final_model_name=final_model_name,
        y_raw_final=y_raw_final,
        best_log_c=float(transform_result["best_log_c"]),
        best_boxcox_c=float(transform_result["best_boxcox_c"]),
        best_boxcox_lambda=float(transform_result["best_boxcox_lambda"]),
        best_yeojohnson_lambda=float(transform_result.get("best_yeojohnson_lambda", 0.0)),
    )
    final_model = fit_ols_model(X_final, final_response_series)

    transformed_model_inference = build_inference_table(
        model=final_model,
        response_transform=final_model_name,
        engineering_step="Pre-engineering final model",
        robust_cov_type="HC3",
    )
    print_section("Pre-Engineering Final Model Inference with HC3 Robust SE")
    print_dataframe(
        transformed_model_inference[["variable", "coef", "ols_f_statistic", "hc3_se", "hc3_f_statistic"]],
        round_digits=6,
    )
    safe_to_csv(
        transformed_model_inference,
        CSV_DIR / "transformed_model_inference.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    print_section("Linearity Diagnostics for the VIF-Filtered Base Model")
    reset_result = linear_reset(final_model, power=2, use_f=True)
    print(f"Ramsey RESET F-statistic: {float(reset_result.fvalue):.6f}")
    reset_f_critical = float(scipy_stats.f.isf(0.05, reset_result.df_num, reset_result.df_denom))
    print(f"Ramsey RESET critical F at 5%: {reset_f_critical:.6f}")

    residuals = final_model.resid
    fitted = final_model.fittedvalues
    lowess_curve = lowess(residuals, fitted, frac=0.35, return_sorted=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(fitted, residuals, alpha=0.35, s=18)
    plt.plot(lowess_curve[:, 0], lowess_curve[:, 1], color="darkorange", linewidth=2, label="LOWESS")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Final Model Residuals vs Fitted with LOWESS")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.legend()
    save_current_figure("08_final_model_lowess_residuals_en.png")

    top_ccpr_vars = transformed_model_inference["variable"].head(6).tolist()
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.flatten()
    for ax, var in zip(axes, top_ccpr_vars):
        component = final_model.params[var] * X_final[var]
        partial_residual = residuals + component
        smoothed = lowess(partial_residual, X_final[var], frac=0.4, return_sorted=True)
        ax.scatter(X_final[var], partial_residual, alpha=0.35, s=18)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="darkorange", linewidth=2)
        ax.set_title(f"Partial Residual Plot: {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Component + Residual")
    for ax in axes[len(top_ccpr_vars):]:
        ax.axis("off")
    save_current_figure("09_final_model_partial_residual_plots_en.png")

    quadratic_candidates = transformed_model_inference["variable"].head(8).tolist()
    quadratic_results = []
    for var in quadratic_candidates:
        centered = X_final[var] - X_final[var].mean()
        quad_col = f"{var}_sq_centered"
        X_aug = X_final.copy()
        X_aug[quad_col] = centered**2
        quad_model = fit_ols_model(X_aug, final_response_series)
        quad_f_test = quad_model.f_test(f"{quad_col} = 0")
        quadratic_results.append(
            {
                "variable": var,
                "quadratic_term_f_stat": float(np.asarray(quad_f_test.fvalue).item()),
                "quadratic_f_critical_5pct": float(scipy_stats.f.isf(0.05, quad_f_test.df_num, quad_f_test.df_denom)),
                "delta_adj_r2": float(quad_model.rsquared_adj - final_model.rsquared_adj),
                "delta_aic": float(final_model.aic - quad_model.aic),
                "delta_bic": float(final_model.bic - quad_model.bic),
            }
        )

    quadratic_df = pd.DataFrame(quadratic_results).sort_values(
        ["quadratic_term_f_stat", "delta_aic"], ascending=[False, False]
    ).reset_index(drop=True)
    print_section("Quadratic-Term Screening for Feature Engineering")
    print_dataframe(quadratic_df.round(6))
    safe_to_csv(quadratic_df, CSV_DIR / "quadratic_term_screening.csv", index=False, encoding="utf-8-sig", float_format="%.16g")

    return {
        "final_model_name": final_model_name,
        "final_response_series": final_response_series,
        "quadratic_screening_df": quadratic_df,
    }


def stage_8_quadratic_engineering(
    X_final: pd.DataFrame,
    final_model_name: str,
    final_response_series: pd.Series,
    y_raw_final: pd.Series,
    raw_df: pd.DataFrame | None = None,
    holdout_test_size: float = HOLDOUT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> None:
    print_section("Sequential Quadratic Feature Engineering on the Final Transformed Model")

    removed_before_engineering = [col for col in FINAL_DROP_VARIABLES if col in X_final.columns]
    if removed_before_engineering:
        print(f"Drop weak non-significant variables before quadratic engineering: {removed_before_engineering}")

    quadratic_sequence = QUADRATIC_SEQUENCE
    comparison_records = []
    engineered_models = []
    # Refit the final engineered model after removing the weakest adjustment terms.
    current_X = X_final.drop(columns=removed_before_engineering, errors="ignore").copy()

    base_model_step, base_metrics = evaluate_engineered_model("Transformed base model", current_X, final_response_series)
    comparison_records.append(base_metrics)
    engineered_models.append(("Transformed base model", base_model_step, current_X.copy()))

    for var in quadratic_sequence:
        if var not in current_X.columns:
            continue
        centered = current_X[var] - current_X[var].mean()
        new_col = f"{var}_sq_centered"
        current_X[new_col] = centered**2
        label = f"Add {var}^2"
        step_model, step_metrics = evaluate_engineered_model(label, current_X, final_response_series)
        comparison_records.append(step_metrics)
        engineered_models.append((label, step_model, current_X.copy()))

    quadratic_progress_df = pd.DataFrame(comparison_records)
    print_section("Quadratic Engineering Progress Table")
    print_dataframe(quadratic_progress_df.round(6))
    feature_engineering_compare_df = (
        quadratic_progress_df[
            quadratic_progress_df["model_label"].isin(["Transformed base model", "Add PctIlleg^2"])
        ][
            [
                "model_label",
                "feature_count",
                "adjusted_r2",
                "aic",
                "bic",
                "jb_stat",
                "bp_f_stat",
                "reset_f",
                "qq_rmse",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )
    safe_to_csv(
        feature_engineering_compare_df,
        CSV_DIR / "feature_engineering_before_after_metrics.csv",
        index=False,
        float_format="%.16g",
    )

    plot_labels = {"Transformed base model", "Add PctIlleg^2"}
    plot_models = [item for item in engineered_models if item[0] in plot_labels]
    fig, axes = plt.subplots(1, len(plot_models), figsize=(12, 5))
    if len(plot_models) == 1:
        axes = [axes]
    for ax, (label, model, design_X) in zip(axes, plot_models):
        resid = model.resid
        fitted = model.fittedvalues
        smoothed = lowess(resid, fitted, frac=0.35, return_sorted=True)
        ax.scatter(fitted, resid, alpha=0.35, s=16)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="darkorange", linewidth=2)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
    plt.tight_layout()
    save_current_figure("10_quadratic_progress_lowess_en.png")

    best_progress_idx = quadratic_progress_df.sort_values(
        ["reset_f", "aic", "qq_rmse"], ascending=[True, True, True]
    ).index[0]
    best_progress_label = quadratic_progress_df.loc[best_progress_idx, "model_label"]
    recommended_engineered = next(item for item in engineered_models if item[0] == best_progress_label)
    recommended_engineered_label, recommended_engineered_model, _ = recommended_engineered

    print_section("Recommended Model After Quadratic Engineering")
    print(f"Recommended engineered model: {recommended_engineered_label}")
    # RESET is used as the primary functional-form criterion at this stage.
    print("Selection rule: prioritize smaller RESET F-statistic, then lower AIC, then smaller QQ RMSE.")

    base_selection_design_df = pd.DataFrame(
        recommended_engineered_model.model.exog,
        columns=recommended_engineered_model.model.exog_names,
        index=X_final.index,
    ).drop(columns=["const"], errors="ignore")
    selection_alpha = 0.05
    robust_cov_type = "HC3"
    sigma2_ref = float(recommended_engineered_model.mse_resid)
    candidate_rows: list[dict] = []
    candidate_store: dict[str, dict] = {}
    candidate_counter = 0

    def sort_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(
            ["all_hc3_significant", "reset_f", "aic", "qq_rmse", "r2_pred", "feature_count"],
            ascending=[False, True, True, True, False, True],
        ).reset_index(drop=True)

    def register_candidate(
        model_label: str,
        design_df: pd.DataFrame,
        candidate_type: str,
        dropped_terms: list[str],
        parent_candidate_id: str | None,
        path_step: int,
    ) -> dict:
        nonlocal candidate_counter

        candidate_id = f"M{candidate_counter}"
        candidate_counter += 1
        model, metrics = evaluate_engineered_model(model_label, design_df, final_response_series)
        inference_df = build_inference_table(
            model=model,
            response_transform=final_model_name,
            engineering_step=model_label,
            robust_cov_type=robust_cov_type,
        )

        protected_linear_terms = {
            col[: -len("_sq_centered")]
            for col in design_df.columns
            if col.endswith("_sq_centered")
        }
        nonsig_df = inference_df[inference_df["hc3_p_value"] > selection_alpha].copy()
        removable_nonsig_df = nonsig_df[
            ~nonsig_df["variable"].isin(protected_linear_terms)
        ].sort_values(["hc3_p_value", "hc3_f_statistic"], ascending=[False, True])

        row = {
            "candidate_id": candidate_id,
            "model_label": model_label,
            "candidate_type": candidate_type,
            "path_step": int(path_step),
            "parent_candidate_id": parent_candidate_id or "",
            "dropped_variables": ", ".join(dropped_terms) if dropped_terms else "None",
            "feature_count": int(design_df.shape[1]),
            "nonsig_term_count_hc3": int(nonsig_df.shape[0]),
            "removable_nonsig_count_hc3": int(removable_nonsig_df.shape[0]),
            "nonsig_variables_hc3": ", ".join(nonsig_df["variable"].tolist()) if not nonsig_df.empty else "None",
            "removable_nonsig_variables_hc3": ", ".join(removable_nonsig_df["variable"].tolist())
            if not removable_nonsig_df.empty
            else "None",
            "max_hc3_p_value": float(nonsig_df["hc3_p_value"].max()) if not nonsig_df.empty else 0.0,
            "all_hc3_significant": bool(nonsig_df.empty),
        }
        row.update(metrics)
        row.update(
            compute_loocv_press_metrics(
                model=model,
                y=final_response_series,
                sigma2_ref=sigma2_ref,
            )
        )

        candidate_rows.append(row)
        candidate_store[candidate_id] = {
            "model": model,
            "design_df": design_df.copy(),
            "inference_df": inference_df.copy(),
            "dropped_terms": dropped_terms.copy(),
            "removable_terms": removable_nonsig_df["variable"].tolist(),
        }
        return row

    baseline_candidate = register_candidate(
        model_label=recommended_engineered_label,
        design_df=base_selection_design_df,
        candidate_type="baseline",
        dropped_terms=[],
        parent_candidate_id=None,
        path_step=0,
    )
    selection_path_ids = [baseline_candidate["candidate_id"]]
    current_candidate_id = baseline_candidate["candidate_id"]
    current_design_df = base_selection_design_df.copy()
    current_dropped_terms: list[str] = []
    path_step = 0

    while candidate_store[current_candidate_id]["removable_terms"]:
        path_step += 1
        trial_ids: list[str] = []
        for drop_var in candidate_store[current_candidate_id]["removable_terms"]:
            next_dropped_terms = current_dropped_terms + [drop_var]
            trial_candidate = register_candidate(
                model_label=f"Drop {' + '.join(next_dropped_terms)}",
                design_df=current_design_df.drop(columns=[drop_var]),
                candidate_type="drop_trial",
                dropped_terms=next_dropped_terms,
                parent_candidate_id=current_candidate_id,
                path_step=path_step,
            )
            trial_ids.append(trial_candidate["candidate_id"])

        trial_df = pd.DataFrame([row for row in candidate_rows if row["candidate_id"] in trial_ids])
        if trial_df.empty:
            break

        best_trial_id = str(sort_candidate_table(trial_df).loc[0, "candidate_id"])
        selection_path_ids.append(best_trial_id)
        current_candidate_id = best_trial_id
        current_design_df = candidate_store[best_trial_id]["design_df"].copy()
        current_dropped_terms = candidate_store[best_trial_id]["dropped_terms"].copy()

    comparison_df = pd.DataFrame(candidate_rows)
    ranked_df = sort_candidate_table(comparison_df.copy())
    ranked_df["selection_rank"] = np.arange(1, len(ranked_df) + 1)
    comparison_df = comparison_df.merge(
        ranked_df[["candidate_id", "selection_rank"]],
        on="candidate_id",
        how="left",
    )
    best_candidate_id = str(ranked_df.loc[0, "candidate_id"])
    comparison_df["selected_into_path"] = comparison_df["candidate_id"].isin(selection_path_ids)
    comparison_df["selected_model"] = comparison_df["candidate_id"] == best_candidate_id
    comparison_df = comparison_df.sort_values("selection_rank").reset_index(drop=True)

    representative_candidate_ids = REPRESENTATIVE_CANDIDATE_IDS
    display_df = comparison_df[comparison_df["candidate_id"].isin(representative_candidate_ids)].copy()
    if display_df.empty:
        display_df = comparison_df.copy()
    display_df = display_df.sort_values("selection_rank").reset_index(drop=True)

    path_df = comparison_df[comparison_df["selected_into_path"]].copy()
    safe_to_csv(
        display_df[
            [
                "candidate_id",
                "model_label",
                "dropped_variables",
                "feature_count",
                "nonsig_term_count_hc3",
                "all_hc3_significant",
                "adjusted_r2",
                "aic",
                "bic",
                "reset_f",
                "qq_rmse",
                "press",
                "rmse_loocv",
                "r2_pred",
                "mallows_cp",
            ]
        ],
        CSV_DIR / "final_model_selection_comparison.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )
    safe_to_csv(
        path_df,
        CSV_DIR / "final_model_selection_path.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )
    safe_to_csv(
        display_df[
            [
                "candidate_id",
                "model_label",
                "feature_count",
                "press",
                "rmse_loocv",
                "r2_pred",
                "mallows_cp",
                "selected_model",
            ]
        ],
        CSV_DIR / "final_model_cross_validation.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    print_section("Candidate Model Comparison After Feature Engineering")
    print_dataframe(
        display_df[
            [
                "candidate_id",
                "model_label",
                "dropped_variables",
                "feature_count",
                "adjusted_r2",
                "aic",
                "reset_f",
                "qq_rmse",
                "r2_pred",
                "nonsig_term_count_hc3",
                "max_hc3_p_value",
                "selection_rank",
                "selected_model",
            ]
        ],
        round_digits=6,
    )

    selected_row = comparison_df.loc[comparison_df["selected_model"]].iloc[0]
    recommended_engineered_label = str(selected_row["model_label"])
    recommended_engineered_model = candidate_store[best_candidate_id]["model"]
    final_design_df = candidate_store[best_candidate_id]["design_df"].copy()

    print_section("Selected Final Model After Candidate Comparison")
    print(f"Selected candidate: {best_candidate_id} - {recommended_engineered_label}")
    print(f"Dropped variables relative to the engineered base model: {selected_row['dropped_variables']}")
    print(
        "Selection rule: prioritize models with all HC3-significant terms, "
        "then smaller RESET F, lower AIC, smaller QQ RMSE, higher LOOCV R^2_pred, and fewer features."
    )

    plot_df = display_df.copy()
    plot_df["plot_color"] = np.where(
        plot_df["selected_model"],
        "#F58518",
        np.where(plot_df["selected_into_path"], "#54A24B", "#4C78A8"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_specs = [
        ("adjusted_r2", "Adjusted R^2"),
        ("aic", "AIC"),
        ("reset_f", "RESET F"),
        ("r2_pred", "LOOCV R^2_pred"),
    ]
    for ax, (metric_col, metric_title) in zip(axes.flatten(), metric_specs):
        ax.bar(plot_df["candidate_id"], plot_df[metric_col], color=plot_df["plot_color"])
        ax.set_title(metric_title)
        ax.set_xlabel("Candidate")
        ax.tick_params(axis="x", rotation=0)
    plt.suptitle("Candidate Model Comparison Metrics")
    plt.tight_layout()
    save_current_figure("11_final_model_selection_metrics.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(plot_df["candidate_id"], plot_df["nonsig_term_count_hc3"], color=plot_df["plot_color"])
    axes[0].set_title("HC3 Non-Significant Term Count")
    axes[0].set_xlabel("Candidate")
    axes[0].set_ylabel("Count")
    axes[1].bar(plot_df["candidate_id"], plot_df["max_hc3_p_value"], color=plot_df["plot_color"])
    axes[1].axhline(selection_alpha, color="crimson", linestyle="--", linewidth=1.5, label="0.05 threshold")
    axes[1].set_title("Largest HC3 p-value")
    axes[1].set_xlabel("Candidate")
    axes[1].set_ylabel("p-value")
    axes[1].legend()
    plt.tight_layout()
    save_current_figure("12_final_model_selection_significance.png")

    engineered_inference = build_inference_table(
        model=recommended_engineered_model,
        response_transform=final_model_name,
        engineering_step=recommended_engineered_label,
        robust_cov_type="HC3",
    )
    print_section("Final Model Inference with HC3 Robust SE")
    print_dataframe(
        engineered_inference[
            ["variable", "coef", "ols_f_statistic", "hc3_se", "hc3_f_statistic"]
        ],
        round_digits=6,
    )
    safe_to_csv(
        engineered_inference,
        CSV_DIR / "final_model_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    overall_f_test_df = build_overall_f_test_summary(
        model=recommended_engineered_model,
        response_transform=final_model_name,
        engineering_step=recommended_engineered_label,
        alpha=0.05,
    )
    print_section("Overall Model F Test")
    print_dataframe(overall_f_test_df, round_digits=6)
    safe_to_csv(
        overall_f_test_df.round(8),
        CSV_DIR / "overall_model_f_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    kfold_df = pd.DataFrame(
        [
            compute_kfold_cv_metrics(
                X_design=final_design_df,
                y=final_response_series,
                k=100,
                random_state=42,
                shuffle=True,
            )
        ]
    )
    safe_to_csv(
        kfold_df,
        CSV_DIR / "final_model_kfold_cv_100.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )
    final_corr_matrix = final_design_df.corr()
    final_high_corr_df = compute_high_corr_pairs(
        X=final_design_df,
        y=pd.Series(final_response_series, index=final_design_df.index, name="final_response"),
        threshold=CORR_THRESHOLD,
    )
    final_vif_df = compute_vif(final_design_df)
    safe_to_csv(final_corr_matrix, CSV_DIR / "final_engineered_correlation_matrix.csv", encoding="utf-8-sig", float_format="%.16g")
    safe_to_csv(final_high_corr_df, CSV_DIR / "final_engineered_high_corr_pairs.csv", index=False, encoding="utf-8-sig", float_format="%.16g")
    safe_to_csv(final_vif_df, CSV_DIR / "final_engineered_vif.csv", index=False, encoding="utf-8-sig", float_format="%.16g")

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        final_corr_matrix,
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
    plt.title("Correlation Heatmap After Feature Engineering")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    save_current_figure("13_final_engineered_correlation_heatmap.png")

    plot_vif_df = final_vif_df.sort_values("VIF", ascending=True)
    plt.figure(figsize=(10, 8.5))
    plt.barh(plot_vif_df["variable"], plot_vif_df["VIF"], color="#4C78A8")
    plt.axvline(VIF_THRESHOLD, color="crimson", linestyle="--", linewidth=1.5, label=f"VIF threshold = {VIF_THRESHOLD:g}")
    plt.xlabel("VIF")
    plt.ylabel("Variable")
    plt.title("VIF After Feature Engineering")
    plt.legend()
    save_current_figure("13_final_engineered_vif_bar.png")

    partial_r2_df = compute_partial_r2_ranking(
        X_design=final_design_df,
        y_series=final_response_series,
        full_model=recommended_engineered_model,
    )
    print_section("Partial R^2 Ranking for Final Model Variable Groups")
    print_dataframe(
        partial_r2_df[
            ["variable_group", "terms_in_group", "partial_r2", "f_statistic", "df_num", "df_denom"]
        ],
        round_digits=6,
    )
    safe_to_csv(
        partial_r2_df,
        CSV_DIR / "partial_r2_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.16g",
    )

    adjustment_dropone_df = build_adjustment_variable_dropone_table(
        X_design=final_design_df,
        y_series=final_response_series,
        full_model=recommended_engineered_model,
        inference_df=engineered_inference,
        partial_r2_df=partial_r2_df,
        alpha=0.05,
    )
    print_section("Drop-One Diagnostics for Non-Significant Adjustment Variables")
    print_dataframe(adjustment_dropone_df, round_digits=6)
    safe_to_csv(
        adjustment_dropone_df.round(8),
        CSV_DIR / "adjustment_variable_dropone.csv",
        index=False,
        encoding="utf-8-sig",
    )

    from evaluation import run_project_evaluation

    run_project_evaluation(
        final_model_name=final_model_name,
        selected_model_label=recommended_engineered_label,
        final_design_df=final_design_df,
        final_response_series=final_response_series,
        y_raw_final=y_raw_final,
        raw_df=raw_df,
        test_size=holdout_test_size,
        random_state=random_state,
    )
