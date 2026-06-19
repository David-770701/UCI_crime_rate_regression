from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special, stats
from scipy.stats import boxcox
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera

from pipeline_config import IMG_DIR, LOGIT_EPS, TARGET


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dataframe(df: pd.DataFrame, round_digits: int | None = None) -> None:
    if round_digits is not None:
        df = df.round(round_digits)
    print(df.to_string())


def safe_to_csv(df: pd.DataFrame, output_path: Path, **kwargs) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, **kwargs)
        print(f"Saved core file: {output_path}")
        return output_path
    except PermissionError:
        # This keeps the pipeline running when a CSV is open in Excel.
        fallback_path = output_path.with_name(f"{output_path.stem}_latest{output_path.suffix}")
        df.to_csv(fallback_path, **kwargs)
        print(f"Could not overwrite locked file: {output_path}")
        print(f"Saved fallback file instead: {fallback_path}")
        return fallback_path


def save_current_figure(filename: str) -> None:
    output_path = IMG_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved image: {output_path}")


def fit_ols_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()


def compute_loocv_press_metrics(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    y: pd.Series | np.ndarray,
    sigma2_ref: float | None = None,
) -> dict:
    y_arr = np.asarray(y, dtype=float)
    resid = np.asarray(model.resid, dtype=float)
    hat_diag = np.asarray(model.get_influence().hat_matrix_diag, dtype=float)
    denom = 1.0 - hat_diag
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    press_resid = resid / denom
    press = float(np.nansum(np.square(press_resid)))
    sst = float(np.sum(np.square(y_arr - float(np.mean(y_arr)))))
    r2_pred = float("nan") if sst <= 0 else float(1.0 - press / sst)
    nobs = int(model.nobs)
    p_params = int(model.df_model) + 1
    sse = float(np.sum(np.square(resid)))
    mse_resid = float(model.mse_resid)
    cp = float("nan")
    if sigma2_ref is not None and sigma2_ref > 0:
        cp = float(sse / sigma2_ref - (nobs - 2 * p_params))

    return {
        "nobs": nobs,
        "p_params": p_params,
        "sse": sse,
        "mse_resid": mse_resid,
        "press": press,
        "rmse_loocv": float(np.sqrt(press / nobs)) if nobs > 0 else float("nan"),
        "r2_pred": r2_pred,
        "mallows_cp": cp,
    }


def compute_kfold_cv_metrics(
    X_design: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int,
    random_state: int = 42,
    shuffle: bool = True,
) -> dict:
    if k < 2:
        raise ValueError("k must be >= 2.")

    X_df = X_design.reset_index(drop=True)
    y_arr = np.asarray(y, dtype=float)
    nobs = int(X_df.shape[0])
    if k > nobs:
        raise ValueError("k cannot be larger than number of observations.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(nobs)
    if shuffle:
        rng.shuffle(indices)

    folds = np.array_split(indices, k)
    press = 0.0
    y_mean = float(np.mean(y_arr))
    sst = float(np.sum(np.square(y_arr - y_mean)))

    for test_idx in folds:
        train_mask = np.ones(nobs, dtype=bool)
        train_mask[test_idx] = False
        train_idx = indices[train_mask[indices]]

        X_train = X_df.iloc[train_idx]
        y_train = y_arr[train_idx]
        X_test = X_df.iloc[test_idx]
        y_test = y_arr[test_idx]

        model = fit_ols_model(X_train, y_train)
        y_pred = model.predict(sm.add_constant(X_test, has_constant="add"))
        press += float(np.sum(np.square(y_test - np.asarray(y_pred, dtype=float))))

    r2_pred = float("nan") if sst <= 0 else float(1.0 - press / sst)
    rmse = float(np.sqrt(press / nobs)) if nobs > 0 else float("nan")

    return {
        "k": int(k),
        "nobs": nobs,
        "press_kfold": float(press),
        "rmse_kfold": rmse,
        "r2_pred_kfold": r2_pred,
        "random_state": int(random_state),
        "shuffle": bool(shuffle),
    }


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["variable", "VIF"])

    X_const = sm.add_constant(X, has_constant="add")
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_df = pd.DataFrame(
        {
            "variable": X_const.columns,
            "VIF": [
                variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])
            ],
        }
    )
    return (
        vif_df[vif_df["variable"] != "const"]
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha_in: float = 0.05,
    alpha_out: float = 0.05,
    verbose: bool = True,
) -> list[str]:
    selected: list[str] = []
    remaining = list(X.columns)

    while True:
        changed = False
        best_f = float("-inf")
        best_f_critical = np.nan
        best_var = None

        # Forward step: add the variable with the strongest partial F improvement.
        for var in remaining:
            try:
                test_vars = selected + [var]
                model = sm.OLS(y, sm.add_constant(X[test_vars], has_constant="add")).fit()
                f_test_result = model.f_test(f"{var} = 0")
                f_value = float(np.asarray(f_test_result.fvalue).item())
                f_critical = float(stats.f.isf(alpha_in, f_test_result.df_num, f_test_result.df_denom))
                if f_value > best_f and f_value > f_critical:
                    best_f = f_value
                    best_f_critical = f_critical
                    best_var = var
            except Exception:
                continue

        if best_var is not None:
            selected.append(best_var)
            remaining.remove(best_var)
            changed = True
            if verbose:
                print(
                    f"Add variable: {best_var:<25} "
                    f"F={best_f:.4f} (critical={best_f_critical:.4f})"
                )

        if selected:
            current_model = sm.OLS(y, sm.add_constant(X[selected], has_constant="add")).fit()
            f_rows: list[dict] = []
            # Backward step: remove variables that no longer pass the F threshold.
            for var in selected:
                f_test_result = current_model.f_test(f"{var} = 0")
                f_value = float(np.asarray(f_test_result.fvalue).item())
                f_critical = float(stats.f.isf(alpha_out, f_test_result.df_num, f_test_result.df_denom))
                f_rows.append(
                    {
                        "variable": var,
                        "f_statistic": f_value,
                        "f_critical": f_critical,
                    }
                )

            if f_rows:
                backward_df = pd.DataFrame(f_rows).sort_values("f_statistic", ascending=True).reset_index(drop=True)
                worst_var = str(backward_df.loc[0, "variable"])
                worst_f = float(backward_df.loc[0, "f_statistic"])
                worst_f_critical = float(backward_df.loc[0, "f_critical"])
                if worst_f < worst_f_critical:
                    selected.remove(worst_var)
                    remaining.append(worst_var)
                    changed = True
                    if verbose:
                        print(
                            f"Remove variable: {worst_var:<25} "
                            f"F={worst_f:.4f} (critical={worst_f_critical:.4f})"
                        )

        if not changed:
            break

    return selected


def compute_feature_f_tests(
    X: pd.DataFrame,
    y: pd.Series,
    selected_vars: list[str],
) -> pd.DataFrame:
    if not selected_vars:
        return pd.DataFrame(columns=["variable", "f_statistic", "df_num", "df_denom"])

    model = fit_ols_model(X[selected_vars], y)
    rows: list[dict] = []
    for var in selected_vars:
        f_test_result = model.f_test(f"{var} = 0")
        f_value = np.asarray(f_test_result.fvalue).item()
        rows.append(
            {
                "variable": var,
                "f_statistic": float(f_value),
                "df_num": float(f_test_result.df_num),
                "df_denom": float(f_test_result.df_denom),
            }
        )

    return pd.DataFrame(rows).sort_values("f_statistic", ascending=False).reset_index(drop=True)


def compute_high_corr_pairs(X: pd.DataFrame, y: pd.Series, threshold: float = 0.70) -> pd.DataFrame:
    corr = X.corr()
    rows: list[dict] = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            corr_value = corr.iloc[i, j]
            if abs(corr_value) > threshold:
                var1 = corr.columns[i]
                var2 = corr.columns[j]
                rows.append(
                    {
                        "Variable 1": var1,
                        "Variable 2": var2,
                        "Correlation": round(float(corr_value), 4),
                        "abs_correlation": abs(float(corr_value)),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Variable 1",
                "Variable 2",
                "Correlation",
                "abs_correlation",
            ]
        )

    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.70,
    vif_threshold: float = 10.0,
) -> tuple:
    model = fit_ols_model(X, y)
    pair_df = compute_high_corr_pairs(X, y, corr_threshold)
    vif_df = compute_vif(X)

    metrics = {
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "high_corr_pair_count": int(pair_df.shape[0]),
        "max_abs_corr": float(pair_df["abs_correlation"].max()) if not pair_df.empty else 0.0,
        "vif_exceed_count": int((vif_df["VIF"] > vif_threshold).sum()) if not vif_df.empty else 0,
        "max_vif": float(vif_df["VIF"].max()) if not vif_df.empty else 0.0,
    }
    return model, vif_df, pair_df, metrics


def select_best_feature_to_drop(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_pool: list[str],
    corr_threshold: float = 0.70,
    vif_threshold: float = 10.0,
) -> tuple[dict, pd.DataFrame]:
    target_corr = pd.concat([X, y], axis=1).corr(numeric_only=True)[y.name].drop(y.name).abs()
    trial_rows: list[dict] = []

    for candidate in sorted(candidate_pool):
        X_trial = X.drop(columns=[candidate])
        _, _, _, metrics = evaluate_feature_set(
            X_trial,
            y,
            corr_threshold=corr_threshold,
            vif_threshold=vif_threshold,
        )
        trial_rows.append(
            {
                "candidate": candidate,
                "removed_target_corr_abs": float(target_corr.get(candidate, np.nan)),
                **metrics,
            }
        )

    # Rank candidate removals by collinearity relief first, then by model quality.
    trial_df = pd.DataFrame(trial_rows).sort_values(
        [
            "vif_exceed_count",
            "max_vif",
            "high_corr_pair_count",
            "max_abs_corr",
            "aic",
            "bic",
            "adj_r2",
            "removed_target_corr_abs",
            "candidate",
        ],
        ascending=[True, True, True, True, True, True, False, True, True],
    ).reset_index(drop=True)

    return trial_df.iloc[0].to_dict(), trial_df


def iterative_vif_filter(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 10.0,
    corr_threshold: float = 0.70,
    protected: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    protected = set(protected or [])
    current_X = X.copy()
    removed: list[dict] = []
    trial_history: list[dict] = []
    vif_history: list[dict] = []

    while True:
        vif_df = compute_vif(current_X)
        over_df = vif_df[vif_df["VIF"] > threshold].copy()
        if over_df.empty:
            break

        corr_abs = current_X.corr().abs()
        candidate_pool = set(over_df["variable"])
        # Also consider highly correlated neighbors of high-VIF variables as drop candidates.
        for var in over_df["variable"]:
            related = corr_abs[var][corr_abs[var] > corr_threshold].index.tolist()
            candidate_pool.update(v for v in related if v != var)

        candidate_pool = sorted(var for var in candidate_pool if var not in protected)
        if not candidate_pool:
            break

        best_choice, trial_df = select_best_feature_to_drop(
            current_X,
            y,
            candidate_pool,
            corr_threshold=corr_threshold,
            vif_threshold=threshold,
        )
        removed_var = best_choice["candidate"]
        removed_var_vif = float(vif_df.loc[vif_df["variable"] == removed_var, "VIF"].iloc[0])

        removed.append(
            {
                "Removed Variable": removed_var,
                "Removed Variable VIF": round(removed_var_vif, 4),
                "Candidate Pool": ", ".join(candidate_pool),
                "VIF>Threshold Before": int(over_df.shape[0]),
                "VIF>Threshold After": int(best_choice["vif_exceed_count"]),
                "Max VIF Before": round(float(vif_df["VIF"].max()), 4),
                "Max VIF After": round(float(best_choice["max_vif"]), 4),
                "High-Corr Pairs After": int(best_choice["high_corr_pair_count"]),
                "Adj R^2 After": round(float(best_choice["adj_r2"]), 4),
                "AIC After": round(float(best_choice["aic"]), 2),
                "BIC After": round(float(best_choice["bic"]), 2),
            }
        )

        current_X = current_X.drop(columns=[removed_var])
        trial_history.append({"stage": f"Before removing {removed_var}", "trial_df": trial_df})
        vif_history.append(
            {
                "stage": f"After removing {removed_var}",
                "removed_variable": removed_var,
                "vif_table": compute_vif(current_X),
            }
        )

    return current_X, pd.DataFrame(removed), trial_history, vif_history


def get_studentized_residuals(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> np.ndarray:
    return np.asarray(model.get_influence().resid_studentized_internal, dtype=float)


def qq_rmse(residuals: pd.Series | np.ndarray) -> float:
    z = np.asarray(residuals, dtype=float)
    z = (z - z.mean()) / z.std(ddof=1)
    n = len(z)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sample = np.sort(z)
    return float(np.sqrt(np.mean((sample - theoretical) ** 2)))


def collect_model_diagnostics(
    label: str,
    transform_name: str,
    X: pd.DataFrame,
    y_transformed: pd.Series | np.ndarray,
    extra_info: dict | None = None,
) -> tuple:
    model = fit_ols_model(X, y_transformed)
    residuals = model.resid
    studentized_residuals = get_studentized_residuals(model)
    fitted = model.fittedvalues
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = het_breuschpagan(residuals, model.model.exog)

    result = {
        "label": label,
        "transform": transform_name,
        "adjusted_r2": float(model.rsquared_adj),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "bp_lm_stat": float(bp_lm_stat),
        "bp_lm_pvalue": float(bp_lm_pvalue),
        "bp_f_stat": float(bp_f_stat),
        "bp_f_pvalue": float(bp_f_pvalue),
        "resid_skew_abs": float(abs(skew)),
        "qq_rmse": qq_rmse(studentized_residuals),
        "resid_fitted_corr_abs": float(abs(np.corrcoef(fitted, residuals)[0, 1])),
    }
    if extra_info:
        result.update(extra_info)
    return model, result


def build_inference_table(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    response_transform: str,
    engineering_step: str | None = None,
    robust_cov_type: str = "HC3",
) -> pd.DataFrame:
    param_index = model.params.index
    robust_result = model.get_robustcov_results(cov_type=robust_cov_type)
    df_resid = float(model.df_resid)

    robust_params = pd.Series(robust_result.params, index=param_index)
    robust_bse = pd.Series(robust_result.bse, index=param_index)
    robust_tvalues = pd.Series(robust_result.tvalues, index=param_index)
    ols_pvalues = pd.Series(2 * stats.t.sf(np.abs(model.tvalues), df=df_resid), index=param_index)
    robust_pvalues = pd.Series(2 * stats.t.sf(np.abs(robust_tvalues), df=df_resid), index=param_index)
    ols_f_stats = pd.Series(np.square(model.tvalues), index=param_index)
    robust_f_stats = pd.Series(np.square(robust_tvalues), index=param_index)
    robust_conf_int = pd.DataFrame(
        robust_result.conf_int(),
        index=param_index,
        columns=[f"{robust_cov_type.lower()}_ci_lower", f"{robust_cov_type.lower()}_ci_upper"],
    )

    inference_df = pd.DataFrame(
        {
            "response_transform": response_transform,
            "engineering_step": engineering_step if engineering_step is not None else "None",
            "variable": param_index,
            "coef": model.params.values,
            "ols_se": model.bse.values,
            "ols_t_value": model.tvalues.values,
            "ols_p_value": ols_pvalues.values,
            "ols_f_statistic": ols_f_stats.values,
            f"{robust_cov_type.lower()}_coef": robust_params.values,
            f"{robust_cov_type.lower()}_se": robust_bse.values,
            f"{robust_cov_type.lower()}_t_value": robust_tvalues.values,
            f"{robust_cov_type.lower()}_p_value": robust_pvalues.values,
            f"{robust_cov_type.lower()}_f_statistic": robust_f_stats.values,
        }
    )
    inference_df = pd.concat([inference_df, robust_conf_int.reset_index(drop=True)], axis=1)
    inference_df = inference_df[inference_df["variable"] != "const"].copy()
    inference_df = inference_df.sort_values(f"{robust_cov_type.lower()}_f_statistic", ascending=False).reset_index(drop=True)
    return inference_df


def evaluate_engineered_model(
    model_label: str,
    X_design: pd.DataFrame,
    y_series: pd.Series | np.ndarray,
) -> tuple:
    model = fit_ols_model(X_design, y_series)
    resid = model.resid
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid)
    bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = het_breuschpagan(resid, model.model.exog)
    reset_res = linear_reset(model, power=2, use_f=True)

    return model, {
        "model_label": model_label,
        "feature_count": int(X_design.shape[1]),
        "adjusted_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "bp_lm_stat": float(bp_lm_stat),
        "bp_lm_pvalue": float(bp_lm_pvalue),
        "bp_f_stat": float(bp_f_stat),
        "bp_f_pvalue": float(bp_f_pvalue),
        "reset_f": float(reset_res.fvalue),
        "reset_pvalue": float(reset_res.pvalue),
        "qq_rmse": qq_rmse(resid),
    }


def refine_model_by_hc3_significance(
    X_design: pd.DataFrame,
    y_series: pd.Series | np.ndarray,
    alpha: float = 0.05,
    robust_cov_type: str = "HC3",
) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    current_X = X_design.copy()
    removal_rows: list[dict] = []

    while True:
        model = fit_ols_model(current_X, y_series)
        inference_df = build_inference_table(
            model=model,
            response_transform="refinement_only",
            engineering_step="None",
            robust_cov_type=robust_cov_type,
        )

        protected_linear_terms = {
            col[: -len("_sq_centered")]
            for col in current_X.columns
            if col.endswith("_sq_centered")
        }
        nonsig_df = inference_df[
            (inference_df[f"{robust_cov_type.lower()}_p_value"] > alpha)
            & (~inference_df["variable"].isin(protected_linear_terms))
        ].copy()

        if nonsig_df.empty:
            return current_X, model, pd.DataFrame(removal_rows)

        drop_row = nonsig_df.sort_values(
            [f"{robust_cov_type.lower()}_p_value", f"{robust_cov_type.lower()}_f_statistic"],
            ascending=[False, True],
        ).iloc[0]
        drop_var = str(drop_row["variable"])

        removal_rows.append(
            {
                "removed_variable": drop_var,
                f"{robust_cov_type.lower()}_p_value_before_drop": float(drop_row[f"{robust_cov_type.lower()}_p_value"]),
                f"{robust_cov_type.lower()}_f_statistic_before_drop": float(drop_row[f"{robust_cov_type.lower()}_f_statistic"]),
                "remaining_feature_count_after_drop": int(current_X.shape[1] - 1),
            }
        )
        current_X = current_X.drop(columns=[drop_var])


def compute_partial_r2_ranking(
    X_design: pd.DataFrame,
    y_series: pd.Series | np.ndarray,
    full_model: sm.regression.linear_model.RegressionResultsWrapper | None = None,
) -> pd.DataFrame:
    if full_model is None:
        full_model = fit_ols_model(X_design, y_series)

    group_map: dict[str, list[str]] = {}
    for col in X_design.columns:
        if col.endswith("_sq_centered"):
            group_name = col[: -len("_sq_centered")]
        else:
            group_name = col
        group_map.setdefault(group_name, []).append(col)

    full_rss = float(np.sum(np.square(full_model.resid)))
    rows: list[dict] = []

    for group_name, terms in group_map.items():
        reduced_X = X_design.drop(columns=terms)
        reduced_model = fit_ols_model(reduced_X, y_series)
        reduced_rss = float(np.sum(np.square(reduced_model.resid)))
        f_value, p_value, df_diff = full_model.compare_f_test(reduced_model)
        # Recompute the p-value explicitly to avoid underflow being written as 0.0 in CSV output.
        p_value = float(stats.f.sf(f_value, df_diff, float(full_model.df_resid)))
        partial_r2 = (reduced_rss - full_rss) / reduced_rss if reduced_rss > 0 else np.nan

        rows.append(
            {
                "variable_group": group_name,
                "terms_in_group": ", ".join(terms),
                "term_count": int(len(terms)),
                "partial_r2": float(partial_r2),
                "f_statistic": float(f_value),
                "df_num": float(df_diff),
                "df_denom": float(full_model.df_resid),
                "p_value": p_value,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["partial_r2", "f_statistic"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_overall_f_test_summary(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    response_transform: str,
    engineering_step: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    def format_f_pvalue(f_value: float, dfn: float, dfd: float) -> str:
        p_value = float(stats.f.sf(f_value, dfn, dfd))
        if np.isfinite(p_value) and p_value > 0:
            return f"{p_value:.16g}"

        half_dfn = dfn / 2.0
        if float(half_dfn).is_integer():
            a = dfd / 2.0
            b = int(round(half_dfn))
            z = dfd / (dfd + dfn * f_value)
            poly_sum = 0.0
            for k in range(b):
                ratio = 1.0
                for j in range(k):
                    ratio *= (a + j) / (a + 1 + j)
                poly_sum += ((-1) ** k) * math.comb(b - 1, k) * ratio * (z**k)

            if poly_sum > 0:
                log_p = a * np.log(z) - np.log(a) - special.betaln(a, b) + np.log(poly_sum)
                log10_p = log_p / np.log(10.0)
                exponent = int(np.floor(log10_p))
                mantissa = 10 ** (log10_p - exponent)
                return f"{mantissa:.16f}e{exponent}"

        return f"< {np.finfo(float).tiny:.16g}"

    f_statistic = float(model.fvalue)
    df_num = float(model.df_model)
    df_denom = float(model.df_resid)
    critical_f = float(stats.f.isf(alpha, df_num, df_denom))
    reject_null = bool(f_statistic > critical_f)

    return pd.DataFrame(
        [
            {
                "response_transform": response_transform,
                "engineering_step": engineering_step,
                "null_hypothesis": "All slope coefficients are jointly equal to 0",
                "alternative_hypothesis": "At least one slope coefficient is not 0",
                "f_statistic": f_statistic,
                "df_num": df_num,
                "df_denom": df_denom,
                "critical_f_5pct": critical_f,
                "p_value": format_f_pvalue(f_statistic, df_num, df_denom),
                "reject_h0_5pct": reject_null,
            }
        ]
    )


def build_adjustment_variable_dropone_table(
    X_design: pd.DataFrame,
    y_series: pd.Series | np.ndarray,
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
    inference_df: pd.DataFrame,
    partial_r2_df: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    nonsig_df = inference_df[inference_df["hc3_p_value"] > alpha].copy()
    if nonsig_df.empty:
        return pd.DataFrame(
            columns=[
                "variable",
                "hc3_p_value",
                "partial_r2",
                "delta_adj_r2_after_drop",
                "delta_aic_after_drop",
                "delta_bic_after_drop",
                "delta_jb_after_drop",
                "delta_bp_f_after_drop",
                "delta_reset_f_after_drop",
                "delta_qq_rmse_after_drop",
                "max_key_coef_pct_change",
                "key_coefficients_changed",
            ]
        )

    if partial_r2_df is None:
        partial_r2_df = compute_partial_r2_ranking(X_design=X_design, y_series=y_series, full_model=full_model)
    partial_r2_map = partial_r2_df.set_index("variable_group")["partial_r2"].to_dict()

    full_metrics = evaluate_engineered_model("full_model", X_design, y_series)[1]
    key_terms = inference_df[inference_df["hc3_p_value"] <= alpha]["variable"].tolist()
    key_terms = [var for var in key_terms if var in X_design.columns]
    full_params = full_model.params.drop(labels=["const"], errors="ignore")

    rows: list[dict] = []
    for variable in nonsig_df["variable"]:
        if variable not in X_design.columns:
            continue

        # Compare each weak variable against the full model one at a time.
        reduced_X = X_design.drop(columns=[variable])
        reduced_model, reduced_metrics = evaluate_engineered_model(f"drop_{variable}", reduced_X, y_series)
        common_key_terms = [term for term in key_terms if term in reduced_model.params.index and term in full_params.index]

        change_rows: list[tuple[str, float]] = []
        for term in common_key_terms:
            baseline = float(full_params[term])
            updated = float(reduced_model.params[term])
            if np.isclose(baseline, 0.0):
                pct_change = abs(updated - baseline)
            else:
                pct_change = abs((updated - baseline) / baseline) * 100
            change_rows.append((term, float(pct_change)))

        change_rows.sort(key=lambda item: item[1], reverse=True)
        top_changed_terms = ", ".join(f"{term} ({pct:.2f}%)" for term, pct in change_rows[:3]) if change_rows else ""
        max_change = change_rows[0][1] if change_rows else 0.0

        rows.append(
            {
                "variable": variable,
                "hc3_p_value": float(nonsig_df.loc[nonsig_df["variable"] == variable, "hc3_p_value"].iloc[0]),
                "partial_r2": float(partial_r2_map.get(variable, np.nan)),
                "delta_adj_r2_after_drop": float(reduced_metrics["adjusted_r2"] - full_metrics["adjusted_r2"]),
                "delta_aic_after_drop": float(reduced_metrics["aic"] - full_metrics["aic"]),
                "delta_bic_after_drop": float(reduced_metrics["bic"] - full_metrics["bic"]),
                "delta_jb_after_drop": float(reduced_metrics["jb_stat"] - full_metrics["jb_stat"]),
                "delta_bp_f_after_drop": float(reduced_metrics["bp_f_stat"] - full_metrics["bp_f_stat"]),
                "delta_reset_f_after_drop": float(reduced_metrics["reset_f"] - full_metrics["reset_f"]),
                "delta_qq_rmse_after_drop": float(reduced_metrics["qq_rmse"] - full_metrics["qq_rmse"]),
                "max_key_coef_pct_change": float(max_change),
                "key_coefficients_changed": top_changed_terms,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["delta_aic_after_drop", "delta_reset_f_after_drop", "max_key_coef_pct_change"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def get_final_response_series(
    final_model_name: str,
    y_raw_final: pd.Series,
    best_log_c: float,
    best_boxcox_c: float,
    best_boxcox_lambda: float,
    best_yeojohnson_lambda: float = 0.0,
) -> pd.Series:
    if final_model_name == "raw y":
        return y_raw_final
    if final_model_name.startswith("log(y +"):
        return pd.Series(np.log(y_raw_final + best_log_c), index=y_raw_final.index, name=f"log_{TARGET}")
    if final_model_name.startswith("Box-Cox("):
        return pd.Series(
            boxcox(y_raw_final + best_boxcox_c, lmbda=best_boxcox_lambda),
            index=y_raw_final.index,
            name=f"boxcox_{TARGET}",
        )
    if final_model_name.startswith("Yeo-Johnson("):
        return pd.Series(
            stats.yeojohnson(y_raw_final, lmbda=best_yeojohnson_lambda),
            index=y_raw_final.index,
            name=f"yeojohnson_{TARGET}",
        )
    if final_model_name.startswith("arcsin(sqrt(y))"):
        return pd.Series(
            np.arcsin(np.sqrt(np.clip(y_raw_final, 0, 1))),
            index=y_raw_final.index,
            name=f"arcsin_sqrt_{TARGET}",
        )
    return pd.Series(
        np.log(
            np.clip(y_raw_final, LOGIT_EPS, 1 - LOGIT_EPS)
            / (1 - np.clip(y_raw_final, LOGIT_EPS, 1 - LOGIT_EPS))
        ),
        index=y_raw_final.index,
        name=f"logit_{TARGET}",
    )
