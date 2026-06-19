# Portfolio Framing Notes

## How This Project Goes Beyond a Basic Regression Assignment

This project is not just a single OLS fit. Its stronger portfolio angle is that it treats regression modeling as a full statistical workflow:

- It starts from raw UCI metadata and data files instead of relying on a pre-cleaned CSV.
- It makes missingness visible, removes high-missingness police-survey variables, and documents the imputation rule.
- It uses influence diagnostics with externally studentized residuals and Cook's distance before final modeling.
- It compares multiple response transformations and chooses the final transformation based on residual diagnostics, not only adjusted R-squared.
- It combines partial F-test stepwise selection, correlation review, and iterative VIF filtering to manage high-dimensional socioeconomic predictors.
- It uses HC3 robust standard errors to make inference less sensitive to heteroskedasticity.
- It adds nonlinear structure through centered quadratic feature engineering and checks functional form with Ramsey RESET.
- It compares candidate models using adjusted R-squared, AIC/BIC, LOOCV PRESS, repeated cross-validation, and robust significance.
- It now includes holdout prediction diagnostics and a LassoCV feature-stability cross-check.
- It generates reproducible CSV tables, diagnostic plots, and a Markdown model report.

## Current Limitations

The main remaining limitation is that the current pipeline is optimized for interpretable statistical modeling rather than fully leakage-proof predictive modeling. Missing-value handling, outlier screening, transformation choice, and feature selection are performed on the analysis dataset before the final holdout check.

This is acceptable for a regression analysis course project, but if the project is framed as a production-grade predictive modeling pipeline, the next improvement would be nested validation:

- Split train/test data at the beginning.
- Learn imputation values, outlier rules, response transformation, feature selection, VIF filtering, and quadratic terms only on the training data.
- Apply the frozen rules to the test data.
- Report test-set metrics only once after all modeling decisions are locked.

Another limitation is that stepwise selection can be unstable. The added Lasso feature-stability table helps address this by checking whether final predictors are also repeatedly selected by a regularized model, but it is still a robustness check rather than a replacement for nested validation.

## Suggested Resume Bullets

- Built an end-to-end interpretable regression pipeline on the UCI Communities and Crime dataset, covering raw data parsing, missingness analysis, outlier diagnostics, feature selection, response transformation, multicollinearity control, and robust inference.
- Improved model validity by comparing log, Box-Cox, Yeo-Johnson, arcsin-sqrt, and logit response transformations, then selecting `log(y + 0.035)` based on residual normality and heteroskedasticity diagnostics.
- Reduced high-dimensional socioeconomic predictors through partial F-test stepwise selection and iterative VIF filtering, then refined the final model with HC3 robust standard errors and quadratic feature engineering.
- Achieved approximately 0.71 adjusted R-squared and 0.71 cross-validated predicted R-squared while maintaining an interpretable 16-variable final model.
- Added holdout prediction diagnostics, LassoCV feature-stability validation, and automated model reporting to convert a course assignment into a reproducible GitHub portfolio project.

## How to Present the Project

The strongest positioning is:

> An interpretable statistical modeling project for violent crime-rate analysis, emphasizing model diagnostics, feature-selection robustness, and reproducible reporting rather than black-box prediction.

Avoid presenting it as a causal crime model. The dataset is observational, normalized, and historically dated, so the safest wording is "association", "predictive signal", "model diagnostics", and "interpretable regression workflow".

## What Makes It Distinct

Common regression projects usually stop at fitting `LinearRegression` or `OLS` and reporting R-squared. This project is more distinctive because it includes:

- Residual-driven response transformation selection
- Influence diagnostics before final modeling
- VIF-aware feature reduction
- HC3 robust inference
- Functional-form testing with Ramsey RESET
- Centered quadratic term engineering
- Partial R-squared ranking for interpretability
- Regularized-model stability cross-check
- Automated model report generation

Those points should be highlighted in the GitHub README and resume because they show statistical judgment, not just library usage.
