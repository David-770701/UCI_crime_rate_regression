# Communities and Crime Regression Modeling

This project builds an interpretable linear regression pipeline for the UCI Communities and Crime dataset. The goal is to explain and predict normalized per-capita violent crime rates using socioeconomic, housing, demographic, and law-enforcement variables.

The analysis was originally developed for a regression analysis course project and has been reorganized into a reproducible portfolio-style project.

## Project Highlights

- Built a complete OLS regression workflow from raw UCI files to final model diagnostics.
- Parsed the original `communities.names` metadata to construct a reusable CSV dataset automatically.
- Removed high-missingness and identifier fields, applied mean imputation, and screened influential observations using externally studentized residuals and Cook's distance.
- Compared response transformations including raw y, `log(y+c)`, Box-Cox, Yeo-Johnson, arcsin-sqrt, and logit transforms.
- Used partial F-test stepwise selection, correlation review, and iterative VIF filtering to control multicollinearity.
- Added quadratic feature engineering and model refinement with HC3 robust standard errors.
- Evaluated candidate models with adjusted R-squared, AIC/BIC, Ramsey RESET, LOOCV PRESS, 100-fold cross-validation, and partial R-squared rankings.
- Added post-selection holdout diagnostics, a pre-split prespecified holdout check, LassoCV feature-stability checks, and an automated Markdown model report.

## Key Result

The final selected model uses `log(ViolentCrimesPerPop + 0.035)` as the response transformation. In the generated results from the current pipeline:

- Final model size: 16 predictors
- Adjusted R-squared: 0.7099
- LOOCV predicted R-squared: 0.7073
- 100-fold CV predicted R-squared: 0.7073
- Holdout raw-scale R-squared: 0.7335
- Holdout raw-scale RMSE: 0.1119
- Pre-split prespecified holdout raw-scale R-squared: 0.6056
- Pre-split prespecified holdout raw-scale RMSE: 0.1374
- Strongest variable groups by robust inference include `PctIlleg`, `agePct65up`, `TotalPctDiv`, `pctWInvInc`, and the centered quadratic term for `PctIlleg`.

These metrics are produced in `outputs/csv/` when the pipeline is run.

The original holdout metrics are computed after the final model specification is selected, so they should be read as a final-model validation check rather than a fully nested estimate of the entire model-selection workflow. The pre-split prespecified holdout splits the raw data before imputation and training-only outlier screening, then reuses the final model specification as a stricter prediction check.

## Repository Structure

```text
.
├── code/
│   ├── crime_regression_pipeline.py   # Main analysis workflow
│   ├── data_preprocessing.py          # Data loading, cleaning, and outlier detection
│   ├── feature_selection.py           # Stepwise selection, correlation, and VIF stages
│   ├── model_diagnostics.py           # Transform comparison, diagnostics, engineering
│   ├── evaluation.py                   # Holdout metrics, Lasso stability, model report
│   ├── pipeline_config.py             # Central paths and modeling constants
│   └── pipeline_utils.py              # Shared statistical helpers
├── docs/
│   └── portfolio_framing.md            # Resume and GitHub positioning notes
├── communities.data                   # Raw UCI data file
├── communities.names                  # Raw UCI metadata and attribute names
├── run_analysis.py                    # Root-level entry point
├── requirements.txt
└── README.md
```

Generated files are intentionally ignored by Git:

- `data/communities.csv`
- `outputs/csv/*.csv`
- `outputs/images/*.png`
- legacy generated files under `code/outputs/`

## How to Run

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the full analysis:

```bash
python run_analysis.py
```

Useful reproducibility options:

```bash
python run_analysis.py --random-state 42 --test-size 0.20
python run_analysis.py --keep-outliers
python run_analysis.py --ignore-warnings
```

The first run creates `data/communities.csv` from the raw UCI files. All model tables and diagnostic plots are written to:

```text
outputs/csv/
outputs/images/
```

Key prediction-check files include:

```text
outputs/csv/final_model_holdout_metrics.csv
outputs/csv/leakage_aware_holdout_metrics.csv
outputs/images/16_leakage_aware_holdout_diagnostics.png
```

## Data Source

Dataset: UCI Communities and Crime

The dataset combines socioeconomic data from the 1990 US Census, law-enforcement data from the 1990 LEMAS survey, and crime data from the 1995 FBI Uniform Crime Report. The target variable is `ViolentCrimesPerPop`, a normalized per-capita violent crime rate.

## Methods

The pipeline follows this sequence:

1. Data parsing and overview
2. Missingness screening and mean imputation
3. Influential observation detection
4. Response transformation comparison
5. Partial F-test stepwise feature selection
6. Correlation and VIF-based multicollinearity filtering
7. Robust-inference diagnostics with HC3 standard errors
8. Quadratic feature engineering
9. Candidate model comparison and cross-validation
10. Post-selection holdout evaluation, pre-split prespecified holdout evaluation, and LassoCV feature-stability check
11. Final coefficient, VIF, partial R-squared, overall F-test, and model-report generation

## Portfolio Framing

This project demonstrates applied regression modeling with an emphasis on interpretability, statistical diagnostics, and reproducibility. It is designed less as a black-box prediction task and more as a transparent model-building workflow for structured social data.
