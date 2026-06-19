# Model Report

## Final Model

- Response transformation: `log(y + 0.035)`
- Selected candidate: `Drop RentLowQ + PctEmploy + PctWorkMom + blackPerCap + NumIlleg`
- Final predictor count: 16
- Holdout metrics are computed after model specification selection, so they are a final-model sanity check rather than a fully nested model-selection estimate.
- The pre-split prespecified holdout splits raw data before imputation and training-only outlier screening, then reuses the selected final model specification.
- Adjusted R-squared: 0.7099
- LOOCV predicted R-squared: 0.7073
- 100-fold CV predicted R-squared: 0.7073
- Holdout raw-scale R-squared: 0.7335
- Holdout raw-scale RMSE: 0.1119
- Holdout raw-scale MAE: 0.0773

## Pre-Split Prespecified Holdout

- Raw-scale R-squared: 0.6056
- Raw-scale RMSE: 0.1374
- Raw-scale MAE: 0.0884
- Random state: 42
- Training outliers removed: 28

## Most Influential Terms

- `PctIlleg`: coef=1.6372, HC3 p-value=5.96e-41
- `agePct65up`: coef=0.9665, HC3 p-value=1.28e-24
- `TotalPctDiv`: coef=0.8601, HC3 p-value=1.07e-21
- `pctWInvInc`: coef=-1.1341, HC3 p-value=1.15e-17
- `PctIlleg_sq_centered`: coef=-1.3379, HC3 p-value=3.97e-17
- `racePctWhite`: coef=-0.5718, HC3 p-value=3.46e-11
- `numbUrban`: coef=0.5001, HC3 p-value=5.04e-11
- `PctOccupManu`: coef=-0.5011, HC3 p-value=2.43e-10

## Partial R-squared Ranking

- `PctIlleg`: partial R-squared=0.0928
- `agePct65up`: partial R-squared=0.0567
- `TotalPctDiv`: partial R-squared=0.0512
- `pctWInvInc`: partial R-squared=0.0415
- `racePctWhite`: partial R-squared=0.0244
- `PctOccupManu`: partial R-squared=0.0231
- `numbUrban`: partial R-squared=0.0163
- `pctUrban`: partial R-squared=0.0161

## Lasso Stability Cross-check

- `PctIlleg`: selected in 100% of repeated CV fits
- `agePct65up`: selected in 100% of repeated CV fits
- `TotalPctDiv`: selected in 100% of repeated CV fits
- `racePctHisp`: selected in 100% of repeated CV fits
- `pctUrban`: selected in 100% of repeated CV fits
- `numbUrban`: selected in 100% of repeated CV fits
- `PersPerRentOccHous`: selected in 100% of repeated CV fits
- `OtherPerCap`: selected in 100% of repeated CV fits

## Key Artifacts

- `outputs/csv/final_model_coefficients.csv`
- `outputs/csv/final_model_holdout_metrics.csv`
- `outputs/csv/leakage_aware_holdout_metrics.csv`
- `outputs/csv/lasso_feature_stability.csv`
- `outputs/images/14_holdout_prediction_diagnostics.png`
- `outputs/images/16_leakage_aware_holdout_diagnostics.png`
- `outputs/images/15_lasso_feature_stability.png`
