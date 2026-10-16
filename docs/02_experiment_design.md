# 2. Experiment design

## Historical comparison

The owner-supplied September comparison evaluated a dummy classifier, logistic regression with three C values and two weighting options, decision trees, random forests and XGBoost. The [frozen report](../MODEL_RESULTS_FROZEN_2026-09-29.md) is the authoritative table.

Unweighted XGBoost was selected using the highest validation AP, **23.48%**, before held-out test reporting. Its test AP was **21.32%**. AP (average precision) summarizes precision across recall levels; it is not accuracy and does not specify a decision threshold.

## Reconstructed fixed experiment

Historical Colab training code, the fitted estimator and exact three-way membership were unavailable. The executable pipeline reconstructs one fixed experiment using the local notebooks and frozen XGBoost settings. It does not recreate the complete historical model search.

Patients are permuted with `default_rng(42)`. The final 20% are test; the final 20% of the remaining patients are validation. Floor cutoffs apply to patient counts. There is no label stratification.

| Partition | Encounters | Patients |
|---|---:|---:|
| Train | 63,974 | 45,078 |
| Validation | 16,123 | 11,270 |
| Test | 20,014 | 14,088 |

The historical encounter counts were 63,925 / 16,175 / 20,011. Different membership prevents a direct claim that changes in AP reflect implementation improvements.

The fixed model uses 150 trees, depth 3, learning rate 0.1, subsample 0.8, column subsample 0.8, binary logistic objective and CPU histogram trees. Class weighting is disabled. No early stopping, additional tuning, operational threshold or test-driven selection is added.

## Leakage controls and evaluation

Split before fitting medians, categorical vocabularies or the estimator. Identifiers and outcome columns never become predictors. Validation and test only transform with fitted preprocessing and receive probabilities. Feature availability is framed at discharge; external validation would still require a real prediction-time audit.

Report AP as the primary metric, with ROC AUC, log loss and Brier score. Training metrics describe fit; held-out metrics describe offline ranking/probability performance for this split. Brier and log loss alone do not establish clinically adequate calibration.

Next: [implementation](03_implementation.md).
