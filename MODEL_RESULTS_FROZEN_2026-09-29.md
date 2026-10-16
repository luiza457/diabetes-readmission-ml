# Frozen readmission-model results — 29 September 2026

## Decision made before viewing the test results

**Selected model:** unweighted XGBoost classifier. Its validation average precision (AP) was **23.48%**, the highest among the tested candidates. The selected settings were 150 boosting rounds, maximum tree depth 3, learning rate 0.1, row subsampling 0.8, feature subsampling 0.8, binary logistic objective, histogram tree method, and random seed 42. No class weight was used.

**Primary comparison metric:** AP, which summarizes precision across recall levels. No operational decision threshold was selected. The 3,500-follow-up limit used in earlier exercises was hypothetical and is not a hospital requirement.

## Locked split and evaluation

The modelling cohort contained 100,111 encounters after excluding 1,652 expired encounters. Patients were kept in separate training, validation, and test sets: 63,925, 16,175, and 20,011 encounters, respectively. Preprocessing was fitted on training data and applied without refitting to validation and test data. Model selection used validation results; the held-out test set was evaluated afterward. No test-set metric is used to change the selected model or its settings.

| Model | Validation AP | Test AP |
|---|---:|---:|
| Dummy baseline | 11.48% | 11.39% |
| Logistic regression, unweighted, C=0.1 | 22.36% | 21.16% |
| Logistic regression, unweighted, C=1.0 | 21.54% | 20.14% |
| Logistic regression, unweighted, C=10.0 | 20.63% | 19.14% |
| Logistic regression, balanced, C=0.1 | 22.14% | 20.98% |
| Logistic regression, balanced, C=1.0 | 21.56% | 20.12% |
| Logistic regression, balanced, C=10.0 | 20.87% | 19.14% |
| Decision tree, unweighted | 21.01% | 19.96% |
| Decision tree, balanced | 21.26% | 19.81% |
| Random forest, unweighted | 21.11% | 19.75% |
| Random forest, balanced | 19.33% | 18.82% |
| **XGBoost, unweighted — selected** | **23.48%** | **21.32%** |
| XGBoost, weighted | 22.94% | 21.29% |

## Interpretation and reporting boundary

The selected XGBoost model's test AP was 21.32%, compared with 11.39% for the dummy baseline. Its test AP was 2.16 percentage points below its validation AP. The best logistic regression's test AP was 21.16%, only 0.16 percentage points below XGBoost. This single test split does not establish a decisive advantage for the more complex model.

The test results are now final reporting evidence, not a source for further tuning or choosing a different winner. No recall, precision, or follow-up workload is claimed for an operational threshold, because none was selected. These results describe held-out patients in the historical study dataset; they are not evidence of performance at a present-day London hospital.

**Source of the test AP values:** the Colab results table supplied by the project owner on 29 September 2026. This file freezes the reported comparison; it is not a serialized fitted-model artifact.
