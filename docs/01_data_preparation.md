# 1. Data preparation

## Prediction question and source

At discharge, predict whether an encounter has a recorded `readmitted == '<30'` outcome. `>30` and `NO` map to zero. `NO` describes recording in this dataset, not proof of no admission anywhere.

The local source contains 101,766 encounters, 50 columns and 71,518 patients. One row is an encounter; repeated patient identifiers represent valid repeat visits. The public dataset covers US hospitals during 1999–2008. See [source attribution](../README.md#data-attribution) and [acquisition script](../scripts/prepare_data.py).

## Audit before changing values

Verify both input hashes, full file size (19,159,383 bytes for the encounter CSV), shape, unique encounter IDs, target categories and repeated patients. The practice notebook records a partial Colab upload; file checks were added to catch it.

Load with `keep_default_na=False`. Only `?` is the missing marker; literal `None` in lab results stays a category. Check types before computing numeric ranges. The cleaning notebook explicitly parses length of stay and reports present values that cannot be converted.

## Reconstructed cohort and features

| Issue | Implemented decision |
|---|---|
| Three `Unknown/Invalid` gender encounters | Exclude those encounters |
| Discharge disposition codes 11, 19, 20, 21 | Exclude 1,652 expired encounters according to the project rule |
| Repeat patient encounters | Retain; keep every encounter for a patient in one split |
| Weight, approximately 96.9% missing | Exclude feature; retain encounters |
| Constant `examide`, `citoglipton` | Exclude features |
| Encounter/patient IDs and outcome | Exclude from predictors; preserve IDs for auditing |
| Eight numeric count/duration fields | Use training medians for missing values |
| Other predictors, including diagnosis and admission codes | Treat as categories; fill missing with `Missing` and learn one-hot vocabulary on train |
| Extreme counts | Flag and retain; unusual does not establish incorrect |

The reconstructed cohort has **100,111 encounters, 70,436 patients and 11,357 positive outcomes**. There are 44 predictors before one-hot encoding: eight numeric and 36 categorical.

The earlier cleaning notebook replaces invalid gender with missing, while the later exploration removes those rows and does not yet apply the expired-discharge rule. These learning stages do not constitute a single frozen pipeline. The reconstruction explicitly combines the documented decisions; its exact rules live in [configuration](../reproducibility_config.json) and [implementation](../train.py).

Next: [experiment design](02_experiment_design.md).
