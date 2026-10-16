# Independent reproducibility review — 14 October 2026

**Outcome: passed in the demonstrated environment.** Two independently launched fresh training processes used the same final code, data, configuration, saved split and pinned environment. All 100,111 predictions matched exactly by encounter ID; every saved metric matched exactly. Final independent review contains **52 passed checks, 0 failed checks** in `results.json`.

## Scope and comparison policy

The reviewer read the actual `train.py`, configuration, dependency pins and documentation, and inspected the original cleaning notebooks and frozen model settings. The implementation summary was not used as proof.

Before training, `comparison_policy.json` fixed exact split membership, patient/sample IDs and labels, and exact prediction/metric equality (`atol=0`, `rtol=0`). Predictions must be finite and matched by stable encounter ID and split. The same pinned single-thread CPU histogram XGBoost execution justifies this requirement. The separately declared independent metric-recomputation allowance was `atol=1e-15`, `rtol=0`; **actual recomputation differences were zero**. No tolerance was relaxed after seeing results.

## Executed commands

From `C:\Users\ldulg\Documents\ChatGPT\Healthcare Project`:

```powershell
.\.venv\Scripts\python.exe review/validate.py
.\.venv\Scripts\python.exe review/validate.py --reuse-runs
```

The first command independently launched these two new processes, captured their return codes (both 0), stdout and stderr, and retained all artifacts:

```powershell
.\.venv\Scripts\python.exe train.py --config reproducibility_config.json --output review/run_a
.\.venv\Scripts\python.exe train.py --config reproducibility_config.json --output review/run_b
```

Each was deliberately started with `PYTHONHASHSEED=12345`, `OMP_NUM_THREADS=4`, and unset `OPENBLAS_NUM_THREADS`. Both entry-point bootstraps successfully restarted before numerical imports, and recorded final seed 42, all configured thread controls 1 and every observed native thread pool at one thread. The environment is Windows, 64-bit Python 3.12.14, with every declared pinned dependency verified against recorded actual package versions.

The second reviewer command rechecked saved evidence and all affected audit checks after correcting reviewer-harness issues described below. It did not replace the original training evidence. Standard reproduction commands and setup are in `../REPRODUCIBILITY.md`; use new output directories because existing run directories are deliberately rejected.

## Numerical results

| Split | Encounters | Patients | AP | ROC AUC | Log loss | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| Train | 63,974 | 45,078 | 0.2714592349591216 | 0.7025789943718265 | 0.32639250662324293 | 0.09400788627968351 |
| Validation | 16,123 | 11,270 | 0.2234885173512859 | 0.673037055724253 | 0.3320303600341865 | 0.0953417253572077 |
| Test | 20,014 | 14,088 | 0.21601058584239669 | 0.6628129916982484 | 0.33189071948409815 | 0.0950023920048646 |

- Exactly equal full split membership in both fresh runs: **100,111 rows, 0 differences**.
- Exactly equal probabilities joined by stable encounter ID: **63,974 train + 16,123 validation + 20,014 test, 0 differing probabilities, maximum absolute difference 0**.
- Every saved metric/count/prevalence exactly equal: **maximum absolute difference 0**.
- Fresh-process reviewer reload of the fitted preprocessor and native model reproduced every probability exactly: **maximum difference 0**.
- Independent metric recomputation from lossless round-trip CSV values, cast to the loaded estimator's observed native float32 probability dtype, matched every saved metric: **maximum difference 0**. The dtype matters for scikit-learn log-loss clipping/reductions.

## Source and behavioural checks

Passed independently:

- Raw data and mapping SHA-256 checks; six original files unchanged from the prechange baseline (three notebooks, source CSV, mapping, frozen report).
- Independent cohort construction preserves original target `<30`, removes exactly three invalid gender encounters and 1,652 expired encounters, and retains **100,111 encounters / 70,436 patients / 11,357 positive targets**.
- Manifest includes every eligible encounter exactly once with the correct patient and target; no patient or sample overlap between partitions. The patient RNG permutation and nested floor cutoffs were independently reconstructed.
- Training-only medians and all 36 categorical vocabularies exactly equal independently computed values from training IDs. Natural heldout-only categories are absent from fitted vocabulary. Literal lab `None` stays a category; identifiers, target, weight and constant medication columns remain excluded, and coded admissions/discharge fields remain categorical.
- Synthetic heldout missing numeric values use saved training medians; unseen categorical values create no columns; transformation does not change learned state. All eight numeric zero entries are explicitly stored in CSR, so clinical zeros do not become implicit missing values to XGBoost.
- Two independently generated manifest pairs match the training membership exactly. Saved membership and sidecar checksums remain unchanged on reuse.
- Negative tests reject missing/duplicate rows, changed encounter ID, changed patient ID, changed target, unknown split, patient overlap, changed configuration identity, altered manifest checksum with otherwise correct identity, missing sidecar, and changed input checksum.
- Run provenance matches independently hashed inputs, configuration, source snapshots and all recorded artifact hashes; both fresh runs used the same code/data/configuration/Python/platform/package versions. Every dependency pin is recorded exactly.

The reviewer inspected the `fit` call site directly: only `frames['train']` reaches learned preprocessing/model fitting; validation and test only call `transform` and `predict_proba`. No early stopping, model selection or test-driven tuning was added.

## Findings and rechecks

No implementation defect requiring remediation remained after the actual-code review and execution. The implementation agent was informed of this result and asked to keep reviewed source unchanged while evidence was finalized.

The initial reviewer harness incorrectly classified two checks as failures: DataFrame `.equals` treated a generated target stored as int8 and the identical CSV target read as int64 as unequal; the original-hash checker did not initially decode the baseline's `file` field. These were **reviewer-harness issues**, with no differing memberships, labels, original file hashes or model outputs. The reviewer replaced the dtype-sensitive checks with exact elementwise membership-value comparisons, corrected baseline decoding and reran all affected checks. All 52 final checks passed. The source/configuration/training artifacts remained unchanged, and the declared comparison tolerances were unchanged.

Synthetic missing-value probes produced a pandas FutureWarning about future replacement downcasting. This did not change values or fitted state under pinned pandas 2.3.3; future pandas versions were not assessed.

## Not run and remaining limitations

Historical Colab source, fitted model and three-way membership are unavailable. The user approved reconstruction from local notebooks/frozen XGBoost settings. Historical split counts/AP were **not** reproduced or validated; the documented reconstructed split is different.

Cross-platform, different CPU, changed Python/package/native-library versions, GPU and multithread executions were not tested. Exact reproducibility evidence is limited to this same pinned environment and code/data/configuration. No broader tolerance was selected or broader claim made. The project has no saved Git commit, so code identity is supported by actual source snapshots/SHA-256 values and recorded Git state. No clinical deployment or generalization assessment was performed.

Evidence: `results.json`, `comparison_policy.json`, `validate.py`, `run_a.stdout.txt`, `run_b.stdout.txt`, both stderr logs, and complete `run_a/` and `run_b/` artifacts.
