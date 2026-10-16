# 4. Results and interpretation

## Historical model selection: frozen 29 September 2026

| Model | Validation AP | Test AP |
|---|---:|---:|
| Dummy baseline | 11.48% | 11.39% |
| Best validation logistic regression, unweighted C=0.1 | 22.36% | 21.16% |
| Selected unweighted XGBoost | **23.48%** | **21.32%** |
| Weighted XGBoost | 22.94% | 21.29% |

The complete 13-candidate table is in [the frozen report](../MODEL_RESULTS_FROZEN_2026-09-29.md). XGBoost exceeded the dummy test AP by 9.93 percentage points, but exceeded the strongest reported logistic test AP by only 0.16 points. A single split without uncertainty estimates does not establish a decisive advantage over the simpler model.

These values were supplied by the project owner from Colab. The report preserves the comparison; the missing historical fitted model, source and membership prevent independent reproduction of those numbers.

## Reconstructed pipeline: reviewed 14 October 2026

| Split | Encounters | AP | ROC AUC | Log loss | Brier score |
|---|---:|---:|---:|---:|---:|
| Train | 63,974 | 27.15% | 0.70258 | 0.32639 | 0.09401 |
| Validation | 16,123 | 22.35% | 0.67304 | 0.33203 | 0.09534 |
| Test | 20,014 | **21.60%** | **0.66281** | **0.33189** | **0.09500** |

Full precision values are in [aggregate metrics](../results/reconstructed_metrics.json). Different reconstructed patient membership means 21.60% is not a corrected or improved version of the historical 21.32%.

Two fresh processes produced exactly the same **100,111 predictions**, split membership and saved metrics. Maximum numerical difference was zero, with `atol=0`, `rtol=0`. Restored artifacts reproduced the probabilities. See [review evidence](../review/REPORT.md) for the complete tested scope and reviewer-harness corrections.

## What remains unresolved

The evidence demonstrates repeatability in the pinned Windows CPU environment. It does not establish performance across hospitals, time periods, demographic groups, CPU types, platforms, dependency upgrades or GPU runs.

No operational decision threshold was chosen. Therefore no specific precision, recall, patient follow-up workload, hospital benefit or safety outcome is claimed. Earlier exercises mentioning 3,500 follow-ups were hypothetical.

Before a deployment study, assess feature availability at discharge, calibration, uncertainty, subgroup performance and external/temporal validation. A serving application and operational logging are future work. These are research/engineering next steps, not completed results.

Next: [LLM harnessing](../llm_harness/README.md).
