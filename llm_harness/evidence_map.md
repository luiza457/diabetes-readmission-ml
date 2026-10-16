# Evidence map

| Claim | Inspectable evidence | Boundary |
|---|---|---|
| Numeric-type assumption was corrected | Parsing and invalid-value reporting in [01_data_cleaning.ipynb](../01_data_cleaning.ipynb) | Colab TypeError is described in project history; original traceback is not bundled |
| Incomplete upload was caught | File-size/shape checks and commentary in [practice notebook](../01_data_cleaning_practice.ipynb) | Not evidence of a current incomplete file |
| Learned preprocessing uses train only | `TrainingPreprocessor.fit`, call site in [train.py](../train.py), [review](../review/REPORT.md) | No general certification of future changes |
| Preprocessor can reload | Stable `train` module identity in [train.py](../train.py), reload checks in [validator](../review/validate.py) | Compatible trusted artifact/environment required |
| Process controls survive Windows launch | `configure_process` in [train.py](../train.py), bootstrap review in [report](../review/REPORT.md) | Other OS/builds were not audited |
| Independent review found matching predictions | [Report](../review/REPORT.md), [aggregate summary](../results/reproducibility_summary.json), [comparison policy](../review/comparison_policy.json) | Full per-encounter artifacts remain local |
| Two reviewer issues were corrected | Findings and rechecks in [report](../review/REPORT.md) | Dtype/schema corrections; tolerances did not change |
| Historical model comparison was frozen | [Frozen report](../MODEL_RESULTS_FROZEN_2026-09-29.md) | Owner-supplied Colab table; code/model/membership unavailable |
| Deployment agent/log verification | Proposed prompt in [templates](prompts.md) | Planned only; no deployment evidence |
