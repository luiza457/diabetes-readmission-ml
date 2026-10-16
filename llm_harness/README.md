# LLM harnessing: development with verification

Here, **LLM harnessing means structuring AI-assisted development with explicit context, constraints, artifacts and verification**. The prediction model is XGBoost. No LLM classifies patient records, generates clinical advice or runs inside the training pipeline.

## Workflow used in this project

| Responsibility | Work and boundary |
|---|---|
| Project owner | Defines the question, challenges assumptions, supplies historical Colab results and approves reconstruction decisions |
| Implementation agent | Inspects actual files, implements the fixed pipeline and records execution artifacts |
| Independent review agent | Reads the implementation, reconstructs expectations from raw inputs, launches fresh processes and compares evidence |
| Final reconciliation | Implementation considers findings; verification corrections are rerun and limitations remain explicit |

The separate implementation/review roles were used for the reproducibility work. Their output is evidenced in the saved pipeline and [review report](../review/REPORT.md). A third deployment/log-checking role was discussed in the project history but was **not implemented**. There is no deployed application or production-monitoring result to report.

## Context and constraints

The agents worked from the local CSVs, cleaning notebooks, frozen model settings and the reproduction contract. Constraints included preserving original files; using the original target; splitting by patient; fitting preprocessing only on train; pinning dependencies; fixing seeds and threads; and declaring comparison tolerances before viewing numerical differences.

Source files, commands, hashes, model parameters, predictions and metrics provide evidence beyond an agent's prose summary. A review must distinguish implemented, executed, passed, failed and unrun work. Historical AP values remain owner-supplied evidence because historical source and membership are missing.

## A concrete correction to generated code

An earlier generated audit called `.min()` / `.max()` on `time_in_hospital` assuming numeric values. A Colab run raised a TypeError with mixed integers and strings. The local CSV loaded with integer dtype, so the exact cause of the Colab state was not established.

The corrected [cleaning notebook](../01_data_cleaning.ipynb) parses numeric values, reports entries that were present but could not be parsed, and computes ranges from valid values:

```python
stay_days = pd.to_numeric(raw['time_in_hospital'], errors='coerce')
invalid_stays = raw['time_in_hospital'].notna() & stay_days.isna()
print(raw.loc[invalid_stays, 'time_in_hospital'])
print(stay_days.min(), stay_days.max())
```

Conversion alone could hide data-quality problems. Reporting invalid values makes the assumption inspectable. The final training pipeline uses strict numeric conversion (`errors='raise'`) on verified source data; the exploratory correction does not imply silently accepting corrupted training inputs.

Other lessons were equally concrete: a partial file upload needed file-size/shape checks; the saved preprocessor needed a stable importable module identity; a Windows startup bootstrap needed a fresh subprocess handoff; and the reviewer had to correct dtype-sensitive equality and baseline-schema checks without relaxing numerical tolerances. The [evidence map](evidence_map.md) links these claims to available artifacts.

## Verification outcome and limits

The retained review reports **52 passed, zero failed**. It compared all 100,111 predictions, metrics and split membership across fresh processes, then checked fitted artifacts and preprocessing. This supports a bounded reproducibility claim; it does not certify all generated code or clinical performance.

[Prompt templates](prompts.md) capture a reusable way to repeat the approach. They are reconstructed templates based on the documented workflow, not verbatim original prompts. Private conversation transcripts and patient rows are not included.
