# Reusable prompts

These are newly written templates reflecting the documented workflow. They are not a transcript or evidence that a particular model version received this exact wording.

## Data preparation

> Inspect the CSVs and notebooks before proposing rules. Verify full file size, shape, missing markers, categories, identifiers and target labels. Preserve original files. Distinguish repeated patients from duplicate encounters. Explain each exclusion and report resulting counts. Treat unexplained extreme values as investigation items, not automatic errors. Identify which decisions remain for the owner.

## Implementation

> Implement the fixed reconstructed model from the supplied configuration and frozen report. Do not claim to reproduce unavailable historical Colab code or membership. Keep all encounters for a patient in one split. Fit preprocessing on training rows only. Pin dependencies and control process-level randomness and native threads. Save configuration, hashes, source snapshots, split pair, preprocessing, fitted artifacts, metrics and predictions keyed by stable ID. Use new output directories and reject changed or incomplete split identities.

## Independent review

> Read the actual source and input artifacts rather than trusting the implementation summary. Fix the comparison policy before running. Launch two fresh processes in the same declared environment and compare exact membership, predictions by encounter ID and saved metrics. Independently reconstruct cohort and split expectations, inspect training-only preprocessing, reload artifacts and recompute metrics. Exercise invalid manifests and changed checksums. Report passed, failed and unrun checks, preserve execution evidence, investigate discrepancies and do not relax tolerance after seeing results.

## Result reporting

> Separate frozen owner-supplied historical metrics from the reconstructed run. Link each claim to an artifact. Explain AP without calling it accuracy. Do not invent a clinical threshold, deployment, patient benefit or present-day hospital performance. Report differences in split membership and missing historical artifacts. Keep future work distinct from completed work.

## Proposed deployment role — future work

> Once a serving application exists, package trusted preprocessing and model artifacts in a specified environment. Check startup, valid and invalid requests and logs. Compare application probabilities with offline pipeline predictions for the same inputs. Record commands and outcomes. Stop short of claiming predictive validity from operational smoke tests alone.
