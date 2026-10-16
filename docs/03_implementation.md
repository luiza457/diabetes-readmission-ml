# 3. Implementation and verification

The reviewed [train.py](../train.py) is a command-line offline training pipeline. Follow [the quick start](../README.md#quick-start) to acquire inputs and run it. [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) retains the original detailed contract and environment-specific commands.

## Execution sequence

1. Validate the configuration, initialize seed/thread controls before numerical imports and verify package pins.
2. Verify the input files by SHA-256, construct the eligible cohort and encode `<30` as the positive label.
3. Generate or validate a persistent patient manifest and its metadata sidecar. Reject changed identities, incomplete pairs or invalid membership.
4. Sort encounters by stable ID and construct train, validation and test frames.
5. Learn numeric medians and categorical vocabulary on training rows only.
6. Build float32 CSR matrices and fit the fixed XGBoost estimator on train.
7. Save the preprocessor, model, resolved parameters, preprocessing audit, configuration, metrics, lossless prediction CSVs and provenance.

CSR numeric zeros are explicitly stored so that XGBoost sees numeric zero. Inactive one-hot entries remain absent and follow XGBoost's missing branches. This convention is deliberate and consistent across splits; switching to dense arrays changes the experiment.

## Artifacts

Each output directory contains `model.ubj`, `preprocessor.joblib`, `metrics.json`, `predictions_<split>.csv`, `splits.csv`, `splits.meta.json`, source copies and `provenance.json`. Provenance records hashes, versions, environment, seed/thread settings and Git state. Load trusted preprocessing artifacts only after `import train`, using compatible dependencies.

The reviewed source is kept at the repository root because source paths and byte hashes are part of the saved historical evidence. Published practice-notebook outputs/counters are cleared without changing its code cells; both original and publication hashes are documented in the integrity manifest. The original run records truthfully indicate that no Git commit existed then. A publication commit does not retrospectively change those records.

## Verification routes

The preserved [original review](../review/REPORT.md) reports 52 checks passed and zero failed on 14 October 2026, including two fresh processes, exact prediction/metric equality, train-only preprocessing, reload, source/artifact hashes and negative split tests. [The summary](../results/reproducibility_summary.json) is an aggregate export of that evidence.

The original [review validator](../review/validate.py) expects full local review artifacts, `review/results.json` and `tmp/reproducibility_source_baseline.json` for reuse; these machine-specific artifacts are intentionally excluded from publication. Its original commands remain part of the historical record.

For a fresh checkout, use [scripts/verify_pipeline.py](../scripts/verify_pipeline.py). It creates two new runs and an independent report under a new output directory, checks disjoint patient groups, exact predictions/metrics, artifact hashes, restored predictions and metric recomputation. This is a portable rerun route with its own check count, not a claim that the original 52-check audit ran on a different machine.

The fast [repository check](../scripts/check_repository.py) verifies original source hashes, Python/notebook code syntax, local documentation links and aggregate evidence consistency. It does not train a model.

Next: [results](04_results.md).
