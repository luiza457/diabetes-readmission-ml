# Reproducing the reconstructed readmission pipeline

This entry point reconstructs a fixed model from the local cleaning notebooks and
`MODEL_RESULTS_FROZEN_2026-09-29.md`, with the owner's approval. The historical
Colab training code, fitted model and three-way membership are unavailable. This
pipeline does **not** reproduce or replace the frozen historical AP results.
Original CSVs, notebooks and the frozen report remain untouched.

## Setup and execution (PowerShell, project directory)

The validated runtime is 64-bit Windows Python **3.12.14**. Use that Python patch
version and the pinned wheel versions for the demonstrated reproducibility scope.
The setup used the bundled interpreter without modifying its installed packages:

```powershell
& 'C:\Users\ldulg\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproducible.txt
.\.venv\Scripts\python.exe train.py --config reproducibility_config.json --output runs/run_1
.\.venv\Scripts\python.exe train.py --config reproducibility_config.json --output runs/run_2
```

Use a new output directory for every invocation; existing directories are rejected.
Paths in the configuration resolve relative to `train.py`, so the source dataset
is always explicit. The first invocation creates `splits/patient_split.csv` and
its `.meta.json` sidecar. Keep and version **both** files with the source and
configuration. Subsequent runs reuse and validate their checksum, exact
configuration/data identity, eligible encounter IDs, patient IDs, labels and
patient disjointness. A changed configuration requires a deliberately new
manifest path, including when only the model settings change. Do not edit a
saved manifest in place. No historical membership is inferred from row counts.

## Reconstruction decisions

* Load the complete 101,766-row CSV with `keep_default_na=False`: literal `None`
  remains a category. Only `?` is missing. SHA-256 checks cover the CSV and mapping.
* Combine the latest notebook's removal of three invalid gender encounters with
  the earlier notebook's expired discharge codes `[11,19,20,21]` (1,652 rows).
  The resulting 100,111 encounters retain repeat patient encounters and numeric
  extremes. The original binary prediction target remains `readmitted == '<30'`.
* Exclude target, encounter/patient identifiers, highly missing weight and the two
  constant medication columns `examide` and `citoglipton`, following the notebooks.
  Use the eight explicit clinical count features as numeric. Everything else,
  including admission/discharge codes, age bands and diagnoses, is categorical.
* Preserve the earlier notebook's `default_rng(42)` patient permutation and nested
  floor cutoffs: the last 20% of patients become test; the last 20% of the remaining
  patients become validation. Fractions describe patients, not encounters; there
  is no target stratification. All encounters of a patient share membership.
  Reconstructed counts are train 63,974 encounters / 45,078 patients, validation
  16,123 / 11,270 and test 20,014 / 14,088. The historical report's encounter
  counts (63,925 / 16,175 / 20,011) describe a different unavailable split.
* Canonically sort encounters by `encounter_id` before fitting and writing outputs.
  Training-only medians fill numeric missing values; constant `Missing` fills
  missing categories. A one-hot encoder learns its vocabulary only on training
  data, with unknown validation/test categories ignored. No scaling, class weights,
  early stopping, tuning or operational threshold is added.
* Preserve sparse one-hot input. XGBoost treats absent CSR entries as missing, so
  inactive one-hot values follow missing branches consistently in all partitions.
  Numeric clinical zeros are explicitly stored in CSR and remain numeric zeros.
  This sparse convention is explicit because unavailable historical training code
  cannot establish whether its final XGBoost input was sparse or dense. Changing
  that representation would change the model and requires a separate study.
* Use the frozen unweighted XGBoost settings: 150 rounds, depth 3, learning rate
  0.1, row/feature subsampling 0.8, binary logistic objective and CPU histogram
  trees. The configuration explicitly fixes remaining execution settings.

## Randomness, artifacts and acceptance criteria

The seed controls the patient permutation, Python random, NumPy global RNG and
XGBoost subsampling. `PYTHONHASHSEED` and thread environment variables are set by
restarting Python before numerical imports when necessary. XGBoost and native
thread pools run with one thread. No GPU or multiprocessing path is used.

Each run saves the full configuration, split pair, training preprocessing audit,
fitted preprocessor and XGBoost model, metrics and train/validation/test prediction
CSVs. Predictions include stable `encounter_id`, `patient_nbr`, original binary
target and split, with binary32 probabilities promoted to binary64 and written
with 17 significant digits for lossless round-trip comparison. Metrics are AP
(primary), ROC AUC, log loss and Brier score; no threshold-dependent clinical
decision is claimed. `preprocessor.joblib` should be loaded after `import train`;
use the same pinned environment and trusted project artifacts.
Metrics consume the original float32 probabilities: cast round-trip CSV values
back to `float32` when recomputing, because dtype affects log-loss clipping.
`model_parameters.json` and `booster_config.json` record resolved default model
hyperparameters and native booster settings in addition to the user configuration.

`provenance.json` records both input hashes, normalized config hash, source hashes,
source copies, artifact hashes, all installed package versions, Python executable
and version, OS/architecture/processor, thread controls and native library details,
command, timestamps, elapsed time, cohort counts and Git status. Git has no saved
commit in this project at implementation time, and original files are untracked;
the run records that fact instead of inventing a code revision. Source snapshots
and their SHA-256 values identify the executed code. `dependencies.txt` records
the actual environment including installer packages.

Before evaluating evidence, the comparison policy is **exact numeric equality**
(`atol=0`, `rtol=0`) for predictions aligned by sample ID and all saved metrics
across fresh processes in the same code/data/configuration/pinned CPU environment.
Splits must match exactly, include every eligible encounter once, and have no
patient overlap. Learned preprocessing must use training rows only. Review must
report pass/fail/unrun checks and investigate differences before any reproducibility
claim. Reading CSV probabilities requires a round-trip parser (pandas
`float_precision='round_trip'`) to preserve the saved numerical values.

This contract makes no exact cross-platform, cross-CPU, cross-version or GPU
claim. Floating point reductions, native builds and libraries can change results
outside this scope. Any broader experiment needs an explicit tolerance chosen
**before** inspecting its comparisons. The reviewer separately permits at most
1e-15 absolute difference for independent metric recomputation from round-trip
CSV values; that is distinct from the exact fresh-process comparison requirement.
The independent evidence lives in `review/` when available.

Implementation references: [XGBoost sparse versus dense semantics](https://xgboost.readthedocs.io/en/release_3.0/faq.html),
[XGBoost parameters](https://xgboost.readthedocs.io/en/release_3.0/parameter.html),
[scikit-learn OneHotEncoder](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
