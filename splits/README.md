# Patient membership

Training creates `patient_split.csv` and `patient_split.meta.json` here using the configured seed and patient permutation. Both files must travel together for reuse. The pipeline checks their hash, configuration/data identity, completeness, labels and patient separation.

The CSV contains encounter and patient identifiers, target and split. It stays local and is excluded from Git. A fresh checkout regenerates it deterministically from the exact verified source inputs and configuration. Preserve both files locally when archiving an experiment; do not edit membership in place.
