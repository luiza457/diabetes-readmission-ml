"""Reconstructed, fixed-model readmission pipeline; see REPRODUCIBILITY.md."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ["train.py", "reproducibility_config.json", "requirements-reproducible.txt",
                "REPRODUCIBILITY.md", "01_data_cleaning.ipynb",
                "01_data_cleaning_practice.ipynb", "02_readmission_cleaning_and_insights.ipynb",
                "MODEL_RESULTS_FROZEN_2026-09-29.md"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def configure_process(seed):
    # Python's hash seed is read at process startup; re-exec before numerical imports.
    desired = {"PYTHONHASHSEED": str(seed), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    if any(os.environ.get(key) != value for key, value in desired.items()):
        environment = dict(os.environ, **desired)
        # subprocess works with Windows venv launcher executables as well as POSIX.
        completed = subprocess.run([sys.executable, *sys.argv], env=environment)
        raise SystemExit(completed.returncode)
    random.seed(seed)
    return desired


def validate_config(config):
    if config["schema_version"] != 1 or not 0 <= config["seed"] < 2**32:
        raise ValueError("Unsupported schema or seed")
    split = config["split"]
    if split["method"] != "patient_permutation_nested" or not all(
        0 < split[key] < 1 for key in ("test_fraction", "validation_fraction_of_remaining")
    ):
        raise ValueError("Unsupported split configuration")
    prep = config["preprocessing"]
    if (prep["numeric_imputation"] != "training_median"
            or prep["categorical_encoding"] != "training_one_hot_unknown_ignore"
            or prep["matrix"] != "csr_float32_explicit_numeric_zeros"):
        raise ValueError("Unsupported preprocessing option")
    required_exclusions = {"encounter_id", "patient_nbr", "readmitted", "weight", "examide", "citoglipton"}
    if not required_exclusions.issubset(prep["excluded_features"]):
        raise ValueError("Identifiers, target and excluded features must remain excluded")
    model = config["model"]
    if model["device"] != "cpu" or model["n_jobs"] != 1 or model["tree_method"] != "hist":
        raise ValueError("This reproducibility contract requires single-thread CPU hist")
    if "random_state" in model or "seed" in model:
        raise ValueError("Use the top-level seed only")


def verify_dependencies():
    for line in (ROOT / "requirements-reproducible.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, version = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"Dependency mismatch: {name} {actual}, required {version}")


def load_cohort(config):
    import pandas as pd
    data = config["data"]
    paths = {"csv": ROOT / data["csv"], "mapping": ROOT / data["mapping"]}
    checksums = {key: sha256(path) for key, path in paths.items()}
    if checksums["csv"] != data["sha256"] or checksums["mapping"] != data["mapping_sha256"]:
        raise ValueError("Source data/mapping checksum mismatch; original inputs have changed")
    raw = pd.read_csv(paths["csv"], keep_default_na=False, low_memory=False)
    if list(raw.shape) != data["expected_shape"] or not raw.encounter_id.is_unique:
        raise ValueError("Unexpected source shape or duplicate encounter IDs")
    if set(raw.readmitted.unique()) != {"<30", ">30", "NO"} or data["positive_label"] != "<30":
        raise ValueError("Unexpected target labels or changed prediction target")
    gender_ok = raw.gender.isin(data["valid_genders"])
    expired = raw.discharge_disposition_id.isin(data["expired_discharge_codes"])
    cohort = raw.loc[gender_ok & ~expired].copy().reset_index(drop=True)
    if len(cohort) != data["expected_cohort_rows"]:
        raise ValueError("Unexpected cohort size")
    cohort["target"] = cohort.readmitted.eq(data["positive_label"]).astype("int8")
    audit = {"raw_rows": len(raw), "invalid_gender_rows": int((~gender_ok).sum()),
             "expired_rows": int(expired.sum()), "eligible_rows": len(cohort),
             "unique_patients": int(cohort.patient_nbr.nunique()),
             "positive_rows": int(cohort.target.sum())}
    return cohort, checksums, audit


def validate_manifest(manifest, cohort):
    import pandas as pd
    expected = cohort[["encounter_id", "patient_nbr", "target"]].sort_values("encounter_id").reset_index(drop=True)
    actual = manifest.sort_values("encounter_id").reset_index(drop=True)
    if set(actual.columns) != {"encounter_id", "patient_nbr", "target", "split"}:
        raise ValueError("Invalid manifest columns")
    if not actual.encounter_id.is_unique or not set(actual.split.unique()) == {"train", "validation", "test"}:
        raise ValueError("Duplicate encounter or invalid/incomplete split labels")
    try:
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Manifest does not exactly match eligible encounters, patients and targets") from error
    if actual.groupby("patient_nbr").split.nunique().max() != 1:
        raise ValueError("Patient leakage: split membership overlaps")
    return actual


def get_manifest(config, cohort, checksums):
    import numpy as np
    import pandas as pd
    path = ROOT / config["split"]["manifest"]
    metadata_path = path.with_suffix(".meta.json")
    identity = {"config_sha256": canonical_hash(config), "data_sha256": checksums,
                "algorithm": "patient_permutation_nested_v1"}
    if path.exists() != metadata_path.exists():
        raise ValueError("Split manifest and metadata must both exist; restore the complete saved pair")
    if path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity or metadata.get("manifest_sha256") != sha256(path):
            raise ValueError("Saved split identity/checksum differs; use a new manifest path for a deliberate new study")
        manifest = pd.read_csv(path)
    else:
        patients = np.random.default_rng(config["seed"]).permutation(cohort.patient_nbr.drop_duplicates().to_numpy())
        outer = int((1 - config["split"]["test_fraction"]) * len(patients))
        inner = int((1 - config["split"]["validation_fraction_of_remaining"]) * outer)
        patient_split = dict.fromkeys(patients[:inner], "train")
        patient_split.update(dict.fromkeys(patients[inner:outer], "validation"))
        patient_split.update(dict.fromkeys(patients[outer:], "test"))
        manifest = cohort[["encounter_id", "patient_nbr", "target"]].copy()
        manifest["split"] = cohort.patient_nbr.map(patient_split)
        manifest = validate_manifest(manifest, cohort)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create prevents silently replacing historical membership.
        with path.open("x", encoding="utf-8", newline="") as stream:
            manifest.to_csv(stream, index=False, lineterminator="\n")
        write_json(metadata_path, {"identity": identity, "manifest_sha256": sha256(path)})
    manifest = validate_manifest(manifest, cohort)
    return manifest, path, metadata_path


class TrainingPreprocessor:
    """All learned medians/categories come from fit's training rows only."""

    def __init__(self, options):
        self.options = options

    def _features(self, frame):
        import numpy as np
        features = frame.drop(columns=self.options["excluded_features"] + ["target"]).copy()
        features = features.replace(self.options["missing_marker"], np.nan)
        return features

    def fit(self, train):
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder
        features = self._features(train)
        self.numeric_columns = self.options["numeric_columns"]
        self.categorical_columns = [c for c in features if c not in self.numeric_columns]
        numeric = features[self.numeric_columns].apply(pd.to_numeric, errors="raise")
        self.medians = numeric.median()
        if self.medians.isna().any():
            raise ValueError("All-missing numeric training feature")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype="float32")
        self.encoder.fit(features[self.categorical_columns].fillna(self.options["categorical_fill"]).astype(str))
        return self

    def transform(self, frame):
        import numpy as np
        import pandas as pd
        from scipy import sparse
        features = self._features(frame)
        numeric = features[self.numeric_columns].apply(pd.to_numeric, errors="raise").fillna(self.medians).to_numpy(dtype=np.float32)
        # Store numeric zero values explicitly: XGBoost treats absent CSR entries as missing.
        n_rows, n_cols = numeric.shape
        numeric_csr = sparse.csr_matrix((numeric.ravel(), np.tile(np.arange(n_cols), n_rows),
                                        np.arange(n_rows + 1) * n_cols), shape=numeric.shape)
        categorical = self.encoder.transform(features[self.categorical_columns].fillna(self.options["categorical_fill"]).astype(str))
        return sparse.hstack([numeric_csr, categorical], format="csr", dtype=np.float32)

    def audit(self):
        return {"numeric_columns": self.numeric_columns, "categorical_columns": self.categorical_columns,
                "training_medians": {key: float(value) for key, value in self.medians.items()},
                "training_categories": {key: values.tolist() for key, values in zip(self.categorical_columns, self.encoder.categories_)},
                "fit_partition": "train", "sparse_semantics": "numeric zeros explicit; absent one-hot entries are XGBoost missing branches"}


def git_details():
    def run(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    return {"head": run("rev-parse", "--verify", "HEAD"), "status": run("status", "--porcelain=v1"),
            "diff": run("diff", "--", "train.py", "reproducibility_config.json", "requirements-reproducible.txt", "REPRODUCIBILITY.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "reproducibility_config.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    controls = configure_process(config["seed"])
    verify_dependencies()
    import joblib
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from threadpoolctl import threadpool_info, threadpool_limits
    from xgboost import XGBClassifier
    np.random.seed(config["seed"])
    started = time.perf_counter()
    cohort, checksums, cohort_audit = load_cohort(config)
    manifest, manifest_path, manifest_meta_path = get_manifest(config, cohort, checksums)
    # Canonical encounter order removes dependence on incidental frame ordering in fitting.
    cohort = cohort.sort_values("encounter_id").reset_index(drop=True)
    cohort = cohort.merge(manifest[["encounter_id", "split"]], on="encounter_id", validate="one_to_one", sort=False)
    frames = {name: cohort.loc[cohort.split.eq(name)].drop(columns="split").copy()
              for name in ("train", "validation", "test")}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    shutil.copy2(manifest_path, output / "splits.csv")
    shutil.copy2(manifest_meta_path, output / "splits.meta.json")
    source_dir = output / "source"
    source_dir.mkdir()
    for filename in SOURCE_FILES:
        shutil.copy2(ROOT / filename, source_dir / filename)
    shutil.copy2(args.config, source_dir / "active_config.json")
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    (output / "dependencies.txt").write_text("".join(f"{name}=={version}\n" for name, version in sorted(packages.items(), key=lambda item: item[0].lower())), encoding="utf-8")
    provenance = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "argv": sys.argv, "cwd": str(Path.cwd()), "python_executable": sys.executable,
                  "python_version": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                  "processor": platform.processor(), "cpu_count": os.cpu_count(), "thread_environment": controls,
                  "packages": packages, "data_sha256": checksums, "config_sha256": canonical_hash(config),
                  "code_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES}, "git": git_details(),
                  "cohort": cohort_audit, "split_sha256": sha256(manifest_path)}
    write_json(output / "provenance.json", provenance)
    with threadpool_limits(limits=1):
        preprocessor = TrainingPreprocessor(config["preprocessing"]).fit(frames["train"])
        train_matrix = preprocessor.transform(frames["train"])
        model = XGBClassifier(**config["model"], random_state=config["seed"])
        model.fit(train_matrix, frames["train"].target.to_numpy())
        model.save_model(output / "model.ubj")
        model_parameters = model.get_params()
        model_parameters["missing"] = "NaN"  # JSON spelling for XGBoost's IEEE NaN marker.
        write_json(output / "model_parameters.json", model_parameters)
        (output / "booster_config.json").write_text(model.get_booster().save_config() + "\n", encoding="utf-8")
        joblib.dump(preprocessor, output / "preprocessor.joblib")
        prep_audit = preprocessor.audit()
        prep_audit["train_encounter_ids_sha256"] = canonical_hash(frames["train"].encounter_id.tolist())
        prep_audit["transformed_feature_count"] = train_matrix.shape[1]
        write_json(output / "preprocessing.json", prep_audit)
        metrics = {}
        for name, frame in frames.items():
            probabilities = model.predict_proba(preprocessor.transform(frame))[:, 1]
            predictions = frame[["encounter_id", "patient_nbr", "target"]].copy()
            predictions["split"] = name
            # Promote binary32 values to binary64 before 17-digit serialization, preserving exact values.
            predictions["probability_readmitted_30d"] = probabilities.astype(np.float64)
            predictions.to_csv(output / f"predictions_{name}.csv", index=False, float_format="%.17g", lineterminator="\n")
            target = frame.target.to_numpy()
            metrics[name] = {"encounters": len(frame), "patients": int(frame.patient_nbr.nunique()),
                             "positive_rows": int(target.sum()), "prevalence": float(target.mean()),
                             "average_precision": float(average_precision_score(target, probabilities)),
                             "roc_auc": float(roc_auc_score(target, probabilities)),
                             "log_loss": float(log_loss(target, probabilities)),
                             "brier_score": float(brier_score_loss(target, probabilities))}
        write_json(output / "metrics.json", metrics)
        provenance["threadpools"] = threadpool_info()
    provenance.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started)
    provenance["artifacts_sha256"] = {str(path.relative_to(output)).replace("\\", "/"): sha256(path)
                                      for path in sorted(output.rglob("*")) if path.is_file() and path.name != "provenance.json"}
    write_json(output / "provenance.json", provenance)
    print(json.dumps({"output": str(output), "elapsed_seconds": provenance["elapsed_seconds"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    # Make the fitted transformer reloadable via import train in another process.
    sys.modules["train"] = sys.modules[__name__]
    TrainingPreprocessor.__module__ = "train"
    main()
