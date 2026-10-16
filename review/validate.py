"""Independent evidence: invokes entry point, reconstructs expectations from raw data."""
import os
os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
import sys
import csv
import json
import hashlib
import subprocess
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
import train

REVIEW = ROOT / "review"
results = {"policy": json.loads((REVIEW / "comparison_policy.json").read_text(encoding="utf-8-sig")), "checks": [], "commands": []}
reuse_runs = "--reuse-runs" in sys.argv
if reuse_runs:
    results["commands"] = json.loads((REVIEW / "results.json").read_text())["commands"]
    results["audit_rerun_reason"] = "Correct reviewer dtype-only manifest comparisons and baseline-schema decoding; exact numerical acceptance unchanged. Source/training artifacts unchanged."

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def check(name, condition, details=None):
    results["checks"].append({"name": name, "status": "passed" if bool(condition) else "failed", "details": details})
    (REVIEW / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{'PASS' if condition else 'FAIL'} {name}", flush=True)

def expect_error(name, fn):
    try:
        fn()
    except (ValueError, RuntimeError) as error:
        check(name, True, str(error))
    else:
        check(name, False, "Invalid input was accepted")

config = load(ROOT / "reproducibility_config.json")
for filename, expected in [(config["data"]["csv"], config["data"]["sha256"]), (config["data"]["mapping"], config["data"]["mapping_sha256"])]:
    check(f"independent source SHA256 {filename}", digest(ROOT / filename) == expected, digest(ROOT / filename))

raw = pd.read_csv(ROOT / config["data"]["csv"], keep_default_na=False)
# Independently encode the decisions in original notebooks; no load_cohort helper.
eligible = raw.loc[raw.gender.isin(["Female", "Male"]) & ~raw.discharge_disposition_id.isin([11, 19, 20, 21])].copy()
eligible["target"] = eligible.readmitted.eq("<30").astype("int8")
check("original cohort decisions and labels", len(raw) == 101766 and len(eligible) == 100111 and eligible.patient_nbr.nunique() == 70436,
      {"raw_rows": len(raw), "invalid_genders": int((~raw.gender.isin(["Female", "Male"])).sum()), "expired_rows": int(raw.discharge_disposition_id.isin([11,19,20,21]).sum()), "eligible_rows": len(eligible), "patients": int(eligible.patient_nbr.nunique()), "positive_rows": int(eligible.target.sum())})

# Fresh processes deliberately begin with conflicting/unset controls. The entry point
# must bootstrap before numerical imports; same final environment/config for both.
for name in ("run_a", "run_b"):
    if reuse_runs:
        check(f"fresh process {name} completed in prior invocation", load(REVIEW/name/"provenance.json")["status"] == "completed")
        continue
    cmd = [sys.executable, "train.py", "--config", "reproducibility_config.json", "--output", f"review/{name}"]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "12345"
    environment["OMP_NUM_THREADS"] = "4"
    environment.pop("OPENBLAS_NUM_THREADS", None)
    completed = subprocess.run(cmd, cwd=ROOT, env=environment, capture_output=True, text=True)
    (REVIEW / f"{name}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (REVIEW / f"{name}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    results["commands"].append({"argv": cmd, "cwd": str(ROOT), "initial_environment_changes": {"PYTHONHASHSEED": "12345", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "unset"}, "returncode": completed.returncode})
    check(f"fresh process {name} completes", completed.returncode == 0, completed.stderr)
    if completed.returncode:
        raise SystemExit("Pipeline execution failed; consult reviewer logs")

a, b = REVIEW / "run_a", REVIEW / "run_b"
ma, mb = pd.read_csv(a / "splits.csv"), pd.read_csv(b / "splits.csv")
check("fresh-process split membership exact", ma.equals(mb), {"rows": len(ma), "differing_rows": int((ma != mb).any(axis=1).sum())})
expected = eligible[["encounter_id", "patient_nbr", "target"]].sort_values("encounter_id").reset_index(drop=True)
actual = ma.sort_values("encounter_id").reset_index(drop=True)
check("manifest complete cohort, stable IDs, patient IDs, target exact", actual.encounter_id.is_unique and np.array_equal(actual[expected.columns].to_numpy(), expected.to_numpy()))
members = {part: set(ma.loc[ma.split.eq(part), "patient_nbr"]) for part in ("train", "validation", "test")}
check("patient and sample partitions disjoint", all(members[x].isdisjoint(members[y]) for x, y in (("train","validation"),("train","test"),("validation","test"))), {part: {"patients": len(ids), "encounters": int(ma.split.eq(part).sum())} for part,ids in members.items()})
patients = np.random.default_rng(42).permutation(eligible.patient_nbr.drop_duplicates().to_numpy())
outer, inner = int(.8 * len(patients)), int(.8 * int(.8 * len(patients)))
check("split independently reconstructed from declared seed and cutoffs", members["train"] == set(patients[:inner]) and members["validation"] == set(patients[inner:outer]) and members["test"] == set(patients[outer:]))

metrics_a, metrics_b = load(a / "metrics.json"), load(b / "metrics.json")
check("fresh-process all metrics exact", metrics_a == metrics_b, {"max_absolute_difference": max(abs(metrics_a[s][k] - metrics_b[s][k]) for s in metrics_a for k in metrics_a[s]), "values": metrics_a})
prep = joblib.load(a / "preprocessor.joblib")
prep_audit = load(a / "preprocessing.json")
train_frame = eligible.loc[eligible.patient_nbr.isin(members["train"])].sort_values("encounter_id").reset_index(drop=True)
numeric = config["preprocessing"]["numeric_columns"]
independent_medians = train_frame[numeric].replace("?", np.nan).apply(pd.to_numeric).median()
check("learned numeric medians equal independent training-only medians", all(float(independent_medians[c]) == prep_audit["training_medians"][c] == float(prep.medians[c]) for c in numeric), independent_medians.to_dict())
features = train_frame.drop(columns=config["preprocessing"]["excluded_features"] + ["target"])
categorical = [c for c in features if c not in numeric]
categories = {c: sorted(features[c].replace("?", "Missing").astype(str).unique().tolist()) for c in categorical}
check("learned vocabulary equals independent training-only categories", categories == prep_audit["training_categories"] and categorical == prep.categorical_columns)
heldout_only = {c: sorted(set(eligible.loc[~eligible.patient_nbr.isin(members["train"]), c].replace("?", "Missing").astype(str)) - set(categories[c])) for c in categorical}
check("natural heldout-only categories are absent from fitted vocabulary", all(not (set(values) & set(prep_audit["training_categories"][c])) for c, values in heldout_only.items()) and any(heldout_only.values()), {c: v for c,v in heldout_only.items() if v})
check("identifiers, target, weight and constants excluded; coded fields categorical", features.shape[1] == 44 and all(c in categorical for c in ("admission_type_id","admission_source_id","discharge_disposition_id")) and not (set(config["preprocessing"]["excluded_features"]) & set(features.columns)))
check("literal lab None remains categorical", "None" in categories["max_glu_serum"] and "None" in categories["A1Cresult"])
before = deepcopy(prep.audit())
probe = train_frame.iloc[:2].copy()
for c in numeric:
    probe[c] = np.nan
probe["gender"] = "REVIEWER_UNSEEN_HELDOUT_GENDER"
matrix = prep.transform(probe)
gender_start = len(numeric) + sum(len(categories[c]) for c in categorical[:categorical.index("gender")])
check("heldout missing numerics use train medians and unknown category creates no feature", np.array_equal(matrix[:, :len(numeric)].toarray(), np.tile(independent_medians.to_numpy(dtype=np.float32), (2,1))) and matrix[:, gender_start:gender_start+len(categories["gender"])].nnz == 0)
zero_probe = train_frame.iloc[:1].copy()
for c in numeric:
    zero_probe[c] = 0
zeros = prep.transform(zero_probe)[:, :len(numeric)]
check("numeric CSR zeros explicitly stored", zeros.nnz == len(numeric) and np.all(zeros.data == 0), {"stored_numeric_entries": zeros.nnz, "numeric_columns": len(numeric)})
check("transforming heldout probes does not change learned state", prep.audit() == before)

with threadpool_limits(limits=1):
    model = XGBClassifier()
    model.load_model(a / "model.ubj")
    for part in ("train", "validation", "test"):
        pa = pd.read_csv(a / f"predictions_{part}.csv", float_precision="round_trip").sort_values("encounter_id").reset_index(drop=True)
        pb = pd.read_csv(b / f"predictions_{part}.csv", float_precision="round_trip").sort_values("encounter_id").reset_index(drop=True)
        probability = "probability_readmitted_30d"
        delta = np.abs(pa[probability].to_numpy() - pb[probability].to_numpy())
        check(f"fresh-process {part} predictions exact by stable sample ID", pa[["encounter_id","patient_nbr","target","split"]].equals(pb[["encounter_id","patient_nbr","target","split"]]) and np.isfinite(delta).all() and np.all(delta == 0), {"rows": len(pa), "differing_predictions": int((delta != 0).sum()), "max_absolute_difference": float(delta.max()), "atol":0,"rtol":0})
        frame = eligible.loc[eligible.patient_nbr.isin(members[part])].sort_values("encounter_id").reset_index(drop=True)
        check(f"{part} prediction membership and labels exact independent cohort", np.array_equal(pa[["encounter_id","patient_nbr","target"]].to_numpy(), frame[["encounter_id","patient_nbr","target"]].to_numpy()) and pa.split.eq(part).all())
        predicted = model.predict_proba(prep.transform(frame))[:, 1]
        check(f"reloaded fitted model and preprocessor reproduce {part} probabilities exactly", np.array_equal(predicted.astype(np.float64), pa[probability].to_numpy()), {"native_dtype": str(predicted.dtype), "max_absolute_difference": float(np.max(np.abs(predicted.astype(np.float64) - pa[probability].to_numpy())))})
        p = pa[probability].to_numpy().astype(predicted.dtype)
        y = pa.target.to_numpy()
        recomputed = {"encounters": len(pa), "patients": pa.patient_nbr.nunique(), "positive_rows": int(y.sum()), "prevalence": float(y.mean()), "average_precision": float(average_precision_score(y,p)), "roc_auc": float(roc_auc_score(y,p)), "log_loss": float(log_loss(y,p)), "brier_score": float(brier_score_loss(y,p))}
        diffs = {k: abs(float(v)-metrics_a[part][k]) for k,v in recomputed.items()}
        check(f"{part} independently recomputed metrics from saved predictions", max(diffs.values()) <= 1e-15, {"max_absolute_difference": max(diffs.values()), "differences": diffs, "atol":1e-15,"rtol":0,"probabilities_cast_to_native_dtype":str(predicted.dtype)})

# Exercise deterministic generation independent of the persistent original pair.
generated = []
for name in ("generated_a", "generated_b"):
    c = deepcopy(config)
    c["split"]["manifest"] = f"review/{name}.csv"
    manifest, path, meta = train.get_manifest(c, eligible, {"csv":digest(ROOT / "diabetic_data.csv"), "mapping":digest(ROOT / "IDS_mapping.csv")})
    generated.append(manifest)
    old = (digest(path), digest(meta))
    reused, _, _ = train.get_manifest(c, eligible, {"csv":digest(ROOT / "diabetic_data.csv"), "mapping":digest(ROOT / "IDS_mapping.csv")})
    check(f"{name} saved manifest reused unchanged", np.array_equal(manifest.to_numpy(),reused.to_numpy()) and old == (digest(path), digest(meta)))
check("independently generated split pairs exact and match training", np.array_equal(generated[0].to_numpy(),generated[1].to_numpy()) and np.array_equal(generated[0].to_numpy(),ma.to_numpy()))

for case in ("missing", "duplicate", "wrong_encounter", "wrong_patient", "wrong_target", "unknown_split", "patient_overlap"):
    bad = ma.copy()
    if case == "missing": bad = bad.iloc[1:]
    elif case == "duplicate": bad = pd.concat([bad,bad.iloc[:1]],ignore_index=True)
    elif case == "wrong_encounter": bad.loc[0,"encounter_id"] += 1
    elif case == "wrong_patient": bad.loc[0,"patient_nbr"] += 1
    elif case == "wrong_target": bad.loc[0,"target"] = 1-bad.loc[0,"target"]
    elif case == "unknown_split": bad.loc[0,"split"] = "other"
    elif case == "patient_overlap":
        repeated_patient = bad.patient_nbr.value_counts().loc[lambda x:x>1].index[0]
        index = bad.index[bad.patient_nbr.eq(repeated_patient)][0]
        bad.loc[index,"split"] = "test" if bad.loc[index,"split"] != "test" else "train"
    expect_error(f"manifest rejects {case}", lambda bad=bad:train.validate_manifest(bad,eligible))

checksums = {"csv":digest(ROOT / "diabetic_data.csv"), "mapping":digest(ROOT / "IDS_mapping.csv")}
c = deepcopy(config)
c["split"]["manifest"] = "review/generated_a.csv"
c["seed"] += 1
expect_error("saved manifest rejects changed configuration", lambda:train.get_manifest(c,eligible,checksums))
corrupt_config = deepcopy(config)
corrupt_config["split"]["manifest"] = "review/corrupt.csv"
(REVIEW / "corrupt.csv").write_bytes((REVIEW / "generated_a.csv").read_bytes() + b"\n")
corrupt_metadata = load(REVIEW / "generated_a.meta.json")
corrupt_metadata["identity"]["config_sha256"] = hashlib.sha256(json.dumps(corrupt_config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(REVIEW / "corrupt.meta.json").write_text(json.dumps(corrupt_metadata),encoding="utf-8")
expect_error("saved manifest rejects altered checksum with correct identity", lambda:train.get_manifest(corrupt_config,eligible,checksums))
partial_config = deepcopy(config)
partial_config["split"]["manifest"] = "review/partial.csv"
(REVIEW / "partial.csv").write_bytes((REVIEW / "generated_a.csv").read_bytes())
expect_error("saved manifest rejects missing sidecar", lambda:train.get_manifest(partial_config,eligible,checksums))
bad_data_config = deepcopy(config)
bad_data_config["data"]["sha256"] = "0"*64
expect_error("training rejects data checksum mismatch", lambda:train.load_cohort(bad_data_config))

for folder in (a,b):
    prov = load(folder / "provenance.json")
    check(f"{folder.name} recorded final process controls and native thread pools", prov["status"] == "completed" and prov["thread_environment"]["PYTHONHASHSEED"] == "42" and all(v=="1" for k,v in prov["thread_environment"].items() if k != "PYTHONHASHSEED") and all(pool["num_threads"] == 1 for pool in prov["threadpools"]), {"python_version":prov["python_version"], "controls":prov["thread_environment"], "threadpools":prov["threadpools"]})
    check(f"{folder.name} input/config/source/artifact provenance independently hashed", prov["data_sha256"] == checksums and all(digest(ROOT/f)==h and digest(folder/"source"/f)==h for f,h in prov["code_sha256"].items()) and all(digest(folder/f)==h for f,h in prov["artifacts_sha256"].items()) and prov["config_sha256"] == train.canonical_hash(load(folder/"config.json")) and prov["split_sha256"] == digest(folder/"splits.csv"))
    pins = [line.split("==") for line in (ROOT/"requirements-reproducible.txt").read_text().splitlines() if line]
    check(f"{folder.name} all pinned dependencies recorded at exact versions", all(prov["packages"].get(package)==version for package,version in pins), {"packages":prov["packages"], "git":prov["git"]})
check("same code, data, config and dependency versions across two fresh processes", all(load(a/"provenance.json")[key] == load(b/"provenance.json")[key] for key in ("data_sha256","config_sha256","code_sha256","packages","python_version","platform","machine","processor","thread_environment")))

baseline = load(ROOT / "tmp/reproducibility_source_baseline.json")
results["original_baseline_schema"] = baseline
# Baseline supplied by root is checked generically because it predates this script.
def baseline_pairs(value):
    if isinstance(value, dict):
        if "file" in value and "sha256" in value:
            yield value["file"],value["sha256"]
            return
        for key,item in value.items():
            if isinstance(item,str) and len(item)==64 and (ROOT/key).is_file():
                yield key,item
            elif isinstance(item,dict) and "sha256" in item:
                path = item.get("path",key)
                if (ROOT/path).is_file(): yield path,item["sha256"]
            else: yield from baseline_pairs(item)
    elif isinstance(value,list):
        for item in value: yield from baseline_pairs(item)
originals = list(baseline_pairs(baseline))
check("original CSVs/notebooks/frozen report unchanged from prechange baseline", len(originals)==6 and all(digest(ROOT/path)==hash_ for path,hash_ in originals), {"matched_original_files": len(originals)})
results["limitations"] = ["No historical Colab code/model/splits available: historical AP and membership not verified.", "Exact evidence applies to this Windows Python 3.12.14 pinned single-thread CPU environment only; other platforms/CPUs/versions/GPU not run.", "No clinical/generalization or operational threshold claim."]
(REVIEW / "results.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
failed = [c["name"] for c in results["checks"] if c["status"] == "failed"]
print(json.dumps({"passed":len(results["checks"])-len(failed),"failed":failed},indent=2),flush=True)
sys.exit(bool(failed))
