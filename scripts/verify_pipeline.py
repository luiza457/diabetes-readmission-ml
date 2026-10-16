"""Run two fresh fixed-model experiments and audit their saved evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New verification directory')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Current two-process execution; separate from historical 52-check review',
              'prediction_atol': 0, 'prediction_rtol': 0, 'checks': []}

    def check(name, passed):
        report['checks'].append({'name': name, 'status': 'passed' if bool(passed) else 'failed'})
        (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        if not passed:
            raise RuntimeError(name)

    config = read(ROOT / 'reproducibility_config.json')
    env = dict(os.environ, PYTHONHASHSEED='12345', OMP_NUM_THREADS='4')
    env.pop('OPENBLAS_NUM_THREADS', None)
    for name in ('run_a', 'run_b'):
        command = [sys.executable, str(ROOT / 'train.py'), '--config',
                   str(ROOT / 'reproducibility_config.json'), '--output', str(output / name)]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (output / f'{name}.stdout.txt').write_text(result.stdout, encoding='utf-8')
        (output / f'{name}.stderr.txt').write_text(result.stderr, encoding='utf-8')
        check(f'{name} fresh process completed', result.returncode == 0)

    # Numerical libraries are loaded only after the subprocesses have completed.
    os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                      'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
    from threadpoolctl import threadpool_limits
    from xgboost import XGBClassifier
    import train

    a, b = output / 'run_a', output / 'run_b'
    for folder in (a, b):
        provenance = read(folder / 'provenance.json')
        check(f'{folder.name} artifact checksums', all(
            hashlib.sha256((folder / path).read_bytes()).hexdigest() == digest
            for path, digest in provenance['artifacts_sha256'].items()))
        check(f'{folder.name} completed with required thread controls',
              provenance['status'] == 'completed'
              and provenance['thread_environment']['PYTHONHASHSEED'] == str(config['seed'])
              and all(v == '1' for k, v in provenance['thread_environment'].items() if k != 'PYTHONHASHSEED')
              and all(p['num_threads'] == 1 for p in provenance['threadpools']))
    ma, mb = pd.read_csv(a / 'splits.csv'), pd.read_csv(b / 'splits.csv')
    check('exact split membership', ma.equals(mb))
    raw = pd.read_csv(ROOT / config['data']['csv'], keep_default_na=False, low_memory=False)
    cohort = raw.loc[raw.gender.isin(['Female', 'Male']) &
                     ~raw.discharge_disposition_id.isin([11, 19, 20, 21])].copy()
    cohort['target'] = cohort.readmitted.eq('<30').astype('int8')
    expected = cohort[['encounter_id', 'patient_nbr', 'target']].sort_values('encounter_id')
    check('independent cohort membership and labels', len(cohort) == 100111 and
          ma.encounter_id.is_unique and np.array_equal(
              ma.sort_values('encounter_id')[expected.columns].to_numpy(), expected.to_numpy()))
    check('patient groups are disjoint', ma.groupby('patient_nbr').split.nunique().max() == 1)
    metrics = read(a / 'metrics.json')
    check('all saved metrics exactly equal', metrics == read(b / 'metrics.json'))
    prep = joblib.load(a / 'preprocessor.joblib')
    with threadpool_limits(limits=1):
        model = XGBClassifier()
        model.load_model(a / 'model.ubj')
        for part in ('train', 'validation', 'test'):
            pa = pd.read_csv(a / f'predictions_{part}.csv', float_precision='round_trip')
            pb = pd.read_csv(b / f'predictions_{part}.csv', float_precision='round_trip')
            pa, pb = [p.sort_values('encounter_id').reset_index(drop=True) for p in (pa, pb)]
            check(f'{part}: exact prediction equality', pa.equals(pb))
            frame = cohort.loc[cohort.encounter_id.isin(ma.loc[ma.split.eq(part), 'encounter_id'])]
            frame = frame.sort_values('encounter_id').reset_index(drop=True)
            check(f'{part}: prediction IDs and labels match cohort',
                  pa.split.eq(part).all() and np.array_equal(
                      pa[['encounter_id', 'patient_nbr', 'target']].to_numpy(),
                      frame[['encounter_id', 'patient_nbr', 'target']].to_numpy()))
            predicted = model.predict_proba(prep.transform(frame))[:, 1]
            saved = pa.probability_readmitted_30d.to_numpy()
            check(f'{part}: restored model predictions exact and finite',
                  np.isfinite(saved).all() and np.array_equal(predicted.astype('float64'), saved))
            y, p = pa.target.to_numpy(), saved.astype(predicted.dtype)
            recomputed = {'average_precision': average_precision_score(y, p),
                          'roc_auc': roc_auc_score(y, p), 'log_loss': log_loss(y, p),
                          'brier_score': brier_score_loss(y, p)}
            check(f'{part}: independent metrics within predeclared 1e-15 allowance',
                  all(abs(float(value) - metrics[part][key]) <= 1e-15 for key, value in recomputed.items()))
    report['passed'] = len(report['checks'])
    report['failed'] = 0
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {report['passed']} checks; report: {output / 'verification.json'}")


if __name__ == '__main__':
    main()
