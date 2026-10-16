"""Check publication integrity using only the Python standard library."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'results/source_integrity.json').read_text())
    for name, expected in manifest.items():
        allowed = expected if isinstance(expected, list) else [expected]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() in allowed, name
    for path in [ROOT / 'train.py', ROOT / 'review/validate.py', *(ROOT / 'scripts').glob('*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in ROOT.glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'{path.name}:cell{i}')
    documents = [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md'),
                 *(ROOT / 'llm_harness').glob('*.md'), ROOT / 'splits/README.md']
    for path in documents:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent / unquote(link.split('#')[0])).exists(), f'{path}: {link}'
    summary = json.loads((ROOT / 'results/reproducibility_summary.json').read_text())
    assert summary['passed'] == 52 and summary['failed'] == 0
    assert len(summary['checks']) == 52
    assert all(c['status'] == 'passed' for c in summary['checks'])
    metrics = json.loads((ROOT / 'results/reconstructed_metrics.json').read_text())
    assert sum(m['encounters'] for m in metrics.values()) == 100111
    assert metrics['test']['average_precision'] == 0.21601058584239669
    print(f'PASS: {len(manifest)} reviewed source hashes, Python/notebook syntax, documentation links and aggregate evidence')


if __name__ == '__main__':
    main()
