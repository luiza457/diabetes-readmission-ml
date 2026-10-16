"""Acquire exact public UCI source files without overwriting existing inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = ('https://archive.ics.uci.edu/static/public/296/'
       'diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use a previously downloaded UCI ZIP')
    args = parser.parse_args()
    data = json.loads((ROOT / 'reproducibility_config.json').read_text())['data']
    expected = {data['csv']: data['sha256'], data['mapping']: data['mapping_sha256']}
    missing = []
    for name, checksum in expected.items():
        path = ROOT / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f'{name}: existing file differs; refusing to overwrite')
            print(f'Verified existing {name}')
        else:
            missing.append(name)
    if not missing:
        return
    if args.archive:
        archive_bytes = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=90) as response:
            archive_bytes = response.read()
    pending = {}
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in missing:
            members = [m for m in archive.namelist() if Path(m).name == name]
            if len(members) != 1:
                raise ValueError(f'Expected exactly one archive member named {name}')
            content = archive.read(members[0])
            if hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError(f'{name}: downloaded bytes differ from frozen input hash')
            pending[name] = content
    # Validate all requested bytes before writing; never extract archive paths.
    for name, content in pending.items():
        with (ROOT / name).open('xb') as stream:
            stream.write(content)
        print(f'Wrote verified {name} ({len(content):,} bytes)')


if __name__ == '__main__':
    main()
