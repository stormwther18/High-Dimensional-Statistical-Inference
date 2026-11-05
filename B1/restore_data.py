"""Restore the losslessly compressed weight table and verify its SHA-256.

Run from any folder: python3 B1/restore_data.py
Only the Python standard library is required. Existing differing data is never
overwritten. The compressed archive is kept for later GitHub uploads.
"""
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    target = ROOT / 'Results/weights.csv'
    archive = ROOT / 'Results/weights.csv.gz'
    expected = json.loads((ROOT / 'SHA256SUMS.json').read_text())['Results/weights.csv']
    if target.exists():
        if digest(target) != expected:
            raise RuntimeError('Existing weights.csv differs from the saved experiment; preserved without overwriting.')
        print('weights.csv is already present and verified.')
        return
    temporary = None
    try:
        h = hashlib.sha256()
        with gzip.open(archive, 'rb') as incoming, tempfile.NamedTemporaryFile(
                mode='wb', prefix='.weights-restore-', suffix='.tmp',
                dir=target.parent, delete=False) as outgoing:
            temporary = Path(outgoing.name)
            for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                h.update(chunk)
                outgoing.write(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError('Compressed data checksum does not match the original weights.csv.')
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print('Restored Results/weights.csv; SHA-256 matches the original experiment.')


if __name__ == '__main__':
    main()
