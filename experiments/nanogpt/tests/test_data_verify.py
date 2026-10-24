import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.sharded_loader import write_shard


def test_empty_data_is_rejected(tmp_path):
    proc = subprocess.run([sys.executable, str(REPO / 'data/finewebedu10B/verify.py'),
                           '--dir', str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 1
    assert 'no training shards' in proc.stdout


def test_requested_hash_check_requires_hashes(tmp_path):
    for name in ['finewebedu_train_000001.bin', 'finewebedu_val_000000.bin']:
        write_shard(str(tmp_path / name), np.arange(20, dtype=np.uint16))
    proc = subprocess.run([sys.executable, str(REPO / 'data/finewebedu10B/verify.py'),
                           '--dir', str(tmp_path), '--sha256'], capture_output=True, text=True)
    assert proc.returncode == 1
    assert 'no manifest SHA256 available' in proc.stdout


def test_partial_download_passes_unless_complete_is_required(tmp_path):
    files = []
    for name in ['finewebedu_train_000001.bin', 'finewebedu_val_000000.bin']:
        write_shard(str(tmp_path / name), np.arange(20, dtype=np.uint16))
        files.append(dict(path=name, size=(tmp_path / name).stat().st_size, sha256=None))
    files.append(dict(path='finewebedu_train_000002.bin', size=1, sha256=None))   # listed, not downloaded
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(files=files)))
    verify = [sys.executable, str(REPO / 'data/finewebedu10B/verify.py'), '--dir', str(tmp_path)]
    proc = subprocess.run(verify, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout
    assert 'not present' in proc.stdout
    proc = subprocess.run(verify + ['--complete'], capture_output=True, text=True)
    assert proc.returncode == 1 and 'missing files listed in manifest' in proc.stdout
