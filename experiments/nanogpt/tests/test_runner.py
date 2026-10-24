import json
import subprocess
import sys
import os
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweeps import run_manifest
from sweeps.experiments import base
from sweeps.run_manifest import outcome, launch, resolve_request, validate_request, prepare_run, build_cmd
from data.sharded_loader import write_shard


@pytest.mark.parametrize('status', ['ok', 'diverged', 'failed', 'oom'])
def test_sigma_collision_never_reuses_previous_run(tmp_path, monkeypatch, status):
    previous = resolve_request(base(sigmas=(0.01,))[0], [], str(tmp_path), None)
    changed = base(sigmas=(0.02,))[0]
    assert previous['run_name'] == changed['run_name']
    out_dir = tmp_path / previous['run_name']
    out_dir.mkdir(parents=True)
    receipt = json.dumps({'status': status, 'config': previous['train_args']})
    (out_dir / 'done.json').write_text(receipt)
    previous['_request']['checkpointing'] = 'auto'
    (out_dir / 'run_request.json').write_text(json.dumps(previous['_request']))
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(changed) + '\n')
    monkeypatch.setattr(run_manifest, 'detect_gpus', lambda: (0, 0.0))
    monkeypatch.setattr(sys, 'argv', ['runner', '--manifest', str(manifest), '--out_root', str(tmp_path)])
    with pytest.raises(ValueError, match='configuration differs'):
        run_manifest.main()
    assert (out_dir / 'done.json').read_text() == receipt
    assert not (tmp_path / 'sweep_status').exists()


def test_request_tracks_extra_overrides_and_data_directory(tmp_path, monkeypatch):
    run = base()[0]
    monkeypatch.setenv('LOOPMUP_DATA', str(tmp_path / 'data_a'))
    original = resolve_request(run, ['--learning_rate=0.001', '--compile=False'], str(tmp_path), None)
    (tmp_path / 'run_request.json').write_text(json.dumps(original['_request']))
    reordered = resolve_request(run, ['--compile=False', '--learning_rate=0.001'], str(tmp_path), None)
    validate_request(reordered, tmp_path)
    for extra, data_dir in [(['--compile=False'], None),
                            (['--compile=False', '--learning_rate=0.002'], None),
                            (['--compile=False', '--learning_rate=0.001'], str(tmp_path / 'data_b'))]:
        with pytest.raises(ValueError, match='configuration differs'):
            validate_request(resolve_request(run, extra, str(tmp_path), data_dir), tmp_path)
    monkeypatch.setenv('LOOPMUP_DATA', str(tmp_path / 'data_b'))
    with pytest.raises(ValueError, match='configuration differs'):
        validate_request(resolve_request(run, ['--learning_rate=0.001', '--compile=False'], str(tmp_path), None), tmp_path)


def test_missing_request_does_not_reuse_unverified_artifacts(tmp_path):
    (tmp_path / 'done.json').write_text(json.dumps({'status': 'ok'}))
    with pytest.raises(ValueError, match='no run_request.json'):
        validate_request(resolve_request(base()[0], [], str(tmp_path), None), tmp_path)


def test_oom_adjustments_preserve_request_and_override_micro_batch(tmp_path):
    run = resolve_request(base()[0], ['--micro_batch=8'], str(tmp_path), None)
    (tmp_path / 'run_request.json').write_text(json.dumps(run['_request']))
    prepare_run(run, 1, 80.0, None)
    run['train_args']['micro_batch'] //= 2
    run['train_args']['loop_activation_checkpointing'] = True
    validate_request(run, tmp_path)
    cmd = build_cmd(run, 1, 'pack', sys.executable, [])
    assert '--micro_batch=4' in cmd and '--micro_batch=8' not in cmd
    assert run['_request']['train_args']['micro_batch'] == 8


def test_runner_skips_matching_run_and_retries_failures(tmp_path, monkeypatch, capsys):
    data = tmp_path / 'data'
    data.mkdir()
    for kind in ('train', 'val'):
        write_shard(str(data / f'finewebedu_{kind}_000000.bin'), np.arange(128) % 64)
    run = dict(run_name='tiny', train_args=dict(
        device='cpu', compile=False, dtype='float32', vocab_size=64, n_embd=16, n_head=2, n_layer=1,
        n_loops=1, block_size=8, micro_batch=2, batch_tokens=16, train_tokens=32,
        eval_tokens_interval=32, eval_tokens_final=32, eval_micro_batch=2, learning_rate=0.001))
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(run) + '\n')
    argv = ['runner', '--manifest', str(manifest), '--out_root', str(tmp_path / 'out'),
            '--data_dir', str(data), '--nproc', '1']
    monkeypatch.setattr(run_manifest, 'detect_gpus', lambda: (0, 0.0))
    monkeypatch.setattr(sys, 'argv', argv)
    assert run_manifest.main() == 0
    out_dir = tmp_path / 'out' / 'tiny'
    request = (out_dir / 'run_request.json').read_bytes()
    metrics = (out_dir / 'metrics.jsonl').read_bytes()
    capsys.readouterr()
    assert run_manifest.main() == 0
    assert '0 to execute' in capsys.readouterr().out
    assert (out_dir / 'metrics.jsonl').read_bytes() == metrics
    for status, extra in [('failed', ['--retry_failed']), ('oom', [])]:
        done = json.loads((out_dir / 'done.json').read_text())
        done['status'] = status
        (out_dir / 'done.json').write_text(json.dumps(done))
        monkeypatch.setattr(sys, 'argv', argv + extra)
        assert run_manifest.main() == 0
        assert json.loads((out_dir / 'done.json').read_text())['status'] == 'ok'
        assert (out_dir / 'run_request.json').read_bytes() == request


def test_nonzero_exit_overrides_success_receipt(tmp_path):
    (tmp_path / 'done.json').write_text(json.dumps({'status': 'ok'}))
    assert outcome({}, str(tmp_path), 1) == 'failed'
    assert json.loads((tmp_path / 'done.json').read_text())['status'] == 'failed'
    assert outcome({}, str(tmp_path), 0) == 'failed'


def test_missing_receipt_is_failure_even_on_zero_exit(tmp_path):
    assert outcome({}, str(tmp_path), 0) == 'failed'
    (tmp_path / 'train.log').write_text('torch.cuda.OutOfMemoryError: CUDA out of memory')
    assert outcome({}, str(tmp_path), 1) == 'oom'


def test_oom_is_read_from_the_current_attempt_only(tmp_path):
    (tmp_path / 'train.log').write_text('torch.cuda.OutOfMemoryError: CUDA out of memory\n')
    proc = launch({}, [sys.executable, '-c', 'import sys; sys.exit(139)'], str(tmp_path), None)
    rc = proc.wait()
    assert outcome({}, str(tmp_path), rc, proc.log_start) == 'failed'
    assert outcome({}, str(tmp_path), rc) == 'oom'


def test_extra_overrides_and_dry_run(tmp_path):
    run = dict(run_name='r', train_args=dict(block_size=1024, batch_tokens=2 ** 13, micro_batch=8, n_embd=64,
                                             n_layer=2, n_loops=1, vocab_size=64, coord_check=True))
    manifest = tmp_path / 'm.jsonl'
    manifest.write_text(json.dumps(run) + '\n')
    out_root = tmp_path / 'out'
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'sweeps' / 'run_manifest.py'),
                           '--manifest', str(manifest), '--out_root', str(out_root), '--mode', 'ddp', '--nproc', '8',
                           '--extra=--device=cpu', '--extra=--compile=False', '--dry_run'],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    cmd = [line for line in proc.stdout.splitlines() if 'train.py' in line][0]
    assert '--device=cpu' in cmd and '--compile=False' in cmd
    assert 'torch.distributed.run' not in cmd and '--micro_batch=8' in cmd   # coordinate checks stay single-process
    assert not out_root.exists()


def test_retry_clears_stale_receipt(tmp_path):
    (tmp_path / 'done.json').write_text(json.dumps({'status': 'oom'}))
    proc = launch({}, [sys.executable, '-c', 'pass'], str(tmp_path), None)
    assert proc.wait() == 0
    assert not (tmp_path / 'done.json').exists()


def test_output_is_live_before_child_exits(tmp_path, capsys):
    gate = tmp_path / 'release'
    script = ("import pathlib, sys, time; "
              "print('live progress'); print('live error', file=sys.stderr); "
              f"gate = pathlib.Path({str(gate)!r})\n"
              "while not gate.exists(): time.sleep(0.01)\n"
              "print('final progress')\nsys.exit(7)")
    proc = launch({}, [sys.executable, '-c', script], str(tmp_path), dict(os.environ))
    observed = ''
    try:
        deadline = time.monotonic() + 5
        while 'live error' not in observed and time.monotonic() < deadline:
            observed += capsys.readouterr().out
            time.sleep(0.01)
        assert proc.poll() is None
        assert 'live progress' in observed and 'live error' in observed
        assert 'live error' in (tmp_path / 'train.log').read_text()
    finally:
        gate.touch()
        rc = proc.wait(timeout=5)
    assert rc == 7
    assert 'final progress' in capsys.readouterr().out
    assert 'final progress' in (tmp_path / 'train.log').read_text()


def test_parallel_output_drains_before_poll_returns_exit(tmp_path, capsys):
    procs = [launch({}, [sys.executable, '-c', f"print('{i}:' + 'x' * 200000)"],
                    str(tmp_path / str(i)), None) for i in range(2)]
    deadline = time.monotonic() + 10
    while any(proc.poll() is None for proc in procs) and time.monotonic() < deadline:
        time.sleep(0.01)
    for i, proc in enumerate(procs):
        assert proc.wait(timeout=5) == 0
        assert f'{i}:' + 'x' * 200000 in (tmp_path / str(i) / 'train.log').read_text()
    output = capsys.readouterr().out
    assert all(f'{i}:' + 'x' * 200000 in output for i in range(2))
