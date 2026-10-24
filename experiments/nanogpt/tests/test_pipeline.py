import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.sharded_loader import write_shard, read_header, ShardSet, SequentialLoader, FixedEvalSet  # noqa: E402
from sweeps.experiments import (COMMON, EXPERIMENTS, NEEDS_ETA, TRAIN_TOKENS, base, dedupe, effective_key,  # noqa: E402
                                loop, make_run)
from sweeps.micro_batch import choose_micro_batch  # noqa: E402
from sweeps.run_manifest import build_cmd, prepare_run  # noqa: E402


@pytest.fixture(scope='module')
def shards(tmp_path_factory):
    d = tmp_path_factory.mktemp('shards')
    rng = np.random.default_rng(0)
    lengths = [1000, 1500, 700]
    for i, n in enumerate(lengths, 1):
        write_shard(str(d / f'finewebedu_train_{i:06d}.bin'), rng.integers(0, 64, size=n))
    write_shard(str(d / 'finewebedu_val_000000.bin'), rng.integers(0, 64, size=3000))
    return d


def build(name):
    return EXPERIMENTS[name](eta_log2=-7.5) if name in NEEDS_ETA else EXPERIMENTS[name]()


def train_py_defaults():
    """Config keys of train.py (top-level assignments before `config_keys`) with their constant defaults."""
    out = {}
    for node in ast.parse(open(os.path.join(REPO, 'train.py')).read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == 'config_keys':
                break
            try:   # constant expressions such as 2 ** 18; anything that needs a module (os, torch) maps to None
                out[name] = eval(compile(ast.Expression(node.value), 'train.py', 'eval'), {'__builtins__': {}})
            except NameError:
                out[name] = None
    return out


def test_header_and_concat(shards):
    files = sorted(str(p) for p in shards.glob('finewebedu_train_*.bin'))
    assert [read_header(f) for f in files] == [1000, 1500, 700]
    ss = ShardSet(str(shards / 'finewebedu_train_*.bin'))
    full = np.concatenate([np.memmap(f, dtype=np.uint16, mode='r', offset=1024) for f in files])
    assert ss.total == 3200
    assert np.array_equal(ss.read(990, 30), full[990:1020])            # crosses shard 1 -> 2
    assert np.array_equal(ss.read(3190, 20), np.concatenate([full[3190:], full[:10]]))  # wraps around


def test_sequential_loader_partition_and_resume(shards):
    pattern = str(shards / 'finewebedu_train_*.bin')
    full = ShardSet(pattern).read(0, 3200).astype(np.int64)
    B, T, world = 2, 8, 3
    loaders = [SequentialLoader(pattern, B, T, rank=r, world_size=world) for r in range(world)]
    seen = []
    for step in range(3):
        for r, ld in enumerate(loaders):
            x, y = ld.next_batch()
            start = step * world * B * T + r * B * T
            assert torch.equal(x.flatten(), torch.from_numpy(full[start:start + B * T]))
            assert torch.equal(y.flatten(), torch.from_numpy(full[start + 1:start + B * T + 1]))
            seen.append(start)
    assert len(set(seen)) == len(seen)                                  # ranks never overlap
    resumed = SequentialLoader(pattern, B, T, rank=1, world_size=world, start_step=3)
    x_r, _ = resumed.next_batch()
    x_c, _ = loaders[1].next_batch()
    assert torch.equal(x_r, x_c)                                         # resume == continue


def test_eval_set_disjoint(shards):
    pattern = str(shards / 'finewebedu_val_*.bin')
    a = FixedEvalSet(pattern, 2048, 4, 8, rank=0, world_size=2)
    b = FixedEvalSet(pattern, 2048, 4, 8, rank=1, world_size=2)
    assert set(a.my_batches).isdisjoint(b.my_batches)
    assert len(a) + len(b) == a.n_batches_total == 2048 // 8 // 4


def test_tiny_train_end_to_end(shards, tmp_path):
    out = tmp_path / 'out'
    args = [sys.executable, os.path.join(REPO, 'train.py'), '--device=cpu', '--compile=False', '--dtype=float32',
            f'--data_dir={shards}', f'--out_root={out}', '--run_name=t', '--vocab_size=64', '--n_embd=32', '--n_head=2',
            '--n_layer=2', '--n_loops=2', '--block_size=8', '--micro_batch=2', '--batch_tokens=32', '--train_tokens=160',
            '--eval_tokens_interval=64', '--eval_tokens_final=128', '--eval_micro_batch=2', '--log_interval=1',
            '--learning_rate=0.01', '--ckpt_every_frac=0.5', '--save_final_checkpoint=True']
    subprocess.run(args, check=True, cwd=REPO, capture_output=True)
    done = json.load(open(out / 't' / 'done.json'))
    assert done['status'] == 'ok' and done['iters_done'] == 5 and done['tokens_seen'] == 160
    assert done['parameterization']['branch_multiplier'] == pytest.approx(0.5)   # N=2, m_L=1, alpha_N=1
    metrics = [json.loads(l) for l in open(out / 't' / 'metrics.jsonl')]
    assert any('val_loss' in m and m.get('final') for m in metrics)
    assert (out / 't' / 'ckpt_last.pt').exists() and (out / 't' / 'ckpt_final.pt').exists()
    # re-running the same run with a longer budget must resume from the periodic checkpoint (step 4), not step 0
    res = subprocess.run(args + ['--train_tokens=192'], check=True, cwd=REPO, capture_output=True, text=True)
    assert 'resumed from' in res.stdout and 'at step 4' in res.stdout
    done2 = json.load(open(out / 't' / 'done.json'))
    assert done2['iters_done'] == 6 and done2['tokens_seen'] == 192


def test_coord_check_mode(shards, tmp_path):
    out = tmp_path / 'out'
    args = [sys.executable, os.path.join(REPO, 'train.py'), '--device=cpu', '--compile=False', '--dtype=float32',
            f'--data_dir={shards}', f'--out_root={out}', '--run_name=cc', '--vocab_size=64', '--n_embd=32', '--n_head=2',
            '--n_layer=2', '--n_loops=3', '--block_size=8', '--micro_batch=2', '--batch_tokens=16', '--train_tokens=160',
            '--coord_check=True', '--coord_steps=2', '--coord_save_params=True', '--learning_rate=0.01']
    subprocess.run(args, check=True, cwd=REPO, capture_output=True)
    rows = [json.loads(l) for l in open(out / 'cc' / 'coord.jsonl')]
    assert len(rows) == 3 and len(rows[0]['act']['resid']) == 3 and len(rows[0]['act']['resid'][0]) == 2
    assert 'update' in rows[0] and 'update' not in rows[-1]
    assert (out / 'cc' / 'params_step0.pt').exists() and (out / 'cc' / 'params_step2.pt').exists()


def test_common_matches_paper_defaults():
    expected = dict(weight_decay=0.1, adam_eps=1e-8, beta1=0.9, beta2=0.95, lr_schedule='wsd', warmup_frac=0.025,
                    decay_frac=0.05, min_lr=0.0, init_std=0.02, branch_mult=1.0, alpha_L=0.5, base_width=256,
                    base_depth=2, batch_tokens=2 ** 18)
    assert {k: COMMON[k] for k in expected} == expected
    assert TRAIN_TOKENS == 11781 * 2 ** 18


def test_config_files_set_only_known_keys(tmp_path, monkeypatch):
    """configurator.py rejects unknown keys in config files, as it does for --key=value."""
    defaults = train_py_defaults()
    configurator = open(os.path.join(REPO, 'configurator.py')).read()
    for name in ('base.py', 'loop.py'):
        ns = dict(defaults)
        monkeypatch.setattr(sys, 'argv', ['train.py', os.path.join(REPO, 'config', name)])
        exec(configurator, ns)
    stale = tmp_path / 'stale.py'
    stale.write_text('n_loop = 4\nn_embd = 512\n')
    monkeypatch.setattr(sys, 'argv', ['train.py', str(stale)])
    with pytest.raises(ValueError, match='n_loop'):
        exec(configurator, dict(defaults))


def test_experiments_deterministic_unique_and_accepted_by_train_py():
    defaults = train_py_defaults()
    for name in EXPERIMENTS:
        a, b = build(name), build(name)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        names = [r['run_name'] for r in a]
        assert len(names) == len(set(names))
        for r in a:
            args = r['train_args']
            assert set(args) <= set(defaults), set(args) - set(defaults)
            for k, v in args.items():   # configurator.py requires the override type to match the default
                assert defaults[k] is None or type(v) is type(defaults[k]), (k, v)
            assert args['n_head'] == r['shape']['d'] // 64
            assert args['learning_rate'] == 2.0 ** r['lr_log2']
            assert r['label'] == f"aN{args['alpha_N']:g}" and r['tag'] == f"{name}/{r['label']}"


def test_experiment_sizes_and_budgets():
    sizes = {name: len(build(name)) for name in EXPERIMENTS}
    assert sizes == dict(coord=96, base=13, loop=117, depth=84, width=21, smoke=1)
    for name in ('base', 'loop', 'depth', 'width'):
        runs = build(name)
        assert {r['seed'] for r in runs} == {1}
        assert {r['train_tokens'] for r in runs} == {r['train_args']['train_tokens'] for r in runs} == {TRAIN_TOKENS}
        assert all(r['train_args']['lr_schedule'] == 'wsd' for r in runs)
    assert {r['alpha_N'] for r in build('depth') + build('width')} == {1.0}
    assert all(r['train_args']['coord_check'] for r in build('coord'))


def test_loop_runs_merge_only_at_one_loop():
    runs = loop(eta_log2=-7.5)
    assert {r['label'] for r in runs} == {'aN0', 'aN0.5', 'aN1', 'aN1.5'}
    merged = [r for r in runs if r.get('aliases')]
    assert len(merged) == 9 and all(r['shape']['N'] == 1 and r['label'] == 'aN1' for r in merged)
    for r in merged:
        assert [(a['label'], a['alpha_N']) for a in r['aliases']] == [('aN0', 0.0), ('aN0.5', 0.5), ('aN1.5', 1.5)]
        assert all(a['run_name'] == r['run_name'].replace('_aN1_', f"_{a['label']}_") for a in r['aliases'])


def test_effective_key_dedupe():
    a = make_run('x', 512, 4, 1, 1.0, -7.0, 1, 1000)
    b = make_run('x', 512, 4, 1, 0.0, -7.0, 1, 1000)    # N=1: N^-alpha_N = 1 for every alpha_N
    c = make_run('x', 512, 4, 2, 0.0, -7.0, 1, 1000)
    d = make_run('x', 512, 4, 2, 1.0, -7.0, 1, 1000)
    assert effective_key(a['train_args']) == effective_key(b['train_args'])
    assert len({effective_key(r['train_args']) for r in (a, c, d)}) == 3
    out = dedupe([a, b, c, d, dict(b)])
    assert [r['run_name'] for r in out] == [a['run_name'], c['run_name'], d['run_name']]
    assert out[0]['aliases'] == [dict(run_name=b['run_name'], label='aN0', alpha_N=0.0)]


def test_base_sigma_names():
    assert all('_sig' not in r['run_name'] for r in base())
    runs = base(sigmas=(0.02, 0.05))
    assert len(runs) == 26 and {r['train_args']['init_std'] for r in runs} == {0.02, 0.05}
    assert runs[0]['run_name'] == 'base/d256_L2_N1_aN1_sig0.02_lr-11.0_s1'


def test_micro_batch_heuristic():
    mb, ck = choose_micro_batch(512, 4, 1, seqs_per_rank=32)          # default 80 GB device
    assert mb == 32 and not ck
    mb8, ck8 = choose_micro_batch(512, 4, 8, seqs_per_rank=32)
    assert mb8 <= mb and not ck8
    mb128, ck128 = choose_micro_batch(512, 16, 8, seqs_per_rank=32)
    assert ck128 and 32 % mb128 == 0


def test_prepare_run_and_command():
    run = next(r for r in loop(eta_log2=-7.5) if r['shape']['N'] == 8)
    prepare_run(run, world=8, mem_gb=0.0, checkpointing_override=None)
    assert run['train_args']['micro_batch'] == choose_micro_batch(512, 4, 8, seqs_per_rank=32, mem_gb=80.0)[0]
    assert run['train_args']['loop_activation_checkpointing'] is False
    cmd = build_cmd(run, 8, 'ddp', 'python', ['--out_root=o'])
    assert '--nproc_per_node=8' in cmd and '--n_loops=8' in cmd and '--alpha_N=1.0' in cmd and cmd[-1] == '--out_root=o'


def test_manifest_cli(tmp_path):
    def make(*extra):
        return subprocess.run([sys.executable, os.path.join(REPO, 'sweeps', 'make_manifest.py'), *extra],
                              cwd=REPO, capture_output=True, text=True)
    assert make('--experiment', 'loop', '--out', str(tmp_path / 'x.jsonl')).returncode == 2       # needs --eta_log2
    assert make('--experiment', 'smoke', '--sigma', '0.02', '--out', str(tmp_path / 'x.jsonl')).returncode == 2
    res = make('--experiment', 'width', '--eta_log2', '-7.5', '--sigma', '0.05', '--seeds', '1,2', '--shards', '2',
               '--out', str(tmp_path / 'width.jsonl'))
    assert res.returncode == 0, res.stderr
    runs = [json.loads(l) for l in open(tmp_path / 'width.jsonl')]
    assert len(runs) == 42 and {r['train_args']['init_std'] for r in runs} == {0.05}
    parts = [len(open(tmp_path / f'width.shard{i}.jsonl').readlines()) for i in range(2)]
    assert parts == [21, 21]
