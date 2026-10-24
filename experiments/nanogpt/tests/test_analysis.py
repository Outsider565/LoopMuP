import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.aggregate import COLUMNS, rows_for  # noqa: E402
from analysis.common import GROUP_KEYS, fit_parabola, group_curves, load_results  # noqa: E402
from diag.coord_check import load_runs  # noqa: E402
from sweeps.experiments import loop  # noqa: E402

ALIASES = [dict(run_name=f'coord/d256_L2_N1_aN{a}_lr-7.0_s1', label=f'aN{a}', alpha_N=float(a)) for a in ('0', '0.5', '1.5')]


def test_coord_aliases_restore_every_alpha(tmp_path):
    run = tmp_path / 'coord' / 'd256_L2_N1_aN1_lr-7.0_s1'
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'config': dict(run_name='coord/d256_L2_N1_aN1_lr-7.0_s1',
        tag='coord/aN1', n_embd=256, n_layer=2, n_loops=1, alpha_N=1.0, seed=1)}))
    (run / 'coord.jsonl').write_text(json.dumps(dict(step=0, act=dict(embedding=1, logits=2, resid=[[3]], attn=[[4]],
        mlp=[[5]], resid_final_rms=6))) + '\n')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(dict(run_name='coord/d256_L2_N1_aN1_lr-7.0_s1', label='aN1', alpha_N=1.0,
                                        aliases=ALIASES)) + '\n')
    rows = load_runs(str(tmp_path), 'coord', str(manifest))
    assert {r['label'] for r in rows} == {'aN0', 'aN0.5', 'aN1', 'aN1.5'}
    assert len(rows) == 4 * 6
    assert {r['label'] for r in load_runs(str(tmp_path), 'coord')} == {'aN1'}


def test_aggregate_expands_aliases():
    config = dict(tag='loop/aN1', n_embd=512, n_layer=4, n_loops=1, alpha_N=1.0, alpha_L=0.5, learning_rate=2 ** -7.5,
                  seed=1, init_std=0.02, train_tokens=100)
    done = dict(run_name='loop/d512_L4_N1_aN1_lr-7.5_s1', status='ok', final_val_loss=3.5, mfu=None)
    rows = rows_for(done, dict(config=config), dict(aliases=ALIASES))
    assert [r['label'] for r in rows] == ['aN1', 'aN0', 'aN0.5', 'aN1.5']
    assert [r['alpha_N'] for r in rows] == [1.0, 0.0, 0.5, 1.5]
    assert rows[0]['alias_of'] == '' and all(r['alias_of'] == done['run_name'] for r in rows[1:])
    assert all(r['experiment'] == 'loop' and r['lr_log2'] == -7.5 and r['N'] == 1 for r in rows)
    assert set(rows[0]) == set(COLUMNS)


def test_base_sigmas_are_separate_curves():
    base = dict(experiment='base', label='aN1', d=256, L=2, N=1, lr_log2=-8)
    rows = [dict(base, init_std=sigma, final_val_loss=loss) for sigma, loss in [(0.02, 3), (0.1, 5)]]
    curves = group_curves(rows)
    assert GROUP_KEYS == ('experiment', 'label', 'd', 'L', 'N', 'init_std') and len(curves) == 2
    assert curves[('base', 'aN1', 256, 2, 1, 0.02)][-8] == [3]


def test_parabola_recovers_optimum():
    xs = np.arange(-10, -4.9, 0.5)
    x_opt, y_opt, ok = fit_parabola(xs, (xs + 7.25) ** 2 + 3.0)
    assert ok and abs(x_opt + 7.25) < 1e-6 and abs(y_opt - 3.0) < 1e-6


def test_aggregate_fit_and_plot_cli(tmp_path):
    """Fabricated done.json files for the loop experiment through aggregate.py, fit_optimum.py and plots.py."""
    out_root, manifest = tmp_path / 'out', tmp_path / 'loop.jsonl'
    runs = [r for r in loop(eta_log2=-7.5) if r['shape']['N'] in (1, 4)]
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in runs))
    for r in runs:
        a = r['train_args']
        d = out_root / r['run_name']
        d.mkdir(parents=True)
        loss = 3.0 + 0.1 * (r['lr_log2'] + 7.5) ** 2 + 0.05 * a['alpha_N'] * (r['lr_log2'] + 7.5) * (a['n_loops'] > 1)
        (d / 'config.json').write_text(json.dumps(dict(config=a)))
        (d / 'done.json').write_text(json.dumps(dict(run_name=r['run_name'], status='ok', final_val_loss=loss,
                                                     best_val_loss=loss, mfu=None)))

    def run(script, *args):
        res = subprocess.run([sys.executable, str(REPO / script), *map(str, args)], cwd=REPO, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        return res.stdout

    results, optima, figs = tmp_path / 'results.csv', tmp_path / 'optima.csv', tmp_path / 'figs'
    run('analysis/aggregate.py', '--out_root', out_root, '--experiments', 'loop', '--manifests', manifest, '--out', results)
    rows = load_results(results)
    assert len(rows) == 4 * 9 * 2 and sum(r['alias_of'] != '' for r in rows) == 3 * 9
    run('analysis/fit_optimum.py', '--results', results, '--out', optima, '--n_boot', 20)
    opt = list(csv.DictReader(open(optima)))
    assert len(opt) == 8 and all(abs(float(o['lr_opt_log2']) + 7.5) < 0.6 for o in opt)
    run('analysis/plots.py', 'lr_transfer', '--results', results, '--experiment', 'loop', '--by', 'N', '--out', figs)
    run('analysis/plots.py', 'best_loss', '--results', results, '--experiment', 'loop', '--by', 'N', '--out', figs)
    run('analysis/plots.py', 'optimum', '--optima', optima, '--experiment', 'loop', '--by', 'N', '--out', figs)
    run('analysis/plots.py', 'heatmap', '--optima', optima, '--experiment', 'loop', '--fix', 'alpha_N=1', '--out', figs)
    assert len(list(figs.glob('*.png'))) == 4
