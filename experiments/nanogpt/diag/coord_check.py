#!/usr/bin/env python3
"""
Aggregate and plot the coordinate checks of the `coord` experiment (`train.py --coord_check=True`).

    python sweeps/make_manifest.py --experiment coord --out outputs/manifests/coord.jsonl
    python sweeps/run_manifest.py --manifest outputs/manifests/coord.jsonl --mode pack --out_root $LOOPMUP_OUT
    python diag/coord_check.py --out_root $LOOPMUP_OUT --experiment coord --manifest outputs/manifests/coord.jsonl --out outputs/figures/coord

A smaller-scale analogue of the paper's Figures 2, 3 (right) and 9. For every alpha_N and every axis (loop count
N at d=256, L=2; depth L at N=1; width d at L=2, N=1) it plots against the axis, one line per training step and
averaged over seeds: the RMS of the final residual stream, the abs-means of the embedding output, last residual
stream, attention / MLP branch outputs and logits, and the RMS change of the final residual stream caused by each
optimizer step (absolute and relative). A quantity is stable along an axis when its lines are flat. Writes
coord_summary.csv and prints, for every alpha_N, the ratio between the largest and the smallest N at the first
and the last optimizer step. The manifest restores the alpha_N labels of runs shared at N=1.

Steps follow coord.jsonl: at step s the forward quantities are measured after s optimizer steps (step 0 is the
initialization), and delta_rms / delta_rel are the change caused by optimizer step s + 1 (step t = s + 1 in the
numbering of the paper's Fig. 9).
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

QUANTITIES = ['resid_rms', 'embedding', 'resid_last', 'attn_mean', 'mlp_mean', 'logits', 'delta_rms', 'delta_rel']
AXES = {
    'loop': dict(var='N', fixed=dict(d=256, L=2)),
    'depth': dict(var='L', fixed=dict(d=256, N=1)),
    'width': dict(var='d', fixed=dict(L=2, N=1)),
}
KEY = dict(d=1, L=2, N=3)   # positions in the aggregate key (label, d, L, N, step, quantity)


def label_text(label):
    a = float(label[2:])
    return rf"$\alpha_N = {a:g}$" + (' (Loop-μP)' if a == 1 else '')


def by_alpha(labels):
    return sorted(set(labels), key=lambda s: float(s[2:]))


def load_runs(out_root, experiment='coord', manifest=None):
    labels = {}
    if manifest:
        with open(manifest) as f:
            for r in map(json.loads, filter(str.strip, f)):
                labels[r['run_name']] = [r['label']] + [a['label'] for a in r.get('aliases', [])]
    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(out_root, experiment, '*', 'config.json'))):
        coord_path = os.path.join(os.path.dirname(cfg_path), 'coord.jsonl')
        if not os.path.exists(coord_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)['config']
        run_labels = labels.get(cfg['run_name'], [f"aN{cfg['alpha_N']:g}"])
        with open(coord_path) as f:
            for line in f:
                r = json.loads(line)
                act = r['act']
                q = dict(embedding=act['embedding'], resid_last=act['resid'][-1][-1],
                         attn_mean=float(np.mean(act['attn'])), mlp_mean=float(np.mean(act['mlp'])),
                         logits=act['logits'], resid_rms=act['resid_final_rms'])
                if 'update' in r:
                    q['delta_rms'] = r['update']['delta_rms']
                    q['delta_rel'] = r['update']['delta_rel']
                for label in run_labels:
                    for name, val in q.items():
                        rows.append(dict(label=label, d=cfg['n_embd'], L=cfg['n_layer'], N=cfg['n_loops'],
                                         seed=cfg['seed'], step=r['step'], quantity=name, value=val))
    return rows


def aggregate(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r['label'], r['d'], r['L'], r['N'], r['step'], r['quantity'])].append(r['value'])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}


def plot(agg, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    steps = sorted({k[4] for k in agg})
    cmap = plt.get_cmap('viridis')
    for axis, spec in AXES.items():
        def shape(x):
            s = dict(spec['fixed'], **{spec['var']: x})
            return s['d'], s['L'], s['N']
        for label in by_alpha(k[0] for k in agg):
            xs = sorted({k[KEY[spec['var']]] for k in agg
                         if k[0] == label and all(k[KEY[f]] == v for f, v in spec['fixed'].items())})
            if len(xs) < 2:
                continue
            fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(3.0 * len(QUANTITIES), 3.0), squeeze=False)
            for ax, q in zip(axes[0], QUANTITIES):
                for si, step in enumerate(steps):
                    ys = [agg.get((label, *shape(x), step, q), (np.nan,))[0] for x in xs]
                    if all(np.isnan(ys)):
                        continue
                    ax.plot(xs, ys, marker='o', ms=3, color=cmap(si / max(1, len(steps) - 1)), label=f'step {step}')
                ax.set_xscale('log', base=2); ax.set_yscale('log')
                ax.set_title(q, fontsize=9); ax.set_xlabel(spec['var'])
                ax.grid(True, alpha=0.3)
            axes[0][0].legend(fontsize=6, ncol=2)
            fixed = ', '.join(f'{k}={v}' for k, v in spec['fixed'].items())
            fig.suptitle(f'{label_text(label)}: coordinate check along {spec["var"]} ({fixed})')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'coord_{axis}_{label}.png'), dpi=150)
            plt.close(fig)


def ratio_summary(agg, step, d=256, L=2):
    print(f"step {step}: value at N=max / value at N=min (d={d}, L={L}); close to 1 = stable across loop counts")
    for label in by_alpha(k[0] for k in agg):
        Ns = sorted({k[3] for k in agg if k[0] == label and k[1] == d and k[2] == L})
        if len(Ns) < 2:
            continue
        parts = []
        for q in ('resid_rms', 'logits', 'delta_rms'):
            lo, hi = agg.get((label, d, L, Ns[0], step, q)), agg.get((label, d, L, Ns[-1], step, q))
            if lo and hi and lo[0] > 0:
                parts.append(f"{q} x{hi[0] / lo[0]:.2f}")
        print(f"  {label:6s} N {Ns[0]}->{Ns[-1]}: " + ', '.join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_root', default=os.environ.get('LOOPMUP_OUT', 'out'))
    ap.add_argument('--experiment', default='coord')
    ap.add_argument('--out', default='figures/coord')
    ap.add_argument('--manifest', help='coord manifest, to expand runs shared by several alpha_N')
    args = ap.parse_args()
    rows = load_runs(args.out_root, args.experiment, args.manifest)
    if not rows:
        raise SystemExit(f'no coord.jsonl under {args.out_root}/{args.experiment}')
    agg = aggregate(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'coord_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'd', 'L', 'N', 'step', 'quantity', 'mean', 'std', 'n'])
        for k in sorted(agg):
            w.writerow(list(k) + list(agg[k]))
    plot(agg, args.out)
    update_steps = sorted({k[4] for k in agg if k[5] == 'delta_rms'}) or [max(k[4] for k in agg)]
    for step in sorted({update_steps[0], update_steps[-1]}):
        ratio_summary(agg, step)
    print(f'wrote {args.out}/coord_summary.csv and coord_*.png ({len(rows)} rows)')


if __name__ == '__main__':
    main()
