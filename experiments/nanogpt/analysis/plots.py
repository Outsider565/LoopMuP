#!/usr/bin/env python3
"""
Figures from results.csv (aggregate.py) and optima.csv (fit_optimum.py).

    python analysis/plots.py lr_transfer --results R --experiment loop --by N --out outputs/figures/loop     # panel per alpha_N
    python analysis/plots.py lr_transfer --results R --experiment depth loop --by L --fix N=8 alpha_N=1 --out outputs/figures/depth
    python analysis/plots.py lr_transfer --results R --experiment width --by d --out outputs/figures/width
    python analysis/plots.py lr_transfer --results R --experiment base --by init_std --out outputs/figures/base
    python analysis/plots.py best_loss   --results R --experiment loop --by N --out outputs/figures/loop     # line per alpha_N
    python analysis/plots.py optimum     --optima outputs/optima.csv --experiment loop --by N --out outputs/figures/loop
    python analysis/plots.py heatmap     --optima outputs/optima.csv --experiment depth loop --fix alpha_N=1 --out outputs/figures/depth

`--experiment` takes one or more experiments (the depth axis combines `depth` with the L=4 runs of `loop`);
`--fix` keeps rows with the given d, L, N, alpha_N or init_std. Lines are seed means with std error bars; stars
mark the argmin of the seed-mean curve.
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import GROUP_KEYS, load_results, group_curves, seed_mean_curve  # noqa: E402

AXES = ('d', 'L', 'N', 'init_std')
IDX = {k: GROUP_KEYS.index(k) for k in AXES}


def label_text(label):
    """'aN1' -> 'α_N = 1 (Loop-μP)'."""
    a = float(label[2:])
    return rf"$\alpha_N = {a:g}$" + (' (Loop-μP)' if a == 1 else '')


def by_alpha(labels):
    return sorted(set(labels), key=lambda s: float(s[2:]))


def parse_fix(fix):
    out = {}
    for item in fix or []:
        k, v = item.split('=')
        if k not in AXES + ('alpha_N',):
            raise SystemExit(f'--fix supports {", ".join(AXES + ("alpha_N",))}, got {k}')
        out[k] = float(v)
    return out


def select(rows, experiments, fix):
    return [r for r in rows if r['experiment'] in experiments and all(float(r[k]) == v for k, v in fix.items())]


def suffix(args):
    return '_'.join(args.experiment) + (f"_by{args.by}" if getattr(args, 'by', None) else '') + \
        ('_' + '_'.join(args.fix) if args.fix else '')


def line_label(args, key):
    return f"{args.by}={key[IDX[args.by]]:g}" + (f" ({key[0]})" if len(args.experiment) > 1 else '')


def lr_transfer(args):
    curves = group_curves(select(load_results(args.results), args.experiment, parse_fix(args.fix)))
    labels = by_alpha(k[1] for k in curves)
    if not labels:
        raise SystemExit('no matching ok runs')
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.4), squeeze=False, sharey=True)
    cmap = plt.get_cmap('viridis')
    for ax, lab in zip(axes[0], labels):
        keys = sorted([k for k in curves if k[1] == lab], key=lambda k: (k[IDX[args.by]], k[0]))
        for j, k in enumerate(keys):
            xs, m, s, n = seed_mean_curve(curves[k])
            color = cmap(j / max(1, len(keys) - 1))
            ax.errorbar(xs, m, yerr=s, marker='o', ms=3, capsize=2, color=color, label=line_label(args, k))
            i = int(np.argmin(m))
            ax.plot(xs[i], m[i], marker='*', ms=12, color=color, markeredgecolor='k')
        ax.set_title(label_text(lab), fontsize=9)
        ax.set_xlabel('base LR (log2)'); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    axes[0][0].set_ylabel('final val loss')
    fig.suptitle(f"{', '.join(args.experiment)}: LR transfer along {args.by}" + (f" ({', '.join(args.fix)})" if args.fix else ''))
    fig.tight_layout()
    save(fig, args.out, 'lr_transfer_' + suffix(args))


def best_loss(args):
    curves = group_curves(select(load_results(args.results), args.experiment, parse_fix(args.fix)))
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for lab in by_alpha(k[1] for k in curves):
        keys = sorted([k for k in curves if k[1] == lab], key=lambda k: k[IDX[args.by]])
        xs, ys, es = [], [], []
        for k in keys:
            lrs, m, s, n = seed_mean_curve(curves[k])
            i = int(np.argmin(m))
            xs.append(k[IDX[args.by]]); ys.append(m[i]); es.append(s[i])
        ax.errorbar(xs, ys, yerr=es, marker='o', ms=4, capsize=2, label=label_text(lab))
    if args.by != 'init_std':
        ax.set_xscale('log', base=2)
    ax.set_xlabel(args.by); ax.set_ylabel('best final val loss (seed mean at argmin LR)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"{', '.join(args.experiment)}: best loss vs {args.by}")
    fig.tight_layout()
    save(fig, args.out, 'best_loss_' + suffix(args))


def load_optima(path, experiments, fix):
    with open(path) as f:
        return select(list(csv.DictReader(f)), experiments, fix)


def optimum(args):
    rows = load_optima(args.optima, args.experiment, parse_fix(args.fix))
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for lab in by_alpha(r['label'] for r in rows):
        rs = sorted([r for r in rows if r['label'] == lab], key=lambda r: float(r[args.by]))
        x = [float(r[args.by]) for r in rs]
        y = [float(r['lr_opt_log2']) for r in rs]
        # no bar when the interval is empty (one seed per learning rate)
        lo = np.nan_to_num([float(r['lr_opt_log2']) - float(r['lr_opt_lo']) for r in rs])
        hi = np.nan_to_num([float(r['lr_opt_hi']) - float(r['lr_opt_log2']) for r in rs])
        line = ax.errorbar(x, y, yerr=[lo, hi], marker='o', ms=4, capsize=2, label=label_text(lab))
        edge = [r.get('at_edge') == 'True' for r in rs]
        ax.scatter([a for a, e in zip(x, edge) if e], [b for b, e in zip(y, edge) if e], s=80,
                   facecolors='none', edgecolors=line[0].get_color(), zorder=3)
    if args.by != 'init_std':
        ax.set_xscale('log', base=2)
    ax.set_xlabel(args.by); ax.set_ylabel('optimal base LR (log2)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"{', '.join(args.experiment)}: optimum vs {args.by} (flat = transfer)\n"
                 "bars: 95% bootstrap over seeds; circled: best LR at the grid edge", fontsize=9)
    fig.tight_layout()
    save(fig, args.out, 'optimum_' + suffix(args))


def heatmap(args):
    rows = load_optima(args.optima, args.experiment, parse_fix(args.fix))
    labels = by_alpha(r['label'] for r in rows)
    if not labels:
        raise SystemExit('no matching optima')
    Ls = sorted({int(r['L']) for r in rows}); Ns = sorted({int(r['N']) for r in rows})
    fig, axes = plt.subplots(2, len(labels), figsize=(3.6 * len(labels), 6), squeeze=False)
    for j, lab in enumerate(labels):
        for i, (field, title) in enumerate((('lr_opt_log2', 'optimal base LR (log2)'), ('loss_opt', 'best loss'))):
            grid = np.full((len(Ls), len(Ns)), np.nan)
            for r in rows:
                if r['label'] == lab:
                    grid[Ls.index(int(r['L'])), Ns.index(int(r['N']))] = float(r[field])
            ax = axes[i][j]
            im = ax.imshow(grid, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns); ax.set_yticks(range(len(Ls))); ax.set_yticklabels(Ls)
            ax.set_xlabel('N'); ax.set_ylabel('L'); ax.set_title(f"{label_text(lab)}: {title}", fontsize=9)
            for a in range(len(Ls)):
                for b in range(len(Ns)):
                    if np.isfinite(grid[a, b]):
                        ax.text(b, a, f"{grid[a, b]:.2f}", ha='center', va='center', fontsize=7, color='w')
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, args.out, 'heatmap_' + suffix(args))


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn, src in (('lr_transfer', lr_transfer, 'results'), ('best_loss', best_loss, 'results'),
                          ('optimum', optimum, 'optima'), ('heatmap', heatmap, 'optima')):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument(f'--{src}', required=True)
        s.add_argument('--experiment', nargs='+', required=True)
        if name != 'heatmap':
            s.add_argument('--by', choices=AXES, default='N')
        s.add_argument('--fix', nargs='*', help='e.g. N=8 L=4 alpha_N=1')
        s.add_argument('--out', default='figures')
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
