#!/usr/bin/env python3
"""
Locate the optimal base learning rate per (experiment, label, d, L, N, init_std) with a quadratic fit in log2-LR
and a bootstrap over seeds.

    python analysis/fit_optimum.py --results outputs/results.csv --out outputs/optima.csv

Columns: experiment, label, alpha_N, d, L, N, init_std, n_lr, n_runs, lr_opt_log2, lr_opt_lo, lr_opt_hi, loss_opt,
loss_lo, loss_hi, lr_argmin_log2, loss_argmin, at_edge. The learning rate transfers along an axis when the optimum
stays within one grid step (0.5 in log2) across that axis. The 95% intervals (*_lo, *_hi) need several seeds per
learning rate (`make_manifest.py --seeds 1,2,3`) and are empty with one seed. at_edge marks a best grid point at
either end of the grid, where the true optimum may lie outside the grid.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import GROUP_KEYS, load_results, group_curves, bootstrap_optimum  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--metric', default='final_val_loss')
    ap.add_argument('--n_boot', type=int, default=200)
    args = ap.parse_args()
    rows = load_results(args.results)
    alpha_of = {r['label']: r['alpha_N'] for r in rows}
    curves = group_curves(rows, keys=GROUP_KEYS, metric=args.metric)
    out = []
    for key in sorted(curves):
        curve = curves[key]
        xs = sorted(curve)
        means = [np.mean(curve[x]) for x in xs]
        i = int(np.argmin(means))
        xo, xlo, xhi, yo, ylo, yhi = bootstrap_optimum(curve, args.n_boot)
        if max(len(v) for v in curve.values()) < 2:   # one seed per LR: resampling cannot give an interval
            xlo = xhi = ylo = yhi = float('nan')
        experiment, label, d, L, N, init_std = key
        out.append(dict(experiment=experiment, label=label, alpha_N=alpha_of[label], d=d, L=L, N=N, init_std=init_std,
                        n_lr=len(xs), n_runs=sum(len(v) for v in curve.values()),
                        lr_opt_log2=round(xo, 3), lr_opt_lo=round(xlo, 3), lr_opt_hi=round(xhi, 3),
                        loss_opt=round(yo, 5), loss_lo=round(ylo, 5), loss_hi=round(yhi, 5),
                        lr_argmin_log2=xs[i], loss_argmin=round(means[i], 5), at_edge=i in (0, len(xs) - 1)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else ['experiment'])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} groups -> {args.out}")
    for r in out:
        print(f"{r['experiment']:6s} {r['label']:6s} d{r['d']} L{r['L']:<3d} N{r['N']:<3d} sigma {r['init_std']:g}  "
              f"lr* 2^{r['lr_opt_log2']:+.2f} [{r['lr_opt_lo']:+.2f},{r['lr_opt_hi']:+.2f}]  "
              f"loss* {r['loss_opt']:.4f} [{r['loss_lo']:.4f},{r['loss_hi']:.4f}]  ({r['n_runs']} runs)"
              + ('  at grid edge' if r['at_edge'] else ''))


if __name__ == '__main__':
    main()
