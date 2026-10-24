"""Shared loading / grouping helpers for analysis scripts (csv + numpy only)."""

import csv
import math
from collections import defaultdict

import numpy as np

GROUP_KEYS = ('experiment', 'label', 'd', 'L', 'N', 'init_std')   # one learning-rate curve per key


def load_results(path, statuses=('ok',)):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if statuses and r['status'] not in statuses:
                continue
            for k in ('d', 'L', 'N', 'seed', 'train_tokens'):
                r[k] = int(r[k])
            for k in ('alpha_N', 'alpha_L', 'lr_log2', 'lr', 'final_val_loss', 'best_val_loss', 'init_std'):
                r[k] = float(r[k]) if r[k] not in ('', 'None', None) else float('nan')
            rows.append(r)
    return rows


def group_curves(rows, keys=GROUP_KEYS, metric='final_val_loss'):
    """-> {group_key: {lr_log2: [loss per seed]}}"""
    curves = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if math.isfinite(r[metric]):
            curves[tuple(r[k] for k in keys)][r['lr_log2']].append(r[metric])
    return curves


def seed_mean_curve(curve):
    xs = sorted(curve)
    means = np.array([np.mean(curve[x]) for x in xs])
    stds = np.array([np.std(curve[x]) if len(curve[x]) > 1 else 0.0 for x in xs])
    ns = np.array([len(curve[x]) for x in xs])
    return np.array(xs), means, stds, ns


def fit_parabola(xs, ys, n_points=5):
    """Quadratic fit in log2-LR around the argmin; returns (x_opt, y_opt, ok). Falls back to the argmin."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        i = int(np.argmin(ys))
        return xs[i], ys[i], False
    i = int(np.argmin(ys))
    lo, hi = max(0, i - n_points // 2), min(len(xs), i + n_points // 2 + 1)
    lo = max(0, min(lo, hi - 3))
    hi = min(len(xs), max(hi, lo + 3))
    a, b, c = np.polyfit(xs[lo:hi], ys[lo:hi], 2)
    if a <= 0:
        return xs[i], ys[i], False
    x_opt = -b / (2 * a)
    x_opt = float(np.clip(x_opt, xs[lo], xs[hi - 1]))
    return x_opt, float(a * x_opt ** 2 + b * x_opt + c), True


def bootstrap_optimum(curve, n_boot=200, seed=0):
    """Resample seeds per LR with replacement, refit; returns (x_opt, x_lo, x_hi, y_opt, y_lo, y_hi)."""
    rng = np.random.default_rng(seed)
    xs = sorted(curve)
    x_opt, y_opt, _ = fit_parabola(xs, [np.mean(curve[x]) for x in xs])
    xo, yo = [], []
    for _ in range(n_boot):
        ys = [np.mean(rng.choice(curve[x], size=len(curve[x]), replace=True)) for x in xs]
        xb, yb, _ = fit_parabola(xs, ys)
        xo.append(xb); yo.append(yb)
    return (x_opt, float(np.percentile(xo, 2.5)), float(np.percentile(xo, 97.5)),
            y_opt, float(np.percentile(yo, 2.5)), float(np.percentile(yo, 97.5)))
