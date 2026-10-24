#!/usr/bin/env python3
"""
Pairwise cosine similarity between residual-stream increments, the analogue of the paper's Figures 3 (left) and 5.

    python diag/loop_cosine.py --run_dir $LOOPMUP_OUT/loop/d512_L4_N8_aN1_lr-7.5_s1 --data_dir $LOOPMUP_DATA --out outputs/figures/cos
    python diag/loop_cosine.py --run_dir ... --weights init          # at initialization (Figure 3, left)
    python diag/loop_cosine.py --run_dir ... --level block           # every block application, not only loop passes

Increments at loop level: delta_n = h(after pass n) - h(after pass n-1), with h(after pass -1) = embedding output
(Figure 5 b-d after training). Increments at block level: one per (pass, layer) application, giving an
(N*L) x (N*L) matrix; with N=1 these are the per-layer increments of an untied model (Figure 5 a). Cosines are
computed per token and averaged over tokens and batches. Trained weights come from ckpt_final.pt (train with
--save_final_checkpoint=True). Writes <out>/<name>.json (matrix, off-diagonal stats) and <out>/<name>.png.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diag.common import load_run, real_batches, hidden_states, loop_boundaries, pairwise_cosine  # noqa: E402


def increments(hidden, L, N, level):
    if level == 'loop':
        b = loop_boundaries(hidden, L, N)
        return [b[i + 1] - b[i] for i in range(N)]
    return [hidden[i + 1] - hidden[i] for i in range(N * L)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--weights', default='auto', help="auto | init | file name inside run_dir")
    ap.add_argument('--data_dir', default=os.environ.get('LOOPMUP_DATA'))
    ap.add_argument('--n_batches', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--level', choices=['loop', 'block'], default='loop')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='figures/cos')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    model, cfg, wpath = load_run(args.run_dir, args.weights, args.device)
    L, N = model.config.n_layer, model.config.n_loops
    M = None
    for X, Y in real_batches(cfg, args.data_dir, args.n_batches, args.batch_size, args.device):
        hidden = hidden_states(model, X, Y)
        m = pairwise_cosine(increments(hidden, L, N, args.level))
        M = m if M is None else M + m
    M = (M / args.n_batches).numpy()
    n = M.shape[0]
    off = M[~torch.eye(n, dtype=torch.bool).numpy()] if n > 1 else M.flatten()
    name = args.name or f"{os.path.basename(args.run_dir.rstrip('/'))}_{args.level}_{'trained' if wpath else 'init'}"
    os.makedirs(args.out, exist_ok=True)
    result = dict(run_dir=args.run_dir, weights=wpath, level=args.level, L=L, N=N, matrix=M.tolist(),
                  offdiag_mean=float(off.mean()), offdiag_min=float(off.min()), offdiag_max=float(off.max()),
                  adjacent_mean=float(sum(M[i, i + 1] for i in range(n - 1)) / max(1, n - 1)))
    with open(os.path.join(args.out, name + '.json'), 'w') as f:
        json.dump(result, f, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.4))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title(f"{args.level}-level increment cosine, L={L}, N={N}\noff-diag mean {result['offdiag_mean']:.3f} "
                 f"[{result['offdiag_min']:.2f}, {result['offdiag_max']:.2f}]", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, name + '.png'), dpi=150)
    print(json.dumps({k: v for k, v in result.items() if k != 'matrix'}, indent=1))


if __name__ == '__main__':
    main()
