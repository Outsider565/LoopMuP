#!/usr/bin/env python3
"""
Residual-stream norm along the loop passes, the analogue of the paper's Figure 6: one trajectory per run directory.

    python diag/hidden_norm.py --run_dirs $LOOPMUP_OUT/loop/d512_L4_N{2,4,8}_aN1_lr-7.5_s1 --data_dir $LOOPMUP_DATA --out outputs/figures/norm

x-axis: loop pass (0 = embedding output, n = after pass n); y-axis: per-token L2 norm of the residual stream
before the final LayerNorm, averaged over tokens and batches. The per-block trajectory is recorded as well.
Uses ckpt_final.pt (train with --save_final_checkpoint=True, e.g. run_manifest.py --extra=--save_final_checkpoint=True)
or `--weights init`. Writes <out>/<name>.json and .png.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diag.common import load_run, real_batches, hidden_states, loop_boundaries  # noqa: E402


def l2(h):
    return float(h.norm(dim=-1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dirs', nargs='+', required=True)
    ap.add_argument('--weights', default='auto')
    ap.add_argument('--data_dir', default=os.environ.get('LOOPMUP_DATA'))
    ap.add_argument('--n_batches', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='figures/norm')
    ap.add_argument('--name', default='hidden_norm')
    args = ap.parse_args()

    results = []
    for rd in args.run_dirs:
        model, cfg, wpath = load_run(rd, args.weights, args.device)
        L, N = model.config.n_layer, model.config.n_loops
        per_pass, per_block = None, None
        for X, Y in real_batches(cfg, args.data_dir, args.n_batches, args.batch_size, args.device):
            hidden = hidden_states(model, X, Y)
            pp = [l2(h) for h in loop_boundaries(hidden, L, N)]
            pb = [l2(h) for h in hidden]
            per_pass = pp if per_pass is None else [a + b for a, b in zip(per_pass, pp)]
            per_block = pb if per_block is None else [a + b for a, b in zip(per_block, pb)]
        results.append(dict(run_dir=rd, weights=wpath, d=model.config.n_embd, L=L, N=N, alpha_N=model.config.alpha_N,
                            per_pass=[v / args.n_batches for v in per_pass],
                            per_block=[v / args.n_batches for v in per_block]))
        print(f"{os.path.basename(rd.rstrip('/'))}: L={L} N={N} L2 norm per pass: "
              + ' '.join(f'{v:.3g}' for v in results[-1]['per_pass']))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.name + '.json'), 'w') as f:
        json.dump(results, f, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for r in results:
        ax.plot(range(len(r['per_pass'])), r['per_pass'], marker='o', ms=3, label=f"L={r['L']} N={r['N']}")
    ax.set_xlabel('pass'); ax.set_ylabel(r'residual-stream norm $\|x\|_2$'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, args.name + '.png'), dpi=150)


if __name__ == '__main__':
    main()
