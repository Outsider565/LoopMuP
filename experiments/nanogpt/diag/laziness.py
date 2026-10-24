#!/usr/bin/env python3
"""
Relative distance from linearization (CompleteP Fig. 6 / Eq. 4) for the hidden states of a looped GPT.

Requires coordinate-check runs saved with --coord_save_params=True (params_step0.pt, params_step<K>.pt, probe_batch.pt),
written to their own output root so that finished coordinate checks without saved weights are not skipped:

    python sweeps/run_manifest.py --manifest outputs/manifests/coord.jsonl --mode pack --out_root $LOOPMUP_OUT/lazy \
        --extra=--coord_save_params=True
    python diag/laziness.py --run_dirs $LOOPMUP_OUT/lazy/coord/d256_L{2,4,8,16,32}_N1_aN1_lr-7.0_s1 --out outputs/figures/lazy

For each residual-stream position i (embedding, then every (loop, layer) application) it computes
    delta_h     = h_i(theta_K) - h_i(theta_0)
    delta_h_lin = J_i(theta_0) (theta_K - theta_0)        (forward-mode JVP)
    ratio_i     = ||delta_h - delta_h_lin|| / ||delta_h_lin||
    feat_i      = ||delta_h|| / ||h_i(theta_0)||           (relative feature change)
A layer is lazy when ratio -> 0 as the model grows. CompleteP finds that the ratio stays Theta(1) as L grows
under alpha_L = 1 and shrinks under alpha_L = 1/2 (the depth exponent of Loop-μP).
"""

import argparse
import json
import os
import sys

import torch
from torch.func import functional_call, jvp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diag.common import load_run  # noqa: E402


def analyse(run_dir, device):
    model, cfg, _ = load_run(run_dir, 'init', device)
    model.set_manual_attention()   # fused SDPA kernels do not support forward-mode AD
    sd0 = torch.load(os.path.join(run_dir, 'params_step0.pt'), map_location=device)
    k_files = [f for f in os.listdir(run_dir) if f.startswith('params_step') and f != 'params_step0.pt']
    if not k_files:
        raise FileNotFoundError(f'{run_dir}: no params_step<K>.pt (run coord check with --coord_save_params=True)')
    kf = sorted(k_files, key=lambda f: int(f[len('params_step'):-3]))[-1]
    sdK = torch.load(os.path.join(run_dir, kf), map_location=device)
    probe = torch.load(os.path.join(run_dir, 'probe_batch.pt'), map_location=device)
    X, Y = probe['X'].to(device), probe['Y'].to(device)
    names = [n for n, _ in model.named_parameters()]
    p0 = {n: sd0[n].float() for n in names}
    pK = {n: sdK[n].float() for n in names}
    dp = {n: pK[n] - p0[n] for n in names}

    def f(params):
        _, _, hidden = functional_call(model, params, (X, Y), {'return_hidden': True})
        return torch.stack([h.float() for h in hidden])

    with torch.no_grad():
        hK = f(pK)
    h0, dh_lin = jvp(f, (p0,), (dp,))
    dh = hK - h0
    flat = lambda t: t.reshape(t.shape[0], -1)
    ratio = (flat(dh - dh_lin).norm(dim=1) / flat(dh_lin).norm(dim=1).clamp_min(1e-12)).tolist()
    feat = (flat(dh).norm(dim=1) / flat(h0).norm(dim=1).clamp_min(1e-12)).tolist()
    L, N = model.config.n_layer, model.config.n_loops
    return dict(run_dir=run_dir, step_file=kf, d=model.config.n_embd, L=L, N=N,
                alpha_N=model.config.alpha_N, alpha_L=model.config.alpha_L,
                ratio=ratio, feature_change=feat, ratio_last=ratio[-1], feature_change_last=feat[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dirs', nargs='+', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='figures/lazy')
    ap.add_argument('--name', default='laziness')
    args = ap.parse_args()
    results = [analyse(rd, args.device) for rd in args.run_dirs]
    for r in results:
        print(f"{os.path.basename(r['run_dir'])}: L={r['L']} N={r['N']} aL={r['alpha_L']:g} aN={r['alpha_N']:g} "
              f"ratio_last={r['ratio_last']:.3f} feature_change_last={r['feature_change_last']:.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.name + '.json'), 'w') as f:
        json.dump(results, f, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for r in results:
        lab = f"L={r['L']} N={r['N']} aN={r['alpha_N']:g} aL={r['alpha_L']:g}"
        axes[0].plot(range(len(r['ratio'])), r['ratio'], marker='.', label=lab)
        axes[1].plot(range(len(r['feature_change'])), r['feature_change'], marker='.', label=lab)
    axes[0].set_title('relative distance from linearization'); axes[1].set_title('relative feature change')
    for ax in axes:
        ax.set_xlabel('residual-stream position (0 = embedding)'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, args.name + '.png'), dpi=150)


if __name__ == '__main__':
    main()
