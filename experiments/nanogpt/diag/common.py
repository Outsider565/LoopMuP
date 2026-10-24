"""Shared helpers for diagnostics: build a model from a run directory, load weights, fetch real batches."""

import glob
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import GPT, GPTConfig  # noqa: E402
from data.sharded_loader import SequentialLoader  # noqa: E402


def load_run(run_dir, weights='auto', device='cpu'):
    """
    Returns (model, config_json). weights: 'auto' picks ckpt_final.pt > params_step*.pt (largest step) >
    ckpt_last.pt > none (random init from the run's seed); or an explicit file name inside run_dir; or 'init'.
    """
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    torch.manual_seed(int(cfg['config'].get('seed', 1)))
    model = GPT(GPTConfig(**cfg['model_args']))
    path = None
    if weights == 'auto':
        cands = [os.path.join(run_dir, 'ckpt_final.pt')]
        steps = sorted(glob.glob(os.path.join(run_dir, 'params_step*.pt')),
                       key=lambda p: int(p.rsplit('step', 1)[1].split('.')[0]))
        cands += steps[::-1] + [os.path.join(run_dir, 'ckpt_last.pt')]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            print(f'{run_dir}: no checkpoint found, using the initialization')
    elif weights not in ('init', None):
        path = weights if os.path.isabs(weights) else os.path.join(run_dir, weights)
    if path:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}
        model.load_state_dict(sd)
    model.to(device).eval()
    return model, cfg, path


def real_batches(cfg, data_dir, n_batches, batch_size, device='cpu', pattern=None):
    c = cfg['config']
    pattern = pattern or os.path.join(data_dir or c['data_dir'], c.get('val_pattern', 'finewebedu_val_*.bin'))
    loader = SequentialLoader(pattern, batch_size, c['block_size'], device=device)
    return [loader.next_batch() for _ in range(n_batches)]


def hidden_states(model, X, Y):
    with torch.no_grad():
        _, _, hidden = model(X, Y, return_hidden=True)
    return [h.float() for h in hidden]


def loop_boundaries(hidden, L, N):
    """Residual stream at the loop boundaries: [embedding, after pass 0, ..., after pass N-1]."""
    return [hidden[0]] + [hidden[1 + n * L + (L - 1)] for n in range(N)]


def pairwise_cosine(deltas):
    """deltas: list of (B, T, d). Returns the (n, n) matrix of per-token cosine similarities, averaged over tokens."""
    n = len(deltas)
    M = torch.zeros(n, n)
    flat = [d.reshape(-1, d.shape[-1]) for d in deltas]
    for i in range(n):
        for j in range(n):
            M[i, j] = torch.nn.functional.cosine_similarity(flat[i], flat[j], dim=-1).mean()
    return M
