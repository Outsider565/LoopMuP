"""
Experiment definitions. Each experiment expands to a list of run records; `make_manifest.py` writes them as
JSONL and `run_manifest.py` executes them with train.py.

Every run uses the parameterization of the paper's Table 1 with alpha_L = 1/2. Runs differ only in shape,
base learning rate, seed and the loop exponent alpha_N of the residual multiplier
branch_mult * N^-alpha_N * m_L^-alpha_L; alpha_N = 1 is Loop-μP. No learning rate or AdamW eps depends on N.

  coord  10-step coordinate checks: N in {1..64} x alpha_N in {0, 1/2, 1, 3/2} at d=256, L=2;
         L in {2..32} and d in {128..1024} at N=1
  base   learning-rate sweep at the base shape d=256, L=2, N=1, where every multiplier is 1
  loop   loop-factor comparison (paper Fig. 4a-d): d=512, L=4, N in {1,2,4,8}, alpha_N in {0, 1/2, 1, 3/2}
  depth  depth transfer under Loop-μP (paper Fig. 4d-f): d=512, L in {2,8,16}, N in {1,2,4,8}
         (the L=4 runs are part of `loop`)
  width  width transfer under Loop-μP: d in {256,512,1024}, L=4, N=4
  smoke  one ~100-step run
"""

import hashlib
import json

HEAD_DIM = 64
VOCAB_SIZE = 50304
TRAIN_TOKENS = 11781 * 2 ** 18   # 3.09B tokens, 11781 steps of 2^18 tokens
LOOP_MUP = 1.0                   # alpha_N of Loop-μP
ALPHA_N = (1.0, 0.0, 0.5, 1.5)   # Loop-μP first, so a run shared by all alpha_N (N=1) keeps the Loop-μP name
NEEDS_ETA = ('loop', 'depth', 'width')

COMMON = dict(
    block_size=1024, vocab_size=VOCAB_SIZE, batch_tokens=2 ** 18,
    base_width=256, base_depth=2, mlp_act='relu2', untie_embeddings=True, bias=False,
    weight_decay=0.1, adam_eps=1e-8, beta1=0.9, beta2=0.95,
    lr_schedule='wsd', warmup_frac=0.025, decay_frac=0.05, min_lr=0.0, grad_clip=0.0,
    eval_tokens_final=2 ** 24, eval_tokens_interval=2 ** 22, eval_every_frac=0.1, log_interval=10,
    init_std=0.02, branch_mult=1.0, dropout=0.0, alpha_L=0.5,
)

# Coordinate checks: one device, batch = micro_batch * block_size (no gradient accumulation).
COORD = dict(coord_check=True, coord_steps=10, micro_batch=8, batch_tokens=8 * 1024, train_tokens=64 * 8 * 1024,
             eval_every_frac=0.0, compile=False)


def lr_grid(center_log2, half_width, step=0.5):
    n = int(round(half_width / step))
    return [round(center_log2 + step * k, 3) for k in range(-n, n + 1)]


def effective_key(args):
    """Numeric identity of a run under Table 1: runs that train identically (all alpha_N at N=1) share a key."""
    N, mL = args['n_loops'], args['n_layer'] / args['base_depth']
    aN, aL = args['alpha_N'], args['alpha_L']
    ident = {k: v for k, v in args.items() if k not in ('run_name', 'tag', 'alpha_N', 'alpha_L', 'branch_mult')}
    ident.update(branch=round(args['branch_mult'] * N ** -aN * mL ** -aL, 12),   # residual multiplier
                 depth_lr=round(mL ** (aL - 1), 12),                            # pre-LN / hidden LR factor
                 depth_eps=round(mL ** -aL, 12))                                # residual-block eps factor
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def make_run(experiment, d, L, N, alpha_N, lr_log2, seed, train_tokens, extra=None, name_extra=''):
    alpha_N = float(alpha_N)
    label = f"aN{alpha_N:g}"
    run_name = f"{experiment}/d{d}_L{L}_N{N}_aN{alpha_N:g}{name_extra}_lr{lr_log2:+.1f}_s{seed}"
    args = dict(COMMON, n_embd=d, n_head=max(1, d // HEAD_DIM), n_layer=L, n_loops=N, alpha_N=alpha_N, seed=seed,
                learning_rate=float(2.0 ** lr_log2), train_tokens=int(train_tokens))
    args.update(extra or {})
    args.update(run_name=run_name, tag=f"{experiment}/{label}")
    return dict(run_name=run_name, experiment=experiment, label=label, tag=args['tag'], alpha_N=alpha_N,
                shape=dict(d=d, L=L, N=N), lr_log2=lr_log2, seed=seed, train_tokens=int(train_tokens),
                train_args=args, key=effective_key(args))


def dedupe(runs):
    """Keep the first run per effective key; it lists the other runs as aliases."""
    first, out = {}, []
    for r in runs:
        keep = first.get(r['key'])
        if keep is None:
            first[r['key']] = r
            out.append(r)
        elif r['run_name'] != keep['run_name'] and all(a['run_name'] != r['run_name'] for a in keep.get('aliases', [])):
            keep.setdefault('aliases', []).append(dict(run_name=r['run_name'], label=r['label'], alpha_N=r['alpha_N']))
    return out


def coord(eta_log2=-7.0, seeds=(1, 2, 3)):
    shapes = [(256, 2, N, aN) for N in (1, 2, 4, 8, 16, 32, 64) for aN in ALPHA_N]
    shapes += [(256, L, 1, LOOP_MUP) for L in (2, 4, 8, 16, 32)]
    shapes += [(d, 2, 1, LOOP_MUP) for d in (128, 256, 512, 1024)]
    return dedupe([make_run('coord', d, L, N, aN, eta_log2, s, COORD['train_tokens'], COORD)
                   for d, L, N, aN in shapes for s in seeds])


def base(seeds=(1,), sigmas=(0.02,), lrs=None):
    lrs = lrs or lr_grid(-8.0, 3.0)   # 2^-11 ... 2^-5
    return dedupe([make_run('base', 256, 2, 1, LOOP_MUP, lr, s, TRAIN_TOKENS, dict(init_std=float(sigma)),
                            name_extra=f"_sig{sigma:g}" if len(sigmas) > 1 else '')
                   for sigma in sigmas for lr in lrs for s in seeds])


def loop(eta_log2, seeds=(1,), sigma=0.02):
    return dedupe([make_run('loop', 512, 4, N, aN, lr, s, TRAIN_TOKENS, dict(init_std=float(sigma)))
                   for aN in ALPHA_N for N in (1, 2, 4, 8) for lr in lr_grid(eta_log2, 2.0) for s in seeds])


def depth(eta_log2, seeds=(1,), sigma=0.02):
    return dedupe([make_run('depth', 512, L, N, LOOP_MUP, lr, s, TRAIN_TOKENS, dict(init_std=float(sigma)))
                   for L in (2, 8, 16) for N in (1, 2, 4, 8) for lr in lr_grid(eta_log2, 1.5) for s in seeds])


def width(eta_log2, seeds=(1,), sigma=0.02):
    return dedupe([make_run('width', d, 4, 4, LOOP_MUP, lr, s, TRAIN_TOKENS, dict(init_std=float(sigma)))
                   for d in (256, 512, 1024) for lr in lr_grid(eta_log2, 1.5) for s in seeds])


def smoke(eta_log2=-7.0):
    return [make_run('smoke', 256, 2, 2, LOOP_MUP, eta_log2, 1, 100 * 2 ** 18,
                     dict(eval_tokens_final=2 ** 22, eval_every_frac=0.5, save_final_checkpoint=True))]


EXPERIMENTS = dict(coord=coord, base=base, loop=loop, depth=depth, width=width, smoke=smoke)


def flops_per_token(d, L, N, T=1024):
    """Approximate training FLOPs per token: looped block matmuls, unembedding and causal attention scores."""
    return 6 * (N * L * 12 * d * d + VOCAB_SIZE * d) + 6 * N * L * d * T


def estimate_gpu_hours(runs, peak_tflops=989.0, mfu=0.35):
    """GPU-hours of the training runs (coordinate checks are not counted)."""
    total = 0.0
    for r in runs:
        if r['train_args'].get('coord_check'):
            continue
        s = r['shape']
        total += flops_per_token(s['d'], s['L'], s['N'], r['train_args']['block_size']) * r['train_tokens']
    return total / (peak_tflops * 1e12 * mfu) / 3600
