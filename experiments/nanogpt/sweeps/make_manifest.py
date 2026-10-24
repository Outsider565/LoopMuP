#!/usr/bin/env python3
"""
Write an experiment's run list as JSONL (one run per line) and print its size and a GPU-hour estimate.

    python sweeps/make_manifest.py --experiment coord --out outputs/manifests/coord.jsonl
    python sweeps/make_manifest.py --experiment base --out outputs/manifests/base.jsonl
    python sweeps/make_manifest.py --experiment loop --eta_log2 -7.5 --out outputs/manifests/loop.jsonl
    python sweeps/make_manifest.py --experiment loop --eta_log2 -7.5 --out loop.jsonl --shards 8   # also loop.shard0..7.jsonl

`--eta_log2` is the base learning rate (log2) chosen from `base` and is required for loop, depth and width.
`--sigma` is the base init std (a comma-separated list for `base`). Manifests are deterministic: regenerate and
diff to review changes.
"""

import argparse
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweeps.experiments import EXPERIMENTS, NEEDS_ETA, estimate_gpu_hours  # noqa: E402


def write_jsonl(path, runs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for r in runs:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--eta_log2', type=float, default=None, help='base LR (log2) chosen from the base sweep')
    ap.add_argument('--sigma', type=str, default=None, help='base init std; comma-separated list for base')
    ap.add_argument('--seeds', type=str, default=None, help='comma-separated seeds, e.g. 1,2')
    ap.add_argument('--shards', type=int, default=1, help='also split into this many round-robin shard files')
    ap.add_argument('--peak_tflops', type=float, default=989.0, help='device peak bf16 TFLOPS for the estimate')
    ap.add_argument('--mfu', type=float, default=0.35)
    args = ap.parse_args()

    kw = {}
    if args.eta_log2 is not None:
        kw['eta_log2'] = args.eta_log2
    if args.sigma:
        sigmas = tuple(float(s) for s in args.sigma.split(','))
        if args.experiment == 'base':
            kw['sigmas'] = sigmas
        elif len(sigmas) == 1:
            kw['sigma'] = sigmas[0]
        else:
            ap.error('a list of --sigma values is only supported for --experiment base')
    if args.seeds:
        kw['seeds'] = tuple(int(s) for s in args.seeds.split(','))
    if args.experiment in NEEDS_ETA and args.eta_log2 is None:
        ap.error(f'--eta_log2 is required for --experiment {args.experiment}')
    fn = EXPERIMENTS[args.experiment]
    unsupported = sorted(set(kw) - set(inspect.signature(fn).parameters))
    if unsupported:
        ap.error(f'--experiment {args.experiment} does not take {", ".join(unsupported)}')
    runs = fn(**kw)

    names = [r['run_name'] for r in runs]
    assert len(names) == len(set(names)), 'duplicate run names'
    write_jsonl(args.out, runs)
    hours = estimate_gpu_hours(runs, args.peak_tflops, args.mfu)
    print(f"{args.experiment}: {len(runs)} runs -> {args.out}; est. {hours:.1f} GPU-h at "
          f"{args.peak_tflops:.0f} TFLOPS x {args.mfu:.0%} MFU")
    if args.shards > 1:
        base, ext = os.path.splitext(args.out)
        for i in range(args.shards):
            part = [r for k, r in enumerate(runs) if k % args.shards == i]
            write_jsonl(f"{base}.shard{i}{ext}", part)
            print(f"  shard {i}: {len(part)} runs, est. {estimate_gpu_hours(part, args.peak_tflops, args.mfu):.1f} GPU-h")


if __name__ == '__main__':
    main()
