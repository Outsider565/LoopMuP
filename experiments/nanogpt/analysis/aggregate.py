#!/usr/bin/env python3
"""
Collect done.json files into one results.csv (one row per run and label).

    python analysis/aggregate.py --out_root $LOOPMUP_OUT --experiments loop depth \
        --manifests outputs/manifests/loop.jsonl outputs/manifests/depth.jsonl --out outputs/results.csv

Runs merged by the manifest dedupe (identical training under several alpha_N, e.g. at N=1) are expanded into
one row per alias, marked alias_of=<executed run>.
"""

import argparse
import csv
import glob
import json
import math
import os

COLUMNS = ['experiment', 'label', 'run_name', 'alias_of', 'd', 'L', 'N', 'alpha_N', 'alpha_L', 'lr_log2', 'lr',
           'seed', 'init_std', 'train_tokens', 'status', 'final_val_loss', 'best_val_loss', 'final_train_loss',
           'tokens_per_second', 'mfu', 'wall_seconds', 'device', 'commit', 'micro_batch']


def load_manifest_index(paths):
    idx = {}
    for p in paths or []:
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    idx[r['run_name']] = r
    return idx


def rows_for(done, cfg, manifest_entry=None):
    c = cfg['config']
    lr = c['learning_rate']
    row = dict(experiment=(c.get('tag') or done['run_name']).split('/')[0], label=f"aN{c['alpha_N']:g}",
               run_name=done['run_name'], alias_of='', d=c['n_embd'], L=c['n_layer'], N=c['n_loops'],
               alpha_N=c['alpha_N'], alpha_L=c['alpha_L'], lr_log2=round(math.log2(lr), 3) if lr > 0 else float('nan'),
               lr=lr, seed=c['seed'], init_std=c['init_std'], train_tokens=c['train_tokens'],
               status=done.get('status'), final_val_loss=done.get('final_val_loss'),
               best_val_loss=done.get('best_val_loss'), final_train_loss=done.get('final_train_loss'),
               tokens_per_second=done.get('tokens_per_second'), mfu=done.get('mfu'),
               wall_seconds=done.get('wall_seconds'), device=done.get('device_name'),
               commit=(done.get('git') or {}).get('commit'), micro_batch=c.get('micro_batch'))
    aliases = (manifest_entry or {}).get('aliases', [])
    return [row] + [dict(row, run_name=a['run_name'], alias_of=done['run_name'], label=a['label'], alpha_N=a['alpha_N'])
                    for a in aliases]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_root', default=os.environ.get('LOOPMUP_OUT', 'out'))
    ap.add_argument('--experiments', nargs='+', required=True)
    ap.add_argument('--manifests', nargs='*', default=[], help='manifests whose aliases are expanded into rows')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    index = load_manifest_index(args.manifests)
    rows = []
    for experiment in args.experiments:
        for done_path in sorted(glob.glob(os.path.join(args.out_root, experiment, '*', 'done.json'))):
            run_dir = os.path.dirname(done_path)
            with open(done_path) as f:
                done = json.load(f)
            with open(os.path.join(run_dir, 'config.json')) as f:
                cfg = json.load(f)
            rows.extend(rows_for(done, cfg, index.get(done['run_name'])))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})
    by_status = {}
    for r in rows:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    print(f"{len(rows)} rows -> {args.out}; status counts {by_status}")


if __name__ == '__main__':
    main()
