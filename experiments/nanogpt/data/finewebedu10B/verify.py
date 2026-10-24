#!/usr/bin/env python3
"""
Verify a directory of llm.c token shards: header magic/version/size consistency, optional sha256 against
manifest.json (written by download.py or by hand: {"files": [{"path", "size", "sha256"}]}), token range < vocab.

    python data/finewebedu10B/verify.py --dir /path/to/finewebedu10B [--sha256] [--complete] [--out verify.json]

Exit code 1 on any problem. Manifest files that are absent (a partial download with --shards) are listed but are
a problem only with --complete. Also checks shards obtained in other ways; without manifest.json only the
structural checks run. The committed manifest.json describes the kjj0/finewebedu10B-gpt2 shards, so shards built
by prepare_from_hf.py belong in another directory.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.sharded_loader import read_header, HEADER_BYTES  # noqa: E402


def sha256_of(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--sha256', action='store_true', help='also hash every file (slow: ~20 GB)')
    ap.add_argument('--complete', action='store_true', help='require every file listed in manifest.json')
    ap.add_argument('--vocab_size', type=int, default=50257)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, '*.bin')))
    manifest = {}
    mpath = os.path.join(args.dir, 'manifest.json')
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = {e['path']: e for e in json.load(f)['files']}
    summary = dict(dir=args.dir, files=[], problems=[])
    if not files:
        summary['problems'].append('no .bin shards found')
    if not any('_train_' in os.path.basename(p) for p in files):
        summary['problems'].append('no training shards found')
    if not any('_val_' in os.path.basename(p) for p in files):
        summary['problems'].append('no validation shards found')
    total_tokens = 0
    for p in files:
        name = os.path.basename(p)
        row = dict(path=name, size=os.path.getsize(p))
        try:
            ntok = read_header(p)
            row['tokens'] = ntok
            total_tokens += ntok
            mm = np.memmap(p, dtype=np.uint16, mode='r', offset=HEADER_BYTES, shape=(ntok,))
            sample = np.concatenate([mm[:1_000_000], mm[-1_000_000:]]) if ntok > 2_000_000 else mm[:]
            row['max_token_sampled'] = int(sample.max())
            if row['max_token_sampled'] >= args.vocab_size:
                summary['problems'].append(f"{name}: token id {row['max_token_sampled']} >= vocab {args.vocab_size}")
        except ValueError as e:
            summary['problems'].append(str(e))
        m = manifest.get(name)
        if args.sha256 and not (m and m.get('sha256')):
            summary['problems'].append(f"{name}: no manifest SHA256 available")
        if m:
            if m['size'] != row['size']:
                summary['problems'].append(f"{name}: size {row['size']} != manifest {m['size']}")
            if args.sha256 and m.get('sha256'):
                row['sha256'] = sha256_of(p)
                row['sha256_ok'] = row['sha256'] == m['sha256']
                if not row['sha256_ok']:
                    summary['problems'].append(f"{name}: sha256 mismatch")
        summary['files'].append(row)
        print(f"{name}: {row.get('tokens', '?'):>12} tokens" + (f"  sha256 {'ok' if row.get('sha256_ok') else 'BAD'}" if 'sha256' in row else ''))
    missing = sorted(set(manifest) - {os.path.basename(p) for p in files})
    summary['missing'] = missing
    if missing and args.complete:
        summary['problems'].append(f"missing files listed in manifest: {missing}")
    elif missing:
        print(f"{len(missing)} file(s) of manifest.json not present (partial download; use --complete to require all)")
    summary.update(n_files=len(files), total_tokens=total_tokens,
                  n_train=sum('_train_' in f for f in files), n_val=sum('_val_' in f for f in files))
    print(f"{summary['n_files']} files ({summary['n_train']} train, {summary['n_val']} val), {total_tokens / 1e9:.3f}B tokens, "
          f"{len(summary['problems'])} problem(s)")
    for pr in summary['problems']:
        print('  PROBLEM:', pr)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=1)
    return 1 if summary['problems'] else 0


if __name__ == '__main__':
    sys.exit(main())
