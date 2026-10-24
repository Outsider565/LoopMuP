#!/usr/bin/env python3
"""
Download the GPT-2-tokenized FineWeb-Edu 10B shards (llm.c format) from the Hugging Face Hub and verify them.

    python data/finewebedu10B/download.py --out data/finewebedu10B            # all 99 train shards + 1 val shard (19.9 GB)
    python data/finewebedu10B/download.py --out data/finewebedu10B --shards 10 # first 10 train shards + val (~2.2 GB)
    python data/finewebedu10B/download.py --out data/finewebedu10B --manifest-only  # write manifest.json (names, sizes, sha256)

Source repo: https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2 (files finewebedu_train_000001..000099.bin,
finewebedu_val_000000.bin; 1024-byte header + 100M uint16 tokens each). Uses only the standard library; resumes
partial files; checks sha256 against the LFS object id listed by the Hub. Set HF_ENDPOINT to use another Hub endpoint.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO_ID = 'kjj0/finewebedu10B-gpt2'
ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')


def hub_tree(repo_id):
    url = f"{ENDPOINT}/api/datasets/{repo_id}/tree/main"
    with urllib.request.urlopen(url, timeout=60) as r:
        entries = json.load(r)
    files = []
    for e in entries:
        if e.get('type') != 'file' or not e['path'].endswith('.bin'):
            continue
        files.append(dict(path=e['path'], size=e['size'], sha256=(e.get('lfs') or {}).get('oid')))
    return sorted(files, key=lambda f: f['path'])


def sha256_of(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(repo_id, entry, out_dir):
    dst = os.path.join(out_dir, entry['path'])
    part = dst + '.part'
    if os.path.exists(dst) and os.path.getsize(dst) == entry['size']:
        return 'exists'
    url = f"{ENDPOINT}/datasets/{repo_id}/resolve/main/{entry['path']}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, 'ab' if have else 'wb') as f:
        if have and r.status != 206:
            f.seek(0); f.truncate()
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    if os.path.getsize(part) != entry['size']:
        raise IOError(f"{entry['path']}: size {os.path.getsize(part)} != {entry['size']}")
    if entry['sha256'] and sha256_of(part) != entry['sha256']:
        os.remove(part)
        raise IOError(f"{entry['path']}: sha256 mismatch, removed partial file")
    os.replace(part, dst)
    return 'downloaded'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=REPO_ID)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shards', type=int, default=0, help='number of train shards to fetch (0 = all)')
    ap.add_argument('--manifest-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = hub_tree(args.repo)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:   # the full listing, whatever --shards is
        json.dump(dict(repo=args.repo, files=files), f, indent=1)
        f.write('\n')
    total_gb = sum(e['size'] for e in files) / 1e9
    print(f"{len(files)} files, {total_gb:.1f} GB in {args.repo}; manifest -> {args.out}/manifest.json")
    if args.manifest_only:
        return
    train = [e for e in files if '_train_' in e['path']]
    val = [e for e in files if '_val_' in e['path']]
    wanted = (train[:args.shards] if args.shards else train) + val
    for i, e in enumerate(wanted, 1):
        st = download(args.repo, e, args.out)
        print(f"[{i}/{len(wanted)}] {e['path']} {e['size'] / 1e6:.0f} MB {st}", flush=True)
    print('done; run data/finewebedu10B/verify.py to re-check headers and hashes')


if __name__ == '__main__':
    sys.exit(main())
