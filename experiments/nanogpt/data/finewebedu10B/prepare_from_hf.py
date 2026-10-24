#!/usr/bin/env python3
"""
Fallback when the pre-tokenized shards are unavailable: tokenize HuggingFaceFW/fineweb-edu (sample-10BT) with the
GPT-2 tokenizer into 100M-token llm.c shards, following karpathy/build-nanogpt fineweb.py.

    python data/finewebedu10B/prepare_from_hf.py --out data/finewebedu10B_hf [--remote_name sample-10BT] [--shard_size 100000000]

Write to a directory other than data/finewebedu10B, whose manifest.json lists the pre-tokenized shards, and point
LOOPMUP_DATA (or --data_dir) at it. Requires `pip install datasets tiktoken` (or `pip install -e "../..[data]"`).
Every document is prefixed with <|endoftext|>; the first shard is the
validation shard (finewebedu_val_000000.bin), the rest are finewebedu_train_000001.bin ... . Roughly 30-60 min of
CPU time with 8+ processes; ~28 GB of parquet is downloaded by `datasets` on the way.
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.sharded_loader import write_shard  # noqa: E402

_enc = None


def tokenize(doc):
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding('gpt2')
    ids = [_enc._special_tokens['<|endoftext|>']] + _enc.encode_ordinary(doc['text'])
    arr = np.array(ids)
    assert (0 <= arr).all() and (arr < 2 ** 16).all()
    return arr.astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--remote_name', default='sample-10BT')
    ap.add_argument('--shard_size', type=int, default=100_000_000)
    ap.add_argument('--procs', type=int, default=max(1, os.cpu_count() // 2))
    args = ap.parse_args()
    from datasets import load_dataset
    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=args.remote_name, split='train')
    buf = np.empty(args.shard_size, dtype=np.uint16)
    count, shard_index = 0, 0

    def flush(n):
        nonlocal shard_index
        split = 'val' if shard_index == 0 else 'train'
        idx = 0 if shard_index == 0 else shard_index
        path = os.path.join(args.out, f'finewebedu_{split}_{idx:06d}.bin')
        write_shard(path, buf[:n])
        print(f'wrote {path} ({n} tokens)', flush=True)
        shard_index += 1

    with mp.Pool(args.procs) as pool:
        for tokens in pool.imap(tokenize, ds, chunksize=16):
            while count + len(tokens) >= args.shard_size:
                take = args.shard_size - count
                buf[count:count + take] = tokens[:take]
                flush(args.shard_size)
                tokens = tokens[take:]
                count = 0
            buf[count:count + len(tokens)] = tokens
            count += len(tokens)
    if count:
        flush(count)


if __name__ == '__main__':
    main()
