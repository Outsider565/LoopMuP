"""
Sequential, deterministic loader for llm.c / modded-nanogpt token shards.

Shard format: 256 int32 header (magic 20240520, version 1, number of tokens) followed by
uint16 tokens. `kjj0/finewebedu10B-gpt2` and `karpathy/fineweb-edu-*-gpt2-token-shards` use it.

Every training step consumes one contiguous window of `world * B * T` tokens from the logical
concatenation of all shards (in sorted filename order); rank r reads the r-th block of `B*T + 1`
tokens of that window. The position depends only on the step index, so resuming is exact and every
run with the same shards sees the same token order regardless of seed or world size.
"""

import glob
import os

import numpy as np
import torch

HEADER_INT32 = 256
HEADER_BYTES = HEADER_INT32 * 4
MAGIC = 20240520
VERSION = 1


def read_header(path):
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INT32)
    if header.size < HEADER_INT32 or header[0] != MAGIC:
        raise ValueError(f"{path}: not an llm.c token shard (magic {header[0] if header.size else 'n/a'})")
    if header[1] != VERSION:
        raise ValueError(f"{path}: unsupported version {header[1]}")
    ntok = int(header[2])
    expected = HEADER_BYTES + 2 * ntok
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(f"{path}: size {actual} != header-implied {expected}")
    return ntok


def write_shard(path, tokens):
    """Write uint16 tokens in llm.c format (used by tests and by data/finewebedu10B/prepare_from_hf.py)."""
    tokens = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(HEADER_INT32, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, tokens.size
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


class ShardSet:
    def __init__(self, pattern):
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no shards match {pattern}")
        self.files = files
        self.lengths = [read_header(f) for f in files]
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.total = int(self.offsets[-1])
        self._mmaps = {}

    def _mmap(self, i):
        if i not in self._mmaps:
            self._mmaps[i] = np.memmap(self.files[i], dtype=np.uint16, mode='r', offset=HEADER_BYTES,
                                       shape=(self.lengths[i],))
        return self._mmaps[i]

    def read(self, start, n):
        """n tokens starting at logical index `start`, wrapping around the end of the corpus."""
        out = np.empty(n, dtype=np.uint16)
        filled = 0
        pos = start % self.total
        while filled < n:
            i = int(np.searchsorted(self.offsets, pos, side='right') - 1)
            local = pos - int(self.offsets[i])
            take = min(n - filled, self.lengths[i] - local)
            out[filled:filled + take] = self._mmap(i)[local:local + take]
            filled += take
            pos = (pos + take) % self.total
        return out


class SequentialLoader:
    def __init__(self, pattern, batch_size, block_size, rank=0, world_size=1, start_step=0, device='cpu'):
        self.shards = ShardSet(pattern)
        self.B, self.T = batch_size, block_size
        self.rank, self.world = rank, world_size
        self.step = start_step
        self.device = device
        self.window = self.world * self.B * self.T
        if self.window + 1 > self.shards.total:
            raise ValueError("corpus smaller than one global batch")

    def next_batch(self):
        start = self.step * self.window + self.rank * self.B * self.T
        buf = self.shards.read(start, self.B * self.T + 1).astype(np.int64)
        buf = torch.from_numpy(buf)
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.step += 1
        if 'cuda' in str(self.device):
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


class FixedEvalSet:
    """First `n_tokens` tokens of a shard set, split into non-overlapping (B, T) batches across ranks."""

    def __init__(self, pattern, n_tokens, batch_size, block_size, rank=0, world_size=1, device='cpu'):
        self.shards = ShardSet(pattern)
        n_tokens = min(n_tokens, self.shards.total - 1)
        self.B, self.T = batch_size, block_size
        n_seq = n_tokens // block_size
        self.n_batches_total = n_seq // batch_size
        self.rank, self.world = rank, world_size
        self.device = device
        # each rank evaluates batches rank, rank+world, ...; the caller averages across ranks
        self.my_batches = list(range(rank, self.n_batches_total, world_size))

    def __len__(self):
        return len(self.my_batches)

    def __iter__(self):
        for b in self.my_batches:
            start = b * self.B * self.T
            buf = torch.from_numpy(self.shards.read(start, self.B * self.T + 1).astype(np.int64))
            x = buf[:-1].view(self.B, self.T).to(self.device)
            y = buf[1:].view(self.B, self.T).to(self.device)
            yield x, y
