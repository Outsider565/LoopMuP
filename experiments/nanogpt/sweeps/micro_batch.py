"""
Memory heuristic for choosing the per-device micro-batch (sequences) and whether to recompute
loop passes in backward. Constants are rough (bf16 autocast, flash SDPA, torch.compile); the runner
retries with a halved micro-batch on OOM, so this only needs to be a good first guess.
"""

BYTES_PER_TOKEN_PER_BLOCK_PER_DIM = 200   # saved activations of one block application, per token, per n_embd
LOGIT_BYTES_PER_TOKEN_PER_VOCAB = 6        # fp32 logits + grad + bf16 copy, per token, per vocab entry
STATIC_OVERHEAD_GB = 6                     # CUDA context, weights, optimizer state, compile workspace
DEFAULT_MEM_GB = 80.0                      # device memory assumed when it cannot be detected


def activation_bytes(seqs, T, d, L, N, V, checkpointing):
    block_apps = L if checkpointing else L * N
    act = seqs * T * block_apps * BYTES_PER_TOKEN_PER_BLOCK_PER_DIM * d
    if checkpointing:
        act += seqs * T * N * d * 4   # loop boundaries kept in fp32
    logits = seqs * T * V * LOGIT_BYTES_PER_TOKEN_PER_VOCAB
    return act + logits


def choose_micro_batch(d, L, N, T=1024, V=50304, seqs_per_rank=32, mem_gb=DEFAULT_MEM_GB, checkpointing=None, max_micro=256):
    """
    Largest power-of-two micro-batch that divides seqs_per_rank and fits into ~70% of device memory.
    checkpointing=None decides automatically (enable when N*L >= 64).
    Returns (micro_batch, checkpointing).
    """
    if checkpointing is None:
        checkpointing = (N * L) >= 64
    budget = (mem_gb - STATIC_OVERHEAD_GB) * 0.7 * 1e9
    mb = 1
    while mb * 2 <= min(seqs_per_rank, max_micro) and seqs_per_rank % (mb * 2) == 0:
        mb *= 2
    while mb > 1 and activation_bytes(mb, T, d, L, N, V, checkpointing) > budget:
        mb //= 2
    return mb, checkpointing


if __name__ == '__main__':
    import sys
    d, L, N = (int(x) for x in sys.argv[1:4])
    print(choose_micro_batch(d, L, N, seqs_per_rank=int(sys.argv[4]) if len(sys.argv) > 4 else 32))
