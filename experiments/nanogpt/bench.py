"""
Throughput benchmark: tokens/s and MFU for one shape, on one device or under torchrun.

    python bench.py --n_embd=512 --n_layer=4 --n_loops=8 --micro_batch=16
    torchrun --standalone --nproc_per_node=8 bench.py --n_embd=512 --n_layer=4 --n_loops=8 --micro_batch=16

Uses random tokens (no data needed) and prints one JSON line. The measured MFU can be passed to
`sweeps/make_manifest.py --mfu` for the GPU-hour estimate. MFU is null when the GPU peak is unknown;
set --peak_flops to provide it.
"""
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from model import GPTConfig, GPT, detect_peak_flops

# -----------------------------------------------------------------------------
n_embd = 512
n_head = 8
n_layer = 4
n_loops = 1
block_size = 1024
vocab_size = 50304
micro_batch = 16
loop_activation_checkpointing = False
mlp_act = 'relu2'
base_width = 256
base_depth = 2
steps = 20
warmup = 5                    # untimed steps
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
peak_flops = 0.0              # 0 => detect from the GPU name
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------
assert steps > warmup, 'steps must exceed warmup'

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    torch.cuda.set_device(device)
else:
    rank, world = 0, 1
torch.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' or dtype == 'float32' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

conf = GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                 n_loops=n_loops, mlp_act=mlp_act, base_width=base_width, base_depth=base_depth,
                 loop_activation_checkpointing=loop_activation_checkpointing)
model = GPT(conf).to(device)
raw = model
optimizer = model.configure_optimizers(0.1, 2 ** -8, (0.9, 0.95), 1e-8, device_type)
if compile:
    model = torch.compile(model)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
x = torch.randint(vocab_size, (micro_batch, block_size), device=device)
y = torch.randint(vocab_size, (micro_batch, block_size), device=device)
peak = peak_flops or detect_peak_flops()

times = []
for k in range(steps):
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with ctx:
        _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    if k >= warmup:
        times.append(time.time() - t0)
dt = sum(times) / len(times)
tokens_per_step = micro_batch * block_size * world
mfu = raw.estimate_mfu(micro_batch, dt, peak)
result = dict(n_embd=n_embd, n_layer=n_layer, n_loops=n_loops, micro_batch=micro_batch, world=world, dtype=dtype,
              compile=compile, checkpointing=loop_activation_checkpointing, step_ms=round(dt * 1000, 1),
              tokens_per_s=round(tokens_per_step / dt), tokens_per_s_per_device=round(tokens_per_step / dt / world),
              mfu=None if mfu is None else round(mfu, 4), peak_flops=peak,
              max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device_type == 'cuda' else None,
              device=torch.cuda.get_device_name() if device_type == 'cuda' else 'cpu')
if rank == 0:
    print(json.dumps(result))
if ddp:
    dist.destroy_process_group()
