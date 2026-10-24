"""
Loop-μP training entry point (single device or DDP).

    python train.py config/base.py --run_name=base --learning_rate=0.00390625
    torchrun --standalone --nproc_per_node=8 train.py config/loop.py --run_name=loop8 --n_loops=8

Token-budgeted: the number of optimizer steps is train_tokens // batch_tokens and gradient accumulation
is derived from micro_batch, block_size and the world size. Every run writes to out_root/run_name/:
config.json, metrics.jsonl, done.json (also on divergence, OOM or a failure during training; not when the
setup before training fails) and optional checkpoints.
Set coord_check=True for the coordinate check (writes coord.jsonl instead of training).
"""

import json
import math
import os
import subprocess
import time
import traceback
from contextlib import nullcontext
from dataclasses import fields

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPTConfig, GPT, detect_peak_flops
from lr_schedule import scheduled_lr
from data.sharded_loader import SequentialLoader, FixedEvalSet

# -----------------------------------------------------------------------------
# I/O
out_root = os.environ.get('LOOPMUP_OUT', 'out')
run_name = 'debug'
tag = ''                      # free-form label written to done.json
resume = True                 # resume from out_dir/ckpt_last.pt when present
save_final_checkpoint = False # write ckpt_final.pt (model weights + config) at the end
ckpt_every_frac = 0.0         # periodic ckpt_last.pt every this fraction of steps; 0 disables
log_interval = 10
diverge_loss = 6.0            # final val loss above this (or non-finite loss) => status 'diverged'
deterministic = False         # deterministic kernels, for reproducibility checks
# data
data_dir = os.environ.get('LOOPMUP_DATA', 'data/finewebedu10B')
train_pattern = 'finewebedu_train_*.bin'
val_pattern = 'finewebedu_val_*.bin'
# budget
train_tokens = 2 ** 29
batch_tokens = 2 ** 18
micro_batch = 32              # sequences per device per micro step
block_size = 1024
eval_every_frac = 0.1         # intermediate evals every this fraction of steps (0 disables)
eval_tokens_interval = 2 ** 22
eval_tokens_final = 2 ** 24
eval_micro_batch = 32
# model
n_layer = 4
n_head = 8
n_embd = 512
vocab_size = 50304
dropout = 0.0
bias = False
init_std = 0.02               # sigma
mlp_act = 'relu2'
untie_embeddings = True
# Loop-μP: residual multiplier branch_mult * n_loops^-alpha_N * m_L^-alpha_L,
# m_n = n_embd / base_width, m_L = n_layer / base_depth
base_width = 256
base_depth = 2
n_loops = 1
alpha_N = 1.0
alpha_L = 0.5
branch_mult = 1.0
loop_activation_checkpointing = False
# optimizer (base values eta, omega, eps at the base shape)
learning_rate = 2 ** -7
lr_schedule = 'wsd'           # wsd | cosine
warmup_frac = 0.025
decay_frac = 0.05             # wsd: final linear decay to min_lr
min_lr = 0.0
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0               # 0 disables
adam_eps = 1e-8
# coordinate check
coord_check = False
coord_steps = 10
coord_save_params = False     # save params_step0.pt / params_stepK.pt (for diag/laziness.py)
# system
seed = 1
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
peak_flops = 0.0              # 0 => detect from the GPU name (MFU is null when unknown)
wandb_log = False
wandb_project = 'loop-mup'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(out_root, run_name)

if deterministic:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank, ddp_local_rank, ddp_world_size, master_process = 0, 0, 1, True
device_type = 'cuda' if 'cuda' in device else 'cpu'

if coord_check:
    assert not ddp, "coord_check runs on a single device"
    compile = False

tokens_per_micro = micro_batch * block_size * ddp_world_size
assert batch_tokens % tokens_per_micro == 0, f"batch_tokens {batch_tokens} not divisible by micro_batch*block_size*world {tokens_per_micro}"
grad_accum = batch_tokens // tokens_per_micro
iters = train_tokens // batch_tokens
warmup_iters = int(round(warmup_frac * iters))
decay_iters = int(round(decay_frac * iters))
eval_interval = max(1, int(round(eval_every_frac * iters))) if eval_every_frac > 0 else 0
ckpt_interval = max(1, int(round(ckpt_every_frac * iters))) if ckpt_every_frac > 0 else 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"run {run_name}: {iters} steps x {batch_tokens:,} tokens (grad_accum {grad_accum}, world {ddp_world_size}), out {out_dir}")
torch.manual_seed(seed + ddp_rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' or dtype == 'float32' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def git_info():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip() != ''
        return dict(commit=commit, dirty=dirty)
    except Exception:
        return dict(commit=None, dirty=None)


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


class JsonlLogger:
    def __init__(self, path, enabled):
        self.f = open(path, 'a') if enabled else None

    def log(self, **row):
        if self.f is not None:
            self.f.write(json.dumps(row) + '\n')
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()


# ----------------------------------------------------------------------------- model
model_args = {f.name: config[f.name] for f in fields(GPTConfig)}
model = GPT(GPTConfig(**model_args))
model.to(device)
raw_model = model
param_counts = model.count_params()
flops_per_token = model.flops_per_token()
peak = peak_flops if peak_flops > 0 else detect_peak_flops()
if master_process:
    print("parameters: total %.2fM, body %.2fM, embedding %.2fM, unembedding %.2fM" % tuple(
        param_counts[k] / 1e6 for k in ('total', 'body', 'embedding', 'unembedding')))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), adam_eps, device_type)
parameterization = model.parameterization_summary(learning_rate, adam_eps, weight_decay)

iter_num = 0
best_val_loss = float('inf')
ckpt_path = os.path.join(out_dir, 'ckpt_last.pt')
if resume and os.path.exists(ckpt_path) and not coord_check:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    if master_process:
        print(f"resumed from {ckpt_path} at step {iter_num}")
    checkpoint = None

if compile:
    if master_process:
        print("compiling the model...")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# ----------------------------------------------------------------------------- data
train_loader = SequentialLoader(os.path.join(data_dir, train_pattern), micro_batch, block_size,
                                rank=ddp_rank, world_size=ddp_world_size,
                                start_step=iter_num * grad_accum, device=device)


def evaluate(n_tokens):
    """Mean next-token loss over the first n_tokens of the validation shards (all ranks participate)."""
    evalset = FixedEvalSet(os.path.join(data_dir, val_pattern), n_tokens, eval_micro_batch, block_size,
                           rank=ddp_rank, world_size=ddp_world_size, device=device)
    model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for X, Y in evalset:
            with ctx:
                _, loss = model(X, Y)
            total[0] += loss.double()
            total[1] += 1
    if ddp:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return (total[0] / total[1]).item(), int(total[1].item()) * eval_micro_batch * block_size


def get_lr(it):
    return scheduled_lr(it, iters, warmup_iters, decay_iters, learning_rate, min_lr, lr_schedule)


def set_lr(lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * group['lr_scale']


def save_checkpoint(path):
    checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss, 'config': config}
    torch.save(checkpoint, path + '.tmp')
    os.replace(path + '.tmp', path)


def finish(status, **extra):
    if not master_process:
        return
    done = dict(run_name=run_name, tag=tag, status=status, iters_planned=iters, params=param_counts,
                flops_per_token=flops_per_token, device_name=torch.cuda.get_device_name() if device_type == 'cuda' else 'cpu',
                world_size=ddp_world_size, git=git_info(), config=config,
                parameterization=parameterization, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    done.update(extra)
    write_json(os.path.join(out_dir, 'done.json'), done)
    print(f"done: status={status} " + ' '.join(f"{k}={v}" for k, v in extra.items() if not isinstance(v, (dict, list))))


if master_process:
    write_json(os.path.join(out_dir, 'config.json'), dict(config=config, model_args=model_args, iters=iters,
                                                          grad_accum=grad_accum, world_size=ddp_world_size,
                                                          parameterization=parameterization))
metrics = JsonlLogger(os.path.join(out_dir, 'metrics.jsonl'), master_process and not coord_check)
wandb_run = None
if master_process and wandb_log and not coord_check:
    import wandb
    wandb_run = wandb.init(project=wandb_project, name=run_name, config=config)


# ----------------------------------------------------------------------------- coordinate check
def run_coord_check():
    """
    coord_steps optimizer steps at constant base LR on real batches. Per step, records abs().mean() of the
    embedding output, every (loop, layer) residual stream, every attention / MLP branch output and the logits,
    plus the RMS change of the final pre-LN residual on a fixed probe batch caused by that step.
    """
    from functools import partial
    branch_calls = {'attn': [], 'mlp': []}

    def hook(module, inp, out, key):
        branch_calls[key].append(out.detach().float().abs().mean().item())

    handles = []
    for blk in raw_model.transformer.h:
        handles.append(blk.attn.register_forward_hook(partial(hook, key='attn')))
        handles.append(blk.mlp.register_forward_hook(partial(hook, key='mlp')))

    probe_X, probe_Y = train_loader.next_batch()
    if coord_save_params:
        torch.save(raw_model.state_dict(), os.path.join(out_dir, 'params_step0.pt'))
        torch.save({'X': probe_X.cpu(), 'Y': probe_Y.cpu()}, os.path.join(out_dir, 'probe_batch.pt'))
    log = JsonlLogger(os.path.join(out_dir, 'coord.jsonl'), True)
    L, N = n_layer, n_loops

    def probe_hidden():
        model.eval()
        with torch.no_grad(), ctx:
            _, _, hidden = model(probe_X, probe_Y, return_hidden=True)
        model.train()
        return hidden[-1].float()

    h_prev = probe_hidden()
    set_lr(learning_rate)
    for step in range(coord_steps + 1):
        # forward on a fresh training batch with hidden states recorded
        X, Y = train_loader.next_batch()
        branch_calls['attn'].clear(); branch_calls['mlp'].clear()
        with ctx:
            logits, loss, hidden = model(X, Y, return_hidden=True)
        act = dict(embedding=hidden[0].float().abs().mean().item(),
                   logits=logits.float().abs().mean().item(),
                   resid=[[hidden[1 + n * L + l].float().abs().mean().item() for l in range(L)] for n in range(N)],
                   attn=[[branch_calls['attn'][n * L + l] for l in range(L)] for n in range(N)],
                   mlp=[[branch_calls['mlp'][n * L + l] for l in range(L)] for n in range(N)],
                   resid_final_rms=hidden[-1].float().pow(2).mean().sqrt().item())
        row = dict(step=step, loss=loss.item(), act=act)
        if step < coord_steps:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            h_new = probe_hidden()
            delta = (h_new - h_prev)
            row['update'] = dict(delta_rms=delta.pow(2).mean().sqrt().item(),
                                 delta_rel=(delta.pow(2).mean().sqrt() / h_prev.pow(2).mean().sqrt().clamp_min(1e-12)).item(),
                                 h_rms=h_new.pow(2).mean().sqrt().item())
            h_prev = h_new
        log.log(**row)
    log.close()
    for h in handles:
        h.remove()
    if coord_save_params:
        torch.save(raw_model.state_dict(), os.path.join(out_dir, f'params_step{coord_steps}.pt'))
    finish('ok', mode='coord_check', coord_steps=coord_steps)


# ----------------------------------------------------------------------------- training loop
def train():
    global iter_num, best_val_loss
    t_start = time.time()
    tokens_at_start = iter_num * batch_tokens
    X, Y = train_loader.next_batch()
    running_mfu = None
    lossf = float('nan')
    t0 = time.time()
    status = 'ok'
    while iter_num < iters:
        lr = get_lr(iter_num)
        set_lr(lr)

        if eval_interval and iter_num > 0 and iter_num % eval_interval == 0:
            val_loss, n_eval = evaluate(eval_tokens_interval)
            if master_process:
                metrics.log(step=iter_num, tokens=iter_num * batch_tokens, val_loss=val_loss, eval_tokens=n_eval, lr=lr)
                if wandb_run: wandb_run.log(dict(step=iter_num, val_loss=val_loss))
                print(f"step {iter_num}: val loss {val_loss:.4f} ({n_eval:,} tokens)")
            best_val_loss = min(best_val_loss, val_loss)
        if ckpt_interval and iter_num > 0 and iter_num % ckpt_interval == 0 and master_process:
            save_checkpoint(ckpt_path)

        for micro_step in range(grad_accum):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum - 1)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / grad_accum
            X, Y = train_loader.next_batch()
            scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        elif (iter_num + 1) % log_interval == 0 or iter_num + 1 == iters:   # steps that are logged below
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        else:
            grad_norm = None
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time(); dt = t1 - t0; t0 = t1
        iter_num += 1
        if iter_num % log_interval == 0 or iter_num == iters:
            lossf = loss.item() * grad_accum   # last micro-batch loss, cheap approximation of the step loss
            mfu = raw_model.estimate_mfu(micro_batch * grad_accum, dt, peak) if iter_num > 5 else None
            if mfu is not None:
                running_mfu = mfu if running_mfu is None else 0.9 * running_mfu + 0.1 * mfu
            if master_process:
                metrics.log(step=iter_num, tokens=iter_num * batch_tokens, train_loss=lossf, lr=lr,
                            grad_norm=grad_norm, step_time=dt, mfu=running_mfu)
                if wandb_run: wandb_run.log(dict(step=iter_num, train_loss=lossf, lr=lr))
                mfu_str = 'n/a' if running_mfu is None else f"{running_mfu * 100:.1f}%"
                print(f"iter {iter_num}/{iters}: loss {lossf:.4f}, lr {lr:.3e}, {dt * 1000:.0f}ms, mfu {mfu_str}")
            diverged = not math.isfinite(lossf)
            if ddp:   # every rank must leave the loop together
                flag = torch.tensor(float(diverged), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                diverged = flag.item() > 0
            if diverged:
                status = 'diverged'
                break

    wall = time.time() - t_start
    tokens_seen = iter_num * batch_tokens
    final_val, n_eval = (float('nan'), 0)
    if status == 'ok':
        final_val, n_eval = evaluate(eval_tokens_final)
        best_val_loss = min(best_val_loss, final_val)
        if not math.isfinite(final_val) or final_val > diverge_loss:
            status = 'diverged'
        if master_process:
            metrics.log(step=iter_num, tokens=tokens_seen, val_loss=final_val, eval_tokens=n_eval, final=True)
            if save_final_checkpoint:
                torch.save({'model': raw_model.state_dict(), 'model_args': model_args, 'config': config,
                            'iter_num': iter_num}, os.path.join(out_dir, 'ckpt_final.pt'))
            if os.path.exists(ckpt_path) and not save_final_checkpoint:
                os.remove(ckpt_path)
    finish(status, final_val_loss=final_val, best_val_loss=best_val_loss, final_train_loss=lossf,
           final_eval_tokens=n_eval, iters_done=iter_num, tokens_seen=tokens_seen, wall_seconds=wall,
           tokens_per_second=(tokens_seen - tokens_at_start) / max(wall, 1e-9), mfu=running_mfu,
           peak_flops_assumed=peak)


try:
    if coord_check:
        run_coord_check()
    else:
        train()
except torch.cuda.OutOfMemoryError as e:
    finish('oom', error=str(e)[:2000], iters_done=iter_num)
    raise
except Exception as e:
    finish('failed', error=traceback.format_exc()[-4000:], iters_done=iter_num)
    raise
finally:
    metrics.close()
    if ddp:
        dist.destroy_process_group()
