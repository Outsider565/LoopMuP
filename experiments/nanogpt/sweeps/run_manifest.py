#!/usr/bin/env python3
"""
Execute the runs of a manifest on one node, idempotently.

    python sweeps/run_manifest.py --manifest loop.jsonl --shard 0/4 --out_root $LOOPMUP_OUT --mode ddp --nproc 8
    python sweeps/run_manifest.py --manifest base.jsonl --mode pack --pack 8           # 8 single-GPU runs at a time

* Completed runs are skipped only when their saved request matches the current configuration.
  Use a different run name or output root when changing a configuration.
* micro_batch is chosen from sweeps/micro_batch.py unless the manifest sets it (device memory is assumed to be
  80 GB when it cannot be detected); on OOM the run is retried with half the micro-batch (up to --max_oom_retries
  times).
* ddp mode: one run at a time, `torchrun --standalone --nproc_per_node NPROC train.py ...`; coordinate checks
  always run in a single process.
  pack mode: up to --pack concurrent single-device runs, slot i on the i-th visible GPU (cycling when --pack
  exceeds the GPU count); use for tiny models.
* Per-run stdout/stderr -> train.log and live console; summary -> out_root/sweep_status/<manifest>.<shard>.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import threading
from ast import literal_eval

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from sweeps.micro_batch import DEFAULT_MEM_GB, choose_micro_batch  # noqa: E402


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_done(out_dir):
    p = os.path.join(out_dir, 'done.json')
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def detect_gpus():
    try:
        import torch
        return torch.cuda.device_count(), (torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0)
    except Exception:
        return 0, 0.0


def fmt(v):
    return str(v) if not isinstance(v, bool) else ('True' if v else 'False')


def build_cmd(run, nproc, mode, python, extra):
    a = dict(run['train_args'])
    overrides = [f"--{k}={fmt(v)}" for k, v in sorted(a.items())] + list(extra)
    train_py = os.path.join(REPO, 'train.py')
    if mode == 'ddp' and nproc > 1:
        return [python, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={nproc}', train_py] + overrides
    return [python, train_py] + overrides


def resolve_request(run, extra, out_root, data_dir):
    """Merge user overrides before scheduling; keep the request separate from OOM adjustments."""
    run = dict(run, train_args=dict(run['train_args']))
    a = run['train_args']
    for arg in extra:
        if not arg.startswith('--') or '=' not in arg:
            raise ValueError('--extra requires --key=value')
        key, value = arg[2:].split('=', 1)
        if key in ('run_name', 'out_root'):
            raise ValueError(f'Set {key} through the manifest or --out_root, not --extra')
        try:
            a[key] = literal_eval(value)
        except (ValueError, SyntaxError):
            a[key] = value
    a['run_name'] = run['run_name']
    a['out_root'] = out_root
    a['data_dir'] = data_dir or a.get('data_dir') or os.environ.get('LOOPMUP_DATA', 'data/finewebedu10B')
    a['data_dir'] = os.path.abspath(os.path.join(REPO, a['data_dir']))
    # Changes to defaults in train.py must also invalidate an earlier request.
    with open(os.path.join(REPO, 'train.py'), 'rb') as f:
        defaults_version = hashlib.sha256(f.read()).hexdigest()
    run['_request'] = dict(train_args=dict(a), defaults_version=defaults_version)
    return run


def validate_request(run, out_dir):
    """Refuse to skip or resume another experiment's artifacts, leaving them untouched."""
    path = os.path.join(out_dir, 'run_request.json')
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)
        if json.dumps(previous, sort_keys=True) != json.dumps(run['_request'], sort_keys=True):
            raise ValueError(f"{run['run_name']}: configuration differs from run_request.json; "
                             'choose a new run_name or --out_root')
    elif os.path.isdir(out_dir) and os.listdir(out_dir):
        raise ValueError(f"{run['run_name']}: existing artifacts have no run_request.json; "
                         'choose a new run_name or --out_root')


def prepare_run(run, world, mem_gb, checkpointing_override):
    a = run['train_args']
    T = a['block_size']
    seqs_per_rank = a['batch_tokens'] // (T * world)
    if seqs_per_rank < 1 or a['batch_tokens'] % (T * world) != 0:
        raise ValueError(f"{run['run_name']}: batch_tokens {a['batch_tokens']} not divisible by block_size*world {T * world}")
    if 'micro_batch' not in a:
        mb, ckpt = choose_micro_batch(a['n_embd'], a['n_layer'], a['n_loops'], T, a['vocab_size'],
                                      seqs_per_rank=seqs_per_rank, mem_gb=mem_gb or DEFAULT_MEM_GB,
                                      checkpointing=checkpointing_override)
        a['micro_batch'] = mb
        a.setdefault('loop_activation_checkpointing', ckpt)
    a['micro_batch'] = min(a['micro_batch'], seqs_per_rank)
    while seqs_per_rank % a['micro_batch'] != 0:
        a['micro_batch'] //= 2
    a.setdefault('eval_micro_batch', a['micro_batch'])
    return run


class LoggedProcess(subprocess.Popen):
    """Drain child output concurrently, including while pack mode polls other runs."""

    def __init__(self, cmd, log, env):
        self._relay = None
        super().__init__(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         cwd=REPO, env=env, text=True, encoding='utf-8', errors='replace')
        self._relay = threading.Thread(target=self._copy_output, args=(log,), daemon=True)
        self._relay.start()

    def _copy_output(self, log):
        with log, self.stdout:
            for line in self.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except OSError:
                    # A disconnected console must not stop draining the child's pipe.
                    pass

    def wait(self, timeout=None):
        rc = super().wait(timeout=timeout)
        if self._relay is not None:
            self._relay.join()
        return rc

    def poll(self):
        rc = super().poll()
        if rc is not None and self._relay is not None:
            self._relay.join()
        return rc


def launch(run, cmd, out_dir, env):
    if '_request' in run:
        validate_request(run, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    request_path = os.path.join(out_dir, 'run_request.json')
    if '_request' in run and not os.path.exists(request_path):
        with open(request_path, 'x') as f:
            json.dump(run['_request'], f, indent=2, sort_keys=True)
    # A retry must produce its own receipt; never reuse an earlier failure or success.
    done_path = os.path.join(out_dir, 'done.json')
    if os.path.exists(done_path):
        os.remove(done_path)
    child_env = dict(os.environ if env is None else env)
    child_env['PYTHONUNBUFFERED'] = '1'
    log_path = os.path.join(out_dir, 'train.log')
    log = open(log_path, 'a')
    try:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        log_start = os.path.getsize(log_path)
        proc = LoggedProcess(cmd, log, child_env)
    except BaseException:
        log.close()
        raise
    proc.log_start = log_start   # outcome() reads only this attempt's part of train.log
    return proc


def outcome(run, out_dir, rc, log_start=0):
    done = read_done(out_dir)
    if done is not None:
        status = done.get('status', 'failed')
        if rc != 0 and status in ('ok', 'diverged'):
            done.update(status='failed', training_status=status, process_returncode=rc)
            path = os.path.join(out_dir, 'done.json')
            with open(path + '.tmp', 'w') as f:
                json.dump(done, f, indent=1)
            os.replace(path + '.tmp', path)
            return 'failed'
        return status
    log_path = os.path.join(out_dir, 'train.log')
    if os.path.exists(log_path):
        with open(log_path, 'rb') as f:
            f.seek(log_start)
            tail = f.read()[-20000:].decode('utf-8', 'replace')
        if 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail:
            return 'oom'
    return 'failed'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--shard', default='0/1', help='i/k: run entries with index %% k == i')
    ap.add_argument('--out_root', default=os.environ.get('LOOPMUP_OUT', 'out'))
    ap.add_argument('--data_dir', default=os.environ.get('LOOPMUP_DATA'))
    ap.add_argument('--mode', choices=['ddp', 'pack'], default='ddp')
    ap.add_argument('--nproc', type=int, default=0, help='devices per ddp run (0 = all visible)')
    ap.add_argument('--pack', type=int, default=0, help='concurrent single-device runs in pack mode (0 = all visible)')
    ap.add_argument('--mem_gb', type=float, default=0.0, help='device memory for the micro-batch heuristic (0 = detect, 80 if unknown)')
    ap.add_argument('--checkpointing', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--max_oom_retries', type=int, default=3)
    ap.add_argument('--retry_failed', action='store_true', help='re-run entries whose last status was failed')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--extra', action='append', default=[],
                    help='an extra --key=value override for every run, written --extra=--key=value; repeat for several')
    ap.add_argument('--dry_run', action='store_true', help='print the train.py commands; run and write nothing')
    args = ap.parse_args()
    args.out_root = os.path.abspath(args.out_root)

    i, k = (int(x) for x in args.shard.split('/'))
    runs = [r for idx, r in enumerate(load_manifest(args.manifest)) if idx % k == i]
    n_gpu, mem_detected = detect_gpus()
    mem_gb = args.mem_gb or mem_detected
    nproc = args.nproc or max(1, n_gpu)
    pack = args.pack or max(1, n_gpu)
    world = nproc if args.mode == 'ddp' else 1
    ckpt_override = {'auto': None, 'on': True, 'off': False}[args.checkpointing]
    # Preflight every entry before writing status files or launching any process.
    unique_runs = {}
    for run in runs:
        run = resolve_request(run, args.extra, args.out_root, args.data_dir)
        run['_request']['checkpointing'] = args.checkpointing
        previous = unique_runs.get(run['run_name'])
        if previous is not None and previous['_request'] != run['_request']:
            raise ValueError(f"{run['run_name']}: conflicting configurations in the manifest")
        validate_request(run, os.path.join(args.out_root, run['run_name']))
        unique_runs[run['run_name']] = run
    runs = list(unique_runs.values())

    status_dir = os.path.join(args.out_root, 'sweep_status')
    if not args.dry_run:
        os.makedirs(status_dir, exist_ok=True)
    status_path = os.path.join(status_dir, f"{os.path.basename(args.manifest)}.{i}of{k}.json")
    summary = dict(manifest=args.manifest, shard=args.shard, host=os.uname().nodename, mode=args.mode,
                   nproc=nproc, started=time.strftime('%Y-%m-%dT%H:%M:%S'), runs={})
    # pack-mode slots map to the GPUs this process may use (respects a CUDA_VISIBLE_DEVICES set by the caller)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids = [s for s in visible.split(',') if s] if visible else [str(g) for g in range(n_gpu)]

    def nproc_of(run):   # coordinate checks always run in a single process
        return 1 if run['train_args'].get('coord_check') else nproc

    def flush():
        if args.dry_run:
            return
        with open(status_path + '.tmp', 'w') as f:
            json.dump(summary, f, indent=1)
        os.replace(status_path + '.tmp', status_path)

    def record(name, **kw):
        summary['runs'].setdefault(name, {}).update(kw)
        flush()

    pending = []
    for run in runs:
        out_dir = os.path.join(args.out_root, run['run_name'])
        done = read_done(out_dir)
        if done is not None and (done.get('status') in ('ok', 'diverged') or (done.get('status') == 'failed' and not args.retry_failed)):
            record(run['run_name'], status=done.get('status'), skipped=True)
            continue
        pending.append(prepare_run(run, min(world, nproc_of(run)), mem_gb, ckpt_override))
    print(f"{len(runs)} runs in shard {args.shard}; {len(pending)} to execute on {n_gpu} GPU(s), mode {args.mode}")

    def run_one(run, device_env):
        out_dir = os.path.join(args.out_root, run['run_name'])
        for attempt in range(args.max_oom_retries + 1):
            cmd = build_cmd(run, nproc_of(run) if args.mode == 'ddp' else 1, args.mode, args.python, [])
            if args.dry_run:
                print(' '.join(cmd))
                return 'dry_run'
            env = dict(os.environ, **device_env)
            t0 = time.time()
            proc = launch(run, cmd, out_dir, env)
            rc = proc.wait()
            st = outcome(run, out_dir, rc, proc.log_start)
            record(run['run_name'], status=st, rc=rc, attempt=attempt, micro_batch=run['train_args']['micro_batch'],
                   seconds=round(time.time() - t0, 1))
            if st == 'oom' and run['train_args']['micro_batch'] > 1 and attempt < args.max_oom_retries:
                run['train_args']['micro_batch'] //= 2
                run['train_args']['eval_micro_batch'] = run['train_args']['micro_batch']
                if not run['train_args'].get('loop_activation_checkpointing'):
                    run['train_args']['loop_activation_checkpointing'] = True
                print(f"OOM on {run['run_name']}; retrying with micro_batch {run['train_args']['micro_batch']}")
                continue
            return st
        return st

    if args.mode == 'ddp':
        for run in pending:
            print(f"-> {run['run_name']}")
            print(f"   {run_one(run, {})}")
    else:
        slots = {}   # gpu index -> (run, proc, t0)
        queue = list(pending)
        while queue or slots:
            for g in range(pack):
                if g in slots:
                    run, proc, t0 = slots[g]
                    if proc.poll() is None:
                        continue
                    st = outcome(run, os.path.join(args.out_root, run['run_name']), proc.returncode, proc.log_start)
                    record(run['run_name'], status=st, rc=proc.returncode, seconds=round(time.time() - t0, 1),
                           micro_batch=run['train_args']['micro_batch'])
                    print(f"   {run['run_name']}: {st}")
                    del slots[g]
                    if st == 'oom' and run['train_args']['micro_batch'] > 1 and run.setdefault('_oom', 0) < args.max_oom_retries:
                        run['_oom'] += 1
                        run['train_args']['micro_batch'] //= 2
                        run['train_args']['eval_micro_batch'] = run['train_args']['micro_batch']
                        run['train_args']['loop_activation_checkpointing'] = True
                        queue.insert(0, run)
                if g not in slots and queue:
                    run = queue.pop(0)
                    cmd = build_cmd(run, 1, 'pack', args.python, [])
                    if args.dry_run:
                        print(' '.join(cmd))
                        continue
                    gpu = gpu_ids[g % len(gpu_ids)] if gpu_ids else None
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu) if gpu is not None else dict(os.environ)
                    print(f"-> [gpu {gpu}] {run['run_name']}")
                    slots[g] = (run, launch(run, cmd, os.path.join(args.out_root, run['run_name']), env), time.time())
            time.sleep(0 if args.dry_run else 5)
            if args.dry_run and not queue:
                break

    summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    flush()
    counts = {}
    for r in summary['runs'].values():
        counts[r.get('status')] = counts.get(r.get('status'), 0) + 1
    print('summary:', counts, '->', status_path)
    return 1 if any(status not in ('ok', 'diverged', 'dry_run') for status in counts) else 0


if __name__ == '__main__':
    sys.exit(main())
