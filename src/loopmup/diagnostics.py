"""Residual-scaling diagnostics behind Figures 2 and 3.

Model: a 12-layer Llama block (d=768, 12 heads, MLP 2048, vocab 128256, tied
embeddings, init std 0.02). A looped stack applies the same 12 layers K times,
and an untied stack has 12 K independent layers. Every attention and MLP branch
is multiplied by K^{-alpha}, so K is the loop count N of a looped stack and the
depth multiplier m_L = L/12 of an untied stack. The 12-layer block is the base
model: m_n = 1 for both stacks and m_L = 1 for the looped stack, so the
initialization of both stacks and the learning rate and epsilon of the looped
stack are the Table 1 values; only the residual multiplier changes with K.

Measurements per (family, alpha, K, seed) on one fixed sequence of random
tokens (batch 1):

* at initialization, the RMS of the residual stream after every 12-layer block
  (x^0 is the embedding output) and the cosine similarity between the block
  increments x^r - x^{r-1} (Figure 2 and Figure 3, left);
* ``steps`` AdamW steps at a learning rate shared across K, recording the RMS
  change of the final pre-norm residual stream between consecutive steps
  (Figure 3, right, which shows the looped stack).

The default learning rate 2e-6 keeps the one-step change proportional to the
learning rate; at a typical language-model rate such as 1.25e-3 the first AdamW
step already changes the residual stream by more than its own norm. The paper
states a shared base learning rate of 1e-4 for this measurement (Figure 9);
pass ``--lr 1e-4`` for that setting. The learning rate is shared across K in
both cases, which is what the comparison of loop factors requires.

Usage:

    loopmup-diagnostics run --family loop --alphas 0 0.5 1 1.5 --out runs
    loopmup-diagnostics run --family deep --alphas 0 0.5 1 1.5 --out runs
    loopmup-diagnostics summarize --in runs --out summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import loopmup

BLOCK_LAYERS = 12
KS = [1, 2, 4, 8, 16, 32, 64]


def make_config(num_layers: int, vocab: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation="sdpa",
    )


def build_model(family: str, k: int, alpha: float, seed: int, vocab: int,
                device):
    torch.manual_seed(seed)
    num_layers = BLOCK_LAYERS if family == "loop" else BLOCK_LAYERS * k
    with torch.device(device):
        model = LlamaForCausalLM(make_config(num_layers, vocab))
    if family == "loop":
        loopmup.loop_layers(model, k)
    loopmup.scale_residual_branches(model, float(k) ** (-alpha))
    return model


class Recorder:
    """Collects x^0 (embedding) and the output of every 12-layer block."""

    def __init__(self, model):
        self.states = []
        self.final = None
        model.model.embed_tokens.register_forward_hook(self._embed)
        layers = model.model.layers
        ends = [layers[i] for i in range(BLOCK_LAYERS - 1, len(layers), BLOCK_LAYERS)]
        for layer in {id(m): m for m in ends}.values():
            layer.register_forward_hook(self._block)
        model.model.norm.register_forward_pre_hook(self._final)

    def _embed(self, _m, _i, out):
        self.states = [out.detach()]

    def _block(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        self.states.append(h.detach())

    def _final(self, _m, inputs):
        self.final = inputs[0].detach().clone()


def rms(x) -> float:
    return x.double().pow(2).mean().sqrt().item()


def run_one(family, alpha, k, seed, args, device):
    model = build_model(family, k, alpha, seed, args.vocab, device)
    rec = Recorder(model)
    model.train()

    gen = torch.Generator().manual_seed(10_000 + seed)
    ids = torch.randint(args.vocab, (1, args.seq_len), generator=gen).to(device)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=args.adam_eps,
        fused=device.type == "cuda",
    )

    out = {"family": family, "alpha": alpha, "k": k, "seed": seed,
           "lr": args.lr, "adam_eps": args.adam_eps, "seq_len": args.seq_len}
    losses, deltas, prev = [], [], None
    for t in range(args.steps + 1):
        training = t < args.steps
        if training:
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
        else:
            with torch.no_grad():
                loss = model(input_ids=ids, labels=ids).loss
        x_final = rec.final
        if t == 0:
            states = rec.states
            assert len(states) == k + 1, (len(states), k)
            assert torch.allclose(states[-1], x_final), "block recorder mismatch"
            out["block_rms"] = [rms(s) for s in states]
            out["energy_init"] = x_final.double().pow(2).mean().item()
            inc = torch.stack([(states[r] - states[r - 1]).flatten().double()
                               for r in range(1, k + 1)])
            inc = inc / inc.norm(dim=1, keepdim=True)
            out["increment_cosine"] = [
                [round(v, 5) for v in row] for row in (inc @ inc.T).tolist()
            ]
        else:
            deltas.append(rms(x_final - prev))
        losses.append(loss.item())
        prev = x_final
        if training:
            loss.backward()
            opt.step()

    out["losses"] = losses
    out["step_delta_rms"] = deltas
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def cmd_run(args) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for alpha in args.alphas:
        path = out_dir / f"{args.family}_a{alpha:g}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done.add((r["k"], r["seed"]))
        with path.open("a") as f:
            for k in args.ks:
                for seed in args.seeds:
                    if (k, seed) in done:
                        continue
                    t0 = time.time()
                    r = run_one(args.family, alpha, k, seed, args, device)
                    r["seconds"] = round(time.time() - t0, 2)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    step1 = (f" d1={r['step_delta_rms'][0]:.4f}"
                             if r["step_delta_rms"] else "")
                    print(f"{args.family} a={alpha:g} K={k} seed={seed} "
                          f"rms0={math.sqrt(r['energy_init']):.4f}{step1} "
                          f"({r['seconds']}s)", flush=True)


def _mean_sem(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def cmd_summarize(args) -> None:
    rows = []
    for path in sorted(Path(args.inp).glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()]

    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["alpha"], r["k"]), []).append(r)

    summary = {}
    for (family, alpha, k), rs in sorted(groups.items()):
        s = summary.setdefault(f"{family}_a{alpha:g}", {
            "family": family, "alpha": alpha, "init": [], "step_change": []})
        mean, sem = _mean_sem([r["energy_init"] for r in rs])
        s["init"].append({"k": k, "num_seeds": len(rs),
                          "mean_energy": mean, "sem_energy": sem})
        for step in range(len(rs[0]["step_delta_rms"])):
            mean, sem = _mean_sem([r["step_delta_rms"][step] for r in rs])
            s["step_change"].append({"k": k, "step": step + 1,
                                     "num_seeds": len(rs),
                                     "mean_rms": mean, "sem_rms": sem})
        # Groups are sorted by k, so this keeps the matrix of the largest k.
        mats = [r["increment_cosine"] for r in rs]
        s["increment_cosine"] = {
            "k": k, "num_seeds": len(rs),
            "mean": [[round(sum(m[i][j] for m in mats) / len(mats), 4)
                      for j in range(k)] for i in range(k)],
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, separators=(",", ":")))
    print(f"wrote {args.out} ({len(rows)} runs, {len(groups)} groups)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--family", choices=["loop", "deep"], required=True)
    r.add_argument("--alphas", type=float, nargs="+", required=True)
    r.add_argument("--ks", type=int, nargs="+", default=KS)
    r.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    r.add_argument("--lr", type=float, default=2e-6)
    r.add_argument("--weight-decay", type=float, default=0.1)
    r.add_argument("--adam-eps", type=float, default=1e-8)
    r.add_argument("--steps", type=int, default=10)
    r.add_argument("--seq-len", type=int, default=128)
    r.add_argument("--vocab", type=int, default=128256)
    r.add_argument("--device", default=None, help="default: cuda if available")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
