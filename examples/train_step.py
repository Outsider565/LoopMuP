"""Take one Loop-muP optimizer step on a small Llama model, then save and reload it."""

import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import loopmup


def main():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
    ))
    # Hyperparameters tuned at width 64 (4 heads of size 16), depth 2, and one loop transfer here.
    cfg = loopmup.LoopMuPConfig(n_loops=4, base_depth=2, base_width=64)
    loopmup.apply(model, cfg)
    optimizer = torch.optim.AdamW(
        loopmup.param_groups(model, cfg, lr=1e-3, weight_decay=0.1, eps=1e-8),
        betas=(0.9, 0.95),
    )
    tokens = torch.randint(128, (2, 32))
    loss = model(input_ids=tokens, labels=tokens).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"Loss: {loss.item():.4f}; distinct layers: {len(loopmup.distinct_layers(model))}")

    with tempfile.TemporaryDirectory() as path:
        loopmup.save_pretrained(model, path)
        reloaded = LlamaForCausalLM.from_pretrained(path)
    loopmup.apply(reloaded, cfg, reinit=False)
    with torch.no_grad():
        diff = (reloaded(tokens).logits - model(tokens).logits).abs().max().item()
    print(f"Reloaded checkpoint; max logit difference: {diff:.1e}")


if __name__ == "__main__":
    main()
