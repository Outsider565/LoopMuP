"""Loop-muP for Hugging Face Llama models.

This file implements the looped column of Table 1 for
``transformers.LlamaForCausalLM``. The L distinct decoder layers are applied
N times with shared weights, while the token embedding, the final norm, and the
language-model head stay outside the loop. Compared with depth-muP for a
non-looped model, looping only multiplies each attention and MLP residual
branch by an extra N^{-1}:

* ``apply`` loops the layers, multiplies every residual branch by
  a N^{-1} m_L^{-alpha_L}, initializes hidden matrices with variance
  sigma^2 m_n^{-1}, and multiplies the logits by m_n^{-1};
* ``param_groups`` returns AdamW parameter groups whose learning rates, weight
  decays, and epsilons follow Table 1. None of them depends on N;
* ``save_pretrained`` writes a checkpoint of the distinct layers. Reload it with
  ``LlamaForCausalLM.from_pretrained`` followed by ``apply(..., reinit=False)``.

Here m_n = d_model / base_width and m_L = L / base_depth, where L counts the
distinct layers. With alpha_L = 1/2 this is Loop-muP, and alpha_L = 1 gives the
looped counterpart of CompleteP. At N = 1, L = base_depth, and
d_model = base_width every factor equals 1, so hyperparameters tuned on the base
model carry over to any loop count.

Looped layers share their ``layer_idx``, so a KV cache would mix the loop
passes. The looped model therefore runs without one and rejects calls that ask
for one (``use_cache=True`` or ``past_key_values``).

``apply`` leaves the attention scale unchanged, so scale the width by adding
attention heads at a fixed head dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoopMuPConfig:
    n_loops: int = 1          # N, number of passes through the layer stack
    base_depth: int = 12      # L_base, distinct layers of the base model
    base_width: int = 768     # n_base, hidden size of the base model
    alpha_L: float = 0.5      # depth exponent: 1/2 for Loop-muP, 1 for CompleteP
    branch_mult: float = 1.0  # a, constant in front of every residual branch


def distinct_layers(model) -> list:
    """The distinct decoder layers of ``model`` in order of first use."""
    return list({id(layer): layer for layer in model.model.layers}.values())


def width_depth_ratios(model, cfg: LoopMuPConfig) -> tuple[float, float]:
    """The width multiplier m_n and the depth multiplier m_L."""
    m_n = model.config.hidden_size / cfg.base_width
    m_L = len(distinct_layers(model)) / cfg.base_depth
    return m_n, m_L


def _reject_kv_cache(_module, _args, kwargs):
    if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
        raise ValueError("a looped model cannot use a KV cache, because its loop "
                         "passes share layer_idx; call it with use_cache=False")


def loop_layers(model, n_loops: int):
    """Apply the decoder layers ``n_loops`` times with shared weights.

    Also turns off the KV cache, which cannot tell the loop passes apart, and
    makes the model raise ``ValueError`` when a call asks for one.
    """
    layers = distinct_layers(model)
    model.model.layers = torch.nn.ModuleList(layers * n_loops)
    # LlamaModel.forward runs layers[:num_hidden_layers].
    model.config.num_hidden_layers = len(layers) * n_loops
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    model.model.register_forward_pre_hook(_reject_kv_cache, with_kwargs=True)
    return model


def scale_residual_branches(model, multiplier: float):
    """Multiply the output of every attention and MLP branch by ``multiplier``."""

    def scale(_module, _inputs, output):
        return output * multiplier

    for layer in distinct_layers(model):
        layer.self_attn.o_proj.register_forward_hook(scale)
        layer.mlp.down_proj.register_forward_hook(scale)
    return model


def apply(model, cfg: LoopMuPConfig, reinit: bool = True):
    """Turn a ``LlamaForCausalLM`` with distinct layers into a Loop-muP model.

    With ``reinit=True`` (a freshly initialized model) the hidden matrices are
    rescaled to variance sigma^2 m_n^{-1}. With ``reinit=False`` (a model
    loaded from a checkpoint written by ``save_pretrained``) the weights are
    kept, and only the loop and the forward multipliers are added. The branch
    and logit multipliers are forward hooks, so they are not stored in
    checkpoints. Call this once per model.
    """
    if (getattr(model, "loopmup_config", None) is not None
            or len(distinct_layers(model)) != len(model.model.layers)):
        raise ValueError("the model is already looped; apply Loop-muP only once")
    if not (isinstance(cfg.n_loops, int) and cfg.n_loops >= 1
            and cfg.base_depth > 0 and cfg.base_width > 0):
        raise ValueError(f"n_loops must be an integer >= 1 and base_depth, "
                         f"base_width positive; got {cfg}")
    m_n, m_L = width_depth_ratios(model, cfg)
    if reinit and m_n != 1:
        with torch.no_grad():
            for layer in distinct_layers(model):
                for p in layer.parameters():
                    if p.ndim >= 2:
                        p.mul_(m_n ** -0.5)
    if m_n != 1:
        model.lm_head.register_forward_hook(lambda _m, _i, out: out / m_n)
    loop_layers(model, cfg.n_loops)
    scale_residual_branches(
        model, cfg.branch_mult / cfg.n_loops * m_L ** -cfg.alpha_L)
    model.loopmup_config = cfg
    return model


def save_pretrained(model, path, **kwargs):
    """Save the distinct layers of a looped model with ``model.save_pretrained``.

    ``kwargs`` are passed on. The looped layer list shares tensors, which
    ``save_pretrained`` rejects, so it is replaced by the distinct layers
    while saving.
    """
    looped, num_layers = model.model.layers, model.config.num_hidden_layers
    layers = distinct_layers(model)
    model.model.layers = torch.nn.ModuleList(layers)
    model.config.num_hidden_layers = len(layers)
    try:
        model.save_pretrained(path, **kwargs)
    finally:
        model.model.layers = looped
        model.config.num_hidden_layers = num_layers


def param_groups(model, cfg: LoopMuPConfig, lr: float, weight_decay: float,
                 eps: float) -> list[dict]:
    """AdamW parameter groups for ``torch.optim.AdamW``.

    ``lr``, ``weight_decay``, and ``eps`` are the base values tuned on the base
    model. PyTorch's AdamW multiplies the weight decay by the learning rate, so
    the hidden weight decay omega m_n keeps their product independent of width.
    Norm gains and biases are not decayed.
    """
    m_n, m_L = width_depth_ratios(model, cfg)
    depth_lr = m_L ** (cfg.alpha_L - 1)
    block_eps = eps / m_n * m_L ** -cfg.alpha_L
    rules = {
        "emb_unemb": dict(lr=lr, weight_decay=weight_decay, eps=eps / m_n),
        "pre_ln": dict(lr=lr * depth_lr, weight_decay=0.0, eps=block_eps),
        "hidden_weight": dict(lr=lr / m_n * depth_lr,
                              weight_decay=weight_decay * m_n, eps=block_eps),
        "hidden_bias": dict(lr=lr * depth_lr, weight_decay=0.0, eps=block_eps),
        "final_ln": dict(lr=lr, weight_decay=0.0, eps=eps / m_n),
    }
    params = {name: [] for name in rules}
    for name, p in model.named_parameters():
        if name.startswith("model.layers."):
            if "norm" in name:
                params["pre_ln"].append(p)
            elif p.ndim >= 2:
                params["hidden_weight"].append(p)
            else:
                params["hidden_bias"].append(p)
        elif name.startswith("model.norm."):
            params["final_ln"].append(p)
        elif name.startswith(("model.embed_tokens.", "lm_head.")):
            params["emb_unemb"].append(p)
        else:
            raise ValueError(f"no Loop-muP rule for parameter {name}")
    return [dict(name=name, params=ps, **rules[name])
            for name, ps in params.items() if ps]
