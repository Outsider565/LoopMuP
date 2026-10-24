"""CPU checks of loopmup on a small Llama model: ``python -m pytest``."""

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import loopmup

BASE = dict(base_depth=2, base_width=64)


def tiny_model(layers=2, width=64, seed=0, tie=False):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=width, intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        tie_word_embeddings=tie))


def groups_by_name(model, cfg):
    return {g["name"]: g for g in loopmup.param_groups(
        model, cfg, lr=1e-3, weight_decay=0.1, eps=1e-8)}


def test_base_model_is_unchanged():
    ids = torch.randint(128, (1, 16))
    model = tiny_model()
    reference = copy.deepcopy(model)
    loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=1, **BASE))
    torch.testing.assert_close(model(ids).logits, reference(ids).logits,
                               rtol=0, atol=0)
    for g in groups_by_name(model, loopmup.LoopMuPConfig(**BASE)).values():
        assert g["lr"] == 1e-3 and g["eps"] == 1e-8


def test_looping_shares_weights():
    model = tiny_model()
    n_params = sum(p.numel() for p in model.parameters())
    loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=4, **BASE))
    assert len(model.model.layers) == 8
    assert len(loopmup.distinct_layers(model)) == 2
    assert sum(p.numel() for p in model.parameters()) == n_params


def test_kv_cache_is_disabled():
    model = tiny_model()
    assert model.config.use_cache and model.generation_config.use_cache
    loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=3, **BASE))
    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False
    assert model(torch.randint(128, (1, 16))).past_key_values is None


def test_explicit_kv_cache_is_rejected():
    ids = torch.randint(128, (1, 8))
    model = loopmup.apply(tiny_model(), loopmup.LoopMuPConfig(n_loops=3, **BASE))
    with pytest.raises(ValueError, match="KV cache"):
        model(ids, use_cache=True)
    with pytest.raises(ValueError, match="KV cache"):
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=2,
                       do_sample=False, use_cache=True, pad_token_id=0)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=2,
                         do_sample=False, pad_token_id=0)
    assert out.shape == (1, 10)


def test_invalid_config_leaves_model_unchanged():
    model = tiny_model()
    with pytest.raises(ValueError):
        loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=0, **BASE))
    assert len(model.model.layers) == model.config.num_hidden_layers == 2
    loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=2, **BASE))


def test_branch_multiplier():
    n, layers, alpha_L = 4, 8, 0.5
    model = tiny_model(layers=layers)
    loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=n, alpha_L=alpha_L, **BASE))
    attn = model.model.layers[0].self_attn.o_proj
    x = torch.randn(3, 64)
    expected = attn.weight @ x.T / n * (layers / BASE["base_depth"]) ** -alpha_L
    torch.testing.assert_close(attn(x), expected.T)


def test_hidden_init_and_reinit_false():
    cfg = loopmup.LoopMuPConfig(n_loops=2, **BASE)
    original = dict(tiny_model(width=128).named_parameters())
    for reinit in (True, False):
        model = loopmup.apply(tiny_model(width=128), cfg, reinit=reinit)
        params = dict(model.named_parameters())
        assert params.keys() == original.keys()
        for name, p in params.items():
            hidden = name.startswith("model.layers.") and p.ndim >= 2
            scale = 2 ** -0.5 if reinit and hidden else 1.0
            torch.testing.assert_close(p, original[name] * scale, rtol=0, atol=0)


def test_apply_twice_raises():
    for n in (1, 3):
        model = loopmup.apply(tiny_model(), loopmup.LoopMuPConfig(n_loops=n, **BASE))
        with pytest.raises(ValueError):
            loopmup.apply(model, loopmup.LoopMuPConfig(n_loops=n, **BASE))


@pytest.mark.parametrize("tie", [False, True])
def test_save_and_reload(tmp_path, tie):
    ids = torch.randint(128, (1, 16))
    cfg = loopmup.LoopMuPConfig(n_loops=3, **BASE)
    model = loopmup.apply(tiny_model(width=128, tie=tie), cfg)
    logits = model(ids).logits
    loopmup.save_pretrained(model, tmp_path)
    assert len(model.model.layers) == model.config.num_hidden_layers == 6
    torch.testing.assert_close(model(ids).logits, logits, rtol=0, atol=0)

    reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
    assert reloaded.config.num_hidden_layers == 2
    loopmup.apply(reloaded, cfg, reinit=False)
    torch.testing.assert_close(reloaded(ids).logits, logits, rtol=0, atol=0)


def test_learning_rates_do_not_depend_on_loops():
    per_n = []
    for n in (1, 2, 8):
        cfg = loopmup.LoopMuPConfig(n_loops=n, **BASE)
        model = loopmup.apply(tiny_model(layers=8), cfg)
        per_n.append({k: (g["lr"], g["eps"], g["weight_decay"])
                      for k, g in groups_by_name(model, cfg).items()})
    assert per_n[0] == per_n[1] == per_n[2]


def test_depth_and_width_rules():
    cfg = loopmup.LoopMuPConfig(n_loops=2, **BASE)
    model = loopmup.apply(tiny_model(layers=8, width=128), cfg)
    g = groups_by_name(model, cfg)
    m_n, m_L = 2.0, 4.0
    assert abs(g["hidden_weight"]["lr"] - 1e-3 / m_n * m_L ** -0.5) < 1e-12
    assert abs(g["hidden_weight"]["weight_decay"] - 0.1 * m_n) < 1e-12
    assert abs(g["hidden_weight"]["eps"] - 1e-8 / m_n * m_L ** -0.5) < 1e-20
    assert abs(g["pre_ln"]["lr"] - 1e-3 * m_L ** -0.5) < 1e-12
    assert g["pre_ln"]["weight_decay"] == g["final_ln"]["weight_decay"] == 0
    assert g["emb_unemb"]["lr"] == g["final_ln"]["lr"] == 1e-3
    assert g["emb_unemb"]["eps"] == g["final_ln"]["eps"] == 1e-8 / m_n


def test_logits_are_divided_by_width_multiplier():
    model = loopmup.apply(tiny_model(width=128), loopmup.LoopMuPConfig(**BASE))
    x = torch.randn(3, 128)
    torch.testing.assert_close(model.lm_head(x), x @ model.lm_head.weight.T / 2)


@pytest.mark.parametrize("tie", [False, True])
def test_every_parameter_in_one_group(tie):
    cfg = loopmup.LoopMuPConfig(n_loops=3, **BASE)
    model = loopmup.apply(tiny_model(tie=tie), cfg)
    grouped = [p for g in loopmup.param_groups(model, cfg, 1e-3, 0.1, 1e-8)
               for p in g["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    torch.optim.AdamW(loopmup.param_groups(model, cfg, 1e-3, 0.1, 1e-8))
