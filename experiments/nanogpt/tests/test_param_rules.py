"""The parameterization in model.py against Table 1 of the paper (looped column)."""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig, _activation, branch_multiplier  # noqa: E402

BASE_WIDTH, BASE_DEPTH = 256, 2
ETA, EPS, WD = 0.01, 1e-8, 0.1


def cfg(d=64, L=2, N=1, aN=1.0, aL=0.5, a=1.0, **kw):
    return GPTConfig(n_embd=d, n_head=max(1, d // 32), n_layer=L, n_loops=N, block_size=16, vocab_size=64,
                     alpha_N=aN, alpha_L=aL, branch_mult=a, **kw)


def groups(c):
    return GPT(c).parameterization_summary(ETA, EPS, WD)['groups']


@pytest.mark.parametrize('d,L,N,aN,aL', [
    (64, 2, 1, 1.0, 0.5), (128, 4, 8, 1.0, 0.5), (512, 8, 4, 1.0, 0.5), (64, 8, 4, 0.5, 0.5),
    (64, 8, 4, 1.5, 0.5), (64, 16, 2, 0.0, 0.5), (128, 4, 8, 1.0, 1.0),
])
def test_rules_match_table(d, L, N, aN, aL):
    c = cfg(d, L, N, aN, aL, a=0.7, bias=True)
    m_n, m_L = d / BASE_WIDTH, L / BASE_DEPTH
    assert math.isclose(branch_multiplier(c), 0.7 * N ** -aN * m_L ** -aL)
    model = GPT(c)
    assert all(math.isclose(blk.branch_multiplier, branch_multiplier(c)) for blk in model.transformer.h)
    g = model.parameterization_summary(ETA, EPS, WD)['groups']
    block_eps = EPS / m_n * m_L ** -aL
    expected = dict(
        emb_unemb=dict(lr=ETA, eps=EPS / m_n, weight_decay=WD),
        pre_ln=dict(lr=ETA * m_L ** (aL - 1), eps=block_eps, weight_decay=0.0),
        hidden_weight=dict(lr=ETA / m_n * m_L ** (aL - 1), eps=block_eps, weight_decay=WD * m_n),
        hidden_bias=dict(lr=ETA * m_L ** (aL - 1), eps=block_eps, weight_decay=0.0),
        final_ln=dict(lr=ETA, eps=EPS / m_n, weight_decay=0.0),
    )
    assert set(g) == set(expected)
    for name, want in expected.items():
        for k, v in want.items():
            assert math.isclose(g[name][k], v, rel_tol=1e-12), (name, k)
    # PyTorch AdamW decays by lr * weight_decay: width-independent for hidden matrices
    assert math.isclose(g['hidden_weight']['lr'] * g['hidden_weight']['weight_decay'], ETA * WD * m_L ** (aL - 1))


def test_no_learning_rate_or_eps_depends_on_loops():
    ref = groups(cfg(128, 4, 1, 1.0))
    for N in (2, 4, 8):
        for aN in (0.0, 0.5, 1.0, 1.5):
            assert groups(cfg(128, 4, N, aN)) == ref
    assert branch_multiplier(cfg(128, 4, 8, 1.0)) == pytest.approx(1 / 8 * 2 ** -0.5)


def test_base_shape_is_plain_adamw():
    g = groups(cfg(256, 2, 1))
    for name, v in g.items():
        assert v['lr'] == ETA and v['eps'] == EPS, name
    assert g['hidden_weight']['weight_decay'] == WD and branch_multiplier(cfg(256, 2, 1)) == 1.0


def test_completep_limit_keeps_hidden_lr_depth_independent():
    for L in (2, 8, 32):
        assert math.isclose(groups(cfg(64, L, 4, aL=1.0))['hidden_weight']['lr'], ETA / (64 / BASE_WIDTH))


def test_init_variances():
    torch.manual_seed(0)
    d = 1024
    model = GPT(GPTConfig(n_embd=d, n_head=16, n_layer=2, block_size=64, vocab_size=512, init_std=0.02, bias=True))
    m_n = d / BASE_WIDTH
    params = dict(model.named_parameters())
    for n, p in params.items():
        if n.endswith(('c_attn.weight', 'c_fc.weight', 'c_proj.weight')):
            assert p.std().item() == pytest.approx(0.02 / math.sqrt(m_n), rel=0.02), n
        elif n in ('transformer.wte.weight', 'transformer.wpe.weight', 'lm_head.weight'):
            assert p.std().item() == pytest.approx(0.02, rel=0.02), n
        elif '.ln_' in n and n.endswith('.weight'):
            assert torch.all(p == 1), n
        else:
            assert torch.all(p == 0), n          # biases


def test_attention_scale_and_logit_multiplier():
    torch.manual_seed(0)
    model = GPT(cfg(128, 2, 2))
    attn = model.transformer.h[0].attn
    x = torch.randn(2, 16, 128)
    B, T, C, H = 2, 16, 128, attn.n_head
    q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in attn.c_attn(x).split(C, dim=2))
    scores = (q @ k.transpose(-2, -1)) / (C // H)                          # 1 / head_dim
    scores = scores.masked_fill(torch.ones(T, T).tril() == 0, float('-inf'))
    ref = attn.c_proj((F.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(attn(x), ref, atol=1e-6)
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        logits, _, hidden = model(idx, return_hidden=True)
        ref_logits = model.lm_head(model.transformer.ln_f(hidden[-1])) / (128 / BASE_WIDTH)
    assert torch.allclose(logits, ref_logits, atol=1e-6)


def test_group_partition_and_optimizer():
    model = GPT(cfg(64, 3, 2, bias=True))
    spec = model.param_groups_spec(WD, EPS)
    names = sorted(n for g in spec for n in g['params'])
    assert names == sorted(n for n, _ in model.named_parameters())
    by = {g['name']: g['params'] for g in spec}
    assert sorted(by['emb_unemb']) == ['lm_head.weight', 'transformer.wpe.weight', 'transformer.wte.weight']
    assert sorted(by['final_ln']) == ['transformer.ln_f.bias', 'transformer.ln_f.weight']
    assert all('.ln_1.' in n or '.ln_2.' in n for n in by['pre_ln']) and len(by['pre_ln']) == 3 * 4
    assert all(n.endswith('.bias') for n in by['hidden_bias']) and len(by['hidden_bias']) == 3 * 4
    assert all(n.endswith('.weight') for n in by['hidden_weight']) and len(by['hidden_weight']) == 3 * 4
    opt = model.configure_optimizers(WD, ETA, (0.9, 0.95), EPS, 'cpu')
    for group, g in zip(opt.param_groups, spec):
        assert math.isclose(group['lr'], ETA * group['lr_scale'])
        assert group['eps'] == g['eps'] and group['weight_decay'] == g['weight_decay']
    summary = model.parameterization_summary(ETA, EPS, WD)
    assert set(summary) == {'config', 'branch_multiplier', 'groups'}
    assert all(set(v) == {'lr', 'eps', 'weight_decay', 'n_params'} for v in summary['groups'].values())


def test_tied_untied_and_activations():
    tied = GPT(cfg(64, 2, 1, untie_embeddings=False))
    assert tied.transformer.wte.weight is tied.lm_head.weight
    assert tied.count_params()['unembedding'] == 0
    untied = GPT(cfg(64, 2, 1))
    assert untied.transformer.wte.weight is not untied.lm_head.weight
    x = torch.randn(3, 5, 8)
    assert torch.allclose(_activation('relu2')(x), torch.relu(x) ** 2)
    assert torch.allclose(_activation('gelu')(x), F.gelu(x))
    with pytest.raises(ValueError):
        _activation('swiglu')
