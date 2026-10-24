import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as model_module  # noqa: E402
from model import GPT, GPTConfig, detect_peak_flops  # noqa: E402


def make(N, **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(n_embd=32, n_head=2, n_layer=2, n_loops=N, block_size=16, vocab_size=64, **kw))


def test_loop_equals_manual_unroll():
    m = make(3, alpha_N=0.3, branch_mult=0.7)
    m.eval()
    x = torch.randint(0, 64, (2, 16)); y = x.clone()
    a = 0.7 * 3 ** -0.3                      # m_L = 1
    with torch.no_grad():
        logits, loss, hidden = m(x, y, return_hidden=True)
        h = m.transformer.wte(x) + m.transformer.wpe(torch.arange(16))
        manual = [h]
        for _ in range(3):
            for blk in m.transformer.h:     # the same two blocks in every pass
                h = h + a * blk.attn(blk.ln_1(h))
                h = h + a * blk.mlp(blk.ln_2(h))
                manual.append(h)
        ref = m.lm_head(m.transformer.ln_f(h)) / (32 / 256)
    assert len(hidden) == 1 + 3 * 2
    for u, v in zip(hidden, manual):
        assert torch.allclose(u, v, atol=1e-6)
    assert torch.allclose(logits, ref, atol=1e-5)
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(ref.view(-1, 64), y.view(-1)), atol=1e-6)


def test_loops_share_weights_and_hidden_shapes():
    m1, m4 = make(1), make(4)
    assert m1.count_params() == m4.count_params()
    assert all(torch.equal(p, q) for p, q in zip(m1.parameters(), m4.parameters()))
    x = torch.randint(0, 64, (2, 16)); y = x.clone()
    with torch.no_grad():
        l1, _ = m1(x, y)
        l1b, _, hidden = m1(x, y, return_hidden=True)
        logits, loss = m1(x)
    assert torch.allclose(l1, l1b)
    assert len(hidden) == 1 + m1.config.n_layer and hidden[0].shape == (2, 16, 32)
    assert logits.shape == (2, 16, 64) and loss is None


def test_activation_checkpointing_matches():
    m = make(4, loop_activation_checkpointing=True)
    x = torch.randint(0, 64, (2, 16)); y = x.clone()
    m.train()
    _, loss_ckpt = m(x, y)
    loss_ckpt.backward()
    g_ckpt = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.config.loop_activation_checkpointing = False
    _, loss = m(x, y)
    loss.backward()
    assert torch.allclose(loss, loss_ckpt, atol=1e-6)
    for a, (n, p) in zip(g_ckpt, m.named_parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6), n


def test_manual_attention_matches_sdpa():
    m = make(2)
    x = torch.randint(0, 64, (2, 16)); y = x.clone()
    with torch.no_grad():
        a = m(x, y)[0]
        m.set_manual_attention()
        b = m(x, y)[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_flops_counts_and_mfu():
    m = make(4)
    c = m.count_params()
    assert c['total'] == c['body'] + c['embedding'] + c['unembedding']
    f1, f4 = make(1).flops_per_token(), m.flops_per_token()
    unemb = 6 * m.lm_head.weight.numel()
    assert abs((f4 - unemb) - 4 * (f1 - unemb)) < 1e-6     # loops multiply the body and attention terms only
    assert m.estimate_mfu(8, 0.5, None) is None
    assert m.estimate_mfu(8, 0.5, 1e12) == pytest.approx(f4 * 16 * 8 / 0.5 / 1e12)


@pytest.mark.parametrize('name,peak', [('NVIDIA H200', 989e12), ('NVIDIA H100 80GB HBM3', 989e12),
                                       ('NVIDIA A100-SXM4-80GB', 312e12), ('NVIDIA GeForce RTX 4090', None)])
def test_detect_peak_flops(monkeypatch, name, peak):
    monkeypatch.setattr(model_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(model_module.torch.cuda, 'get_device_name', lambda *a: name)
    assert detect_peak_flops() == peak


def test_detect_peak_flops_without_gpu(monkeypatch):
    monkeypatch.setattr(model_module.torch.cuda, 'is_available', lambda: False)
    assert detect_peak_flops() is None
