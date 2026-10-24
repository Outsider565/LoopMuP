"""
GPT-2-style language model with the Loop-μP parameterization.

The stack of n_layer distinct blocks is applied n_loops (N) times with shared weights. With
m_n = n_embd / base_width and m_L = n_layer / base_depth, the scaling rules are those of Table 1 of the paper:

    residual branches   x + a N^-alpha_N m_L^-alpha_L Branch(LN(x))     (Loop-μP: alpha_N = 1, alpha_L = 1/2)
    init                N(0, sigma^2) for the embeddings and unembedding, N(0, sigma^2 m_n^-1) for hidden matrices
    logits              lm_head(ln_f(x)) m_n^-1
    learning rate       eta m_n^-1 m_L^(alpha_L-1) for hidden matrices, eta m_L^(alpha_L-1) for block norms and
                        hidden biases, eta for the embeddings, the unembedding and the final norm
    weight decay        omega m_n for hidden matrices
    AdamW epsilon       eps m_n^-1 m_L^-alpha_L in residual blocks, eps m_n^-1 for the embeddings and unembedding

No learning rate and no epsilon depends on N; alpha_N only enters the residual multiplier (the paper's
loop-factor comparison varies it and keeps every other rule). alpha_L = 1 is the looped counterpart of
CompleteP. Entries that Table 1 leaves open follow nanoGPT-mup: norm gains start at 1 and biases at 0 (neither is
rescaled), the embeddings and unembedding use weight decay omega while norms and biases are not decayed, the
final norm uses the embedding epsilon, the initialization is a plain normal (the paper's experiments use a
truncated normal), and attention uses the μP scale 1 / head_dim (head_dim is 64 in every experiment, so this is a
constant factor). Adapted from EleutherAI/nanoGPT-mup (branch completep), a fork of karpathy/nanoGPT.
"""

import inspect
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304         # GPT-2 vocabulary (50257) padded to a multiple of 64
    n_layer: int = 12               # distinct blocks L
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    init_std: float = 0.02          # sigma
    mlp_act: str = 'relu2'          # gelu | relu | relu2
    untie_embeddings: bool = True
    base_width: int = 256           # m_n = n_embd / base_width
    base_depth: int = 2             # m_L = n_layer / base_depth
    n_loops: int = 1                # N: passes through the block stack
    alpha_N: float = 1.0
    alpha_L: float = 0.5
    branch_mult: float = 1.0        # a
    loop_activation_checkpointing: bool = False   # recompute each loop pass in backward

    def __post_init__(self):
        assert self.n_loops >= 1, 'n_loops must be >= 1'
        assert self.n_embd % self.n_head == 0, 'n_embd must be divisible by n_head'


def width_depth_ratios(config):
    """(m_n, m_L) = (n_embd / base_width, n_layer / base_depth)."""
    return config.n_embd / config.base_width, config.n_layer / config.base_depth


def branch_multiplier(config):
    """a N^-alpha_N m_L^-alpha_L, applied to both residual branches of every block."""
    _, m_L = width_depth_ratios(config)
    return config.branch_mult * config.n_loops ** -config.alpha_N * m_L ** -config.alpha_L


def param_group(name):
    """Table 1 group of a GPT parameter, by name."""
    if name in ('transformer.wte.weight', 'transformer.wpe.weight', 'lm_head.weight'):
        return 'emb_unemb'
    if name.startswith('transformer.ln_f.'):
        return 'final_ln'
    if name.startswith('transformer.h.'):
        module, kind = name.split('.')[-2:]
        if module in ('ln_1', 'ln_2'):
            return 'pre_ln'
        if module in ('c_attn', 'c_fc', 'c_proj'):
            return 'hidden_weight' if kind == 'weight' else 'hidden_bias'
    raise ValueError(f'unhandled parameter {name}')


def _activation(name):
    if name == 'gelu':
        return F.gelu
    if name == 'relu':
        return F.relu
    if name == 'relu2':
        return lambda x: F.relu(x).square()
    raise ValueError(f"unknown mlp_act {name!r}; use gelu | relu | relu2")


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = True   # fused SDPA; GPT.set_manual_attention() switches to explicit softmax

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        scale = 1.0 / k.size(-1)   # 1 / head_dim
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                               is_causal=True, scale=scale)
        else:
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            y = self.attn_dropout(F.softmax(att, dim=-1)) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = _activation(config.mlp_act)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.branch_multiplier = branch_multiplier(config)

    def forward(self, x):
        x = x + self.branch_multiplier * self.attn(self.ln_1(x))
        x = x + self.branch_multiplier * self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.m_n, self.m_L = width_depth_ratios(config)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if not config.untie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if param_group(name) == 'hidden_weight':
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(self.m_n))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def count_params(self):
        body = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith('transformer.h.') or n.startswith('transformer.ln_f.'))
        embedding = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        unembedding = self.lm_head.weight.numel() if self.config.untie_embeddings else 0
        return dict(total=body + embedding + unembedding, body=body, embedding=embedding, unembedding=unembedding)

    def _run_stack(self, x):
        for block in self.transformer.h:
            x = block(x)
        return x

    def forward(self, idx, targets=None, return_hidden=False):
        """
        Returns (logits, loss), with loss None when targets is None. return_hidden=True also returns the
        residual stream: hidden[0] is the embedding output and hidden[1 + n*L + l] the stream after block l
        of loop pass n (before ln_f). return_hidden runs without activation checkpointing.
        """
        t = idx.size(1)
        assert t <= self.config.block_size, f"sequence length {t} exceeds block_size {self.config.block_size}"
        pos = torch.arange(t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        hidden = [x] if return_hidden else None
        checkpoint = self.config.loop_activation_checkpointing and self.training and torch.is_grad_enabled()
        for _ in range(self.config.n_loops):
            if return_hidden:
                for block in self.transformer.h:
                    x = block(x)
                    hidden.append(x)
            elif checkpoint:
                x = torch.utils.checkpoint.checkpoint(self._run_stack, x, use_reentrant=False)
            else:
                x = self._run_stack(x)
        logits = self.lm_head(self.transformer.ln_f(x)) / self.m_n
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return (logits, loss, hidden) if return_hidden else (logits, loss)

    def set_manual_attention(self):
        """Use explicit softmax attention (forward-mode AD in diag/laziness.py does not support fused SDPA)."""
        T = self.config.block_size
        for block in self.transformer.h:
            mask = torch.tril(torch.ones(T, T, device=block.attn.c_attn.weight.device)).view(1, 1, T, T)
            block.attn.register_buffer('mask', mask, persistent=False)
            block.attn.flash = False

    # ------------------------------------------------------------------ optimizer

    def param_groups_spec(self, weight_decay, adam_eps):
        """
        Optimizer groups of Table 1, with parameter names. lr_scale multiplies the base learning rate (also
        under the schedule in train.py); weight_decay and eps are the per-group AdamW values.
        """
        m_n, m_L, alpha_L = self.m_n, self.m_L, self.config.alpha_L
        depth_lr = m_L ** (alpha_L - 1)
        block_eps = adam_eps / m_n * m_L ** -alpha_L
        groups = dict(
            emb_unemb=dict(lr_scale=1.0, weight_decay=weight_decay, eps=adam_eps / m_n),
            pre_ln=dict(lr_scale=depth_lr, weight_decay=0.0, eps=block_eps),
            hidden_weight=dict(lr_scale=depth_lr / m_n, weight_decay=weight_decay * m_n, eps=block_eps),
            hidden_bias=dict(lr_scale=depth_lr, weight_decay=0.0, eps=block_eps),
            final_ln=dict(lr_scale=1.0, weight_decay=0.0, eps=adam_eps / m_n),
        )
        names = {g: [] for g in groups}
        for n, p in self.named_parameters():
            if p.requires_grad:
                names[param_group(n)].append(n)
        return [dict(name=g, params=names[g], **groups[g]) for g in groups if names[g]]

    def configure_optimizers(self, weight_decay, learning_rate, betas, adam_eps, device_type):
        params = dict(self.named_parameters())
        optim_groups = [dict(params=[params[n] for n in g['params']], lr=learning_rate * g['lr_scale'],
                             lr_scale=g['lr_scale'], weight_decay=g['weight_decay'], eps=g['eps'])
                        for g in self.param_groups_spec(weight_decay, adam_eps)]
        fused = device_type == 'cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=adam_eps,
                                 **(dict(fused=True) if fused else {}))

    def parameterization_summary(self, learning_rate, adam_eps, weight_decay):
        """Effective per-group values and parameter counts (written to config.json and done.json)."""
        params = dict(self.named_parameters())
        groups = {g['name']: dict(lr=learning_rate * g['lr_scale'], eps=g['eps'], weight_decay=g['weight_decay'],
                                  n_params=sum(params[n].numel() for n in g['params']))
                  for g in self.param_groups_spec(weight_decay, adam_eps)}
        return dict(config=asdict(self.config), branch_multiplier=branch_multiplier(self.config), groups=groups)

    # ------------------------------------------------------------------ throughput

    def flops_per_token(self):
        """Training FLOPs per token (forward + backward); the attention-score term counts the causal half."""
        cfg = self.config
        body = self.count_params()['body'] - self.transformer.ln_f.weight.numel()
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        return 6 * (cfg.n_loops * body + self.lm_head.weight.numel()) + 6 * cfg.n_loops * L * H * Q * T

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_flops):
        """Model FLOPs utilization of one iteration over fwdbwd_per_iter sequences; None if peak_flops is unknown."""
        if not peak_flops:
            return None
        return self.flops_per_token() * self.config.block_size * fwdbwd_per_iter / dt / peak_flops


# Dense bf16 peak per GPU; names are matched longest first, so 'H100 PCIe' takes precedence over 'H100' (SXM).
PEAK_BF16_FLOPS = {'H200': 989e12, 'H100 PCIe': 756e12, 'H100': 989e12, 'A100': 312e12}


def detect_peak_flops():
    """Dense bf16 peak of the current GPU, or None when it is not listed in PEAK_BF16_FLOPS."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for key in sorted(PEAK_BF16_FLOPS, key=len, reverse=True):
        if key in name:
            return PEAK_BF16_FLOPS[key]
    return None
