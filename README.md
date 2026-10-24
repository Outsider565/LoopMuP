# Loop-μP

Code for **[Loop-μP: Residual Scaling and Hyperparameter Transfer in Looped Transformers](https://arxiv.org/abs/2606.18524)**.

Loop-μP combines residual scaling with width and depth parameterization for Transformers that repeat a shared stack of layers.

## Installation

```bash
git clone https://github.com/Outsider565/LoopMuP.git
cd LoopMuP
pip install -e ".[test]"
```

Requires Python 3.10 or later and Transformers 4.57. The tests use Transformers 4.57.1.

## Quick start

```bash
python examples/train_step.py
```

The example takes one optimizer step on a small Llama model, saves it, and reloads it. To apply Loop-μP to a freshly initialized model:

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import loopmup

model = LlamaForCausalLM(LlamaConfig(
    vocab_size=128, hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
))
cfg = loopmup.LoopMuPConfig(n_loops=4, base_depth=2, base_width=64)
loopmup.apply(model, cfg)
optimizer = torch.optim.AdamW(
    loopmup.param_groups(model, cfg, lr=1e-3, weight_decay=0.1, eps=1e-8),
    betas=(0.9, 0.95),
)
```

Apply the recipe once per model. `base_width` and `base_depth` describe the base model; `n_loops` sets the number of passes through the shared layers. Grow width by adding attention heads at a fixed head dimension. Use `use_cache=False` with the looped model.

## Checkpoints

```python
loopmup.save_pretrained(model, "checkpoints/demo")
model = LlamaForCausalLM.from_pretrained("checkpoints/demo")
loopmup.apply(model, cfg, reinit=False)
```

Use `loopmup.save_pretrained` to save the distinct layers, then reapply the same configuration with `reinit=False` after loading.

## Residual-scaling diagnostics

```bash
loopmup-diagnostics run --family loop --alphas 0 0.5 1 1.5 --lr 1e-4 --out outputs/diagnostics
loopmup-diagnostics run --family deep --alphas 0 0.5 1 1.5 --lr 1e-4 --out outputs/diagnostics
loopmup-diagnostics summarize --in outputs/diagnostics --out outputs/summary.json
loopmup-plot --summary outputs/summary.json --out outputs/figures
```

Set `--ks`, `--seeds`, and `--steps` to choose the sweep size. Use a separate output directory for each configuration.

## nanoGPT experiments

See [experiments/nanogpt](experiments/nanogpt/README.md) for data preparation, training, learning-rate sweeps, and analysis.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/loopmup/__init__.py` | Llama parameterization and checkpoint helpers |
| `src/loopmup/diagnostics.py` | Residual-scaling diagnostics |
| `src/loopmup/plot_figures.py` | Plotting |
| `examples/train_step.py` | One training step, save, and reload |
| `experiments/nanogpt/` | nanoGPT training, sweeps, analysis, and diagnostics |
| `tests/` | CPU tests for the Llama implementation |

## Tests

```bash
python -m pytest
cd experiments/nanogpt
python -m pytest tests
```

## Citation

```bibtex
@article{wang2026loopmup,
  title={Loop-$\mu$P: Residual Scaling and Hyperparameter Transfer in Looped Transformers},
  author={Wang, Shaowen and Li, Bingrui and Zhu, Tansheng and Zhang, Ge and Huang, Wenhao and Yan, Shen and Li, Jian},
  journal={arXiv preprint arXiv:2606.18524},
  year={2026}
}
```

## License

[MIT](LICENSE). The nanoGPT code keeps its original MIT notice in [experiments/nanogpt/LICENSE](experiments/nanogpt/LICENSE).
