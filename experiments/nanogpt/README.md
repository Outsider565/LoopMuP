# nanoGPT experiments

Code for the language-modeling experiments in **[Loop-μP: Residual Scaling and Hyperparameter Transfer in Looped Transformers](https://arxiv.org/abs/2606.18524)**, built on nanoGPT and FineWeb-Edu.

Run the commands below from this directory after installing the repository with `pip install -e "../..[test]"`. Training uses CUDA; the tests run on a CPU.

## Data

```bash
python data/finewebedu10B/download.py --out data/finewebedu10B
python data/finewebedu10B/verify.py --dir data/finewebedu10B --sha256
```

The downloader fetches GPT-2-tokenized [FineWeb-Edu shards](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2), resumes partial downloads, and checks SHA-256 hashes. Use `--shards 10` to download the first ten training shards. Set `LOOPMUP_DATA` to use another data directory.

## Training

```bash
python train.py config/base.py --run_name=base_demo
torchrun --standalone --nproc_per_node=8 train.py config/loop.py --run_name=loop_demo --learning_rate=0.0055
```

`n_layer` sets the number of distinct layers and `n_loops` the number of shared passes. Loop-μP uses `alpha_N=1.0` and `alpha_L=0.5`. Override training options with `--key=value`.

The number of optimizer steps is `train_tokens // batch_tokens`. `batch_tokens` must be divisible by `micro_batch * block_size * world_size`. Outputs are saved under `$LOOPMUP_OUT/<run_name>` (default `out/`): configuration, metrics, and a completion record. Use `--ckpt_every_frac=0.1` for periodic checkpoints and `--save_final_checkpoint=True` to save the final weights.

## Sweeps

Tune the base learning rate:

```bash
export LOOPMUP_DATA=$PWD/data/finewebedu10B LOOPMUP_OUT=$PWD/out
python sweeps/make_manifest.py --experiment base --out outputs/manifests/base.jsonl
python sweeps/run_manifest.py --manifest outputs/manifests/base.jsonl --mode pack
python analysis/aggregate.py --experiments base --out outputs/base.csv
python analysis/fit_optimum.py --results outputs/base.csv --out outputs/base_optima.csv
```

Then sweep loop count, depth, or width around the fitted optimum. For example, with a base learning rate of 2^-7.5:

```bash
python sweeps/make_manifest.py --experiment loop --eta_log2 -7.5 --out outputs/manifests/loop.jsonl
python sweeps/run_manifest.py --manifest outputs/manifests/loop.jsonl --mode ddp --nproc 8 \
  --extra=--save_final_checkpoint=True
```

The available experiments are `base`, `loop`, `depth`, `width`, `coord`, and `smoke`. Use `make_manifest.py --seeds 1,2,3` for multiple seeds.

`--mode pack` runs independent jobs across GPUs; `--mode ddp` runs each job on multiple GPUs. `--shard i/k` splits a manifest across nodes. `--extra=--key=value` overrides a training option, and `--dry_run` previews commands.

The runner reuses completed results only when the saved request matches the configuration and overrides. Choose a new run name or output root when changing the configuration. Interrupted runs can be restarted with the same command; `--retry_failed` retries failed runs, and out-of-memory retries reduce the micro-batch automatically.

## Analysis and diagnostics

```bash
python analysis/aggregate.py --experiments loop --manifests outputs/manifests/loop.jsonl --out outputs/results.csv
python analysis/fit_optimum.py --results outputs/results.csv --out outputs/optima.csv
python analysis/plots.py lr_transfer --results outputs/results.csv --experiment loop --by N --out outputs/figures
python sweeps/make_manifest.py --experiment coord --out outputs/manifests/coord.jsonl
python sweeps/run_manifest.py --manifest outputs/manifests/coord.jsonl --mode pack
python diag/coord_check.py --experiment coord --manifest outputs/manifests/coord.jsonl --out outputs/figures/coord
```

`diag/hidden_norm.py`, `diag/loop_cosine.py`, and `diag/laziness.py` provide further diagnostics. Each script accepts `--help`.

## Layout

| Path | Contents |
| --- | --- |
| `model.py` | GPT model and Loop-μP optimizer groups |
| `train.py`, `configurator.py`, `config/` | Training and configurations |
| `lr_schedule.py` | Learning-rate schedules |
| `data/` | Data loading, download, and verification |
| `sweeps/` | Experiment definitions, manifests, and runner |
| `analysis/`, `diag/` | Analysis, plotting, and diagnostics |
| `tests/` | CPU tests |

## Tests

```bash
python -m pytest tests
```

## Attribution

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT; see [LICENSE](LICENSE)) is the base of the model, training script, and configurator.
- [nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup) by EleutherAI provides the μP and CompleteP implementation that `model.py` adapts.
- *Don't be lazy: CompleteP enables compute-efficient deep transformers* (Dey et al.) is the untied depth recipe in the left column of Table 1, and the source of the linearization measure in `diag/laziness.py`.
- The data are [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) with the GPT-2 tokenizer. `prepare_from_hf.py` follows [build-nanogpt](https://github.com/karpathy/build-nanogpt).
