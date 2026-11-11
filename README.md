<p align="center">
  <img src="assets/cover.png" alt="做 LLM 啦" width="100%">
</p>

# CS336 Spring 2026 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-NVIDIA%20B200-76B900?logo=nvidia&logoColor=white">
  <img alt="Validation loss" src="https://img.shields.io/badge/val%20loss-3.3347-2a78d6">
</p>

## TRUMP-786

**Tensor-Resonant Ultra-Micro Predictor**

A byte-level BPE tokenizer, a pre-norm decoder-only Transformer, an AdamW
optimizer, and a small training stack written from scratch for CS336
Assignment 1.

| | |
|---|---|
| Final OpenWebText validation loss | **3.3347** |
| Training budget | 25,000 steps · 1.64B tokens |
| Wall-clock | 43m48s on one B200 180GB |
| Context length | 512 |
| Parameters | 8 layers · 768 width · 12 heads |
| Vocabulary | 32,000 byte-level BPE |

## Architecture

```text
Decoder-only Transformer
  pre-norm residual blocks
  RMSNorm
  RoPE
  causal multi-head self-attention
  SwiGLU feed-forward, d_ff = 2048
  separate input embedding and LM head
```

The implementation includes the tokenizer, BPE training, data pipeline,
attention, RoPE, SwiGLU, optimizer, learning-rate schedule, gradient clipping,
checkpointing, and sampling code. No `torch.nn` transformer or optimizer
implementations are used.

## Training recipe

```text
batch size        128
context length    512
steps             25,000
tokens            1,638,400,000
optimizer         AdamW
betas             (0.9, 0.95)
epsilon           1e-8
weight decay      0.01
gradient clipping 1.0
learning rate     cosine, 2.8e-3 -> 1e-4, warmup 1,000
precision         bf16 autocast
hardware          1 × NVIDIA B200 180GB
```

The final run is logged in the public W&B project:

[TRUMP-786 on W&B](https://forge.coreweave.com/wandb/whengeason-southeast-university/CS336-a1-B200-owt/runs/f1a7pz60)

## Quick start

```bash
uv sync --python 3.12

# unit tests
uv run pytest

# train the final configuration on a B200
uv run python -m cs336_basics.train \
  --config configs/leaderboard/owt_B200_45min.yaml \
  --device cuda

# interactive generation with the local CPU build
uv run python cs336_basics/generate.py
```

The training configs expect:

```text
data/owt_train.npy
data/owt_val.npy
```

## Repository layout

```text
cs336_basics/          tokenizer, model, optimizer, training and generation
configs/               TinyStories, OWT, LR/batch sweeps and ablations
configs/leaderboard/   final 45-minute B200 configuration
leaderboard_submission/ score row, PR description and wall-clock curve
tests/                 unit tests and adapters
笔记/                   implementation notes and experiment records
assets/                README artwork
```

The final checkpoint is kept outside Git in:

```text
checkpoints/owt_B200/checkpoint_25000.pt
```

## Notes

- [Tokenizer](笔记/01-Tokenizer.md)
- [Model](笔记/02-Model.md)
- [Training](笔记/03-Training.md)
- [Generation](笔记/04-Generation.md)
- [OWT BPE](笔记/05-OWT-BPE-训练记录.md)
- [Training records](笔记/06-训练记录.md)
- [Scale OWT / TRUMP-786](笔记/07-scale-owt.md)

The complete written report is in
[笔记/CS336 A1 · 从零实现语言模型.pdf](笔记/CS336%20A1%20·%20从零实现语言模型.pdf).
The handwritten notes are in
[笔记/手写笔记.pdf](笔记/手写笔记.pdf).

---

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
