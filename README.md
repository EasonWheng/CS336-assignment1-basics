<p align="center">
  <img src="assets/cover.png" alt="做 LLM 啦" width="100%">
</p>

<h1 align="center">CS336 A1 · 从零训练语言模型</h1>

<p align="center">从 UTF-8 字节到 Transformer 续写，再到单卡 B200 的 OpenWebText 实验。</p>

<p align="center">
  <img alt="CS336 Spring 2026" src="https://img.shields.io/badge/CS336-Spring%202026-8C1515?style=flat-square">
  <img alt="PyTorch 2.11" src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="NVIDIA B200" src="https://img.shields.io/badge/NVIDIA-B200-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="仅限个人非商业自学" src="https://img.shields.io/badge/use-personal%20study%20only-6D5ACF?style=flat-square">
</p>

<p align="center">
  <a href="#trump-786">B200 实验</a> ·
  <a href="#从字节到续写">代码导览</a> ·
  <a href="#开始运行">开始运行</a> ·
  <a href="#学习资料">学习资料</a> ·
  <a href="#许可">许可</a>
</p>

## TRUMP-786

**Tensor-Resonant Ultra-Micro Predictor**

<table>
  <tr>
    <td align="center"><strong>3.33471</strong><br>OpenWebText 验证 loss</td>
    <td align="center"><strong>43 分 48 秒</strong><br>单张 B200 · 最终验证点</td>
    <td align="center"><strong>16.38 亿</strong><br>训练 token</td>
  </tr>
</table>

![TRUMP-786 训练和验证 loss 随墙钟时间下降的曲线](leaderboard_submission/images/trump786_learning_curve.png)

模型采用 **8 层 · 768 宽度 · 12 个注意力头**，上下文为 **512 token**，配套 **32,000 项字节级 BPE 词表**。正式训练跑了 25,000 步，每步处理 128 × 512 个 token。AdamW 配合 BF16 autocast，学习率预热 1,000 步，再从 `2.8e-3` 按余弦降至 `1e-4`。 [完整配置](configs/leaderboard/owt_B200_45min.yaml) · [榜单提交材料](leaderboard_submission/PR_DESCRIPTION.md)

## 从字节到续写

| 环节 | 实现入口 |
| --- | --- |
| 学习 BPE 合并规则，把文本编码成 token ID | [tokenizer.py](cs336_basics/tokenizer.py) · [tokenize_data.py](scripts/tokenize_data.py) |
| 把 token 送入 pre-norm Transformer，计算下一个 token 的 logits | [model.py](cs336_basics/model.py) |
| 计算交叉熵，按学习率调度更新参数并保存检查点 | [nn_utils.py](cs336_basics/nn_utils.py) · [optimizer.py](cs336_basics/optimizer.py) · [train.py](cs336_basics/train.py) |
| 用温度与 Top-p 采样，把生成的 ID 解码成文本 | [generate.py](cs336_basics/generate.py) |

```text
UTF-8 原文 → 字节级 BPE → Token ID → Embedding
                                    ↓
                  RMSNorm · 因果注意力 + RoPE · SwiGLU
                                    ↓
                     LM Head → 下一 token → 续写
```

## 开始运行

在仓库根目录用 Python 3.12 配好环境并运行已有测试。

```bash
uv sync --python 3.12
uv run pytest
```

训练前，从 TinyStories 或 Stanford 的 OpenWebText 样本下载原文，并用配套词表生成 `.npy` token 文件。分词脚本里的数据路径需指向本机。 [数据目录说明](data/数据说明.txt) · [分词脚本](scripts/tokenize_data.py) · [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) · [OpenWebText 样本](https://huggingface.co/datasets/stanford-cs336/owt-sample)

准备好 `data/owt_train.npy` 和 `data/owt_val.npy` 后，可以启动 B200 实验。模型检查点由训练生成，保存在 Git 仓库之外。

```bash
uv run python -m cs336_basics.train \
  --config configs/leaderboard/owt_B200_45min.yaml \
  --device cuda
```

## 学习资料

| 想看什么 | 从这里读 |
| --- | --- |
| 作业要求 | [官方指导书 PDF](cs336_assignment1_basics.pdf) · [中文稿](cs336_assignment1_basics_中文.md) |
| 完整学习记录 | [《从零实现语言模型》PDF](笔记/CS336%20A1%20·%20从零实现语言模型.pdf) · [手写笔记](笔记/手写笔记.pdf) |
| 实现细节 | [分词](笔记/01-Tokenizer.md) · [模型](笔记/02-Model.md) · [训练](笔记/03-Training.md) · [生成](笔记/04-Generation.md) |
| 实验记录 | [OWT BPE](笔记/05-OWT-BPE-训练记录.md) · [TinyStories 与 OWT](笔记/06-训练记录.md) · [B200 规模化](笔记/07-scale-owt.md) |

## 许可

本项目的原创实现、配置与笔记以**个人非商业自学**为授权范围。允许个人下载、运行和私下修改，用于非商业自学。商用、机构使用、付费教学、再分发及提交为作业或竞赛成果的规定见完整条款。 [个人非商业自学许可](LICENSE)

Stanford 课程材料、起始代码、测试及其他第三方内容适用各自的原有许可。 [Stanford MIT 原始声明](LICENSES/Stanford-MIT.txt)
