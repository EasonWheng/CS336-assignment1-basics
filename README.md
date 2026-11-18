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

### 01 · 字节级 BPE

```mermaid
flowchart LR
    A["UTF-8 文本"] --> B["按文档分块<br/>并行预分词"]
    B --> C["汇总词频和相邻 pair"]
    C --> D["反复合并最高频 pair"]
    D --> E["词表与 merge rank"]
    E --> F["编码 · uint16 .npy"]
    classDef data fill:#16384a,stroke:#55c7dd,color:#fff
    classDef step fill:#2e2456,stroke:#a795ff,color:#fff
    class A,E,F data
    class B,C,D step
```

词表从 256 个单字节和预留的特殊 token 开始。worker 沿文档边界分工统计，主进程汇总后执行合并，同频 pair 按字典序取较大者。编码时按已学到的 merge rank 合并。训练和验证的 token ID 分别落盘，供训练循环抽取窗口。 [分词器](cs336_basics/tokenizer.py) · [数据编码](scripts/tokenize_data.py)

### 02 · Transformer 的张量流

```mermaid
flowchart TB
    X["输入残差流 · B × T × D"] --> N1["RMSNorm"]
    N1 --> QKV["Q / K / V 投影与分头"]
    QKV --> A["Q、K 加 RoPE<br/>因果注意力和输出投影"]
    X --> R1["残差相加"]
    A --> R1
    R1 --> N2["RMSNorm"]
    N2 --> F["SwiGLU"]
    F --> R2["残差相加"]
    R1 --> R2
    R2 --> Y["输出残差流 · B × T × D"]
    classDef io fill:#16384a,stroke:#55c7dd,color:#fff
    classDef step fill:#2e2456,stroke:#a795ff,color:#fff
    classDef skip fill:#244636,stroke:#77d4a7,color:#fff
    class X,Y io
    class N1,QKV,A,N2,F step
    class R1,R2 skip
```

| 张量 | 通用形状 | TRUMP-786 训练配置 |
| --- | --- | --- |
| 输入和目标 ID | B × T | 128 × 512 |
| 残差流 | B × T × D | 128 × 512 × 768 |
| 分头后的 Q、K、V | B × H × T × dₕ | 128 × 12 × 512 × 64 |
| 注意力分数 | B × H × T × T | 128 × 12 × 512 × 512 |
| 词表 logits | B × T × V | 128 × 512 × 32,000 |

RoPE 旋转每个头中 Q、K 的相邻维度对。因果掩码 `M` 将未来位置的分数设为负无穷，当前 token 只聚合自身和过去位置的 V。八层 block 后，残差流经过最终 RMSNorm 和独立的 LM head，得到词表 logits。 [模型实现](cs336_basics/model.py)

```math
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
```

### 03 · 下一 token 预测与优化

```mermaid
flowchart LR
    A[".npy token 流"] --> B["随机窗口<br/>目标右移一位"]
    B --> C["前向 · BF16 autocast"]
    C --> D["稳定交叉熵 · FP32 logits"]
    D --> E["反向传播与全局梯度裁剪"]
    E --> F["AdamW"]
    F -.-> C
    F --> G["定期验证和存档"]
    classDef data fill:#16384a,stroke:#55c7dd,color:#fff
    classDef step fill:#2e2456,stroke:#a795ff,color:#fff
    class A,B,G data
    class C,D,E,F step
```

例如输入 <code>[the, cat, sat]</code>，目标是 <code>[cat, sat, down]</code>。<code>get_batch</code> 为两者共用同一批随机窗口起点，目标索引再加一。交叉熵先减最大 logit，再用 <code>gather</code> 取正确 token 的分数。训练中对多个位置的损失求平均，经反向传播、全局 L2 梯度裁剪和 AdamW 更新。 [取 batch](cs336_basics/data.py) · [交叉熵](cs336_basics/nn_utils.py) · [训练循环](cs336_basics/train.py)

```math
\ell(z,y)=\log\!\left(\sum_j e^{z_j-m}\right)-(z_y-m),\qquad m=\max_j z_j
```

### 04 · 从最后一个 logit 向量续写

每轮取最后一个位置在整个词表上的分数，应用温度和 Top-p，再把采到的 token 接回输入。随着上下文增长，下一步看到的是原文加上已经生成的内容。遇到结束符或长度上限就退出。温度为零时直接取最高分 token。 [采样实现](cs336_basics/generate.py)

```text
原文 → 编码 → 最后位置的 logits → 温度 / Top-p → 新 token
               ↑                                   │
               └────────── 把新 token 接回序列 ──────┘
```

### 05 · 两种语料和三次长跑

| 配置 | 层数 | 上下文 | 步数 | 训练 token | 各自验证集 loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyStories | 4 | 256 | 40,000 | 327.68M | 1.3514 |
| OpenWebText 基线 | 4 | 256 | 40,000 | 327.68M | 4.1755 |
| TRUMP-786 | 8 | 512 | 25,000 | 1,638.40M | **3.33471** |

TinyStories 用来跑通训练，OpenWebText 基线把同一套流程搬到网页语料。B200 正式跑每步处理 65,536 token，到 5,000 步累计 327.68M token 时，验证 loss 为 3.7484。两组 OpenWebText 实验同时改变了模型形状、上下文和 batch。B200 全程按最终验证点的墙钟时间折算约 623k token/s。 [训练记录](笔记/06-训练记录.md) · [规模化实验](笔记/07-scale-owt.md)

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
