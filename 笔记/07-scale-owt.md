# Scale OWT

2026-11-11。单卡 B200 跑了 25,000 步，OWT 验证 loss 3.33471，最后一个验证点在 43 分 48 秒。

## 机器

B200 在 Nebius `us-central1`，Regular 实例：

```text
GPU: 1 × B200 NVLink, 180GB HBM (183,359 MiB)
CPU: 20 核
内存: 224 GiB
磁盘: 200 GiB SSD
系统: Ubuntu 24.04 LTS, CUDA 13.0
价格: $8.52/hour
```

Python 3.12.3，`torch 2.11.0+cu130`，`torch.cuda.is_available()` 为 True，`torch.cuda.get_arch_list()` 包含 `sm_100`。

4090 在 AutoDL：

```text
GPU: RTX 4090, 24GB
CPU: 20 核
内存: 90GB
```

## 现在的位置

4 层、512 维、17M 非嵌入参数的 OWT 40k 已经跑完：327.68M token，val 4.1755。生成能保持新闻句式和局部语法，实体和篇章会漂。这个模型继续加 token 的收益会变小，下一步把参数量、token 数、上下文一起放大。曲线和生成原文在 `06-训练记录.md`。

## 4090 代理

先在 4090 上跑 8 层、768 维的代理模型，避免一上来就占 B200。配置：

```text
layers=8, d_model=768, heads=12, d_ff=2048
context=512, batch=16, steps=40000
max_lr=5e-4, min_lr=5e-5, warmup=1000
```

每步 8192 token，40k 步共 327.68M token，和 TinyStories 的 40k 预算一致。约 106M 参数；按核算公式，激活约 8.1GB，权重、梯度、优化器状态约 1.6GB，总显存约 9.7GB。

配置文件：

```text
configs/pre/owt_proxy_8l768_40k.yaml
configs/pre/owt_proxy_8l768_lr3e-4_5k.yaml
configs/pre/owt_proxy_8l768_lr5e-4_5k.yaml
configs/pre/owt_proxy_8l768_lr1e-3_5k.yaml
```

5k 扫描结果：

| max_lr | 最终 val |
|---:|---:|
| 3e-4 | 4.6886 |
| 5e-4 | 4.5327 |
| 1e-3 | 4.5066 |
| 3e-3 | 5.5719 |

`1e-3` 最好，`3e-3` 已经明显失稳，稳定边缘在 1e-3 附近。`5e-4` 只差 0.026，落在短训练噪声里。

扫描只选起点。选中后在 4090 跑 `owt_proxy_8l768_40k.yaml`，看 327.68M token 下的曲线，再把学习率带到 B200。4090 代理与 B200 正式训练都用 8 层、768 维。换卡后主要变动的是 batch、学习率和训练 token 预算。

## B200 初始预算

45 分钟排行榜先用 8L / 768：

```text
layers=8, d_model=768, heads=12, d_ff=2048
context=512, batch=64, tokens/step=32,768
max_lr=2e-3, min_lr=2e-4, warmup=400
```

配置在 `configs/leaderboard/`。先跑 `owt_b200_bench500.yaml` 测 token/s，再按 45 分钟换算步数；目标 18,300 步、约 600M token。第二枪是 `owt_45min_8l768_lr1.5e-3.yaml`，只在第一枪高于 4.0 时跑。

`configs/hi/owt_scale.yaml` 是更大的规模化配置：

```text
layers=24, d_model=1280, heads=32, d_ff=3392
context=512, batch=16, steps=40000
max_lr=3e-4, min_lr=3e-5, warmup=1000
```

约 554M 参数，batch 16 的 fp32 激活约 45.6GB，bf16 激活约 22.8GB，B200 180GB 能放下。327.68M token 下 token/参数只有 0.69，容量偏大；这组只作为规模化对照，45 分钟榜单先跑 8L / 768。

## 学习率

8L / 768、batch 16 的 5k 扫描：`3e-4` 4.6886，`5e-4` 4.5327，`1e-3` 4.5066，`3e-3` 5.5719。`1e-3` 最好，`3e-3` 明显失稳。

45 分钟榜单用 batch 64，按平方根放大到 `2e-3`；第一枪曲线抬头就退到 `1.5e-3`。24L / 1280 这种更大的模型从 `3e-4` 起，不能把 1e-3 直接搬过去。

### B200 batch 探针

8L / 768、context 512、2000 步：

| batch | max_lr | 2000 步 val | 现象 |
|---:|---:|---:|---|
| 128 | 2.8e-3 | 4.3193 | 还在下降 |
| 256 | 4.0e-3 | 5.6771 @1000 | 明显失稳 |

batch 256 的 `4e-3` 已经越过稳定边缘。它的边缘低于 4e-3，具体在 3e-3 还是 3.5e-3 还没夹准。batch 128 的 2.8e-3 在 2k 步还正常，不能说明 45 分钟长训练也稳定。

这组数据只用来记录边缘：`batch 256 + 4e-3` 不作为正式配置。

## B200 45 分钟结果

最终模型叫 TRUMP-786，全称 Tensor-Resonant Ultra-Micro Predictor。配置在 `configs/leaderboard/owt_B200_45min.yaml`，检查点在 `checkpoints/owt_B200/checkpoint_25000.pt`。

三枪都在同一张 B200 上，context 512，验证口径一致：

| 配置 | 步数 | token | max_lr | min_lr | 最终 val | runtime |
|---|---:|---:|---:|---:|---:|---:|
| batch 64 | 23,000 | 753.7M | 2e-3 | 2e-4 | 3.5159 | 22 分 16 秒 |
| batch 128 | 25,500 | 1.67B | 2.8e-3 | 2.8e-4 | 3.3399 | 44 分 39 秒 |
| batch 128 | 25,000 | 1.64B | 2.8e-3 | 1e-4 | 3.3347 | 43 分 48 秒 |

batch 128 的 25,000 步这组作为榜单成绩。每步处理 128 × 512 = 65,536 token，全程 1,638,400,000 token。W&B 在最后一个验证点记录 2628.03 秒，按整轮折算约 623k token/s。稳定段的 643k token/s 用来估步数，实际排预算时还要留出启动、定期验证和存盘的时间。

### 25,000 步的验证曲线

训练从头开始，warmup 1,000 步，学习率从峰值 2.8e-3 按余弦降到 1e-4。训练和验证都用 context 512。前向走 BF16 autocast，交叉熵接收转回 FP32 的 logits，模型参数和 AdamW 状态保留 FP32。每 500 步从验证集抽 20 个 batch 求平均，每个验证点合计 1,310,720 个目标 token。

| 步数 | 累计训练 token | 验证 loss |
|---:|---:|---:|
| 5,000 | 327.68M | 3.7484 |
| 10,000 | 655.36M | 3.5663 |
| 15,000 | 983.04M | 3.4680 |
| 20,000 | 1,310.72M | 3.3727 |
| 25,000 | 1,638.40M | 3.33471 |

头 5,000 步已经用掉 327.68M token，和旧的 4 层模型跑 40,000 步的 token 量相同。旧模型在这个预算下的验证 loss 是 4.1755，这轮到 3.7484。两轮的模型宽度、层数、batch 和学习率都变了。后半程还能继续降，20,000 到 25,000 步再投 327.68M token，loss 降了约 0.038。

两次 batch 128 的正式训练只差约 0.0052。第二次把学习率下限从 2.8e-4 降到 1e-4，同时把总步数从 25,500 改成 25,000，余弦衰减的长度也变了。这个差值先记成两套配置的成绩差，单独量尾部学习率的收益还得固定总步数。

batch 64 的那轮在约 22 分钟结束，处理了 753.7M token。batch 128 那轮用了约 44 分钟，处理量超过两倍。这个差距同时包含 batch 和运行时长的变化。跑 batch 256 的探针时，吞吐只比 128 高约 4%，配上 4e-3 又出现失稳，正式训练就留在 batch 128。


### 生成与榜单

最终模型生成仍是英文新闻风格，局部流畅，实体和事实会漂。榜单只看验证损失，生成质量另算。

提交材料放在：

```text
CS336/CS336_Assignments/assignment1-basics/leaderboard_submission/
```

W&B run：

https://wandb.ai/whengeason-southeast-university/cs336-a1/runs/f1a7pz60

## 待办

- [x] 4090 跑 5k 学习率扫描，记录最终 val 和曲线。
- [x] 4090 用选中的学习率跑 40k 代理，已完成。
- [x] B200 跑 500 步基准，按 token/s 定 45 分钟步数。
- [x] B200 跑 45 分钟第一枪，必要时跑第二枪。
- [x] B200 结果记录在 `07-scale-owt.md`。
