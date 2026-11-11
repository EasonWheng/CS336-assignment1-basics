# Scale OWT

2026-11-10。4090 的 5k 学习率扫描已经跑完；B200 代码和环境已准备，数据还在上传。

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

扫描只选起点。选中后在 4090 跑 `owt_proxy_8l768_40k.yaml`，看 327.68M token 下的曲线，再把学习率带到 B200。代理模型的结果不能直接当最终结果，模型规模和 B200 不同。

## B200 目标

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

## 待办

- [x] 4090 跑 5k 学习率扫描，记录最终 val 和曲线。
- [ ] 4090 用选中的学习率跑 40k 代理，记录 val 和生成。
- [ ] B200 跑 500 步基准，按 token/s 定 45 分钟步数。
- [ ] B200 跑 45 分钟第一枪，必要时跑第二枪。
- [ ] 把 B200 结果和 bits/byte 补回 `06-训练记录.md`。
