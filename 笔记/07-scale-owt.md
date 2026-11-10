# Scale OWT

2026-11-10。4090 代理配置已经建好，训练还没开始；B200 目标配置也还没开跑。

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

三份 5k 配置是学习率扫描：

| max_lr | min_lr | token |
|---:|---:|---:|
| 3e-4 | 3e-5 | 40.96M |
| 5e-4 | 5e-5 | 40.96M |
| 1e-3 | 1e-4 | 40.96M |

扫描只选起点。选中后在 4090 跑 `owt_proxy_8l768_40k.yaml`，看 327.68M token 下的曲线，再把学习率带到 B200。代理模型的结果不能直接当最终结果，模型规模和 B200 不同。

## B200 目标

`configs/hi/owt_scale.yaml` 现在是：

```text
layers=16, d_model=1024, heads=64, d_ff=2688
context=512, batch=16, steps=40000
max_lr=1e-3, min_lr=1e-4, warmup=1000
```

每步 8192 token，40k 步同样是 327.68M token。约 267M 参数；激活约 45.5GB，权重、梯度、优化器状态约 4.3GB，总计约 50GB。B200 的 180GB HBM 能放下，4090 24GB 会 OOM。

开跑前改这几项：

- `num_heads` 从 64 改成 32，head_dim 从 16 变成 32。
- `max_lr` 从 1e-3 改成 5e-4，等 4090 扫描结果再定。
- `checkpoint_interval` 从 500 改成 2000 或 5000。267M 参数一个检查点约 3.2GB，500 步一次会写出约 256GB。
- `eval_interval` 从 100 改成 200 或 500，减少大模型验证开销。
- `checkpoint_dir` 和 `run_name` 换成独立名字，避免覆盖已有 OWT 结果。
- 训练循环现在是 fp32，B200 上用 bf16 autocast 可以省显存、提速度。

## 学习率

4 层 512 维 OWT 的最优学习率是 1e-3，这个数不能直接搬到 8 层 768 维或 16 层 1024 维。模型变宽、变深，batch 和上下文也变了，最优学习率会移动。5k 扫描在 4090 上一小时左右能跑完；B200 上跑错学习率，返工成本高得多。

扫描看两点：同一步数下哪档 val 最低，曲线有没有尖峰。差距在噪声内时选中间档，B200 上再确认一次，不重复三档扫描。

## 待办

- [ ] 4090 跑 5k 学习率扫描，记录最终 val 和曲线。
- [ ] 4090 用选中的学习率跑 40k 代理，记录 val 和生成。
- [ ] B200 上按 180GB 显存配置跑放大模型，记录实际耗时和费用。
- [ ] 把代理结果、B200 结果和 bits/byte 补回 `06-训练记录.md`。
