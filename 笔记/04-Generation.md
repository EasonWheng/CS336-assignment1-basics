# 文本生成

2026-11-04 · 玩具张量验证

`cs336_basics/generate.py`。解码一步取模型输出最后一个位置的 logits，经过温度缩放与 top-p 截断，再用 `torch.multinomial` 采样，把 token 追加到序列末尾。

### 温度缩放

`temperature_scaled_softmax(x, temperature)` 先计算 `x / temperature`，再在最后一维做 `softmax`。τ<1 分布更尖锐，τ>1 更平坦；τ→0 退化为 one-hot。

τ=0 直接构造 one-hot：`argmax(dim=-1, keepdim=True)` 取最后一个维度的最大值位置，`torch.zeros_like(x)` 建同形状零向量，`scatter_(-1, idx, 1)` 写入 1。`argmax` 遇到并列取第一个下标。τ<0 没有定义，抛出 `ValueError`。

玩具：logits `[2.0, 1.0, 0.5, 0.1]`、τ=0.5 得到 `[0.8282, 0.1121, 0.0412, 0.0185]`，和为 0.99999994，float32 的累积误差。

### top-p

先对温度缩放后的概率 `q` 排序，累加，找到累加和首次达到 p 的位置 k；取该位置的概率作为 cutoff。在原始 `q` 上保留 `q >= cutoff`，其余置 0，再除以保留部分之和完成归一化。

玩具 p=0.8：cumsum 为 `[0.8282, 0.9402, 0.9815, 1.0000]`，k=0，cutoff=0.8282，保留后是 `[1, 0, 0, 0]`。p=0.9：k=1，cutoff=0.1121，保留前两个概率并按 `q >= cutoff` 归一化为 `[0.8808, 0.1192, 0, 0]`。

`torch.sort(q, descending=True)` 返回排序后的值与原始下标；只保留值会丢失 token 位置。需要严格取前 k+1 个元素时保留 `indices`，用 `indices[:k+1]` 建 mask；并列概率较多时，`q >= cutoff` 会多保留一些，集合总概率略高于 p。

### 采样

`torch.multinomial(prob, 1)` 按概率权重抽一个下标，`prob` 必须是一维且和为 1；批量输入时用 `(batch, vocab)` 并设置 `num_samples=1`。采样是随机的，固定 `torch.manual_seed` 才能复现。

### 解码循环

prompt 编码成 token ID 后送入模型，取最后一个位置的 logits；采样得到下一个 token，追加到序列；遇到 `<|endoftext|>`（ID 256）或达到最大生成长度停止。整个过程放在 `model.eval()` 与 `torch.no_grad()`（或 `torch.inference_mode()`）下。

朴素实现每步重算整个前缀。模型 17M 参数、上下文 256，在 CPU 上生成几百个 token 可以接受。KV cache 是进一步加速的手段：prefill 一次存下每层 K/V，后续只输入新 token；需要修改 attention 与 Block 的接口。作业的第 6 节只要求正确性，没有要求缓存。

