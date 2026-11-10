# 文本生成

2026-11-04 · 玩具张量验证

`cs336_basics/generate.py`。解码一步取模型输出最后一个位置的 logits，经过温度缩放与 top-p 截断，再用 `torch.multinomial` 采样，把 token 追加到序列末尾。

## 温度缩放

`temperature_scaled_softmax(x, temperature)` 先计算 `x / temperature`，再在最后一维做 `softmax`。τ<1 分布更尖锐，τ>1 更平坦；τ→0 退化为 one-hot。

τ=0 直接构造 one-hot：`argmax(dim=-1, keepdim=True)` 取最后一个维度的最大值位置，`torch.zeros_like(x)` 建同形状零向量，`scatter_(-1, idx, 1)` 写入 1。`argmax` 遇到并列取第一个下标。τ<0 没有定义，抛出 `ValueError`。

玩具：logits `[2.0, 1.0, 0.5, 0.1]`、τ=0.5 得到 `[0.8282, 0.1121, 0.0412, 0.0185]`，和为 0.99999994，float32 的累积误差。

## top-p

先对温度缩放后的概率 `q` 排序，累加，找到累加和首次达到 p 的位置 k；取该位置的概率作为 cutoff。在原始 `q` 上保留 `q >= cutoff`，其余置 0，再除以保留部分之和完成归一化。

玩具 p=0.8：cumsum 为 `[0.8282, 0.9402, 0.9815, 1.0000]`，k=0，cutoff=0.8282，保留后是 `[1, 0, 0, 0]`。p=0.9：k=1，cutoff=0.1121，保留前两个概率并按 `q >= cutoff` 归一化为 `[0.8808, 0.1192, 0, 0]`。

`torch.sort(q, descending=True)` 返回排序后的值与原始下标；只保留值会丢失 token 位置。需要严格取前 k+1 个元素时保留 `indices`，用 `indices[:k+1]` 建 mask；并列概率较多时，`q >= cutoff` 会多保留一些，集合总概率略高于 p。

## 采样

`torch.multinomial(prob, 1)` 按概率权重抽一个下标，`prob` 必须是一维且和为 1；批量输入时用 `(batch, vocab)` 并设置 `num_samples=1`。采样是随机的，固定 `torch.manual_seed` 才能复现。

## 一步怎么循环

prompt 编码成 token ID 后送入模型，取最后一个位置的 logits；采样得到下一个 token，追加到序列；遇到 `<|endoftext|>`（ID 256）或达到最大生成长度停止。整个过程放在 `model.eval()` 与 `torch.no_grad()`（或 `torch.inference_mode()`）下。

朴素实现每步重算整个前缀。模型 17M 参数、上下文 256，在 CPU 上生成几百个 token 可以接受。KV cache 是进一步加速的手段：prefill 一次存下每层 K/V，后续只输入新 token；需要修改 attention 与 Block 的接口。作业的第 6 节只要求正确性，没有要求缓存。

## decoding 怎么写

`decoding(prompt, max_token_num, context_len, temperature, top_p, model, device, eos_id)` 接收 `(1, S)` 的 token 张量。先分配 `(1, max_len)` 的缓冲区，`max_len = min(max_token_num + S, context_len)`；prompt 长度超过 `context_len` 时抛出 `ValueError`。prompt 写入前 S 个位置，`pos` 记下一个写入位置。

循环在 `model.eval()` 与 `torch.no_grad()` 下执行：`input_ids = buffer[:, :pos]` 是视图，模型输出 `(1, pos, V)`，取 `[:, -1, :]`；`sampling` 返回 `(1, 1)`；遇到 EOS 就停止；否则写入 `buffer[:, pos:pos+1]`，`pos` 加一。返回 `buffer[:, :pos].clone()`，不把未写入的尾部带出去。

`handle_user_input` 管字符串和张量之间的转换。它按 `tiny` 或 `owt` 选词表，`eos_id = tokenizer.encode("<|endoftext|>")[0]`；用 `parse_args(["--config", path])` 和 `load_config` 拿配置，`build_model` 建模型；`torch.load(checkpoint_path, map_location=device)` 读检查点，`load_state_dict(ckpt["model"])`，再 `model.to(device)`。返回的闭包把 prompt 编码成 `(1, S)`，调 `decoding`，最后用 `tokenizer.decode` 转回文本。

## 踩过的坑

EOS ID 不能从 `tokenizer.merges` 取；`merges` 是列表，没有 `.get`。用 `tokenizer.encode("<|endoftext|>")[0]` 或 `reversed_vocab[b"<|endoftext|>"]`。

`prompt.size(0)` 是 batch 维；输入 `(1, S)` 时它恒为 1。长度用 `size(-1)`。

`torch.empty(max_token_num)` 是一维，后面的 `buffer[:, :len_prompt]` 会报二维索引错误；缓冲区建成 `(1, max_len)`。

返回整个 `buffer` 会把未初始化尾部一起返回；切片到 `pos` 并 `clone()`。

`torch.load(config.checkpoint_dir)` 把目录当文件；`checkpoint_dir` 是目录，需要具体的 `checkpoint_5000.pt`。

模型在 `config.device`，缓冲区与 `input_ids` 在 `device`；两者不同会报设备不一致，加载后调用 `model.to(device)`。

`Tokenizer.encode` 对空字符串返回空列表，模型前向的形状 `(1, 0)` 会失败；空 prompt 直接拒绝，或者补一个 BOS。

