# Model 基础构件

2026-10-25 · test_linear、test_embedding、test_rmsnorm、test_silu、test_swiglu、test_rope、test_scaled_dot_product_attention、test_4d_scaled_dot_product_attention、两组 MHA 测试通过

## Linear

`cs336_basics/model.py:11`。权重存成 `(d_out, d_in)`，方向和 `tests/adapters.py:19` 的 `weights` 标注一致，装权重不用转置。

```python
self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
```

第一版 `torch.empty` 没接 `device` 和 `dtype`，两个构造参数只存进了 `self.device` / `self.dtype`。CPU + float32 下默认值刚好一样，`test_linear` 通过；换成 `Linear(8, 16, dtype=torch.bfloat16)`，`weight.dtype` 还是 float32。

初始化 σ = sqrt(2/(d_in+d_out))，边界取 `±3σ`。`trunc_normal_` 的 `a`、`b` 填绝对值，传 `-3*std` 和 `3*std`。`d_in=512, d_out=256` 时 σ≈0.0510，初始化后 `abs().max()` 0.153。

```python
return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
```

`...` 兜住前导维度，`(2,5,d_in)` 进 `(2,5,d_out)` 出。

## Embedding

`cs336_basics/model.py:31`。权重 `(vocab_size, d_model)`，初始化 `N(0,1)` 截断 `±3`。

第一版前向照 Linear 的样子写：

```python
einsum(token_ids, self.weight, "... seq_len, vocab_size emb_dim -> ... seq_len emb_dim")
```

形状对，数值错。`vocab_size` 在输入里出现、没进输出，einsum 沿它求和，8×4 的表上结果等于 `ids.unsqueeze(-1) * W.sum(0)`。

改成按 id 取行：

```python
return self.weight[token_ids]
```

`(2,5)` 的 id 得到 `(2,5,d_model)`。10000×512 的表取 1024 个 id，索引 0.058ms，`torch.stack` 拼列表 2.27ms。

## RMSNorm

`cs336_basics/model.py:51`。参数只有一个 `gain`，`torch.ones` 初始化，`eps=1e-5`。

第一版前向写成 `sum(value ** 2 for value in x)`。Python 的 `sum` 从 0 开始，沿第 0 维遍历，`(4,12,64)` 的输入被沿 batch 加总，除以 `d_model` 之后是 `(12,64)`，再和 `x` 相除按 batch 广播，四个样本互相污染。改成沿最后一维求均方：

```python
RMS_x = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
```

`keepdim=True` 保住尺寸 1，`(4,12,1)` 才能和 `(4,12,64)` 广播。中间一版漏了 `+ eps` 和开方，拿均方直接除，输出尺度跟着输入走。

adapter 第一次报：

```
TypeError: ones() received an invalid combination of arguments - got (int, dtype=NoneType, device=Tensor)
```

`RMSNorm(d_model, eps, weights)` 按位置传参，第三个参数是 `device`。构造只收形状，权重走 `load_state_dict({"gain": weights})`（`tests/adapters.py:388`）。

前向开头 `x.to(torch.float32)`，结尾 `.to(in_type)` 降回原 dtype。

## SiLU

`cs336_basics/model.py:78`：

```python
return in_features * torch.sigmoid(in_features)
```

和 `F.silu` 对拍，三种形状最大偏差 1e-7 量级。

## SwiGLU

`cs336_basics/model.py:82`。三个 Linear：`w1`、`w3` 是 `(d_ff, d_model)`，`w2` 是 `(d_model, d_ff)`。

```python
return self.w2(SiLU(self.w1(x)) * self.w3(x))
```

初始化不用另写，三个 Linear 各自按 3.3.1 的规则建参数。

`d_ff` 从构造参数进，测试传 128（`tests/conftest.py:247`）。`(8/3) × 64 ≈ 171`，按最近的 64 倍数会算成 192，写死成自动推导的话 `load_state_dict` 形状对不上。

adapter 用 `.data` 直接赋三份权重（`tests/adapters.py:89`），没走 `state_dict`。

## softmax

`cs336_basics/model.py:127`。

```python
tmp = x - x.max(dim=dim, keepdim=True).values
return torch.exp(tmp) / torch.sum(torch.exp(tmp), dim=dim, keepdim=True)
```

`x.max(dim=...)` 返回 `(values, indices)` 的命名元组，只要值就取 `.values`，或者用 `torch.amax`。`keepdim=True` 少了，减法和除法都广播不上。输入里放一个 1000，不减最大值会得到 `inf / inf = nan`。

## 缩放点积注意力

`cs336_basics/model.py:131`。

```python
tmp_attn = einsum(query, key, "... n d_k, ... m d_k -> ... n m") / (dk ** 0.5)
tmp_attn.masked_fill_(~mask, float("-inf"))
```

`dk` 取 `query.size(-1)`，和 `d_v` 无关。掩码的 True 表示关注，取反后把 False 的位置填 `-inf`，`exp(-inf)` 精确等于 0。三阶和四阶输入两个测试都过了。

## RoPE

`cs336_basics/model.py:99`。缓冲区 `(max_seq_len, d_k//2)`，`register_buffer(..., persistent=False)`。

第一版指数写成 `theta ** (+arange(0, d_k, 2) / d_k)`，少一个负号，频率随 k 递增。`d_k=64, theta=10000` 时第 32 对每走一步转 7499 弧度。改回负指数后 `freqs[0]=1`、`freqs[-1]≈1.3e-4`。

前向第一版往 `rearrange` 出来的视图里写：

```python
pair_x[..., 0] = d_1
pair_x[..., 0] = d_2
```

两行都落在第 0 列，`b` 一直没更新，输入张量也被改掉。训练时反向传播会报 inplace 相关的错误。改成 `torch.stack([d_1, d_2], dim=-1)` 重建张量。

MHA 里 Q 是 `(batch, head, seq, d_k)`，测试传的 `token_positions` 是 `(1, seq)`，cos 取出来 `(1, seq, half)`，靠广播覆盖 batch 和 head 两维。

## 因果多头自注意力

`cs336_basics/model.py:147`。四个投影都是 `Linear(d_model, d_model)`，投影后拆头：

```python
rearrange(Q, "... seq (head dkv) -> ... head seq dkv", head=num_heads)
```

第一版写成 `"... (head dkv) -> ... head dkv"`，得到 `(batch, seq, head, dkv)`。注意力把最后两维当 `(queries, d_k)`，头被当成查询位置、序列被当成批次，算出来是四个头之间的注意力。输出形状不变，100% 不匹配。

第二个问题是没加因果掩码。adapter 构造时不传 mask，12 个键全部可见，不匹配率 91.6%。两个测试的期望第一行都是 `-0.240338`，位置 0 的旋转是恒等变换，只有第一行只关注自己时才会相等。

掩码在 forward 里按当前序列长度现造：

```python
mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
```

现在建在 CPU 上，上 GPU 训练时要补 `device=in_features.device`。

## adapters

Linear、Embedding、RMSNorm 三个 adapter 构造后装权重，key 跟属性名对上，写错时报 `Missing key(s) in state_dict: "weight"`。SwiGLU 用 `.data` 直接赋值，SiLU 直接调函数（`tests/adapters.py:403`）。MHA 的两个 adapter 装四份投影权重，带 RoPE 的那个多构造一个 rope 对象传进构造函数。

## 下一步

只剩 TransformerBlock 和 TransformerLM 的组装。
