# Model 基础构件

2026-10-25 · test_linear、test_embedding、test_rmsnorm、test_silu、test_swiglu 通过

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

## adapters

Linear、Embedding、RMSNorm 三个 adapter 构造后装权重，key 跟属性名对上，写错时报 `Missing key(s) in state_dict: "weight"`。SwiGLU 用 `.data` 直接赋值，SiLU 直接调函数（`tests/adapters.py:403`）。

## 下一步

RoPE 和缩放点积注意力，往上是 MHA、Block、LM。
