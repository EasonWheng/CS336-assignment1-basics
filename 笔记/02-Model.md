# Model 基础构件

2026-10-25 · test_linear、test_embedding 通过

## Linear

`cs336_basics/model.py:9`。权重存成 `(d_out, d_in)`，方向和 `tests/adapters.py:19` 的 `weights` 标注一致，装权重不用转置。

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

`cs336_basics/model.py:29`。权重 `(vocab_size, d_model)`，初始化 `N(0,1)` 截断 `±3`。

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

## adapters

`tests/adapters.py:34` 和 `:57` 两处都是先构造模块并装入 `weights`，再调前向。key 跟属性名对上，写错时报 `Missing key(s) in state_dict: "weight"`。

## 下一步

写 RMSNorm（`model.py:47` 已开头）、RoPE、MHA、SwiGLU，再组装 Block 和 LM。`data.py`、`nn_utils.py`、`optimizer.py`、`serialization.py` 还是占位注释。
