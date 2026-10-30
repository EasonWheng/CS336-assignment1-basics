# Model 基础构件

2026-10-25 · model.py 的 13 项测试全部通过，含 test_transformer_lm 与 test_transformer_lm_truncated_input

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

`cs336_basics/model.py:51`。参数只有一个 `weight`，`torch.ones` 初始化，`eps=1e-5`。

第一版前向写成 `sum(value ** 2 for value in x)`。Python 的 `sum` 从 0 开始，沿第 0 维遍历，`(4,12,64)` 的输入被沿 batch 加总，除以 `d_model` 之后是 `(12,64)`，再和 `x` 相除按 batch 广播，四个样本互相污染。改成沿最后一维求均方：

```python
RMS_x = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
```

`keepdim=True` 保住尺寸 1，`(4,12,1)` 才能和 `(4,12,64)` 广播。中间一版漏了 `+ eps` 和开方，拿均方直接除，输出尺度跟着输入走。

adapter 第一次报：

```
TypeError: ones() received an invalid combination of arguments - got (int, dtype=NoneType, device=Tensor)
```

`RMSNorm(d_model, eps, weights)` 按位置传参，第三个参数是 `device`。构造只收形状，权重走 `load_state_dict({"weight": weights})`（`tests/adapters.py:406`）。

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

adapter 用 `.data` 直接赋三份权重（`tests/adapters.py:93`），没走 `state_dict`。

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

`cs336_basics/model.py:147`。四个投影都是 `Linear(d_model, d_model)`，属性名对齐参考权重：`q_proj`、`k_proj`、`v_proj`、`output_proj`。投影后拆头：

```python
rearrange(Q, "... seq (head dkv) -> ... head seq dkv", head=num_heads)
```

第一版写成 `"... (head dkv) -> ... head dkv"`，得到 `(batch, seq, head, dkv)`。注意力把最后两维当 `(queries, d_k)`，头被当成查询位置、序列被当成批次，算出来是四个头之间的注意力。输出形状不变，100% 不匹配。

第二个问题是没加因果掩码。adapter 构造时不传 mask，12 个键全部可见，不匹配率 91.6%。两个测试的期望第一行都是 `-0.240338`，位置 0 的旋转是恒等变换，只有第一行只关注自己时才会相等。

掩码在 forward 里按当前序列长度现造，设备跟着输入走：

```python
mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=in_features.device))
```

第一版没写 `device`，CPU 上测试全过，搬到 GPU 会在加掩码时报设备不一致。

## TransformerBlock

`cs336_basics/model.py:179`。两个 RMSNorm、一个 MHA、一个 SwiGLU，属性名用 `ln1`、`ln2`、`attn`、`ffn`，和参考 key 一致。

第一版第二个残差写成 `in_features + out_ffn`，第一个子层的输出被丢掉。正确链路是 `adder1 = in_features + out_attn`，`fnl_result = adder1 + out_ffn`，对应手册式 (25)(26)。

RoPE 在 Block 里构造后传给 MHA，`d_k = d_model // num_heads`。位置用 `torch.arange(seq_len, dtype=torch.long)`，adapter 不传。

`run_transformer_block` 直接 `load_state_dict(weights)`（`tests/adapters.py:299`），key 必须一字不差：

```
attn.q_proj.weight    ln1.weight    ffn.w1.weight
```

为此改了一轮命名：Block 里 `rsm1/rsm2/attention/FFN` 改成 `ln1/ln2/attn/ffn`，MHA 的 `w_q/w_k/w_v/w_o` 改成 `q_proj/...`，RMSNorm 的参数 `gain` 改成 `weight`。`run_rmsnorm` 的 key 跟着换成 `"weight"`（`:406`），MHA 两处 adapter 也同步改了投影名。

## TransformerLM

`cs336_basics/model.py:205`。四个属性名 `token_embeddings`、`layers`、`ln_final`、`lm_head`，对应参考权重的四类前缀。

`layers` 用 `nn.ModuleList`。普通 `list` 里的模块不会被注册，`parameters()`、`.to()`、`state_dict()` 都看不到。换之后 key 自动是 `layers.0.attn.q_proj.weight`、`layers.1...` 这种形式。

前向四段：embedding → 循环过 block → `ln_final` → `lm_head`。序列长度取输入的实际长度，`context_length` 只决定 RoPE 缓冲区大小和位置上限，截断输入那项测试靠的就是这一点。

图 1 在 LM head 后面画了 Softmax，第一版照着加了，两个 LM 测试都失败。`tests/_snapshots/test_transformer_lm.npz` 里存的是 logits：`min -15.59`，每行求和是 -47339 这种量级，概率的行和应当是 1。图里那步属于生成流程，模型边界停在 LM head。训练时交叉熵内部还会做一次 `log_softmax`。

## 核算函数

`cs336_basics/model.py:235` 起四个函数，算写作业用的参数量、内存和 FLOPs，不参与模型前向。

参数量每块 `d_model × (4·d_model + 3·d_ff + 2)`，块外加两份 `vocab_size × d_model` 和 `d_model`。GPT-2 XL 配置得到 1,640,452,800，48 个 block 占 90.2%。乘 4 字节是 6,561,811,200 B，约 6.56 GB 或 6.11 GiB。

`block_matmul_FLOPs` 第一版漏了 Q、K、V 三个投影，只算分数矩阵、输出投影和 FFN，单块得 54,106,521,600。补上 `6·n·d_model²` 之后单块 69,835,161,600，整模型加 LM head 是 3,516,769,894,400，约 3.52 TFLOPs。

四项占比：QKV 投影 22.5%、分数与加权 9.6%、输出投影 7.5%、FFN 60.4%。

注释里抄闭式时把外面的 `d_model` 漏了，手写推导是对的。写全应该是

```python
2 * n * d_model * (4 * d_model + 2 * n + 3 * d_ff)
```

代入配置得 69,835,161,600，和逐项相加一致。这个式子已经包含 QKV 投影，不需要另加。全部是前向的量，训练按三倍估。

## adapters

Linear、Embedding、RMSNorm 三个 adapter 构造后装权重，属性名和参考 key 对上，写错时报 `Missing key(s) in state_dict: "weight"`。SwiGLU 用 `.data` 直接赋值（`tests/adapters.py:93`），SiLU 直接调函数（`:421`）。MHA 的两个 adapter 装四份投影权重，带 RoPE 的那个多构造一个 rope 对象。Block 和 LM 都是一句 `load_state_dict(weights)` 装完（`:299`、`:383`）。

## 下一步

模型部分完成。剩下 `nn_utils.py` 的 cross_entropy 和 gradient_clipping、`optimizer.py`、`data.py`、`serialization.py`。
