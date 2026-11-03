2026-11-01 · test_adamw、test_get_lr_cosine_schedule、test_gradient_clipping、test_get_batch、test_checkpointing 通过，训练部分随 4.1–5.2 增补

## 学习率

`experiments/lr_tuning.py` 抄的是手册 4.2.1 的式 (20)：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t)$$

玩具问题是 `(weights**2).mean()`，`weights` 为 10×10，初值取 `5 * torch.randn`。四组学习率 1、1e1、1e2、1e3 各运行 10 步，每组用同一个种子，只有 `lr` 不同。

初始损失 24.17，种子 42。`lr=1` 和 `lr=1e1` 下损失下降很慢，十步后分别还剩 20.00 和 3.25；`lr=1e2` 四步内降到 1.5e-16，是最快的一组；`lr=1e3` 时损失每一步放大 361 倍，第十步到 2.2e18，训练发散。

乘上多少可以直接计算。`L = mean(w²)` 摊在 100 个元素上，`∇L = 2w/100 = w/50`，式 (20) 变成

$$w \leftarrow w - \frac{\alpha}{50\sqrt{t+1}} w = w\left(1 - \frac{\alpha}{50\sqrt{t+1}}\right)$$

第一步 `t=0` 的系数是 `1 - α/50`：α=1 得 0.98，α=10 得 0.8，α=100 得 −1，α=1000 得 −19。损失正比于 `w²`，第一步之后损失与初值之比就是系数平方，四组实测依次是 0.96、0.64、1.00、361，对应 0.98²、0.8²、(−1)²、(−19)²。α=100 的第二步系数是 −0.414，step2/step1 是 0.1716 = 0.414²；第四步系数正好是 0，`w` 归零，损失掉到 1.5e-16。

## 优化器的演化

小批量先解决方差。`g_t = (1/B) Σ_{i∈B_t} g_i` 是全量梯度的无偏估计，方差比单样本降 `B` 倍，今天说的「SGD」默认就是这一种。

动量用来对付峡谷地形。`v_t = βv_{t-1} + (1-β)g_t`，`θ_t = θ_{t-1} - ηv_t`，β 取 0.9。垂直方向的梯度相邻两步反号，进 `v` 后相互抵消；沿谷底方向同号，`v` 一路累积。PyTorch 的 `SGD(momentum=0.9)` 写成 `v = βv + g`，稳态下 `v` 是 `1/(1-β)` 倍，等效学习率放大 10 倍，两种写法折算学习率后等价。NAG 把梯度取在 `θ - ηβv_{t-1}` 的预判位置上：动量冲过头时那里梯度已经反号，能提前刹车。

Adagrad 让每个参数有自己的步长。`s_t = s_{t-1} + g_t²`，更新是 `θ -= ηg_t/(√s_t + ε)`。梯度常年大的参数分母大、步子小，稀有特征对应的参数步子大。麻烦在 `s` 只增不减，练久了所有参数的步长都趋近 0。

RMSProp 把累加换成滑动平均，`s_t = ρs_{t-1} + (1-ρ)g_t²`，ρ 取 0.9 或 0.99。近期梯度小，`s` 跟着小，步长能回升。

Adam 同时平滑两个矩。`m` 用 β₁、`v` 用 β₂，两个都从 0 起步，前几步偏小，要除以 `1-β^t` 修正。分母是 `√v̂ + ε`，ε 取 1e-8。LLaMA 和 GPT-3 用 (0.9, 0.95)。

L2 正则把 `λθ` 加进梯度，Adam 随后把整个梯度除以 `√v̂`，衰减跟着被缩放：梯度方差大的参数衰减得轻，方差小的几乎按 `ηλ` 全额衰减，`λ` 不再等于「每步乘 `(1-ηλ)`」。把 `λθ` 从梯度里拿出来单独执行 `θ ← θ - ηλθ`，不碰 `m`、`v`，就是 AdamW。

### 手册规定的写法

Algorithm 1 的 `t` 从 1 开始，衰减在矩估计更新之前执行。偏差修正折进了步长：

$$\alpha_t = \alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}, \qquad \theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v} + \varepsilon}$$

`m`、`v` 用的是未修正的值，ε 也没被修正。测试注释说不同的衰减写法「原则上等价，浮点上不同」，所以测试先比较 `torch.optim.AdamW`，差在 1e-4 以内直接通过，对不上再比较参考快照。

构造签名按测试传：`opt_class(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)`，运行 1000 步 3→2 的线性回归，比较最后的 `weight`。

## AdamW 实现

`cs336_basics/optimizer.py:7`。构造参数 `lr=1e-3`、`betas=(0.9, 0.95)`、`eps=1e-8`、`weight_decay=0.3`，四个超参数各自检查范围后组合成 `defaults` 交给 `super().__init__`。betas 取 (0.9, 0.95) 是 LLaMA 的口径。

三份状态在 `step` 第一次用到时写入 `state`。`.get` 的默认值当初始值，`m`、`v` 取 0，`t` 取 1：

```python
t = state.get("t", 1)
m = state.get("m", 0)
v = state.get("v", 0)
```

`.get` 只读不写，真正保存下来靠后面的 `state["m"] = ...`、`state["v"] = ...`、`state["t"] = t + 1`。`t` 记的是当前步数，第一次进来就是 1，α_t 要用当前值，末尾再加一。α_t 依赖每个参数各自的 `t`，放在参数循环里计算。

第一版把新的 m、v 写入 state 之后，参数更新那行读的还是进来时的局部变量：

```python
state["m"] = betas[0] * m + (1 - betas[0]) * grad
state["t"] = t + 1
p.data -= alpha_t * m / (v ** 0.5 + eps)     # 旧的 m、v
```

第一步 `m`、`v` 都是 0，更新量 `alpha_t * 0 / (0 + eps)` 是 0，参数原地不动；后面每一步都用前一步的矩，整体滞后一拍。1000 步运行完 6 个权重里有 2 个对不上，最大绝对差 2.96e-4，超过 1e-4 的容差。写入 state 之后重新取值再更新，测试通过。

## 余弦学习率调度

`optimizer.py:108`。`t < Tw` 时线性上升到 `a_max`。从 `Tw` 到 `Tc` 按余弦降到 `a_min`，`Tc` 之后保持 `a_min`。

```python
if t < Tw:
    return a_max * (t / Tw)
elif Tw <= t <= Tc:
    return a_min + 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (a_max - a_min)
else:
    return a_min
```

两段能够衔接。`t = Tw` 时角度是 0，`cos` 取 1，代进去得 `a_min + (a_max - a_min) = a_max`；`t = Tc` 时角度 π，`cos` 取 −1，剩 `a_min`。

测试用 `a_max=1, a_min=0.1, Tw=7, Tc=21`，给了 22 个期望值。逐点执行完最大偏差 2.8e-17。`t=0` 是 0，`t=7` 是 1.0，`t=21` 回到 0.1，`t=22` 往后都是 0.1。

## 梯度裁剪

`nn_utils.py:29`。测试在 `tests/test_nn_utils.py:60`，6 个 5×5 的参数，最后一个 `requires_grad_(False)`。

```python
parameters = list(parameters)
total = sum((p.grad ** 2).sum() for p in parameters if p.grad is not None) ** 0.5
clip_coef = min(max_l2_norm / (total + 1e-6), 1.0)
for p in parameters:
    if p.grad is not None:
        p.grad.mul_(clip_coef)
```

第一版写成 `sum(p.grad ** 2 for p in parameters) ** 0.5`。`sum` 加的是张量，逐元素相加之后还是张量，`** 0.5` 又逐元素开方，两个 5×5 的梯度得到 5×5 的结果。要每个参数先 `.sum()` 归约成 0 维再相加。

第二版写成 `parameters.data.mul_(clip_coef)`，`parameters` 是 `Iterable`，测试传元组，报 `AttributeError: 'tuple' object has no attribute 'data'`；那里即使有 `.data`，改的也是参数本身。

第三版漏了 `if p.grad is not None`，冻结参数的 `p.grad` 是 `None`，`None ** 2` 报 `TypeError`。`parameters` 还会被遍历两次，传生成器时第二遍是空的，开头 `list(parameters)` 一并解决。

和 `torch.nn.utils.clip_grad_norm_` 对照测试，6 个参数、`max_l2_norm=1e-2`，逐位相同，`max diff 0.0`。五个未冻结的参数各 `0.004472`，总范数 `√5 × 0.004472 = 0.01`。

## 数据批次

`data.py:7`。测试在 `tests/test_data.py`。

```python
starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
idx = starts[:, None] + np.arange(context_length)
x = torch.as_tensor(dataset[idx], device=device).long()
y = torch.as_tensor(dataset[idx + 1], device=device).long()
```

`randint` 的 `high` 是排他的，最大起点为 `len - context_length - 1`，`y` 的最后一个 token 恰好落在数组末尾。测试用 `arange(100)` 和 `context_length=7`，起点要覆盖到 92；写成 `len - context_length - 1` 当 `high` 只能到 91。

`y` 是 `x` 往后移动一位：`x` 取 `data[i:i+L]`，`y` 取 `data[i+1:i+L+1]`，索引矩阵加一即可。

`as_tensor` 的 `device` 参数让张量创建时直接落在目标设备上，不经过先创建 CPU 张量再 `.to()`。分词数据是 `uint16`，embedding 的索引要 int64，后面连接一个 `.long()`。测试传 `"cuda:99"` 时要求抛出错误，`device` 没真正用上就过不了。

还有一段统计检查：1000 轮、每轮 32 条，每个起点出现的次数要落在 `1000×32/93` 的 ±5σ 内，固定批次或只取前几个窗口都会失败。

## 检查点

`serialization.py:9` 和 `:44`。测试运行三层 MLP 加 AdamW 十步，保存到 `tmp_path/checkpoint.pt`，再新建一组模型和优化器读取回来，逐参数比对。

```python
dict_to_save = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
}
torch.save(dict_to_save, out)
```

保存 `state_dict()`，不保存模块对象。`torch.save(model)` 会把这个模块整体序列化，读取的时候要求类定义完全对得上；`state_dict()` 是普通字典，`load_state_dict` 会检查键和形状，少一项就报错 `Missing key(s) in state_dict`。

优化器的 `state_dict` 有两部分：`param_groups` 保存超参数，`state` 保存每个参数的 `m`、`v`、`t`。只保存模型的话，恢复之后矩估计全丢，前几步的更新会偏。

返回的迭代次数用来衔接余弦调度的 `t` 和日志里的横轴。`torch.save`、`torch.load` 都收路径或类文件对象，测试传的是路径，训练脚本里也可以传 `io.BytesIO`。

## AdamW 训练核算

`optimizer.py` 里两个核算函数，`resource_AdamW` 和 `calc_flops`，不参与训练。

参数量按 `d_ff = (8/3)·d_model` 计算，GPT-2 XL 是 `P = 1,635,537,600`。3.5 节那张表写的 `d_ff = 4288` 得到 `1,640,452,800`，差 0.3%，写答案时前后统一用 8/3。

内存是 `h(B) = 16.151·B + 26.169` GB，`B` 为批大小。常数项 `16P` 拆开是参数 6.542 GB、梯度 6.542 GB、`m` 和 `v` 13.084 GB；一次项每个样本 `4.04e9` 个激活元素。80 GB 内 `(80 − 26.169)/16.151 = 3.33`，批大小取 3。

一步 AdamW 每参数 14 次逐元素运算，`14P = 2.290e10` FLOPs，梯度不在这 14 次里。

训练时间从单序列前向 `3.5067e12` 出发，乘 3 得前向加反向，再乘批大小 1024、步数 400,000，总计 `4.309e21` FLOPs。除以 `495e12 × 0.5`，得 `1.741e7` 秒，4836 小时，201.5 天。

`calc_flops` 第一版写成了 `(vocab_size, context_len, num_layers)`，调用时 `context_len=1024` 进了层数、`num_layers=48` 进了 `d_model`，计算得到 `6.98e8`，不报错只是小三十倍。AdamW 和上下文长度无关，形参里出现 `context_len` 就是信号。修改好后是 `3 × count_matmul_FLOPs(...) × batch_size`。

作业实际训练用的是 7.2 节的小配置：4 层、16 头、`d_model=512`、`d_ff=1344`、词表 10000、上下文 256、1700 万参数、3.28 亿 token，B200 上 20–30 分钟。低资源版本缩减到 4000 万 token，CPU 约 1 小时 22 分。

测试适配器全部接入完成。剩下 5.3 的训练脚本和第 6 节的生成。

## 训练脚本（5.3）

`cs336_basics/train.py:20`。`parse_args` 只保留三个参数：`--config` 默认 `configs/tinystories.yaml`，`--resume` 默认 `None`，`--device` 默认 `None`，`choices` 限制为 `cpu` 与 `cuda`。三十个超参数从命令行移进 YAML，命令行只负责选择配置文件、覆盖恢复路径与运行设备。

`load_config`（`:28`）用 `yaml.safe_load` 读取，检查 `model`、`optimizer`、`data`、`run`、`checkpoint` 五个顶层键，按顺序调用 `merged.update(section)` 合并成平铺字典，再交给 `argparse.Namespace` 构造。命令行提供的 `resume` 与 `device` 覆盖文件值。`yaml.safe_load` 的隐式类型要留意：`1e-3` 解析成字符串，写成 `1.0e-3` 才得到浮点数。

`validate_config`（`:57`）收集全部错误再一次性抛出。检查覆盖数值范围与跨字段关系。数值范围包括正整数 12 项（`vocab_size`、`context_length`、`d_model`、`num_heads` 等）、非负整数（`warmup_steps`、`seed`）、正数（`max_lr`、`eps`、`rope_theta`）、非负数（`min_lr`、`weight_decay`、`grad_clip`）。跨字段关系包括 `d_model % num_heads == 0`、`min_lr <= max_lr`、`warmup_steps < total_steps`。数据路径与恢复路径用 `os.path.isfile` 检查。`bool` 从整数检查里排除。

## 配置文件

`configs/tinystories.yaml` 与 `configs/owt.yaml`。TinyStories 的词表 10000，OWT 的词表 32000，两者模型架构与优化器数值相同。OWT 的 BPE 需要约 100GB 内存，本机 48GB 内存不够，`data/owt_train.npy` 与 `data/owt_valid.npy` 尚未生成，这份配置暂时不能运行。

`toknizer_params/` 里只有 TinyStories 的词表与合并文件。`scripts/tokenize_data.py` 仍是空文件，`data/train.npy` 与 `data/val.npy` 还没有生成，`validate_config` 的文件检查会在这里失败。

