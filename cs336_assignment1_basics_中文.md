# CS336 作业一（基础）：构建 Transformer 语言模型

版本 26.0.3

CS336 教学团队

2026 年春季

> 说明：本文是 handout 的中文翻译。代码、函数签名、公式、样例文本和参考文献保留原文，方便对照原 PDF 阅读。

---

## 1 作业概览

本次作业要求你从零搭建训练一个标准 Transformer 语言模型（LM）所需的全部组件，并实际训出几个模型。

**你需要实现的内容**

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数与 AdamW 优化器（第 4 节）
4. 训练循环，支持模型状态和优化器状态的序列化与加载（第 5 节）

**你需要跑起来的内容**

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器编码数据集，把文本转成整数 ID 序列。
3. 在 TinyStories 上训练 Transformer LM。
4. 用训练好的 Transformer LM 生成样本，并评估困惑度。
5. 在 OpenWebText 上训练模型，把得到的困惑度提交到排行榜。

**你可以使用的东西**

我们期望每个组件都由你从零实现。具体来说，除了下面这些，你不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）¹
- `torch.optim.Optimizer` 基类

其他 PyTorch 定义都可以用。如果你想用某个函数或类，又拿不准是否被允许，可以到 Slack 上问。拿不准的时候，先想想用它是否会损害这份作业「从零实现」的初衷。

**关于 AI 工具的说明**

AI 目前可以完全自主地解决作业的许多部分，这会让学生更难深入投入课程材料、从中获得学习。

使用 AI 工具回答高层次的概念问题，或者提供函数签名、库 API 这类底层的编程文档，是允许的。但是，用 AI 工具实现作业的任何部分都是不允许的。这包括编码 agent（例如 Cursor Agents、Codex、Claude Code）和 AI 自动补全（例如 Cursor Tab、GitHub Copilot）。使用 AI agent 时，确保它读取所提供的 `AGENTS.md` 文件；使用聊天机器人时，也应把这份提示词一并附上。

¹ 完整列表见 pytorch.org/docs/stable/nn.html#containers。

我们强烈建议你在做作业时关闭 IDE 里的 AI 自动补全（例如 Cursor Tab、GitHub Copilot）（非 AI 的自动补全，例如补全函数名，完全没问题）。以前的学生反馈，关掉 AI 自动补全之后更容易深入理解材料。

完整的 AI 政策见这份文档。

**代码长什么样**

作业代码和本 handout 都放在 GitHub 上：

github.com/stanford-cs336/assignment1-basics

请 `git clone` 这个仓库。如果有更新，我们会通知你，届时 `git pull` 拿到最新版本即可。

1. `cs336_basics/*`：你写代码的地方。注意这里本来没有代码，你完全可以从零自由发挥。
2. `adapters.py`：定义了一组你的代码必须具备的功能。对于每一个功能（例如 scaled dot product attention），你只需在其中（例如 `run_scaled_dot_product_attention`）调用自己的实现。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑，这里只是胶水代码。
3. `test_*.py`：包含你必须通过的全部测试（例如 `test_scaled_dot_product_attention`），它们会调用 `adapters.py` 中定义的钩子。不要修改测试文件。

**如何提交**

提交时运行 `make_submission.sh` 生成提交用的 zip 文件。如果你有不想放进提交包的大数据文件或检查点，记得把它们加到脚本的排除列表中。

你需要在 Gradescope 上提交以下文件：

- `writeup.pdf`：回答所有书面问题。请排版后再提交。
- `code.zip`：你写的全部代码。

要提交到排行榜，向下面这个仓库提 PR：

github.com/stanford-cs336/assignment1-basics-leaderboard

详细的提交说明见排行榜仓库里的 `README.md`。

**数据集从哪里获取**

本次作业使用两个预处理好的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两个数据集各自是一个大型纯文本文件。

如果你跟着课程做，可以在 compute guide 里找到下载数据集的说明。

如果你是自学，可以用 `README.md` 里的命令下载这些文件。

> **低资源提示：开篇**
>
> 在整套课程作业中，我们会给出一些在 GPU 资源较少甚至没有 GPU 时完成作业的建议。例如，我们会建议缩小数据集或模型规模，或者说明如何在 Mac 集成显卡或 CPU 上跑训练代码。这些「低资源提示」会放在像这样的蓝色框里。即使你是有课程机器使用权限的斯坦福在册学生，这些建议也能帮你更快迭代、节省时间，建议读一读。

> **低资源提示：在 Apple Silicon 或 CPU 上做作业一**
>
> 用官方参考实现，我们可以在 36 GB 内存的 Apple M4 Max 芯片上训出一个能生成相当流畅文本的 LM：Metal GPU（MPS）上不到 5 分钟，CPU 上约 30 分钟。如果这些名词对你来说没什么概念，不用担心。你只需要知道，如果你有一台配置还算新的笔记本，实现正确且高效，就能训出一个小 LM，生成简单的儿童故事，流畅度还不错。
>
> 作业后文会说明在 CPU 或 MPS 上需要做哪些调整。

---

## 2 字节对编码（BPE）分词器

作业的第一部分，我们要训练并实现一个字节级 BPE 分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们把任意（Unicode）字符串表示成字节序列，在这个字节序列上训练 BPE 分词器。随后用这个分词器把文本（字符串）编码成 token（整数序列），供语言模型使用。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，把字符映射到整数码点。截至 Unicode 17.0（2025 年 9 月发布），该标准在 172 种书写系统中定义了 159,801 个字符。例如，字符 "s" 的码点是 115（通常写作 U+0073，其中 U+ 是约定前缀，0073 是 115 的十六进制），字符 "牛" 的码点是 29275。在 Python 中，可以用 `ord()` 函数把单个 Unicode 字符转换成它的整数表示，用 `chr()` 函数把整数码点转换成对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**问题（unicode1）：理解 Unicode（1 分）**

(a) `chr(0)` 返回的是哪个 Unicode 字符？

交付物：一句话回答。

(b) 这个字符的字符串表示（`__repr__()`）和打印出来的表示有何不同？

交付物：一句话回答。

(c) 这个字符出现在文本里时会发生什么？可以在 Python 解释器里试一试下面这些，看看是否符合你的预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

交付物：一句话回答。

### 2.2 Unicode 编码

Unicode 标准定义了字符到码点（整数）的映射，但直接在码点上训练分词器并不现实：词表会大到难以承受（约 15 万项），而且极其稀疏（很多字符非常罕见）。因此我们改用 Unicode 编码，把 Unicode 字符转换成字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页）。

把 Unicode 字符串编码成 UTF-8 可以用 Python 的 `encode()`；要取出 `bytes` 对象底层的字节值，可以遍历它（例如调用 `list()`）；反过来，用 `decode()` 可以把 UTF-8 字节串解码回 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

把 Unicode 码点转换成字节序列（例如通过 UTF-8 编码），本质上就是把一串码点（21 位整数，有 159,801 个合法值）变成一串字节值（0 到 255 的整数）。256 项的词表规模可管理得多。使用字节级分词时，也不用担心 out-of-vocabulary 的问题，因为任何输入文本都能表示成 0 到 255 的整数序列。

**问题（unicode2）：Unicode 编码（3 分）**

(a) 相对于 UTF-16 或 UTF-32，选用 UTF-8 编码的字节来训练分词器有哪些理由？比较一下这几种编码在不同输入字符串上的输出会有所帮助。

交付物：一到两句话回答。

(b) 下面这个（错误的）函数本想将 UTF-8 字节串解码成 Unicode 字符串。它错在哪里？给出一个会产生错误结果的输入字节串例子。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

```python
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

交付物：一个会使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串，并用一句话解释该函数为何不正确。

(c) 给出一个不能解码成任何 Unicode 字符的两字节序列。

交付物：一个例子，并附一句话解释。

### 2.3 子词分词

字节级分词虽然缓解了词级分词器的 out-of-vocabulary 问题，但把文本切成字节会使输入序列极长。这会拖慢模型训练：一句 10 个词的句子，在词级语言模型里可能只有 10 个 token，在字符级模型里却可能有 50 个甚至更多（取决于词的长度）。更长的序列意味着模型每一步都要做更多计算。而且，在字节序列上做语言建模本身就更困难，因为更长的输入序列会在数据中造成长程依赖。

子词分词介于词级分词器和字节级分词器之间。注意字节级分词器的词表只有 256 项（字节取值 0 到 255）。子词分词器用更大的词表换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在原始训练文本中频繁出现，把它作为一项加进词表，就能把这个 3 token 的序列压缩成单个 token。

怎么挑选要加进词表的子词单元？R. Sennrich 等人 [3] 提出使用字节对编码（BPE；P. Gage [5]），这是一种压缩算法：迭代地把出现最频繁的字节对替换（"合并"）成一个新的、尚未使用的索引。注意这个算法加入子词 token 的方式，是为了最大化输入序列的压缩率——如果一个词在输入文本里出现得足够多，它就会变成单个子词单元。

用 BPE 构建词表的子词分词器通常就叫 BPE 分词器。本次作业中，我们实现字节级 BPE 分词器：词表项是字节或合并后的字节序列，在 out-of-vocabulary 处理和输入序列长度两方面都兼顾了。构建 BPE 分词器词表的过程，被称为「训练」BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包含三个主要步骤。

**词表初始化**

分词器词表是字节串 token 到整数 ID 的一一映射。由于训练的是字节级 BPE 分词器，初始词表就是所有字节的集合。字节值共有 256 种，所以初始词表大小为 256。

**预分词**

有了词表之后，原则上可以统计文本中相邻字节共同出现的次数，从最频繁的字节对开始合并。但这样计算开销很大，因为每合并一次都要把整个语料过一遍。此外，直接在语料上跨字符合并字节，可能产生只在标点上不同的 token（例如 `dog!` 和 `dog.`）。这些 token 会得到完全不同的 token ID，尽管它们的语义很可能高度相似（只差一个标点）。

为了避免这个问题，我们先对语料做预分词。你可以把它理解为对语料做一次粗粒度分词，方便统计字符对的出现频率。例如，单词 `'text'` 可能是一个出现 10 次的预分词单元。这样统计字符 `'t'` 和 `'e'` 相邻出现了多少次时，只需看到单词 `'text'` 里 `'t'` 和 `'e'` 相邻，就把计数加 10，不必通读整个语料。由于训练的是字节级 BPE 模型，每个预分词单元表示为一个 UTF-8 字节序列。

R. Sennrich 等人 [3] 最初的 BPE 实现只按空白切分（即 `s.split(" ")`）。基于 SentencePiece 的分词器（例如 Llama 1 和 2 的分词器）至今仍在用这种方法。

现代分词器大多采用基于正则的预分词器，这一做法来自 GPT-2；A. Radford 等人 [6]。我们使用的正则来自 github.com/openai/tiktoken/pull/234/files，是原始正则的一个稍整洁的版本：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

用这个预分词器交互式地切一些文本，有助于体会它的行为：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过在代码里用的时候，应当用 `re.finditer`，这样构建「预分词单元 → 计数」映射时不必把所有预分词结果都存下来。

**计算 BPE 合并**

现在输入文本已经转成了预分词单元，每个预分词单元也表示成了 UTF-8 字节序列，可以开始计算 BPE 合并（即训练 BPE 分词器）了。概括地说，BPE 算法迭代地统计每一对字节，找出出现频率最高的字节对（"A"、"B"），然后把每一处（"A"、"B"）合并，即替换成新 token "AB"。这个新合并出的 token 加入词表；因此 BPE 训练结束后的最终词表大小 = 初始词表大小（本例为 256）+ 训练过程中执行的 BPE 合并次数。为了 BPE 训练的效率，我们不考虑跨越预分词边界的字节对。² 计算合并时，若多个字节对频率相同，按确定性规则打破平局：优先选择字典序更大的字节对。例如，如果 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 的出现频率并列最高，我们合并 ("BA", "A")：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**特殊 token**

某些字符串（例如 `<|endoftext|>`）常用来编码元信息（例如文档之间的边界）。编码文本时，通常希望把某些字符串当作「特殊 token」处理，它们永远不被拆成多个 token，始终作为一个整体保留。例如，序列结束字符串 `<|endoftext|>` 应当始终保留为单个 token（即单个整数 ID），这样才能知道何时停止从语言模型生成。这些特殊 token 必须加入词表，从而拥有固定的 token ID。

R. Sennrich 等人 [3] 的 Algorithm 1 包含一个低效的 BPE 分词器训练实现（基本就是我们上面概述的步骤）。作为第一个练习，实现并测试这个函数有助于检验你的理解。

**示例（bpe_example）：BPE 训练示例**

下面是 R. Sennrich 等人 [3] 中的一个风格化示例。考虑由以下文本组成的语料：

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

词表中还有一个特殊 token `<|endoftext|>`。

*词表*

词表初始化为特殊 token `<|endoftext|>` 加上 256 个字节值。

*预分词*

为了简单、专注于合并过程，本例假设预分词就是按空白切分。预分词并计数后，得到频率表：

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

用 `dict[tuple[bytes, ...], int]` 表示比较方便，例如 `{(l,o,w): 5, …}`。注意即使单个字节，在 Python 中也是 `bytes` 对象。Python 没有表示单字节的 byte 类型，就像没有表示单字符的 char 类型一样。

*合并*

先看每一对相邻字节，把它们所在单词的频率求和：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。字节对 `('e', 's')` 和 `('s', 't')` 并列，取字典序较大的 `('s', 't')`。合并后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

第二轮中，`(e, st)` 是最常见的字节对（计数为 9），合并后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续下去，最终得到的合并序列是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果只取 6 次合并，得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，词表元素是 `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。

有了这个词表和合并集合，单词 `newest` 会分词为 `[ne, west]`。

² 注意 R. Sennrich 等人 [3] 的原始 BPE 公式中指定了加入一个词尾 token。训练字节级 BPE 模型时我们不加词尾 token，因为模型词表已经包含了所有字节（包括空白和标点）。既然空格和标点都被显式表示，学到的 BPE 合并自然会反映这些词边界。

### 2.5 动手实验：BPE 分词器训练

现在在 TinyStories 数据集上训练一个字节级 BPE 分词器。数据集的定位和下载说明见第 1 节。开始之前，建议先浏览一下 TinyStories 数据集，了解数据长什么样。

**并行化预分词**

你会发现预分词是主要的性能瓶颈。用内置的 `multiprocessing` 库并行化，可以加速预分词。具体建议：并行实现中把语料分块，并保证块的边界落在某个特殊 token 的开头。可以直接使用下面链接里的起始代码来获取分块边界，再把工作分发给各进程：

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

这样分块总是合法的，因为我们从不跨文档边界合并。作业范围内，你都可以按这种方式切分。不必担心拿到一个不含 `<|endoftext|>` 的超大语料的边缘情况。

**预分词之前先去掉特殊 token**

用正则模式（`re.finditer`）做预分词之前，应当先把语料（或并行实现中的分块）里的所有特殊 token 去掉。要按特殊 token 切分，保证不会跨过它们所界定的文本发生合并。例如，语料（或分块）是 `[Doc 1]<|endoftext|>[Doc 2]` 时，应当按特殊 token `<|endoftext|>` 切分，分别对 `[Doc 1]` 和 `[Doc 2]` 预分词，这样就不会跨文档边界合并。换句话说，训练时特殊 token 是硬性的切分边界，但它们本身不应计入合并统计。可以用 `re.split`、以 `"|".join(special_tokens)` 为分隔符实现（注意 `|` 可能出现在特殊 token 中，要用 `re.escape` 小心处理）。测试 `test_train_bpe_special_tokens` 会检查这一点。

**优化合并步骤**

上面风格化示例中的朴素 BPE 训练实现很慢，因为每次合并都要遍历所有字节对来找最频繁的那个。可是每次合并之后，计数会发生变化的字节对只有与合并对重叠的那些。因此可以给所有字节对的计数建索引、增量更新，不必显式遍历每一对字节来统计频率，从而显著加速 BPE 训练。这种缓存做法能带来可观的提速；不过要注意，BPE 训练中的合并部分在 Python 里无法并行化。

> **低资源提示：性能分析**
>
> 用 `cProfile`、`py-spy` 这类性能分析工具找出实现中的瓶颈，集中优化这些地方。

> **低资源提示：「缩小规模」**
>
> 与其一上来就在完整的 TinyStories 上训练分词器，建议先在数据的一个小子集（「调试数据集」）上训练。例如可以改用 TinyStories 的验证集训练分词器，它是 2.2 万篇文档，而不是 212 万篇。这体现了一个通用策略：开发时尽可能缩小规模来加速，例如用更小的数据集、更小的模型等。调试数据集或超参数配置的规模需要仔细权衡：调试集要大到能复现完整配置下的瓶颈（这样你做的优化才能推广），又要小到不至于跑太久。

**问题（train_bpe）：BPE 分词器训练（15 分）**

交付物：写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。函数至少要处理以下输入参数：

**输入**

- `input_path: str`：BPE 分词器训练数据文本文件的路径。
- `vocab_size: int`：正整数，定义最终词表的最大规模（包括初始字节词表、合并产生的词表项以及所有特殊 token）。
- `special_tokens: list[str]`：要加入词表的字符串列表。训练时把它们当作硬边界，阻止跨其所在范围的合并；但计算合并统计时不把它们算进去。

BPE 训练函数应返回得到的词表和合并列表：

**输出**

- `vocab: dict[int, bytes]`：分词器词表，从 int（词表中的 token ID）到 bytes（token 字节）的映射。
- `merges: list[tuple[bytes, bytes]]`：训练产生的 BPE 合并列表。每一项是一个字节元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排列。

要用我们提供的测试检验你的 BPE 训练函数，先实现测试适配器 `[adapters.run_train_bpe]`，然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应当能通过全部测试。可选地（可能相当耗时），你可以用某种系统语言实现训练方法的若干关键部分，例如 C++（可考虑 cppyy 或 nanobind）或 Rust（用 PyO3）。如果这么做，注意哪些操作需要拷贝、哪些可以直接读 Python 内存，并留下构建说明，或者确保仅靠 `pyproject.toml` 就能构建。另外注意 GPT-2 的正则在多数正则引擎里支持得不好，支持的那些也常常太慢。我们验证过 Oniguruma 速度尚可且支持负向前瞻，而 Python 的 `regex` 包如果要说的话还更快。

**问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）**

(a) 在 TinyStories 数据集上训练字节级 BPE 分词器，最大词表规模设为 10,000。记得把 TinyStories 的 `<|endoftext|>` 特殊 token 加入词表。把得到的词表和合并序列化到磁盘，便于后续查看。训练用了多少时间和内存？词表里最长的 token 是什么？它是否合理？

资源需求：≤30 分钟（无 GPU），≤30 GB 内存

> **提示** 预分词用多进程并行，再加上下面两个事实，BPE 训练应该能在 2 分钟内完成：
>
> (a) 数据文件中 `<|endoftext|>` token 分隔文档。
> (b) 在应用 BPE 合并之前，`<|endoftext|>` token 已作为特殊情况处理。

交付物：一到两句话回答。

(b) 对你的代码做性能分析。分词器训练过程中哪一部分最耗时？

交付物：一到两句话回答。

接下来，尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。和前面一样，建议先浏览数据集，了解其内容。

**问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）**

(a) 在 OpenWebText 数据集上训练字节级 BPE 分词器，最大词表规模设为 32,000。把得到的词表和合并序列化到磁盘，便于后续查看。词表里最长的 token 是什么？它是否合理？

资源需求：≤12 小时（无 GPU），≤100 GB 内存

交付物：一到两句话回答。

(b) 比较并对比在 TinyStories 和 OpenWebText 上训练得到的分词器。

交付物：一到两句话回答。

### 2.6 BPE 分词器：编码与解码

上一部分我们实现了在输入文本上训练 BPE 分词器的函数，得到了词表和 BPE 合并列表。现在要实现一个 BPE 分词器：加载给定的词表和合并列表，用它们把文本与 token ID 互相转换。

#### 2.6.1 编码文本

用 BPE 编码文本的过程与训练 BPE 词表的过程相仿，主要有几步。

**第 1 步：预分词。** 先对序列做预分词，并把每个预分词单元表示成 UTF-8 字节序列，和 BPE 训练时一样。我们在每个预分词单元内部把这些字节合并成词表元素，各预分词单元独立处理（不跨预分词边界合并）。

**第 2 步：应用合并。** 然后按创建顺序，把 BPE 训练得到的词表元素合并序列应用到预分词单元上。

**示例（bpe_encoding）：BPE 编码示例**

假设输入字符串是 `'the cat ate'`，词表为 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的合并为 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，预分词器把字符串切成 `['the', ' cat', ' ate']`。然后对每个预分词单元应用 BPE 合并。

第一个预分词单元 `'the'` 初始表示为 `[b't', b'h', b'e']`。查看合并列表，第一个适用的合并是 `(b't', b'h')`，据此把该预分词单元变成 `[b'th', b'e']`。再回到合并列表，下一个适用的合并是 `(b'th', b'e')`，把它变成 `[b'the']`。最后再看合并列表，已经没有适用于该字符串的合并了（整个预分词单元已合并成单个 token），合并过程结束。对应的整数序列是 `[9]`。

对剩下的预分词单元重复这个过程：预分词单元 `' cat'` 应用合并后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`；最后一个预分词单元 `' ate'` 应用合并后为 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此输入字符串的最终编码结果为 `[9, 7, 1, 5, 10, 3]`。

**特殊 token**

分词器编码文本时应当能正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存方面的考虑**

假设要分词的文本文件很大，无法全部放进内存。为了高效地处理这个大文件（或其他数据流），需要把它切成可管理的块，逐块处理，使内存复杂度为常数，而不随文本大小线性增长。这样做时，必须保证 token 不跨越块边界，否则分词结果会与一次性在内存中处理整个序列的朴素方法不同。

#### 2.6.2 解码文本

要把整数 token ID 序列解码回原始文本，只需在词表中查出每个 ID 对应的项（一个字节序列），拼接起来，再把这些字节解码成 Unicode 字符串。注意输入的 ID 不保证能映射到合法的 Unicode 字符串（用户可以输入任意整数 ID 序列）。当输入 token ID 无法产生合法 Unicode 字符串时，应当把畸形字节替换为官方 Unicode 替换字符 U+FFFD。³ `bytes.decode` 的 `errors` 参数控制 Unicode 解码错误的处理方式，使用 `errors='replace'` 会自动把畸形数据替换成替换标记。

³ 关于 Unicode 替换字符的更多信息见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

**问题（tokenizer）：实现分词器（15 分）**

交付物：实现一个 `Tokenizer` 类，给定词表和合并列表，能把文本编码成整数 ID，把整数 ID 解码回文本。分词器还应支持用户提供的特殊 token（若词表中没有，则追加到词表末尾）。推荐使用以下接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```

从给定的词表、合并列表和（可选的）特殊 token 列表构造分词器。该函数应接受以下参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

类方法，从序列化的词表和合并列表（格式与你的 BPE 训练代码输出相同）以及（可选的）特殊 token 列表构造并返回 `Tokenizer`。该方法应接受以下额外参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

把输入文本编码成 token ID 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定字符串的可迭代对象（例如 Python 文件句柄），返回一个惰性产出 token ID 的生成器。对于无法直接载入内存的大文件，为了实现内存高效的分词，这个方法是必需的。

```python
def decode(self, ids: list[int]) -> str
```

将 token ID 序列解码为文本。

要用我们提供的测试检验你的 `Tokenizer`，先实现测试适配器 `[adapters.get_tokenizer]`，然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应当能通过全部测试。

### 2.7 实验

**问题（tokenizer_experiments）：分词器实验（4 分）**

(a) 从 TinyStories 和 OpenWebText 中各采样 10 篇文档，用你前面训练好的 TinyStories 和 OpenWebText 分词器（词表规模分别为 10K 和 32K）把这些文档编码成整数 ID。两个分词器的压缩比（bytes/token）各是多少？

交付物：一到两句话回答。

(b) 如果用 TinyStories 分词器去编码 OpenWebText 样本，会发生什么？比较压缩比，或者定性地描述现象。

交付物：一到两句话回答。

(c) 估算分词器的吞吐量（例如每秒处理多少字节）。给 Pile 数据集（825GB 文本）做分词需要多久？

交付物：一到两句话回答。

(d) 用你的 TinyStories 和 OpenWebText 分词器，分别把对应的训练集和开发集编码成整数 token ID 序列。后面训练语言模型时要用到。建议把 token ID 序列化为 `uint16` 数据类型的 NumPy 数组。为什么 `uint16` 是合适的选择？

交付物：一到两句话回答。

---

## 3 Transformer 语言模型架构

语言模型接收一个批量的整数 token ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），返回（批量的）词表上的归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中预测分布是针对每个输入 token 的下一个词的分布。训练语言模型时，我们用这些下一词预测来计算实际下一词与预测下一词之间的交叉熵损失。推理时从语言模型生成文本，我们取最后一个时间步（即序列的最后一项）的下一词预测分布来生成序列的下一个 token（例如取概率最大的 token、从分布中采样等），把生成的 token 加入输入序列，如此重复。

本部分中，你要从零构建这个 Transformer 语言模型。我们先对模型做高层次描述，再逐步展开各个组件。

### 3.1 Transformer LM

给定 token ID 序列，Transformer 语言模型用输入嵌入把 token ID 转成稠密向量，把嵌入后的 token 送入 `num_layers` 个 Transformer 块，最后通过一个可学习的线性投影（「输出嵌入」或「LM head」）产生预测的下一 token logits。示意图见图 1。

> 图 1：我们的 Transformer 语言模型总览。
>
> 图 2：pre-norm Transformer 块。

**Token 嵌入**

第一步，Transformer 把（批量的）token ID 序列嵌入成一组包含 token 身份信息的向量（图 1 中的红色块）。

具体来说，给定 token ID 序列，Transformer 语言模型用 token 嵌入层产生一组向量。嵌入层接收形状为 `(batch_size, sequence_length)` 的整数张量，输出形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

**Pre-norm Transformer 块**

嵌入之后，激活值经过若干结构相同的神经网络层。标准的 decoder-only Transformer 语言模型由 `num_layers` 个相同的层组成（通常称为 Transformer「块」）。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，返回同形状的输出。每个块在序列范围内聚合信息（通过自注意力），再对信息做非线性变换（通过前馈层）。

经过 `num_layers` 个 Transformer 块后，我们取最终的激活值，把它转成词表上的分布。

我们实现的是「pre-norm」Transformer 块（详见 3.4 节），它在最后一个 Transformer 块之后还需要做一次层归一化，以保证输出尺度合适。

归一化之后，用一个标准的可学习线性变换，把 Transformer 块的输出转换成预测的下一 token logits（参见例如 A. Radford 等人 [7] 的式 2）。

### 3.2 附注：批处理、Einsum 与高效计算

整个 Transformer 中，我们会对许多类似批次的输入执行相同的计算。例如：

- **批次元素**：对每个批次元素施加相同的 Transformer 前向操作。
- **序列长度**：RMSNorm、前馈这类「逐位置」操作在序列的每个位置上执行完全相同的计算。
- **注意力头**：在「多头」注意力操作中，注意力运算在多个注意力头上批量执行。

用一种符合人体工学的方式来执行这些操作、充分利用 GPU 并且易于阅读和理解，是很有价值的。许多 PyTorch 操作可以接受张量开头额外的「类批次」维度，并高效地沿这些维度重复/广播运算。

例如，假设我们在做逐位置、批量的运算。有一个形状为 `(batch_size, sequence_length, d_model)` 的「数据张量」D，想让它与形状为 `(d_model, d_model)` 的矩阵 A 做批量向量-矩阵乘法。此时 `D @ A` 会执行批量矩阵乘法，这是 PyTorch 中一个高效的原语，`(batch_size, sequence_length)` 两个维度会被批量处理。

因此，设计函数时最好假定它可能收到额外的类批次维度，并把这些维度放在 PyTorch 形状的开头。为了把张量组织成这种可批量处理的形式，可能需要多步 `view`、`reshape` 和 `transpose`。这会有点烦，而且常常难以看懂代码在做什么、你的张量是什么形状。

更顺手的选择是使用 `torch.einsum` 里的 einsum 记法，或者使用与框架无关的库，如 einops 或 einx。两个关键操作是 `einsum`（可对任意维度的输入张量做张量收缩）和 `rearrange`（可对任意维度重排、拼接、拆分）。机器学习中几乎所有运算都是「维度腾挪」和「张量收缩」的组合，外加偶尔的（通常是逐点的）非线性函数。这意味着用 einsum 记法写代码，可读性和灵活性通常好得多。

我们强烈建议在本课程中学习并使用 einsum 记法。之前没接触过 einsum 的同学请用 einops（文档见此处），已经熟悉 einops 的同学请学习更通用的 einx（见此处）。⁴ 这两个包在提供的环境里都已安装。

下面给出几个使用 einsum 记法的例子。这些是 einops 文档的补充，建议先读文档。

⁴ 值得注意，einops 的支持面很广，einx 则尚未经过同等程度的实战检验。如果在 einx 中发现限制或 bug，可以自在地退回到 einops 加一些普通 PyTorch。

**示例（einstein_example1）：用 einops.einsum 做批量矩阵乘法**

```python
import torch
from einops import rearrange, einsum
## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?
## Einsum is self-documenting and robust
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**示例（einstein_example2）：用 einops.rearrange 做广播运算**

有一批图像，对每张图像，我们要根据某个缩放因子生成 10 个变暗版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
## Reshape and multiply
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**示例（einstein_example3）：用 einops.rearrange 做像素混合**

假设有一批图像，表示为形状 `(batch, height, width, channel)` 的张量，我们要在图像所有像素上做线性变换，但每个通道独立进行。线性变换表示为形状 `(height * width, height * width)` 的矩阵 B。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)
## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改用 einops：

```python
height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你想玩点花的：用 `einx.dot`（einx 中相当于 `einops.einsum` 的写法）一口气完成：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

上面第一种实现可以通过在前后加注释来标明输入输出形状，但这样很笨重，也容易出错。用 einsum 记法，文档即实现。

Einsum 记法能处理任意输入批次维度，还有一个关键好处：自带文档。使用 einsum 记法的代码中，输入输出张量的相关形状一目了然。至于剩下的张量，可以考虑使用 Tensor 类型提示，例如 jaxtyping 库（并非 JAX 专用）。

关于 einsum 记法的性能影响，我们会在作业二中详细讨论；现在只需知道它们几乎总是比替代方案更好。

#### 3.2.1 数学记号与内存顺序

很多机器学习论文在记号中使用行向量，这种表示与 NumPy 和 PyTorch 默认的 row-major 内存顺序契合得很好。用行向量时，线性变换写作

$$y = xW^\top \tag{1}$$

其中 W ∈ ℝ^{d_out × d_in} 按 row-major 存储，x ∈ ℝ^{1 × d_in} 是行向量。注意这样可以通过增加 x 的最外层维度来批量输入，即把向量输入 x 换成矩阵输入 X ∈ ℝ^{batch × d_in}。

在线性代数中更常见的是列向量，线性变换写作

$$y = Wx \tag{2}$$

其中 W ∈ ℝ^{d_out × d_in} 按 row-major 存储，x ∈ ℝ^{d_in} 是列向量。这种情况下要批量输入，批次维度得放在最后，即 x 要换成矩阵 X̃ ∈ ℝ^{d_in × batch}。

本次作业的数学记号主要使用列向量，因为数学惯例如此。要记住：如果你想用普通的矩阵乘法记号，就得像式 1 中的行向量约定那样对矩阵做转置，因为 PyTorch 使用 row-major 内存顺序。如果用 einsum 做线性代数运算，只要轴标注正确，这就不是问题。顺带一提，Matlab、Julia、Fortran 等其他语言/线性代数包使用 column-major 内存顺序，批次维度放在最后，而 Python 及相关包采用了 C 标准的 row-major 顺序。

### 3.3 基础构件：Linear 与 Embedding 模块

#### 3.3.1 参数初始化

有效训练神经网络往往需要仔细初始化模型参数——糟糕的初始化会导致梯度消失或爆炸等不良行为。pre-norm Transformer 对初始化异常地稳健，但初始化仍会显著影响训练速度和收敛。本次作业已经很长，细节留到作业三，这里先给出在大多数情况下都很好用的近似初始化：

- **Linear 权重**：𝒩(μ = 0, σ² = 2/(d_in + d_out))，截断在 [−3σ, 3σ]。
- **Embedding**：𝒩(μ = 0, σ² = 1)，截断在 [−3, 3]。
- **RMSNorm**：全 1。

应当用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 乃至一般神经网络的基本构件。首先，实现你自己的 `Linear` 类，继承 `torch.nn.Module`，执行线性变换：

$$y = Wx \tag{3}$$

注意，与大多数现代 LLM 一致，我们不包含偏置项。

**问题（linear）：实现 linear 模块（1 分）**

交付物：实现一个继承 `torch.nn.Module`、执行线性变换的 `Linear` 类。实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，只是没有 bias 参数或 bias 形参。推荐接口如下：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造线性变换模块。该函数应接受以下参数：

- `in_features: int`：输入的最后一维大小
- `out_features: int`：输出的最后一维大小
- `device: torch.device | None = None`：存放参数所在的设备
- `dtype: torch.dtype | None = None`：参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入施加线性变换。

务必做到：

- 继承 `nn.Module`
- 调用父类构造函数
- 把参数构造并保存为 W（而不是 Wᵀ），放进 `nn.Parameter`
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

初始化按上面的设置，用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试 `Linear` 模块，实现测试适配器 `[adapters.run_linear]`。适配器应把给定的权重加载进你的 `Linear` 模块，可以用 `Module.load_state_dict`。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如前所述，Transformer 的第一层是嵌入层，把整数 token ID 映射到 d_model 维向量空间。我们实现一个继承 `torch.nn.Module` 的自定义 `Embedding` 类（因此不要用 `nn.Embedding`）。`forward` 方法应当用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID，对形状为 `(vocab_size, d_model)` 的嵌入矩阵做索引，取出每个 token ID 对应的嵌入向量。

**问题（embedding）：实现 embedding 模块（1 分）**

交付物：实现继承 `torch.nn.Module`、执行嵌入查找的 `Embedding` 类。实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。推荐接口如下：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造嵌入模块。该函数应接受以下参数：

- `num_embeddings: int`：词表大小
- `embedding_dim: int`：嵌入向量维度，即 d_model
- `device: torch.device | None = None`：存放参数所在的设备
- `dtype: torch.dtype | None = None`：参数的数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token ID 对应的嵌入向量。

务必做到：

- 继承 `nn.Module`
- 调用父类构造函数
- 把嵌入矩阵初始化为 `nn.Parameter`
- 保存嵌入矩阵时以 d_model 作为最后一维
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

初始化同样用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的实现，实现测试适配器 `[adapters.run_embedding]`，然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：多头自注意力机制和逐位置前馈网络（[A. Vaswani et al., 2017]，3.1 节）。

原始 Transformer 论文中，模型在两个子层各自周围使用残差连接，随后接层归一化。这种架构通常称为「post-norm」Transformer，因为层归一化作用在子层输出上。后来大量工作发现，把层归一化从每个子层的输出移到每个子层的输入（并在最后一个 Transformer 块之后再加一次层归一化），能提升 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——这种「pre-norm」Transformer 块的可视化见图 2。于是每个 Transformer 块子层的输出通过与子层输入的残差连接相加（A. Vaswani et al. [8]，5.4 节）。pre-norm 的直觉是：从输入嵌入一直到 Transformer 最终输出，存在一条干净的、不经过任何归一化的「残差流」，据说这能改善梯度流动。这种 pre-norm Transformer 如今已成为语言模型的标准做法（例如 GPT-3、LLaMA、PaLM 等），所以我们也实现这一变体。下面逐一讲解 pre-norm Transformer 块各组件，并按顺序实现。

#### 3.4.1 均方根层归一化

A. Vaswani 等人 [8] 的原始 Transformer 使用层归一化 [J. L. Ba et al., 2016] 来归一化激活值。沿用 H. Touvron 等人 [12]，我们用均方根层归一化（RMSNorm；B. Zhang et al. [13]，式 4）来做层归一化。给定激活值向量 a ∈ ℝ^{d_model}，RMSNorm 按如下方式重新缩放每个激活值 a_i：

$$\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} g_i \tag{4}$$

其中 RMS(a) = √( (1/d_model) Σ_{i=1}^{d_model} a_i² + ε )。这里 g_i 是可学习的「增益」参数（共 d_model 个），ε 是常固定为 1e-5 的超参数。

应当在平方之前把输入升到 `torch.float32` 以防溢出。总体而言，`forward` 方法应当长这样：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
```

**问题（rmsnorm）：均方根层归一化（1 分）**

交付物：把 RMSNorm 实现为一个 `torch.nn.Module`。推荐接口如下：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 RMSNorm 模块。该函数应接受以下参数：

- `d_model: int`：模型的隐藏维度
- `eps: float = 1e-5`：用于数值稳定性的 epsilon
- `device: torch.device | None = None`：存放参数所在的设备
- `dtype: torch.dtype | None = None`：参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，返回同形状张量。

注意：如上所述，执行归一化前记得把输入升到 `torch.float32`（之后再降回原 dtype）。

要测试你的实现，实现测试适配器 `[adapters.run_rmsnorm]`，然后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

> 图 3：SiLU（又称 Swish）与 ReLU 激活函数对比。

原始 Transformer 论文（A. Vaswani et al. [8] 3.3 节）中，Transformer 的前馈网络由两个线性变换构成，中间夹一个 ReLU 激活（ReLU(x) = max(0, x)）。在那个原始架构里，内层前馈层的维度通常是输入维度的 4 倍。

现代语言模型相对这个原始设计通常有两处主要变化：换用另一种激活函数，并引入门控机制。具体来说，我们实现被 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用的「SwiGLU」激活：它把 SiLU（常称为 Swish）激活与一种称为门控线性单元（GLU）的门控机制结合起来。我们还按照自 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来的现代 LLM 做法，去掉线性层里有时会用的偏置项。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} \tag{5}$$

由图 3 可见，SiLU 与 ReLU 相似，但在零点处光滑。

门控线性单元（GLU）最初由 Y. N. Dauphin 等人 [19] 定义为：一个线性变换经过 sigmoid 后与另一个线性变换逐元素相乘：

$$\mathrm{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x \tag{6}$$

其中 ⊙ 表示逐元素乘法。据称 GLU 通过「为梯度提供线性通路，同时保留非线性能力」来缓解深层架构中的梯度消失问题。

把 SiLU/Swish 与 GLU 合在一起，就得到 SwiGLU，我们将把它用于前馈网络：

$$\mathrm{FFN}(x) = \mathrm{SwiGLU}(x, W_1, W_2, W_3) = W_2(\mathrm{SiLU}(W_1 x) \odot W_3 x) \tag{7}$$

其中 x ∈ ℝ^{d_model}，W_1, W_3 ∈ ℝ^{d_ff × d_model}，W_2 ∈ ℝ^{d_model × d_ff}，通常取 d_ff = (8/3) d_model。具体实现中，为了硬件效率，把它四舍五入到附近的 64 的倍数即可。

N. Shazeer [20] 最早提出把 SiLU/Swish 激活与 GLU 结合起来，并通过实验表明 SwiGLU 在语言建模任务上优于 ReLU、无门控的 SiLU 等基线。作业后面会让你比较 SwiGLU 与 SiLU。尽管我们给出了一些关于这些组件的启发式论证（论文中也提供了更多支持证据），保持经验主义的视角仍然有益：Shazeer 论文中如今一句著名的话是

> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

（我们无法解释这些架构为何奏效；我们将其成功，如同其他一切，归功于神的恩典。）

**问题（positionwise_feedforward）：实现逐位置前馈网络（2 分）**

交付物：实现 SwiGLU 前馈网络，由 SiLU 激活函数和一个 GLU 组成。

注意：在这一特定场景中，为了数值稳定性，你可以放心使用 `torch.sigmoid`。

实现中应当把 d_ff 设为约 (8/3) × d_model，同时保证内层前馈层维度是 64 的倍数，以充分利用硬件。要用我们提供的测试检验实现，需实现测试适配器 `[adapters.run_swiglu]`，然后运行 `uv run pytest -k test_swiglu`。

#### 3.4.3 相对位置嵌入

为了把位置信息注入模型，我们实现旋转位置嵌入（Rotary Position Embeddings）[J. Su et al., 2021]，通常称为 RoPE。对于位置 i 上的查询 token，q^(i) = W_q x^(i) ∈ ℝ^d，我们施加成对旋转矩阵 R_i，得到 q'^(i) = R_i q^(i) = R_i W_q x^(i)。这里 R_i 把嵌入元素按对 q^(i)_{2k−1:2k} 当作二维向量，旋转角度 θ_{i,k} = i / Θ^{(2k−2)/d}，其中 k ∈ {1, …, d/2}，Θ 为某个常数。因此 R_i 可看作 d × d 的分块对角矩阵，对角块为 R_i^k，k ∈ {1, …, d/2}：

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix} \tag{8}$$

于是得到完整的旋转矩阵

$$R_i = \begin{pmatrix} R_i^1 & 0 & 0 & \cdots & 0 \\ 0 & R_i^2 & 0 & \cdots & 0 \\ 0 & 0 & R_i^3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_i^{d/2} \end{pmatrix} \tag{9}$$

其中 0 表示 2 × 2 零矩阵。虽然可以构造完整的 d × d 矩阵，好的解法应当利用这个矩阵的性质来更高效地实现变换。由于我们只关心给定序列内 token 的相对旋转，cos(θ_{i,k}) 和 sin(θ_{i,k}) 的计算结果可以跨层、跨不同批次复用。如果想优化，可以让所有层共用一个 RoPE 模块，在 `init` 时用 `self.register_buffer(persistent=False)` 建立二维的 sin、cos 预计算缓冲区，而不是用 `nn.Parameter`（我们不希望学习这些固定的余弦、正弦值）。对 q^(i) 做的旋转过程，同样以相应的 R_j 施加到 k^(j) 上。注意这一层没有可学习参数。

**问题（rope）：实现 RoPE（2 分）**

交付物：实现类 `RotaryPositionalEmbedding`，对输入张量施加 RoPE。推荐接口如下：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE 模块，按需创建缓冲区。

- `theta: float`：RoPE 的 Θ 值
- `d_k: int`：查询向量和键向量的维度
- `max_seq_len: int`：将要输入的最大序列长度
- `device: torch.device | None = None`：存放缓冲区所在的设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，返回同形状张量。需容忍 x 具有任意数量的批次维度。可以假定 token 位置是形状为 `(..., seq_len)` 的张量，指定 x 沿序列方向的 token 位置。

应当用 token 位置沿序列维度对你的（可能预计算好的）cos、sin 张量做切片。

要测试你的实现，完成 `[adapters.run_rope]`，确保通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

现在实现 A. Vaswani 等人 [8]（3.2.1 节）描述的缩放点积注意力。作为预备，Attention 操作的定义会用到 softmax：它把一组未归一化的分数向量变成归一化分布：

$$\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)} \tag{10}$$

注意 exp(v_i) 在数值较大时可能变成 inf（于是 inf/inf = NaN）。注意到 softmax 对「给所有输入加上任意常数 c」不变，就能避免这一问题。利用这一性质做数值稳定：通常从 v 的每个元素中减去 v 的最大元素，使新的最大元素为 0。下面你来实现 softmax，并用这个技巧保证数值稳定。

**问题（softmax）：实现 softmax（1 分）**

交付物：写一个对张量施加 softmax 的函数。函数接收两个参数：一个张量和一个维度 i，对输入张量的第 i 维施加 softmax。输出张量与输入张量形状相同，但第 i 维上是归一化的概率分布。用「从第 i 维的所有元素中减去该维最大值」的技巧避免数值稳定性问题。

要测试你的实现，完成 `[adapters.run_softmax]`，确保通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在可以从数学上定义 Attention 操作了：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \tag{11}$$

其中 Q ∈ ℝ^{n × d_k}，K ∈ ℝ^{m × d_k}，V ∈ ℝ^{m × d_v}。Q、K、V 都是该操作的输入——注意它们不是可学习参数。

**掩码**：有时对注意力操作的输出做掩码会很方便。掩码形状应为 M ∈ {True, False}^{n × m}，这个布尔矩阵的第 i 行表示查询 i 应当关注哪些键。按惯例（也稍显令人困惑），位置 (i, j) 处为 True 表示查询 i 确实关注键 j，为 False 表示不关注。换句话说，在值为 True 的 (i, j) 对上「信息可以流动」。例如，考虑 1 × 3 的掩码矩阵 `[[True, True, False]]`，唯一的查询向量只关注前两个键。

从计算上讲，用掩码比在子序列上分别计算注意力高效得多。做法是在 softmax 之前的取值（QKᵀ/√d_k）上，给掩码矩阵中所有为 False 的位置加上 −∞。

**问题（scaled_dot_product_attention）：实现缩放点积注意力（5 分）**

交付物：实现缩放点积注意力函数。实现应能处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他类批次维度（若提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。类批次维度的讨论见 3.2 节。

实现还应支持可选的、用户提供的形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 True 的位置的注意力概率之和应为 1，掩码值为 False 的位置的注意力概率应为 0。

要用我们提供的测试检验实现，需实现测试适配器 `[adapters.run_scaled_dot_product_attention]`。`uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试你的实现，`uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试。

#### 3.4.5 因果多头自注意力

我们实现 A. Vaswani 等人 [8] 3.2.2 节描述的多头自注意力。回顾一下，从数学上多头注意力的施加定义如下：

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) \tag{12}$$

$$\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i) \tag{13}$$

其中 Q_i、K_i、V_i 是 Q、K、V 沿嵌入维度按大小 d_k 或 d_v 切出的第 i 片，i ∈ {1, …, h}。Attention 即 3.4.4 节定义的缩放点积注意力。由此可以构造多头自注意力操作：

$$\mathrm{MultiHeadSelfAttention}(x) = W_O \mathrm{MultiHead}(W_Q x, W_K x, W_V x) \tag{14}$$

这里可学习参数是 W_Q ∈ ℝ^{h d_k × d_model}、W_K ∈ ℝ^{h d_k × d_model}、W_V ∈ ℝ^{h d_v × d_model}、W_O ∈ ℝ^{d_model × h d_v}。由于多头注意力操作中会把 Q、K、V 切片，可以认为 W_Q、W_K、W_V 沿输出维度按头切分。做完之后，键、值、查询的投影应当总共只用三次矩阵乘法来实现。⁵

**因果掩码**

你的实现应阻止模型关注序列中未来的 token。换言之，给定 token 序列 t_1, …, t_n，对前缀 t_1, …, t_i（i < n）计算下一词预测时，模型不应能访问（关注）位置 t_{i+1}, …, t_n 的 token 表示，因为推理生成文本时它拿不到这些 token（而这些未来 token 会泄露真实下一词的身份，使语言建模预训练目标变得没有意义）。对输入 token 序列 t_1, …, t_n，一种朴素做法是跑 n 次多头自注意力（对应序列的 n 个不同前缀）。我们改用因果注意力掩码，允许 token i 关注序列中所有 j ≤ i 的位置。可以用 `torch.triu` 或广播的索引比较来构造掩码，并利用 3.4.4 节的缩放点积注意力实现已支持注意力掩码这一点。

**应用 RoPE**

RoPE 应施加于查询向量和键向量，不施加于值向量。此外，头维度应当按批次维度处理，因为多头注意力中每个头独立计算注意力。这意味着对每个头，查询向量和键向量都应施加完全相同的 RoPE 旋转。

**问题（multihead_self_attention）：实现因果多头自注意力（5 分）**

⁵ 进阶目标：尝试把键、查询、值的投影合并成单个权重矩阵，这样只需一次矩阵乘法。

交付物：把因果多头自注意力实现为 `torch.nn.Module`。实现至少应接受以下参数：

- `d_model: int`：Transformer 块输入的维度。
- `num_heads: int`：多头自注意力使用的头数。

沿用 A. Vaswani 等人 [8]，令 d_k = d_v = d_model / h。要用我们提供的测试检验实现，实现测试适配器 `[adapters.run_multihead_self_attention]`，然后运行 `uv run pytest -k test_multihead_self_attention`。

### 3.5 完整的 Transformer LM

先来组装 Transformer 块（可以回看图 2）。一个 Transformer 块包含两个「子层」，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。每个子层中，我们先做 RMSNorm，再做主要运算（MHA/FF），最后加上残差连接。

具体来说，Transformer 块的前半部分（第一个「子层」）应当实现下面这组更新，由输入 x 产生输出 y：

$$y = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)) \tag{15}$$

**问题（transformer_block）：实现 Transformer 块（3 分）**

按 3.4 节描述、图 2 所示实现 pre-norm Transformer 块。你的 Transformer 块至少应接受以下参数：

- `d_model: int`：Transformer 块输入的维度。
- `num_heads: int`：多头自注意力使用的头数。
- `d_ff: int`：逐位置前馈内层的维度。

要测试你的实现，实现适配器 `[adapters.run_transformer_block]`，然后运行 `uv run pytest -k test_transformer_block`。

交付物：能通过所给测试的 Transformer 块代码。

现在按图 1 的高层示意图把这些块组装起来。按照 3.1 节对嵌入的描述，把它送入 `num_layers` 个 Transformer 块，再送入最后的层归一化和 LM head，得到词表上的未归一化分布（即 logits）。

**问题（transformer_lm）：实现 Transformer LM（3 分）**

该把一切拼起来了！按 3.1 节描述、图 1 所示实现 Transformer 语言模型。你的实现至少应接受前面提到的 Transformer 块的全部构造参数，以及以下额外参数：

- `vocab_size: int`：词表大小，用于确定 token 嵌入矩阵的维度。
- `context_length: int`：最大上下文长度，用于确定 RoPE sin/cos 缓冲区的维度。
- `num_layers: int`：Transformer 块的层数。

要用我们提供的测试检验实现，先实现测试适配器 `[adapters.run_transformer_lm]`，然后运行 `uv run pytest -k test_transformer_lm`。

交付物：能通过上述测试的 Transformer LM 模块。

**资源核算**

理解 Transformer 各部分如何消耗算力和内存很有用。下面走一遍基本的「FLOPs 核算」步骤。Transformer 中绝大多数 FLOPs 来自矩阵乘法，所以核心方法很简单：

1. 写出 Transformer 一次前向传播中的所有矩阵乘法。
2. 把每个矩阵乘法换算成所需 FLOPs。

第二步中，下面这条规则很有用：

> **规则**：给定 A ∈ ℝ^{m × n} 和 B ∈ ℝ^{n × p}，矩阵乘积 AB 需要 2mnp 个 FLOPs。

理由：(AB)[i, j] = A[i, :] · B[:, j]，这个点积需要 n 次加法和 n 次乘法（2n 个 FLOPs）。矩阵乘积 AB 有 m × p 个元素，所以总 FLOPs 为 (2n)(mp) = 2mnp。

做下一题之前，可以先过一遍你 Transformer 块和 Transformer LM 的每个组件，列出所有矩阵乘法及其对应的 FLOPs 开销。

**问题（transformer_accounting）：Transformer LM 资源核算（5 分）**

(a) 考虑用本次作业架构搭建的 GPT-2 XL 规模模型，配置如下：

```
vocab_size:     50,257
context_length: 1,024
num_layers:     48
d_model:        1,600
num_heads:      25
d_ff:           4,288（最接近 (8/3) × 1,600 的 64 的倍数）
```

假设用这个配置构造模型。模型有多少可训练参数？假设每个参数用单精度浮点数表示，仅加载这个模型需要多少内存？

交付物：一到两句话回答。

(b) 指出 GPT-2 XL 形状模型完成一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列有 `context_length` 个 token。

交付物：矩阵乘法列表（附说明），以及所需 FLOPs 总数。

(c) 基于上面的分析，模型哪些部分消耗的 FLOPs 最多？

交付物：一到两句话回答。

(d) 对 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复上述分析。随着模型规模增大，Transformer LM 的哪些部分在总 FLOPs 中的占比变大或变小？

交付物：对每个模型给出各组件及其 FLOPs 占比（占一次前向传播总 FLOPs 的比例）。另外用一到两句话描述模型规模变化如何改变各组件的 FLOPs 占比。

(e) 取 GPT-2 XL，把上下文长度增大到 16,384。一次前向传播的总 FLOPs 如何变化？各模型组件的 FLOPs 相对贡献如何变化？

交付物：一到两句话回答。

---

## 4 训练 Transformer LM

现在我们有了数据预处理（分词器）和模型（Transformer）的步骤。剩下的就是搭建支持训练的全部代码，包括：

- **损失**：定义损失函数（交叉熵）。
- **优化器**：定义最小化该损失的优化器（AdamW）。
- **训练循环**：搭建加载数据、保存检查点、管理训练所需的全部基础设设。

### 4.1 交叉熵损失

回顾一下，Transformer 语言模型对每个长度为 m + 1 的序列 x，以及 i = 1, …, m，定义分布 p_θ(x_{i+1} | x_{1:i})。给定由长度 m + 1 的序列组成的训练集 D，我们定义标准的交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D|m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i}) \tag{16}$$

（注意 Transformer 一次前向传播就会给出所有 i = 1, …, m 的 p_θ(x_{i+1} | x_{1:i})。）

具体来说，Transformer 对每个位置 i 计算 logits o_i ∈ ℝ^{vocab_size}，从而得到：⁶

$$p(x_{i+1} \mid x_{1:i}) = \mathrm{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\mathrm{vocab\_size}} \exp(o_i[a])} \tag{17}$$

交叉熵损失通常是针对 logits 向量 o_i ∈ ℝ^{vocab_size} 和目标 x_{i+1} 定义的。⁷

⁶ 注意 o_i[k] 指向量 o_i 中索引 k 处的值。
⁷ 这对应于 x_{i+1} 上的 Dirac delta 分布与预测分布 softmax(o_i) 之间的交叉熵。

实现交叉熵损失和 softmax 一样，需要在数值问题上多加小心。

**问题（cross_entropy）：实现交叉熵（1 分）**

交付物：写一个计算交叉熵损失的函数，接收预测 logits（o_i）和目标（x_{i+1}），计算交叉熵 ℓ_i = −log softmax(o_i)[x_{i+1}]。函数应处理以下事项：

- 为了数值稳定性减去最大元素。
- 尽可能把 log 与 exp 相消。
- 处理额外的批次维度，并返回批次上的平均值。与 3.2 节一致，我们假定类批次维度总在最前面，位于词表大小维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 测试。

**困惑度**

交叉熵足以支撑训练，但评估模型时，我们还希望报告困惑度。对长度为 m 的序列，若交叉熵损失为 ℓ_1, …, ℓ_m：

$$\mathrm{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right) \tag{18}$$

### 4.2 SGD 优化器

有了损失函数，开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。从随机初始化的参数 θ_0 出发，对每一步 t = 0, …, T−1 执行如下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t) \tag{19}$$

其中 B_t 是从数据集 D 中随机采样的一个数据批次，学习率 α_t 和批大小 |B_t| 都是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

实现优化器时，我们继承 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法：

`def __init__(self, params, ...)` 负责初始化优化器。这里 `params` 是要优化的参数集合（也可能是参数组，以便用户对模型不同部分使用不同超参数，例如不同的学习率）。务必把 `params` 传给基类的 `__init__` 方法，基类会保存这些参数供 `step` 使用。可以按优化器的需要接收额外参数（例如学习率很常见），并作为字典传给基类构造函数，字典的键由你为这些参数取名。

`def step(self)` 执行一次参数更新。训练循环中，它会在反向传播之后被调用，因此你能拿到最后一个批次的梯度。这个方法应遍历每个参数张量 p 并原地修改它们，即设置 `p.data`（保存该参数对应张量），修改依据是梯度 `p.grad`（如果存在），后者表示损失对该参数的梯度。

PyTorch 优化器 API 有些微妙之处，用例子解释更容易。为了让例子更丰富，我们实现 SGD 的一个小变体：学习率随训练衰减，从初始学习率 α 开始，随时间步逐步变小：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t) \tag{20}$$

看看这个版本的 SGD 如何实现为一个 PyTorch Optimizer：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
```

`__init__` 中，我们把参数和默认超参数传给基类构造函数（参数可能按组给出，每组可以有不同超参数）。如果参数只是单个 `torch.nn.Parameter` 集合，基类构造函数会创建单个组并赋予默认超参数。然后在 `step` 中，我们遍历每个参数组，再遍历组内每个参数，应用式 20。这里把迭代次数保存为与每个参数关联的状态：先读取，用于梯度更新，再更新它。该 API 规定用户可以传入一个可调用 closure，在优化器步进前重新计算损失。我们要用的优化器不需要它，但为了符合 API 还是加上了。

为验证效果，可以用下面这个最小的训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
```

这是训练循环的典型结构：每次迭代计算损失，运行一步优化器。训练语言模型时，可学习参数来自模型（在 PyTorch 中，`m.parameters()` 给出这个集合），损失在一个采样批次上计算，而训练循环的基本结构保持不变。

**问题（learning_rate_tuning）：调学习率（1 分）**

我们将会看到，影响训练最大的超参数之一是学习率。在玩具例子中实际验证一下。用另外三个学习率值 `1e1`、`1e2`、`1e3` 运行上面的 SGD 示例，只跑 10 次训练迭代。各个学习率下损失如何变化？衰减得更快、更慢，还是发散（即训练过程中不断增大）？

交付物：用一两句话描述观察到的行为。

### 4.3 AdamW

现代语言模型通常不用 SGD，而用更复杂的优化器。近期使用的优化器大多是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生版本。我们使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛采用。AdamW 对 Adam 的修改在于：以与梯度更新解耦的方式加入权重衰减（每次迭代把参数往 0 拉），从而改善正则化。我们按 I. Loshchilov 等人 [23] 的 Algorithm 2 实现 AdamW。

AdamW 是有状态的：对每个参数，它维护一阶矩和二阶矩的滑动估计。因此 AdamW 用额外的内存换取更好的稳定性和收敛性。除学习率 α 外，AdamW 还有一对控制矩估计更新的超参数 (β_1, β_2)，以及权重衰减率 λ。典型设置取 (β_1, β_2) = (0.9, 0.999)，而 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 等大语言模型常用 (0.9, 0.95)。算法可写为如下形式，其中 ε 是一个很小的值（例如 10⁻⁸），用于在 v 出现极小值时改善数值稳定性：

```
Algorithm 1: AdamW Optimizer
1  init(θ)                              ▷ Initialize learnable parameters
2  m ← 0                                ▷ Initial value of the first moment vector; same shape as θ
3  v ← 0                                ▷ Initial value of the second moment vector; same shape as θ
4  for t = 1, …, T do
5      Sample batch of data B_t
6      g ← ∇_θ ℓ(θ; B_t)                ▷ Compute the gradient of the loss
7      α_t ← α √(1 − β_2^t) / (1 − β_1^t)  ▷ Compute adjusted α for iteration t
8      θ ← θ − α λ θ                    ▷ Apply weight decay
9      m ← β_1 m + (1 − β_1) g          ▷ Update the first moment estimate
10     v ← β_2 v + (1 − β_2) g²         ▷ Update the second moment estimate
11     θ ← θ − α_t m / (√v + ε)         ▷ Apply moment-adjusted weight updates
12 end for
```

Algorithm 1：AdamW 优化器伪代码。

注意 t 从 1 开始。下面由你实现这个优化器。

**问题（adamw）：实现 AdamW（2 分）**

交付物：把 AdamW 实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接收学习率 α，以及 β、ε、λ 这些超参数。为帮助你保存状态，基类 `Optimizer` 提供了字典 `self.state`，把 `nn.Parameter` 对象映射到保存该参数所需任意信息的字典（对 AdamW 来说就是矩估计）。实现 `[adapters.get_adamw_cls]`，确保通过 `uv run pytest -k test_adamw`。

**问题（adamw_accounting）：AdamW 训练的资源核算（2 分）**

计算运行 AdamW 需要多少内存和算力。假设每个张量都用 float32。

(a) 运行 AdamW 的峰值内存是多少？按参数、激活值、梯度、优化器状态的内存占用分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表达。假设 d_ff = (8/3) × d_model。

为简化，计算激活值内存时只考虑以下组件：

- Transformer 块
  - RMSNorm
  - 多头自注意力子层：QKV 投影、QKᵀ 矩阵乘法、softmax、值的加权求和、输出投影
  - 逐位置前馈（SwiGLU）：W_1、W_2、门控分支上的 SiLU、逐元素乘积、W_3
- 最后的 RMSNorm
- 输出嵌入
- logits 上的交叉熵

交付物：参数、激活值、梯度、优化器状态各自的代数表达式，以及总和。

(b) 对 GPT-2 XL 形状的模型代入你的答案，得到只依赖 `batch_size` 的表达式。在 80GB 内存内能用的最大批大小是多少？

交付物：一个形如 a·batch_size + b 的表达式（a、b 为具体数值），以及最大批大小。

(c) 运行一步 AdamW 需要多少 FLOPs？

交付物：一个代数表达式，附简要理由。

(d) 模型 FLOPs 利用率（MFU）定义为观测吞吐量（每秒 token 数）与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 在「float32」（实际上是 TensorFloat-32，真实精度为「bfloat19」）运算下的理论峰值为 495 teraFLOP/s。假设能达到 50% MFU，在单张 H100 上训练 GPT-2 XL 400K 步、批大小 1024 需要多久？沿用 J. Kaplan 等人 [25] 和 J. Hoffmann 等人 [26]，假设反向传播的 FLOPs 是前向传播的两倍。

交付物：训练所需小时数，附简要理由。

### 4.4 学习率调度

使损失下降最快的学习率在训练过程中往往会变化。训练 Transformer 时，通常会使用学习率调度：开始用较大的学习率、在初期做更快的更新，随着训练推进慢慢衰减到较小值。⁸ 本次作业实现 LLaMA 训练所用的余弦退火调度 [H. Touvron et al., 2023]。

调度器就是一个函数：接收当前步 t 及相关参数（例如初始学习率和最终学习率），返回第 t 步梯度更新所用的学习率。最简单的调度是常数函数，对任意 t 都返回同一学习率。

余弦退火学习率调度接收：(i) 当前迭代 t；(ii) 最大学习率 α_max；(iii) 最小（最终）学习率 α_min；(iv) 预热迭代数 T_w；(v) 余弦退火的最终迭代 T_c。第 t 次迭代的学习率定义为：

- **预热（Warm-up）**：若 t < T_w，则 α_t = (t / T_w) α_max。
- **余弦退火（Cosine annealing）**：若 T_w ≤ t ≤ T_c，则 α_t = α_min + (1/2)(1 + cos((t − T_w)/(T_c − T_w) · π))(α_max − α_min)。
- **退火后（Post-annealing）**：若 t > T_c，则 α_t = α_min。

⁸ 有时也会使用学习率回升（restart）的调度，帮助跳出局部极小值。

**问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1 分）**

写一个函数，接收 t、α_max、α_min、T_w、T_c，按上面的调度返回学习率 α_t。然后实现 `[adapters.get_lr_cosine_schedule]`，确保通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

训练时有时会遇到产生很大梯度的样本，使训练变得不稳定。实践中常用的一种缓解手段是梯度裁剪：在每次反向传播之后、优化器步进之前，对梯度的范数施加限制。

给定（所有参数的）梯度 g，计算其 ℓ2 范数 ‖g‖₂。若该范数小于最大值 M，则保持 g 不变；否则把 g 按 M/(‖g‖₂ + ε) 的比例缩小（ε 是一个小量，如 10⁻⁶，用于数值稳定）。注意缩放后的范数会略小于 M。

**问题（gradient_clipping）：实现梯度裁剪（1 分）**

写一个实现梯度裁剪的函数。函数接收参数列表和最大 ℓ2 范数，原地修改每个参数的梯度。使用 ε = 10⁻⁶（PyTorch 默认值）。然后实现适配器 `[adapters.run_gradient_clipping]`，确保通过 `uv run pytest -k test_gradient_clipping`。

---

## 5 训练循环

现在终于可以把前面搭好的主要组件组合起来：分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据（例如你在 `tokenizer_experiments` 中准备好的数据）是一个 token 序列 x = (x_1, …, x_n)。即使源数据由多个独立文档组成（例如不同的网页或源码文件），常见做法也是把它们全部拼接成一个 token 序列，并在之间加上分隔符（例如 `<|endoftext|>` token）。

数据加载器把它变成一批批的流，每批包含 B 个长度为 m 的序列，以及对应的、长度同样为 m 的下一 token 序列。例如 B = 1、m = 3 时，`([x_2, x_3, x_4], [x_3, x_4, x_5])` 可以是一个批次。

这样加载数据能从多方面简化训练。首先，任意 1 ≤ i ≤ n − m 都给出合法的训练序列，采样非常方便。其次，所有训练序列长度相同，不需要 padding，硬件利用率更高（增大批大小 B 也能提高利用率）。最后，采样训练数据时不必把整个数据集载入内存，因此可以轻松处理内存装不下的大数据集。

**问题（data_loading）：实现数据加载（2 分）**

交付物：写一个函数，接收 NumPy 数组 x（存 token ID 的整数数组）、`batch_size`、`context_length` 以及 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），返回一对张量：采样得到的输入序列，以及对应的下一 token 目标。两个张量的形状都应为 `(batch_size, context_length)`，包含 token ID，并放在指定设备上。要用我们提供的测试检验实现，先实现测试适配器 `[adapters.run_get_batch]`，然后运行 `uv run pytest -k test_get_batch`。

> **低资源提示：在 CPU 或 Apple Silicon 上加载数据**
>
> 如果打算在 CPU 或 Apple Silicon 上训练 LM，需要把数据放到正确的设备上（后面模型也要用同一设备）。
>
> 在 CPU 上用 `'cpu'` 设备字符串，在 Apple Silicon（M* 芯片）上用 `'mps'`。
>
> 关于 MPS 的更多资料：
>
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

如果数据集大到内存装不下怎么办？可以用名为 `mmap` 的 Unix 系统调用，把磁盘上的文件映射到虚拟内存，访问该内存位置时才惰性加载文件内容。这样就能「假装」整个数据集都在内存里。NumPy 通过 `np.memmap`（或给 `np.load` 加 `mmap_mode='r'` 标志，前提是当初用 `np.save` 保存了数组）实现这一点，返回一个类数组对象，访问时才按需加载项。训练中从数据集（NumPy 数组）采样时，务必以内存映射模式加载数据集（`np.memmap` 或 `np.load` 的 `mmap_mode='r'`，取决于数组的保存方式）。还要确保指定的 dtype 与所加载数组匹配。显式验证内存映射数据看起来正确（例如没有超出预期词表大小的值）会有帮助。

### 5.2 检查点

除了加载数据，训练过程中还需要保存模型。跑任务时，我们常希望能从中途停止的训练运行中恢复（例如任务超时、机器故障等）。即便一切顺利，之后也可能需要中间模型（例如事后研究训练动态、从不同训练阶段的模型采样等）。

检查点应包含恢复训练所需的全部状态。至少需要能恢复模型权重。如果使用有状态优化器（如 AdamW），还需要保存优化器状态（例如 AdamW 的矩估计）。最后，要恢复学习率调度，还需要知道停止时的迭代次数。PyTorch 让保存这些都很容易：每个 `nn.Module` 都有 `state_dict()` 方法，返回包含全部可学习权重的字典，之后可以用配套的 `load_state_dict()` 恢复；`torch.optim.Optimizer` 同理。此外，`torch.save(obj, dest)` 可以把对象（例如一个字典，值里既有张量也有整数这类普通 Python 对象）转储到文件（路径）或类文件对象，之后用 `torch.load(src)` 载回内存。

**问题（checkpointing）：实现模型检查点（1 分）**

实现以下两个函数来加载和保存检查点：

```python
def save_checkpoint(model, optimizer, iteration, out)
```

把模型、优化器和迭代次数中的全部状态转储到类文件对象 `out`。可以用模型和优化器的 `state_dict` 方法取得相关状态，用 `torch.save(obj, out)` 把 obj 写入 out（PyTorch 这里支持路径或类文件对象）。常见做法是让 obj 为字典，但只要之后能加载，格式随意。

该函数期望以下参数：

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model, optimizer)
```

从 `src`（路径或类文件对象）加载检查点，并从中恢复模型和优化器状态。函数应返回检查点中保存的迭代次数。可以用 `torch.load(src)` 取回 `save_checkpoint` 中保存的内容，用模型和优化器的 `load_state_dict` 方法把它们恢复到之前的状态。

该函数期望以下参数：

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，确保通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在可以把你实现的全部组件整合进主训练脚本了。让不同超参数下的训练运行易于启动（例如通过命令行参数传入）会很值得，后面你会反复用它们研究不同选择对训练的影响。

**问题（training_together）：整合起来（4 分）**

交付物：写一个脚本，运行训练循环，在用户提供的输入上训练模型。具体来说，建议训练脚本至少支持：

- 能配置和控制各种模型与优化器超参数。
- 用 `np.memmap` 内存高效地加载大型训练集和验证集。
- 把检查点序列化到用户指定的路径。
- 定期记录训练和验证性能（例如输出到控制台和/或 Weights and Biases 之类的外部服务）。⁹

⁹ wandb.ai

---

## 6 生成文本

现在能训练模型了，最后一块是用模型生成文本。回顾一下，语言模型接收长度为 `sequence_length` 的（可能带批次的）整数序列，产生大小为 `(sequence_length, vocab_size)` 的矩阵，序列中每个元素是预测该位置之后下一个 token 的概率分布。下面写几个函数，把它变成新序列的采样方案。

**Softmax**

按标准约定，语言模型的输出是最后一个线性层的输出（「logits」），因此要通过 softmax 把它变成归一化概率，即前面式 10 所示的操作。

**解码**

要从模型生成文本（解码），我们把一个前缀 token 序列（「prompt」）交给模型，让它给出词表上的概率分布，用来预测序列的下一个 token。然后从这个词表项分布中采样，确定下一个输出 token。

具体来说，解码过程的一步应接收序列 x_{1…t}，通过下面的式子返回 token x_{t+1}：

$$P(x_{t+1} = i \mid x_{1\dots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)} \tag{21}$$

$$v = \mathrm{TransformerLM}(x_{1\dots t})_t \in \mathbb{R}^{\mathrm{vocab\_size}} \tag{22}$$

其中 `TransformerLM` 是模型，接收长度为 `sequence_length` 的序列，产生大小为 `(sequence_length, vocab_size)` 的矩阵；我们取该矩阵的最后一个元素，因为要找的是第 t 个位置上的下一 token 预测。

反复从这些单步条件分布中采样（把上一步生成的输出 token 追加到下一步解码的输入中），直到生成序列结束 token `<|endoftext|>`（或达到用户指定的最大生成 token 数），就得到了基本的解码器。

**解码技巧**

我们要实验的是小模型，小模型有时会生成质量很低的文本。两个简单的解码技巧可以缓解这些问题。第一，温度缩放：用温度参数 τ 修改 softmax，新的 softmax 为

$$\mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{\mathrm{vocab\_size}} \exp(v_j / \tau)} \tag{23}$$

注意 τ → 0 时，v 中最大的元素占据主导，softmax 的输出退化为集中在最大元素上的 one-hot 向量。

第二，另一个技巧是核采样（nucleus sampling）或 top-p 采样：通过截断低概率 token 来修改采样分布。设 q 是（温度缩放后的）softmax 得到的、大小为 `vocab_size` 的概率分布。超参数为 p 的核采样按下面的式子产生下一个 token：

$$P(x_{t+1} = i \mid q) = \begin{cases} \dfrac{q_i}{\sum_{j \in V(p)} q_j} & \text{if } i \in V(p) \\ 0 & \text{otherwise} \end{cases} \tag{24}$$

其中 V(p) 是满足 Σ_{j ∈ V(p)} q_j ≥ p 的最小索引集合。可以先把概率分布 q 按大小排序，从词表中最大的元素开始选取，直到达到目标水平 p，从而方便地算出这个集合。

**问题（decoding）：解码（3 分）**

交付物：实现一个从语言模型解码的函数。建议支持以下特性：

- 为用户提供的 prompt 生成补全（即接收 x_{1…t}，采样补全直到遇到 `<|endoftext|>`）。
- 允许用户控制最大生成 token 数。
- 给定期望的温度值，在采样前对预测的下一 token 分布施加 softmax 温度缩放。
- 给定用户指定的阈值，做 top-p 采样（[A. Holtzman et al., 2020]，又称核采样）。

---

## 7 实验

现在把一切拼起来，在预训练数据集上训练（小）语言模型。

### 7.1 如何做实验与交付物

理解 Transformer 各架构组件背后道理的最好方式，就是亲自修改它并运行。动手经验无法替代。

为此，能够快速、一致地做实验并记录所做之事很重要。为了快速实验，我们会在小规模模型（约 1700 万总参数）和简单数据集（TinyStories）上跑许多实验；为了一致，你会系统地消融组件、变化超参数；为了记录，我们会要求你提交实验日志以及每个实验对应的学习曲线。

为了能提交损失曲线，务必定期评估验证损失，并同时记录步数和实际耗时（wall-clock time）。你可能会发现 Weights and Biases 之类的日志基础设施很有帮助。

**问题（experiment_log）：实验记录（3 分）**

为你的训练和评估代码搭建实验跟踪基础设施，以便相对于梯度步数和实际耗时跟踪实验与损失曲线。

交付物：实验的日志基础设施代码，以及本节下面各作业问题的实验日志（记录你试过的所有内容）。

### 7.2 TinyStories

从很简单的数据集开始（TinyStories；R. Eldan et al. [1]），模型训练得快，也能观察到一些有意思的行为。获取该数据集的说明见第 1 节。数据样例如下。

**示例（tinystories_example）：TinyStories 中的一个例子**

（以下为原文，保留不译）

> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

#### 7.2.1 超参数调优

我们会给出一些非常基础的起始超参数，要求你为其他超参数找到合适的设置。

**词表大小** 10000。典型词表规模在几万到几十万之间。你应当变化这个值，观察词表和模型行为如何变化。

**上下文长度** 256。TinyStories 这类简单数据集可能不需要很长的序列，但后面处理 OpenWebText 数据时，你可能想变化这个值。试着改变它，观察每次迭代运行时间和最终困惑度受到的影响。

**d_model** 512。这比许多小型 Transformer 论文使用的 768 维略小，但会让运行更快。

**d_ff** 1344。约为 (8/3) d_model，同时是 64 的倍数，对 GPU 性能友好。

**RoPE theta 参数 Θ** 10000。

**层数和头数** 4 层，16 头。合起来约有 1700 万非嵌入参数，是相当小的 Transformer。

**处理的总 token 数** 327,680,000（你的批大小 × 总步数 × 上下文长度应大致等于这个值）。

以下其他超参数需要通过试错找到合适的默认值：学习率、学习率预热、AdamW 的其他超参数（β_1、β_2、ε）以及权重衰减。这些超参数的典型取值可在 D. P. Kingma 等人 [22] 中找到。

#### 7.2.2 整合起来

现在可以把一切拼起来了：拿到训练好的 BPE 分词器，对训练集做分词，放进你写的训练循环里跑。重要提示：如果你的实现正确且高效，上述超参数在 1 张 B200 GPU 上应带来约 20–30 分钟的运行时间。如果运行时间远超这个值，请检查数据加载、检查点或验证损失代码是否成为瓶颈，以及实现是否做了恰当的批处理。

#### 7.2.3 调试模型架构的技巧

强烈建议你熟悉 IDE 内置的调试器（例如 VSCode/Zed），它比用 print 调试更省时间。如果使用文本编辑器，可以用 ipdb 之类工具。调试模型架构时还有几点好做法：

- 开发任何神经网络架构时，常见的入门步骤是先过拟合单个小批次。如果实现正确，应当能很快把训练损失压到接近 0。
- 在模型各组件中设置调试点，检查中间张量的形状是否符合预期。
- 监控激活值、模型权重和梯度的范数，确保它们没有爆炸或消失。

**问题（learning_rate）：调学习率（2 B200 小时）（3 分）**

学习率是最值得调的超参数之一。基于你训练好的基础模型，回答以下问题：

(a) 对学习率做超参数扫描，报告最终损失（若优化器发散，也要注明）。

交付物：多个学习率对应的学习曲线，并解释你的超参数搜索策略。

交付物：一个在 TinyStories 上验证损失（每 token）不超过 1.45 的模型。

> **低资源提示：在 CPU 或 Apple Silicon 上只训练几步**
>
> 如果在 cpu 或 mps 上运行，应把处理的总 token 数降到 40,000,000，这足以产生相当流畅的文本。也可以把目标验证损失从 1.45 放宽到 2.00。
>
> 用调好学习率的官方参考代码在 M4 Max 芯片、36 GB 内存上运行时，我们用批大小 × 总步数 × 上下文长度 = 32 × 5000 × 256 = 40,960,000 个 token，cpu 上耗时 1 小时 22 分钟，mps 上 36 分钟。在第 5000 步时验证损失为 1.80。
>
> 一些额外提示：
>
> - 使用 N 个训练步时，建议调整余弦学习率衰减调度，让衰减恰好在第 N 步结束（即到达最小学习率）。
> - 使用 mps 时不要用 TF32 kernel，即不要设置 `torch.set_float32_matmul_precision('high')`（在 cuda 设备上你可能会这么设）。我们在 mps 上尝试启用 TF32 kernel（torch 2.9.0），发现后端有时会静默使用有问题的 kernel，导致训练不稳定。
> - 可以用 `torch.compile` JIT 编译模型来加速训练：
>   - 在 cpu 上：`model = torch.compile(model)`
>   - 在 mps 上，可用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。截至 torch 2.9.0，mps 不支持 Inductor 编译。

(b) 有种民间说法认为最佳学习率位于「稳定边缘」。研究学习率开始发散的点与你的最佳学习率之间的关系。

交付物：学习率逐渐增大（至少包含一次发散运行）的学习曲线，以及对其与收敛速度关系的分析。

接下来变化批大小，看看训练会如何。批大小很重要——更大的批让我们能做更大的矩阵乘法，从而提高 GPU 效率，但批大小总是越大越好吗？跑几个实验看看。

**问题（batch_size_experiment）：批大小变化（1 B200 小时）（1 分）**

把批大小从 1 变化到 GPU 内存上限。中间至少尝试几个不同批大小，包括 64、128 这类典型值。

交付物：不同批大小运行的学习曲线。必要时应重新调优学习率。

交付物：几句话讨论你对批大小及其对训练影响的发现。

有了解码器，现在可以生成文本了！我们会从模型生成文本，看看效果如何。作为参照，你的输出应当至少和下面的例子一样好。

**示例（ts_generate_example）：TinyStories 语言模型的样本输出**

（以下为原文，保留不译）

> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 如果使用处理 4000 万 token 的低资源配置，生成的文本应当仍像英语，但流畅度不如上面。例如，我们在 4000 万 token 上训练的 TinyStories 语言模型的样本输出如下：
>
> > Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> >
> > "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"
> >
> > As Sue watched the sky moved, what it was. She

以下是精确的问题陈述和我们的要求：

**问题（generate）：生成文本（1 分）**

用你的解码器和训练好的检查点，报告模型生成的文本。你可能需要调整解码器参数（温度、top-p 等）来获得流畅的输出。

交付物：至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），并简要评论这段输出的流畅度，以及至少两个影响输出好坏的因素。

### 7.3 消融与架构修改

理解 Transformer 的最好方式就是亲手修改并观察其行为。下面做几个简单的消融和修改。

**消融 1：层归一化**

人们常说层归一化对 Transformer 训练的稳定性很重要。但我们也许想冒点险：把每个 Transformer 块中的 RMSNorm 去掉，看看会发生什么。

**问题（layer_norm_ablation）：去掉 RMSNorm 训练（0.5 B200 小时）（1 分）**

去掉 Transformer 中所有 RMSNorm 并训练。在之前的最优学习率下会发生什么？用更低的学习率能否获得稳定性？

交付物：去掉 RMSNorm 训练时的学习曲线，以及最佳学习率下的学习曲线。

交付物：几句话评论 RMSNorm 的影响。

接下来研究另一个乍看随意的层归一化选择。pre-norm Transformer 块定义为

$$z = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)) \tag{25}$$

$$y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)) \tag{26}$$

这是对原始 Transformer 架构少数几处「共识性」修改之一，原始实现采用 post-norm 做法：

$$z = \mathrm{RMSNorm}(x + \mathrm{MultiHeadSelfAttention}(x)) \tag{27}$$

$$y = \mathrm{RMSNorm}(z + \mathrm{FFN}(z)) \tag{28}$$

把实现改回 post-norm，看看会发生什么。

**问题（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时）（1 分）**

把 pre-norm Transformer 实现改成 post-norm，训练并观察结果。

交付物：post-norm Transformer 的学习曲线，与 pre-norm 对比。

可以看到，层归一化对 Transformer 的行为影响很大，连层归一化的位置都很重要。

**消融 2：位置嵌入**

接着研究位置嵌入对模型性能的影响。具体来说，把我们的基础模型（带 RoPE）与完全不含位置嵌入的版本（NoPE）做对比。研究表明，decoder-only transformer（即我们实现的带因果掩码的模型）理论上可以在没有被显式提供位置嵌入的情况下推断相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。下面用实验检验 NoPE 与 RoPE 的表现。

**问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）**

把带 RoPE 的 Transformer 实现改成完全去掉位置嵌入信息，观察结果。

交付物：对比 RoPE 与 NoPE 性能的学习曲线。

**消融 3：SwiGLU 与 SiLU**

接下来沿用 N. Shazeer [20]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络，检验门控在前馈网络中的重要性：

$$\mathrm{FFN}_{\mathrm{SiLU}}(x) = W_2 \mathrm{SiLU}(W_1 x) \tag{29}$$

回顾一下，我们的 SwiGLU 实现把内层前馈层维度设为约 d_ff = (8/3) d_model（同时保证 d_ff mod 64 = 0，以利用 GPU 张量核心）。在这个消融基线中，你的 FFN_SiLU 实现应当改为 d_ff = 4 × d_model，以大致匹配默认 SwiGLU 前馈网络的参数量（后者有三个、而前者只有两个权重矩阵）。

**问题（swiglu_ablation）：SwiGLU 与 SiLU（0.5 B200 小时）（1 分）**

交付物：参数量大致匹配的 SwiGLU 与 SiLU 前馈网络性能对比学习曲线。

交付物：几句话讨论你的发现。

> **低资源提示：GPU 资源有限的线上学生应在 TinyStories 上测试修改**
>
> 作业余下部分将转向更大规模、噪声更多的网页数据集（OpenWebText），实验架构修改，并（可选地）向课程排行榜提交结果。
>
> 在 OpenWebText 上把 LM 训到流畅需要很长时间，因此我们建议 GPU 资源有限的线上学生继续在 TinyStories 上测试修改（用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

现在转向一个更标准的、由网页抓取构建的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一个小样本也以单个文本文件提供：如何获取见第 1 节。

下面是 OpenWebText 中的一个例子。注意文本更真实、更复杂、更多样。你也许想浏览训练数据集，感受网络抓取语料的训练数据长什么样。

**示例（owt_example）：OWT 中的一个例子**

（以下为原文，保留不译）

> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. […]

注意：这个实验可能需要重新调优学习率、批大小等超参数。

**问题（main_experiment）：OWT 实验（2 B200 小时）（2 分）**

用与 TinyStories 相同的模型架构和总训练迭代次数在 OpenWebText 上训练你的语言模型。效果如何？

交付物：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 损失的差异——这些损失应如何解读？

交付物：OpenWebText LM 生成的文本，格式与 TinyStories 输出相同。文本流畅度如何？模型和算力预算都相同，为什么输出质量更差？

### 7.5 你自己的修改 + 排行榜

恭喜你走到这里，快完成了！接下来尝试改进 Transformer 架构，看看你的超参数和架构在全班同学中处于什么水平。

**排行榜规则**

除以下几条外没有其他限制：

**运行时间**：提交最多可在 B200 上运行 45 分钟。如果使用 SLURM 或 Modal，可以在提交脚本里强制这一限制。

**数据**：只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你想怎么做都可以。

如果想找些实现灵感，可以看看这些资源：

- 最先进的开源 LLM 系列，例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库（github.com/KellerJordan/modded-nanogpt），社区成员在上面发布了许多用于「速通」小规模语言模型预训练的修改。例如，一个可追溯到原始 Transformer 论文的常见修改是把输入嵌入和输出嵌入的权重绑定在一起（见 A. Vaswani 等人 [8] 第 3.4 节，以及 A. Chowdhery 等人 [16] 第 2 节）。如果尝试权重绑定，可能需要降低 embedding/LM head 初始化的标准差。

在跑完整的 45 分钟之前，先在 OpenWebText 的一个小小子集或 TinyStories 上测试这些修改。

需要说明的是，你在这个排行榜上发现有效的某些修改，未必能推广到更大规模的预训练。课程的 scaling law 单元会进一步探讨这个想法。

**问题（leaderboard）：排行榜（10 B200 小时）（6 分）**

按上述排行榜规则训练一个模型，目标是在 0.75 B200 小时内最小化语言模型的验证损失。

交付物：最终记录的验证损失；对应的学习曲线，横轴为实际耗时（wall-clock time）且清楚显示小于 45 分钟；以及你所做工作的描述。我们期望排行榜提交至少超过 5.0 损失这一朴素基线。在这里提交：github.com/stanford-cs336/assignment1-basics-leaderboard。

---

## 参考文献

（保留原文，不译）

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer`s Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl
