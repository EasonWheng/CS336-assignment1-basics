# Tokenizer

2026-10-24 · 23 passed, 2 skipped（Windows 没有 rlimit）

`train_bpe` 学出 vocab 和 merges 之后，剩下的事都在 `Tokenizer` 里：文本进来换成一串 id，id 出去换回文本。代码在 `cs336_basics/tokenizer.py`。

## BPE 训练

`train_bpe(input_path, vocab_size, special_tokens)` 返回 `(vocab, merges)`。`vocab` 是 `dict[int, bytes]`，0–255 是单字节，256 起是特殊 token；`merges` 是 `list[tuple[bytes, bytes]]`。encode 用 `merges` 建 rank 表，decode 只查 `vocab`。

文件先按 `<|endoftext|>` 的边界切成 chunk。worker 在 `process_chunk` 里预分词，把每个预 token 转成 `tuple[bytes, ...]` 并计数；父进程用 `Pool.starmap` 收回各块的 `local_counts`，再合成 `indices`。特殊 token 不进这个计数表，单独映射到 id。

合并循环维护 `global_counts: dict[pair, [频次, 包含该 pair 的词集合]]`。每轮选频次最高的 pair，平局取字典序更大的；`merge()` 重建受影响的词，更新相邻 pair。词表大小达到 `vocab_size` 后返回。

## 类里保存了什么

构造函数收 vocab 和 merges，类里再建两个字典：

```python
self.reversed_vocab = {v: k for k, v in self.vocab.items()}
self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
```

`reversed_vocab` 给 encode 收尾用，拿到一串字节要马上知道它是第几号。`merge_ranks` 记每个 pair 的名次，名次就是它在 merges 里的下标，越小越早合并。判断一个 pair 该不该合并，看它在不在这个字典里就够了。

两个字典都在 `__init__` 里建。encode 每处理一个预 token 都要查询，放进构造函数省得每次重新计数。`reversed_vocab` 的前提是 vocab 里没有两个 id 对应同一串字节，训练出来的 vocab 正好满足。

保存文件用 latin-1。0–255 每个字节都能对应到 U+0000–U+00FF 的一个字符，读取回来再转回字节不会丢东西。写入 JSON 的时候带 `ensure_ascii=True`，Windows 上 `open` 记得设置 `encoding="utf-8"` 和 `newline="\n"`，默认编码是 cp936。

## 先切分特殊 token

特殊 token 不能进预分词器。`<|endoftext|>` 进去会变成 `<|`、`endoftext`、`|>` 三段，vocab 里根本没有这三个东西。

```python
specials = sorted(self.special_tokens, key=len, reverse=True)
split_pat = "(" + "|".join(re.escape(tok) for tok in specials) + ")"
pieces = re.split(split_pat, text)
```

排序按长度倒序排列，`<|endoftext|><|endoftext|>` 要排在 `<|endoftext|>` 前面。`re.escape` 是必须的，不然 `<` 和 `|` 会被当成正则语法。括号也不能省略，没有捕获组的话 `re.split` 会把分隔符一起丢弃。训练用的 `remove_special_token` 就是故意不带括号的，那边正好要丢掉分隔符。

切分之后一段一段检查。`piece not in special_set` 为假，说明这一段本身就是特殊 token，直接查询 `reversed_vocab`；剩下的交给 `pre_tokenize`。

## 合并

预 token 先拆分成一个个单字节。合并只在当前这一个预 token 里做，跨预 token 的相邻字节不算。

```python
candidate_pair = Counter()
for pair_in in zip(elem[:-1], elem[1:]):
    if self.merge_ranks.get(pair_in, -1) >= 0:
        candidate_pair[pair_in] += 1
if not candidate_pair:
    break
merge_pair = min(candidate_pair, key=self.merge_ranks.get)
elem = self_merge(elem, merge_pair)
```

每一轮计数一遍相邻 pair，只保留 `merge_ranks` 里有的，挑选名次最小的，然后把所有不重叠的出现替换掉。选择 `min`。第一版写成了 `max(candidate_pair, key=self.merge_ranks.get)`，合并顺序颠倒，运行得到的 id 和 tiktoken 对不上。

`.get(pair, -1)` 这个默认值容易写错。写成 0 的话，不在 merges 里的 pair 也满足 `>= 0`，每轮都被选中，`while True` 再也出不来。

合并本身是两个指针的事：配上了就跳过两个，配不上走一格。`aaa` 里的 `(a, a)` 只合左边那对。`self_merge` 返回新列表，不修改传进去的那个，所以返回结果必须接收，漏了就是原地打转。

## encode_iterable

按行读取文件，chunk 边界会切断预 token。切分成 `hel` 和 `lo` 是常见的一种，编码结果和 `hello` 不一样。行尾的 `\n` 更容易遗漏：单独一次调用里就一个 token，两次连起来会合并成 `Ċ Ċ`，`gpt2_merges.txt` 第 373 条，`tinystories_sample_5M.txt` 里出现了 1771 次。

所以要留缓冲区，每次只把前面确定的那一段输出。

```python
keep = max(max((len(tok) for tok in self.special_tokens or ()), default=0) - 1, 0)

for chunk in iterable:
    buffer += chunk
    matches = list(pre_tokenize(buffer))
    if not matches:
        continue
    X = len(buffer) - keep
    cut = 0
    for match in matches:
        if match.start() <= X:
            cut = match.start()
    for tok in sorted(self.special_tokens or (), key=len, reverse=True):
        p = buffer.find(tok)
        while p != -1:
            if p < cut < p + len(tok):
                cut = p
            p = buffer.find(tok, p + 1)

    yield from self.encode(buffer[:cut])
    buffer = buffer[cut:]

if buffer:
    yield from self.encode(buffer)
```

切点得满足三个条件。

得是某个 match 的 `start()`。正则表达式从左往右执行，前面的判定已经定了，只有最后一个会被后来的 chunk 续长。切在 match 起点上，最后一个 match 就完整留在缓冲区里。

得在 `len(buffer) - keep` 之前，`keep` 是最长特殊 token 长度减一。没收齐的特殊 token 一定贴着末尾，最长 `L-1` 个字符，起点不可能早于 `len - (L-1)`。

还不能掉进完整的特殊 token 里面。预分词不认特殊 token，`<|endoftext|>` 内部照样有 match 起点。切分在那里，`encode` 收到的前半截会被 `re.split` 当成普通文本，后半截再也拼接不回完整的特殊 token。退回它的起点 `p` 就行，`encode` 会按特殊 token 重新切分段。

`keep` 管不到中间。它只保证末尾那截没被切，完整的特殊 token 落在 buffer 中间时，离末尾远得很，`keep` 条件随便就满足了。第二个循环专门补这个。

`find` 要从 `p + 1` 接着找。同一行里有两个特殊 token 的时候，第一个没被切到，不代表第二个安全。

第一版这里写的是 `matches[-1].start() - keep`。减完的位置一般不对应任何 match，等于把一个预分词从中间劈开。roundtrip 测试只比较 decode 出来的字节，看不出来；tiktoken 那条逐 id 比较，当场就报错。

输出用 `yield from`。`yield` 一个列表出去，调用方 append 完得到的是一堆嵌套列表。循环结束记得把缓冲区剩下的再编码一次。

## decode

```python
byte_list = list(map(self.vocab.get, ids))
string = b"".join(byte_list).decode("utf-8", errors="replace")
```

先拼接再解码。单个 token 里可能只有半个 UTF-8 字符，🙃 是四个字节，BPE 很可能把它切分成两个 token，单独解码其中一个直接抛出 `UnicodeDecodeError`。拼接成完整字节流就没有问题。`errors="replace"` 是给本来就不合法的输入准备的，tiktoken 也这么处理。

## Windows 上的测试环境

`tests/test_tokenizer.py` 第一行就 `import resource`，这模块只有 Unix 有。两个内存测试在 Windows 上会被 `skipif` 跳过，可 import 发生在收集阶段，还是报 `ModuleNotFoundError`。往 `.venv/Lib/site-packages/` 放入一个 `resource.py`，里面放入一个 `RLIMIT_AS` 和两个抛出 `NotImplementedError` 的函数就够了，跳过的测试根本不会调。

另一个错误是 `KeyError: '臓'`。测试里的 helper 读 `gpt2_merges.txt` 没有设置 encoding，中文 Windows 默认 cp936，UTF-8 的 fixture 直接被解码成乱码，解出来的字符不在 `gpt2_byte_decoder` 表里。`gpt2_vocab.json` 全是 `\uXXXX` 转义，纯 ASCII，所以只有 merges 这一行出现问题。`setx PYTHONUTF8 1`，然后彻底退出 PyCharm 再开，`open()` 就默认 UTF-8 了。

## 跑测试

```text
cd C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics
.\.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -q
```

23 passed，2 skipped，跳过的是两个内存测试。

测试挂了先看这几条：`test_roundtrip_*` 只依赖 encode 和 decode 的字节往返，先让这条通过；`test_*_matches_tiktoken` 逐 id 对比，合并顺序和切点的毛病都在这暴露；`test_encode_iterable_*` 放最后。

训练产物在 `toknizer_params/`，`vocab_TinyStoriesV2-GPT4.json` 一万项，`merges_TinyStoriesV2-GPT4.json` 9743 条。代码在 `EasonWheng/CS336-assignment1-basics`。

## 数据分词

`scripts/tokenize_data.py`。分词器用 `Tokenizer.from_files` 构造，传入 `toknizer_params/vocab_TinyStoriesV2-GPT4.json`、`merges_TinyStoriesV2-GPT4.json` 与 `special_tokens=["<|endoftext|>"]`。合并文件一行一个 JSON 数组，`load_tokenizer` 逐行解析；直接对合并文件调用 `json.load` 会抛出 `Extra data`。

文本以 `utf-8` 编码打开，文件对象交给 `encode_iterable`。这个方法按行接收字符串，内部保留缓冲区，处理行边界与特殊 token。返回生成器，必须在 `with` 块内消费；块外再遍历会抛出 `ValueError: I/O operation on closed file`。

ID 用 `array.array("H")` 累积，每个元素占 2 字节。编码结束后用 `np.frombuffer(ids, dtype=np.uint16)` 转成 NumPy 数组，调用 `np.save` 写入 `data/tiny_train.npy` 与 `data/tiny_val.npy`。`configs/tinystories.yaml` 的 `train_data` 与 `val_data` 指向这两个文件。

### 几个 API 的实际行为

| API | 作用 | 参数与返回 |
| --- | --- | --- |
| `Tokenizer.from_files(vocab, merges, special_tokens=None)` | 读取词表与合并文件，构造分词器 | 类方法；返回 `Tokenizer` |
| `encode_iterable(iterable)` | 流式编码 | 参数是字符串的可迭代对象；返回生成器，逐个产出整数 |
| `encode(text)` | 编码单个字符串 | 返回 `list[int]` |
| `itertools.islice(iterator, n)` | 从迭代器取出前 n 项 | 返回迭代器；不复制剩余部分 |
| `array.array("H")` | 动态整数缓冲区 | 每个元素占 2 字节；`extend` 批量追加 |
| `np.frombuffer(buffer, dtype=np.uint16)` | 把缓冲区解释成数组 | 不复制数据；返回与缓冲区共享内存的 `ndarray` |
| `np.save(path, array)` | 写入 `.npy` 文件 | 路径没有 `.npy` 后缀时自动追加；保存 dtype 与形状 |
| `np.load(path, mmap_mode="r")` | 读取 `.npy` 文件 | 返回 `np.memmap`，按需读取磁盘 |
| `os.path.getsize(path)` | 返回文件字节数 | 不打开文件；文本模式读取的字符数不能代替 |

`np.frombuffer` 返回视图，与缓冲区共享内存。修改返回数组会修改原缓冲区；玩具验证：`b[0] = 9` 之后 `array.array` 的第一个元素也变成 9。传入 `bytes` 时视图只读，传入 `array.array` 时可写。

`np.save` 写入 `.npy` 头与二进制数据，`np.load` 从头里读出 dtype 与形状。刚才验证：`np.save("x", arr)` 生成 `x.npy`，`np.load("x.npy", mmap_mode="r")` 返回 `memmap`，dtype 为 `uint16`，形状与保存前一致。

`scripts/tokenize_data.py` 拆成两个函数。`build_tokenizer(vocab_path, merge_path, special_tokens)` 构造分词器；`tokenize_file(tokenizer, file_path, save_path)` 接收分词器、文本路径与输出路径。分词器通过参数传入，函数不依赖全局变量。`tokenize_file` 用 `islice` 每次取 10 万个 token，累计到 `array.array("H")`，每 100 万 token 输出一次进度。

worker 的步骤是：`find_chunk_boundaries` 返回偏移列表，`task = (start, end)`；worker 打开文件，`seek(start)` 后读 `end - start` 字节，`decode("utf-8", errors="ignore")`；`encode_iterable` 产出的 id 用 `array("H")` 累积，最后 `np.frombuffer(..., dtype=np.uint16)`、`np.save` 写入 `.npy`。

`f.read()` 传给 `encode_iterable` 会按字符遍历整个字符串，缓冲区反复处理，复杂度接近平方；大文件必须传入文件对象。

`if __name__ == "main"` 的判断永远为假，Python 的入口名是 `"__main__"`。`import array` 之后调用 `array("H")` 会抛出 `TypeError: 'module' object is not callable`，构造器是 `array.array`。

调用验证集时写成 `tokenize_file(tiny_val_save, tiny_val_save)`，把输出路径当成输入文本。第一次运行时 `open` 抛出 `FileNotFoundError`；生成过文件后会把二进制 `.npy` 当成 UTF-8 文本读取。输入路径与输出路径分开传。

查产物的时候看了这几项：词表 10000 项，ID 0–9999；合并 9743 对；特殊 token `<|endoftext|>` 的 ID 是 256；0–255 对应单字节。每个合并对的两部分在合并时已经存在，9743 个结果全部在词表里，没有重复；词表等于基础字节、合并结果与特殊 token 的并集。最长 token 是 `b' accomplishment'`，15 字节，ID 7160。

TinyStories 训练集 2.1GB、15,600,056 行，验证集 22MB。OWT 训练集 12GB、94,568,884 行，BPE 已经训练完，峰值 RSS 8.58GB；OWT 分词器使用 32000 词表，属于另一份产物。

### 分词结果

TinyStories 训练集 2,227,753,162 字节，编码后 541,229,169 个 token，压缩比 4.1161 bytes/token；`data/tiny_train.npy` 为 1,082,458,466 字节。验证集 22,502,601 字节，编码后 5,465,880 个 token，压缩比 4.1169 bytes/token；`data/tiny_val.npy` 为 10,931,888 字节。两个数组的 dtype 都是 `uint16`，最大 ID 9999，训练集最小 ID 9，验证集最小 ID 10。

`scripts/tokenize_data.py` 增加 `compression_ratio(token_path, text_path)`：`np.load(..., mmap_mode="r")` 返回内存映射数组，`len(tokens)` 是 token 数；`os.path.getsize(text_path)` 返回源文件字节数；两者相除得到 bytes/token，打印用 `f"{ratio:.4f}"` 保留四位小数。

单进程编码从 10:29 运行到 15:55。前段与其他进程争用 CPU，后段单核吞吐约 56,000 token/s。训练配置取 5000 步 × batch 32 × 上下文 256 = 40,960,000 个 token，占训练集的 7.6%。

`data/` 在 `.gitignore` 里，原始文本与 `.npy` 不进入版本库；`data/数据说明.txt` 用 `git add -f` 单独提交。

### 并行分词

分词是 CPU 密集型任务。当前机器 i7-12700H 有 14 个核心、20 个逻辑处理器；单进程运行时，进程的 CPU 时间占经过时间的 96%，吞吐约 42,000–48,000 token/s，其余核心空闲。Python 的 GIL 让多线程在同一时刻只执行一个线程的字节码，线程切换不会提高吞吐；并行要用 `multiprocessing`，每个进程有独立的解释器与 GIL。

切分点必须落在 `<|endoftext|>` 之后。先以二进制模式分块扫描文件，记录每个分隔符结束位置的字节偏移，再把偏移列表均分成 N 段，每段是一个 `(start, end)` 区间。区间边界都是 ASCII 字节，二进制定位不会切坏 UTF-8 字符；每段以完整文档结束，跨段合并不存在。

工作进程自己打开文件，`seek(start)`，循环 `readline()` 到 `end`，把每行用 `utf-8` 解码后产出。这个生成器交给 `encode_iterable`，缓冲区照常处理行边界与特殊 token。进程之间不共享文件句柄，也不共享分词器；用 `Pool` 的 `initializer` 让每个进程构造一次 `Tokenizer`。

`Pool.imap` 按区间顺序返回数组，父进程按顺序拼接。TinyStories 总量约 1GB，直接 `np.concatenate` 后 `np.save` 即可。OWT 约 30 亿 token、6GB，父进程持有全部数组再拼接会翻倍内存；让工作进程各自写临时 `.npy`，父进程用 `np.lib.format.open_memmap` 分配最终文件，再逐段写入。

Windows 的 `multiprocessing` 使用 spawn，工作函数与初始化函数必须是模块级函数，入口放在 `if __name__ == "__main__"` 下面。i7-12700H 有 6 个性能核，进程数取 6 左右；OWT 机器按内存与核心数调整。

### 分块大小的实测

用真实 TinyStories 分词器和 1、4、10MB 合成文本测峰值内存。`encode_iterable(f)` 按行读取，Python 峰值 0.1MB；`encode_iterable([chunk])` 把整段作为一个元素，1MB 文本峰值 85.9MB，4MB 文本 344MB，约 86MB 每 1MB，线性增长。

`pre_tokenize(buffer)` 会为整个 buffer 生成匹配列表，buffer 多大，峰值就跟到多大。TinyStories 切 16 块时每块 139MB，单个 worker 峰值十几 GB；8 个进程同时运行，48GB 内存被吃光，机器蓝屏。

把块数加到 1024，每块约 2MB，单 worker 峰值约 170MB，8 个进程约 1.4GB。OWT 12GB 切 1024 块每块约 12MB，单 worker 约 1GB，8 个进程约 8GB；内存更紧时继续加块，或让 worker 按行产出。

`Pool` 的进程数由 `tokenize_file` 最后一个参数决定。第一次调用显式写了 `2`，只启动两个 worker；改成 16 后用满。默认值是 8，显式参数覆盖默认。

任务里只放 `(file_path, start, end)`，`tokenizer` 通过 `Pool` 的 `initializer` 让每个进程构造一次，不再按任务 pickle。进程启动时把分词器存进模块级全局变量，worker 从全局取。`imap` 保持区间顺序，父进程按顺序写 `.bin`，再用 `open_memmap` 转成 `.npy`；`out.flush()` 之后 `del out`、`del raw`、`gc.collect()`，最后删除 `.bin`。

### 并行分词的结果

TinyStories 训练集切成 256 块，每块 8,702,160 字节，`num_processes=16`。从 10:09:08 到 10:29:39 生成 `data/tiny_train.npy`，用时 20 分 31 秒，整块编码得到 541,229,347 个 token，聚合吞吐约 44 万 token/s。worker 改成按行产出后重新生成，最终文件是 541,229,169 个 token，与单进程结果一致。

单进程那次从 10:29 运行到 15:55，墙钟 5 小时 26 分，平均约 27,750 token/s；按墙钟比较快约 15.8 倍。性能模式下纯单核约 53,000–56,000 token/s，按纯单核算约 8 倍。16 个 worker 超过 6 个性能核，单 worker 只有约 27,500 token/s，所以加速比低于 16。

整块编码得到的 541,229,347 比按行编码的 541,229,169 多 178 个 token，差 0.00003%，来自块内行尾的处理；解码结果一致，总量与压缩比不变。

### 分词吞吐

2026-11-06 的 OWT 训练集分词用 16 个 worker。15:17 到 15:37 的一小时写出 2 GiB 的 `data/owt_train.bin`，即 1,073,741,824 个 token，298,262 token/s，约 3.0×10^5 token/s。`.bin` 用 `uint16`，写盘速度 0.57 MiB/s；乘压缩比 4.4045 得到原文输入吞吐 1.31 MB/s。16 个 worker 合计占用约 14.1 个核，单个 worker 约 1.86×10^4 token/s。

TinyStories 训练集 20 分 31 秒编码 541,229,347 个 token，4.4×10^5 token/s；原文 2,227,753,162 字节，输入吞吐 1.81 MB/s，写盘 0.88 MB/s，单个 worker 2.75×10^4 token/s。32000 词表的合并表 31,743 对，10000 词表 9,743 对，OWT 的单个 token 处理更慢。

Pile 825GB 按 1.31 MB/s 估算，需要 6.3×10^5 秒，约 7.29 天。这个估算使用本机 16 进程的吞吐，换机器要按核数和单 worker 速率重算。

### 编码粒度与序列差异

第一个差异在第 3,614,265 个 token：旧文件是 4942（`b' \n'`），新文件是 32、10（`b' '`、`b'\n'`），解码结果完全相同。

旧文件用 `encode_iterable(f)` 按行喂，行尾的 `" \n"` 在整个字符串末尾，预分词把它当成一个整体，BPE 合并成 4942；新文件 worker 把整块 8.7MB 当成一个字符串，换行后面接着下一行的文本，预分词把空格与换行拆成 32、10。词表与合并文件没有变，差异来自编码粒度。

用真实 2MB 片段验证：整文件 `encode_iterable(f)`、切 4 块、切 8 块都得到 486,605 个 token，解码与原文逐字符一致。差异只影响行尾的分段，不影响可逆性。

## 分词器实验（2.7）

每个数据集随机抽 10 篇完整文档，文档边界为 `<|endoftext|>`，统计时把分隔符也算进字节与 token。

种子固定为 0 时，循环十次读的是同一批文档，三行输出完全一致：4.1175、4.4045、3.3059。`sample_file` 里写的是 `random.Random(0)`，每次调用都重新播种，`sample(range(doc_num), 10)` 返回同样的下标；分词器编码确定，结果逐位相同。标准差为 0 来自固定种子，不能说明样本方差小。另外四次运行得到四组不同数值，五组结果合在一起统计。

TinyStories 的 10000 词表编码自己的样本：均值 4.1493 bytes/token，样本标准差 0.0224，范围 4.1175–4.1768。OWT 的 32000 词表编码自己的样本：均值 4.4570，样本标准差 0.0877，范围 4.4045–4.6122；4.6122 这一组偏高，去掉后均值 4.4182、标准差 0.0150，中位数 4.4277。

用 TinyStories 的 10000 词表编码同一批 OWT 样本：均值 3.3066，样本标准差 0.0439，范围 3.2386–3.3562。相对 OWT 自己的 4.4570 下降 25.8%，相同字节多出 34.8% 的 token。网页文本里的长词、专有名词、URL 与数字在 TinyStories 的合并表里覆盖不足，片段被拆得更碎；字节级 BPE 不会出现无法编码的字符。熟悉语料的两组均值都在 4 以上，跨语料编码降到 3.31。

两个词表分别是 10000 与 32000，ID 上限 9999 与 31999，都小于 65536，`uint16` 的 0–65535 足够表示。训练时 `get_batch` 再把索引转成 int64。

## 简答题（unicode1/2）

`chr(0)` 返回 U+0000，也就是 NUL，字符串表示是 `'\x00'`。`repr` 会把它显示成转义的 `\x00`，`print` 直接输出一个不可见控制字符；把它拼进句子打印时，终端看不到这个字符，但 `repr` 能看到。

UTF-8 对 ASCII 保持单字节，英文网页最紧凑；没有字节序和 BOM 问题；变长编码适合按字节处理，也正好配合 byte-level BPE。UTF-16 需要处理代理对和字节序，UTF-32 每个码点固定 4 字节，英文文本会明显膨胀。

逐字节解码函数把每个字节单独交给 `decode("utf-8")`，多字节字符会被拆开。一个字节不是完整的 UTF-8 字符，单独解码会抛 `UnicodeDecodeError`。例如 `'牛'.encode('utf-8')` 是 `b'\xe7\x89\x9b'`，三个字节分开都不能单独解码。

不能解码的两字节序列可以取 `b'\xff\xff'`。`0xff` 不是合法的 UTF-8 起始字节，紧跟的字节也不能组成合法序列。
