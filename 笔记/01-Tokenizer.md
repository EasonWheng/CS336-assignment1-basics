# Tokenizer

2026-10-24 · 23 passed, 2 skipped（Windows 没有 rlimit）

`train_bpe` 学出 vocab 和 merges 之后，剩下的事都在 `Tokenizer` 里：文本进来换成一串 id，id 出去换回文本。代码在 `cs336_basics/tokenizer.py`。

## 类里存了什么

外面传进来 vocab 和 merges，类里另外留两张表：

```python
self.reversed_vocab = {v: k for k, v in self.vocab.items()}
self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
```

`reversed_vocab` 给 encode 收尾用，拿到一串字节要马上知道它是第几号。`merge_ranks` 记每个 pair 的名次，名次就是它在 merges 里的下标，越小越早合并。判断一个 pair 该不该合并，看它在不在这个字典里就够了。

两张表都放在 `__init__` 里建。encode 每处理一个预 token 都要查，放进构造函数省得每次重新数。`reversed_vocab` 的前提是 vocab 里没有两个 id 对应同一串字节，训练出来的 vocab 正好满足。

存文件用 latin-1。0–255 每个字节都能对应到 U+0000–U+00FF 的一个字符，读回来再转回字节不会丢东西。写 JSON 的时候带 `ensure_ascii=True`，Windows 上 `open` 记得写 `encoding="utf-8"` 和 `newline="\n"`，默认编码是 cp936。

## 先切特殊 token

特殊 token 不能进预分词器。`<|endoftext|>` 进去会变成 `<|`、`endoftext`、`|>` 三段，vocab 里根本没有这三个东西。

```python
specials = sorted(self.special_tokens, key=len, reverse=True)
split_pat = "(" + "|".join(re.escape(tok) for tok in specials) + ")"
pieces = re.split(split_pat, text)
```

排序按长度倒着来，`<|endoftext|><|endoftext|>` 要排在 `<|endoftext|>` 前面。`re.escape` 是必须的，不然 `<` 和 `|` 会被当成正则语法。括号也不能省，没有捕获组的话 `re.split` 会把分隔符一起扔掉。训练用的 `remove_special_token` 就是故意不带括号的，那边正好要丢掉分隔符。

切开之后一段一段看。`piece not in special_set` 为假，说明这一段本身就是特殊 token，直接查 `reversed_vocab`；剩下的交给 `pre_tokenize`。

## 合并

预 token 先摊成一个个单字节。合并只在当前这一个预 token 里做，跨预 token 的相邻字节不算，这是预分词的意义。

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

每一轮数一遍相邻 pair，只留 `merge_ranks` 里有的，挑名次最小的，然后把所有不重叠的出现换掉。选 `min`。第一版写成了 `max(candidate_pair, key=self.merge_ranks.get)`，合并顺序颠倒，跑出来的 id 和 tiktoken 对不上。

`.get(pair, -1)` 这个默认值容易写错。写成 0 的话，不在 merges 里的 pair 也满足 `>= 0`，每轮都被选中，`while True` 再也出不来。

合并本身是两个指针的事：配上了就跳过两个，配不上走一格。`aaa` 里的 `(a, a)` 只合左边那对。`self_merge` 返回新列表，不修改传进去的那个，所以返回值得接住，漏了就是原地打转。

## encode_iterable

按行读文件，chunk 边界会切断预 token。切成 `hel` 和 `lo` 是常见的一种，编码结果和 `hello` 不一样。行尾的 `\n` 更容易漏：单独一次调用里就一个 token，两次连起来会合并成 `Ċ Ċ`，`gpt2_merges.txt` 第 373 条，`tinystories_sample_5M.txt` 里出现了 1771 次。

所以要留缓冲区，每次只把前面确定的那截放出去。

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

切点不好随便挑，有三个条件。

得是某个 match 的 `start()`。正则从左往右跑，前面的判定已经定了，只有最后一个会被后来的 chunk 续长。切在 match 起点上，最后一个 match 就完整留在缓冲区里。

得在 `len(buffer) - keep` 之前，`keep` 是最长特殊 token 长度减一。没收齐的特殊 token 一定贴着末尾，最长 `L-1` 个字符，起点不可能早于 `len - (L-1)`。

还不能掉进完整的特殊 token 里面。预分词不认特殊 token，`<|endoftext|>` 内部照样有 match 起点。切在那儿，`encode` 收到的前半截会被 `re.split` 当成普通文本，后半截再也拼不回完整的特殊 token。退回它的起点 `p` 就行，`encode` 会按特殊 token 重新切段。

`keep` 管不到中间。它只保证末尾那截没被切，完整的特殊 token 落在 buffer 中间时，离末尾远得很，`keep` 条件随便就过了。第二个循环专门补这个。

`find` 要从 `p + 1` 接着找。同一行里有两个特殊 token 的时候，第一个没被切到，不代表第二个安全。

第一版这里写的是 `matches[-1].start() - keep`。减完的位置一般不对应任何 match，等于把一个预分词从中间劈开。roundtrip 测试只比 decode 出来的字节，看不出来；tiktoken 那条逐 id 比，当场就报。

输出用 `yield from`。`yield` 一个列表出去，调用方 append 完拿到的是一堆嵌套列表。循环结束记得把缓冲区剩下的再编一次。

## decode

```python
byte_list = list(map(self.vocab.get, ids))
string = b"".join(byte_list).decode("utf-8", errors="replace")
```

先拼再解。单个 token 里可能只有半个 UTF-8 字符，🙃 是四个字节，BPE 很可能把它切成两个 token，单独解其中一个直接抛 `UnicodeDecodeError`。拼成完整字节流就没事了。`errors="replace"` 是给本来就不合法的输入准备的，tiktoken 也这么处理。

## 测试环境

`tests/test_tokenizer.py` 第一行就 `import resource`，这模块只有 Unix 有。两个内存测试在 Windows 上会被 `skipif` 跳过，可 import 发生在收集阶段，还是报 `ModuleNotFoundError`。往 `.venv/Lib/site-packages/` 扔一个 `resource.py`，里面放个 `RLIMIT_AS` 和两个抛 `NotImplementedError` 的函数就够了，跳过的测试根本不会调。

另一个错误是 `KeyError: '臓'`。测试里的 helper 读 `gpt2_merges.txt` 没写 encoding，中文 Windows 默认 cp936，UTF-8 的 fixture 直接被解成乱码，解出来的字符不在 `gpt2_byte_decoder` 表里。`gpt2_vocab.json` 全是 `\uXXXX` 转义，纯 ASCII，所以只有 merges 这行出事。`setx PYTHONUTF8 1`，然后彻底退出 PyCharm 再开，`open()` 就默认 UTF-8 了。

## 结果

```text
cd C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics
.\.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -q
```

23 passed，2 skipped，跳过的是两个内存测试。

出问题的时候按这个顺序看：`test_roundtrip_*` 只依赖 encode 和 decode 的字节往返，先把这条走通；`test_*_matches_tiktoken` 逐 id 对比，合并顺序和切点的毛病都在这暴露；`test_encode_iterable_*` 放最后。

训练产物在 `toknizer_params/`，`vocab_TinyStoriesV2-GPT4.json` 一万项，`merges_TinyStoriesV2-GPT4.json` 9743 条。代码在 `EasonWheng/CS336-assignment1-basics`。
