from collections import defaultdict, Counter
from dataclasses import dataclass
import os
import json
from pathlib import Path
from multiprocessing import Pool
from collections.abc import Iterable, Iterator
try:
    from .pretokenization import find_chunk_boundaries
except ImportError:
    from pretokenization import find_chunk_boundaries
import regex as re

def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(string.encode("utf-8"))
    # num_bytes = len(list(map(int, string.encode("utf-8"))))
    # num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]  # index -> bytes
    merges: list[tuple[bytes, bytes]]  # bytes -> index

# 训练分词器之前需要预分词
def pre_tokenize(string: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, string)     # 注意返回的是一个迭代器


def remove_special_token(string: str, special_tokens: list[str]) -> list[str]:
    assert special_tokens, "special tokenizer should not be empty"
    if not special_tokens:
        return [string]
    specials = sorted(special_tokens, key=len, reverse=True)  # 长的排前面
    split_pat = "|".join(re.escape(tok) for tok in specials)
    return re.split(split_pat, string)


def process_chunk(*args):
    start, end, input_path, special_tokens = args
    local_counts = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    chunks = remove_special_token(chunk, special_tokens)        # 这是一个一堆str组成的list：[chunk, chunk...]
    all_words_iter = [pre_tokenize(chunk) for chunk in chunks]  # 这是由迭代器组成的list
    # 每一个元素都是迭代器，对应一个chunk预分词的结果
    for chunk in all_words_iter:
        for word in chunk:
            local_counts[tuple(bytes([x]) for x in word.group().encode("utf-8"))] += 1
    print("何意味，哈机密-pre-分词进程返回了")
    return local_counts


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = ({x: bytes([x]) for x in range(256)}
                               | {(256 + i): special_tokens[i].encode("utf-8")
                                  for i in range(len(special_tokens))})

    current_vocab_size = len(vocab)

    file_size = os.path.getsize(input_path)
    if file_size < 1_000_000:
        num_chunks = 1
        workers = 1
    else:
        workers = min(14, os.cpu_count() or 1)
        num_chunks = min(64, max(workers, file_size // 1_000_000))

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    tasks = [(start, end, input_path, special_tokens) for start, end
             in zip(boundaries[:-1], boundaries[1:])]

    global_counts_origin = Counter()
    if workers == 1:
        for task in tasks:
            global_counts_origin.update(process_chunk(*task))
    else:
        with Pool(processes=workers) as pool:
            for local_counts in pool.starmap(process_chunk, tasks):
                global_counts_origin.update(local_counts)
    # global_counts_origin just like this: {(b"l",n"o",b"w"): 5, ...}
    # print("哈哈哈预分词结束了，开始低效的建立词表")    # 这里应当是 的 -> 地
    indices = global_counts_origin
    global_counts = count_adjacent_pairs(global_counts_origin)
    for i in range(current_vocab_size, vocab_size):
        print(f"这是第{i}个vocab, gogogo!!!")
        # global_counts返回的是dict[tuple[bytes, bytes], list[int, set[tuple[bytes, ...]]]]
        max_pair = max(global_counts, key=lambda p: (global_counts[p][0], p))  # tuple[bytes, bytes], 作业要求平局取字典序更大的 pair
        new_index = i

        merges.append(max_pair)
        vocab[new_index] = max_pair[0] + max_pair[1]

        words_changed = global_counts.get(max_pair)[1]  # 是set[tuple[bytes, ...]]
        indices, global_counts = merge(indices, global_counts, max_pair, words_changed)

    return vocab, merges

"""主函数里：

  words_changed = global_counts.get(max_pair)[1]
  merge(indices, global_counts, max_pair, words_changed)

  传给 merge 的是这个 set 对象本身，不是拷贝。merge 里的参数 words_changed 和
  global_counts[max_pair][1] 指向同一块内存，和变量名写在哪个函数里没有关系。

  所以 merge 里执行：

  global_counts[old_pair][1].discard(word)

  当 old_pair == max_pair 时：

  global_counts[old_pair][1]  →  主函数 get 到的那个 set  →  merge 的参数 words_changed

  三者是同一个对象。discard 修改它，外层 for word in words_changed 正在遍历的也是它，于是
  RuntimeError。

  同理，global_counts 也是主函数的同名对象，merge 改它主函数立刻能看到。你在 merge
  里看到的和主函数里的是同一份数据，函数的边界不会复制可变对象。
  """
def merge(indices: dict[tuple[bytes, ...], int],
          global_counts: dict[tuple[bytes, bytes], list[int, set[tuple[bytes, ...]]]],
          pair: tuple[bytes, bytes],
          words_changed: set[tuple[bytes, ...]]):
    touched = set()
    for word in list(words_changed):
        current_word = []
        f_old = indices[word]
        for old_pair in zip(word[:-1], word[1:]):
            global_counts[old_pair][0] -= f_old
            global_counts[old_pair][1].discard(word)
            touched.add(old_pair)
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                current_word.append(pair[0] + pair[1])
                i += 2
                continue
            current_word.append(word[i])
            i += 1

        new_word = tuple(current_word)
        indices[new_word] += indices[word]
        indices.pop(word)

        for new_pair in zip(new_word[:-1], new_word[1:]):
            global_counts[new_pair][0] += f_old
            global_counts[new_pair][1].add(new_word)

    for old_pair in touched:
        if global_counts[old_pair][0] <= 0:
            global_counts.pop(old_pair)

    return indices, global_counts


def count_adjacent_pairs(indices: dict[tuple[bytes, ...], int]) \
        -> dict[tuple[bytes, bytes], list[int, set[tuple[bytes, ...]]]]:   # 这里维护一份倒排索引
    """Return a dictionary mapping each adjacent pair of tokens in `indices` to the number of times it occurs."""
    counts = defaultdict(lambda: [0, set()])
    for word, freq in indices.items():
        for pair in zip(word[:-1], word[1:]):
            counts[pair][0] += freq
            counts[pair][1].add(word)

    return counts  # 返回的是dict[tuple[bytes, bytes], list[int, set[tuple[bytes, ...]]]]


def save_tokenizer(vocab, merges, vocab_path, merges_path):
    vocab_path, merges_path = Path(vocab_path), Path(merges_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    # vocab: {0: b"\x00", ..., 256: b"<|endoftext|>"}
    vocab_json = {str(i): tok.decode("latin-1") for i, tok in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=True)

    # merges: 每行一个 pair
    with open(merges_path, "w", encoding="utf-8", newline="\n") as f:
        for a, b in merges:
            f.write(json.dumps([a.decode("latin-1"), b.decode("latin-1")],
                               ensure_ascii=True) + "\n")


def load_tokenizer(vocab_path, merges_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {int(i): tok.encode("latin-1") for i, tok in raw.items()}

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = json.loads(line)
            merges.append((a.encode("latin-1"), b.encode("latin-1")))

    return vocab, merges


class Tokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""

    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        vocab_from_file, merges_from_file = load_tokenizer(vocab_filepath, merges_filepath)
        return Tokenizer(vocab_from_file, merges_from_file, special_tokens)


    def encode(self, text: str) -> list[int]:
        def self_merge(indices: list[bytes], pair: tuple[bytes, bytes]):
            new_seq = []
            i = 0
            n = len(indices)
            while i < n:
                if i < n - 1 and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                    new_seq.append(pair[0] + pair[1])
                    i += 2
                    continue
                new_seq.append(indices[i])
                i += 1
            return new_seq

        special_set = set(self.special_tokens or ())
        if not self.special_tokens:  # None 或 []
            pieces = [text]  # 直接整段走预分词
        else:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            split_pat = "(" + "|".join(re.escape(tok) for tok in specials) + ")"
            pieces = re.split(split_pat, text)

        all_words_iter = [pre_tokenize(piece) if piece not in special_set
                          else piece for piece in pieces]  # 这是由迭代器组成的list，但是特殊token请注意直接保留
        result = []

        for word in all_words_iter:
            if isinstance(word, str):
                result.append(self.reversed_vocab[word.encode("utf-8")])
                continue

            tmp = [[bytes([y]) for y in x.group().encode("utf-8")] for x in word]
            for elem in tmp:
                while True:
                    candidate_pair = Counter()
                    for pair_in in zip(elem[:-1], elem[1:]):
                        if self.merge_ranks.get(pair_in, -1) >= 0:
                            candidate_pair[pair_in] += 1
                    if not candidate_pair:
                        break
                    merge_pair = min(candidate_pair, key=self.merge_ranks.get)
                    elem = self_merge(elem, merge_pair)

                for byte_pair in elem:
                    result.append(self.reversed_vocab[byte_pair])

        return result


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
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


    def decode(self, ids: list[int]) -> str:
        byte_list = list(map(self.vocab.get, ids))
        string = b"".join(byte_list).decode("utf-8", errors="replace")
        return string


if __name__ == "__main__":
    path_fast = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    path_slow = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\owt_train.txt"
    vocab_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json"
    merge_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"
    vocab_path_owt = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_owt.json"
    merge_path_owt = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_owt.json"

    config_long_or_short = "long"

    choose = {"short": path_fast, "long": path_slow}
    choose_path = {"long": [vocab_path_owt, merge_path_owt], "short":[vocab_path_TinyStoriesV2_GPT4, merge_path_TinyStoriesV2_GPT4]}
    params = train_bpe(choose[config_long_or_short], 32000, ["<|endoftext|>"])
    vocab, merges = params

    path_params = choose_path[config_long_or_short]
    save_tokenizer(vocab, merges, path_params[0], path_params[1])
    vocab, merges = load_tokenizer(path_params[0], path_params[1])
