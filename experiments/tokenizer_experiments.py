# 从两个数据集各采样 10 篇、用两个分词器编码、比较 bytes/token
import random
import os
from scripts.tokenize_data import build_tokenizer


def sample_file(path, num):
    assert num > 0, "num must be > 0"
    delim = b"<|endoftext|>"
    bounds = [0]
    consumed = 0
    carry = b""

    with open(path, "rb") as f:
        print(f"[PID {os.getpid()}] 正在读取文档!")
        while chunk := f.read(4_000_000):
            data = carry + chunk
            base = consumed - len(carry)
            start = 0
            while True:
                i = data.find(delim, start)
                if i == -1:
                    break
                bounds.append(base + i + len(delim))
                start = i + len(delim)
            consumed += len(chunk)
            carry = data[-(len(delim) - 1):]

    if bounds[-1] != consumed:
        bounds.append(consumed)

    doc_num = len(bounds) - 1
    index = random.Random(0).sample(range(doc_num), num)
    doc_boundary = [[bounds[idx], bounds[idx + 1]] for idx in index]
    doc = []

    with open(path, "rb") as f:
        for pair in doc_boundary:
            f.seek(pair[0])
            doc.append(f.read(pair[1] - pair[0]).decode("utf-8", errors="replace"))

    return doc


def sample_tiny():
    path = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    num = 10
    doc = sample_file(path, num)

    return doc


def sample_owt():
    path = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\owt_train.txt"
    num = 10
    doc = sample_file(path, num)

    return doc


def tokenize_tiny_sample():
    print("正在对TinyStoriesV2-GPT4-train.txt进行操作!")
    vocab_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json"
    merge_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"

    tokenizer = build_tokenizer(vocab_path_TinyStoriesV2_GPT4, merge_path_TinyStoriesV2_GPT4, ["<|endoftext|>"])

    doc = sample_tiny()
    result = [tokenizer.encode(text) for text in doc]
    token_len = sum(len(token_list) for token_list in result)
    raw_len = sum(len(text.encode("utf-8")) for text in doc)
    compression_ratio = raw_len / token_len

    # for i, text in enumerate(doc):
    #     print(f"第 {i} 个样本:\n{text}")

    print(f"compression ratio = {compression_ratio:.4f}")
    return doc, result


def tokenize_owt_sample():
    print("正在对owt_train.txt进行操作!")
    vocab_owt = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_owt.json"
    merge_owt = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_owt.json"

    tokenizer = build_tokenizer(vocab_owt, merge_owt, ["<|endoftext|>"])

    doc = sample_owt()
    result = [tokenizer.encode(text) for text in doc]
    token_len = sum(len(token_list) for token_list in result)
    raw_len = sum(len(text.encode("utf-8")) for text in doc)
    compression_ratio = raw_len / token_len

    # for i, text in enumerate(doc):
    #     print(f"第 {i} 个样本:\n{text}")

    print(f"compression ratio = {compression_ratio:.4f}")
    return doc, result


# 用 TinyStories 分词器编码 OWT 样本
def tokenize_cross_sample():
    print("正在使用 TinyStories 分词器编码 OWT 样本!")
    vocab_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json"
    merge_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"

    tokenizer = build_tokenizer(vocab_path_TinyStoriesV2_GPT4, merge_path_TinyStoriesV2_GPT4, ["<|endoftext|>"])

    doc = sample_owt()
    result = [tokenizer.encode(text) for text in doc]
    token_len = sum(len(token_list) for token_list in result)
    raw_len = sum(len(text.encode("utf-8")) for text in doc)
    compression_ratio = raw_len / token_len

    # for i, text in enumerate(doc):
    #     print(f"第 {i} 个样本:\n{text}")

    print(f"使用 TinyStories 分词器编码 OWT 样本的 compression ratio = {compression_ratio:.4f}")
    return doc, result


if __name__ == "__main__":
    for _ in range(10):
        tokenize_tiny_sample()
        tokenize_owt_sample()
        tokenize_cross_sample()     # 用 TinyStories 分词器编码 OWT 样本
