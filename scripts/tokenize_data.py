import gc

import os
from multiprocessing import Pool

from itertools import islice

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries
import numpy as np
from numpy.lib.format import open_memmap
import array

#   train_data: data/tiny_train.npy
#   val_data: data/tiny_val.npy
def build_tokenizer(vocab_path, merge_path, spectoken):
    return Tokenizer.from_files(vocab_path, merge_path, spectoken)


def tokenize_worker(task):
    file_path, start, end, tokenizer = task

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    tokens = tokenizer.encode_iterable([chunk])
    ids = array.array("H")
    total = 0
    while True:
        piece = list(islice(tokens, 100_000))
        if not piece:
            break
        ids.extend(piece)
        total += len(piece)
        if total % 1000_000 < 100_000:
            print(f"[进程 {os.getpid()}] 已处理 {total} 个token")

    print(f"[进程 {os.getpid()}] 已结束，共处理 {total} 个token")
    return ids


def tokenize_file(tokenizer, file_path, save_path, worker, num_processes=8):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 512, b"<|endoftext|>")
    # desired_num_chunks 请务必不要设置地太小，否则可能造成内存爆炸，电脑蓝屏

    tasks = [(file_path, start, end, tokenizer) for start, end
             in zip(boundaries[:-1], boundaries[1:])]
    raw_path = save_path + ".bin"
    os.remove(raw_path)     # 确保是最新的产物
    with Pool(processes=num_processes) as pool:
        with open(raw_path, "wb") as f:
            for ids in pool.imap(worker, tasks):
                arr = np.asarray(ids, dtype="<u2")
                f.write(arr.tobytes())

    fnl_path = save_path + ".npy"
    total = os.path.getsize(raw_path) // 2

    raw = np.memmap(
        raw_path,
        dtype="<u2",
        mode="r",
        shape=(total, ),
    )

    out = open_memmap(
        fnl_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total, ),
    )

    step = 1_000_000
    for i in range(0, total, step):
        print(f"[{i / total * 100:.4f}%] 正在制作.npy文件")
        out[i:i + step] = raw[i:i + step]

    out.flush()
    del out
    del raw
    gc.collect()
    os.remove(raw_path)


def compression_ratio(token_path, text_path)->float:
    tokens = np.load(token_path, mmap_mode="r")
    origin_size = os.path.getsize(text_path)
    # bytes / compression_ratio = tokens
    compress_r = origin_size / len(tokens)
    print(f"分词器在该文件的 compression ratio: {compress_r:.4f}")
    return compress_r


def build_tiny_BPE():
    vocab_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json"
    merge_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"

    tokenizer = build_tokenizer(vocab_path_TinyStoriesV2_GPT4, merge_path_TinyStoriesV2_GPT4, ["<|endoftext|>"])

    tiny_train = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    tiny_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_train"
    tokenize_file(tokenizer, tiny_train, tiny_save, tokenize_worker, 16)

    tiny_val = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    tiny_val_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_val"
    tokenize_file(tokenizer, tiny_val, tiny_val_save, tokenize_worker, 16)


def compression_ratio_tiny():
    tiny_train = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    tiny_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_train.npy"

    tiny_val = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    tiny_val_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_val.npy"
    print("tiny-train:")
    compression_ratio(tiny_save, tiny_train)
    print("tiny-val:")
    compression_ratio(tiny_val_save, tiny_val)


def print_tokens(path, size):
    tokens = np.load(path, mmap_mode="r")
    print(f"何意味，分词结果前 {size} 个:")
    for token in tokens[:size]:
        print(token)


def print_tokens_tiny(size):
    print_tokens(r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_train.npy", size)
    print_tokens(r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_val.npy", size)


if __name__ == "__main__":
    build_tiny_BPE()
    compression_ratio_tiny()
    print_tokens_tiny(5)
