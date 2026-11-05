import os
from multiprocessing import Pool

from itertools import islice

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries
import numpy as np
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

    tokens = tokenizer.encode_iterable(chunk)
    return array.array("H").extend(tokens)


def tokenize_file(tokenizer, file_path, save_path, worker, num_processes=8):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(file_path, 2 * num_processes, b"<|endoftext|>")

    tasks = [(file_path, start, end, tokenizer) for start, end
             in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        for ids in pool.imap(worker, tasks):




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
    tiny_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_train.npy"
    tokenize_file(tokenizer, tiny_train, tiny_save)

    tiny_val = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    tiny_val_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\tiny_val.npy"
    tokenize_file(tokenizer, tiny_val, tiny_val_save)


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
    # build_tiny_BPE()
    compression_ratio_tiny()
    print_tokens_tiny(5)
