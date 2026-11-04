from itertools import islice

from cs336_basics.tokenizer import Tokenizer
import numpy as np
import array

#   train_data: data/train.npy
#   val_data: data/val.npy
def build_tokenizer(vocab_path, merge_path, spectoken):
    return Tokenizer.from_files(vocab_path, merge_path, spectoken)


def tokenize_file(tokenizer, file_path, save_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tokens = tokenizer.encode_iterable(f)
        ids = array.array("H")
        total = 0
        while True:
            chunk = list(islice(tokens, 100_000))
            if not chunk:
                break
            ids.extend(chunk)
            total += len(chunk)
            if total % 1_000_000 < 100_000:
                print(f"已处理 token: {total}")

    arr = np.frombuffer(ids, dtype=np.uint16)
    np.save(save_path, arr)


if __name__ == "__main__":
    vocab_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json"
    merge_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"

    tokenizer = build_tokenizer(vocab_path_TinyStoriesV2_GPT4, merge_path_TinyStoriesV2_GPT4, ["<|endoftext|>"])

    tiny_train = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    tine_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\train.npy"
    tokenize_file(tokenizer, tiny_train, tine_save)

    tiny_val = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    tiny_val_save = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\val.npy"
    tokenize_file(tokenizer, tiny_val, tiny_val_save)
