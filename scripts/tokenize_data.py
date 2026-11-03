from cs336_basics.tokenizer import train_bpe, Tokenizer
import numpy as np
import array

#   train_data: data/train.npy
#   val_data: data/val.npy

if __name__ == "__main__":
    vocab_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json"
    merge_path_TinyStoriesV2_GPT4 = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"

    tokenizer = Tokenizer.from_files(vocab_path_TinyStoriesV2_GPT4, merge_path_TinyStoriesV2_GPT4, ["<|endoftext|>"])

    tiny_train = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    with open(tiny_train, "r", encoding="utf-8") as f:
        tokens = tokenizer.encode_iterable(f)
        ids = array.array("H")
        ids.extend(tokens)

    arr = np.frombuffer(ids, dtype=np.uint16)
    np.save("data/train.npy", arr)

    tiny_val = r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    with open(tiny_val, "r", encoding="utf-8") as f:
        tokens = tokenizer.encode_iterable(f)
        ids = array.array("H")
        ids.extend(tokens)

    arr = np.frombuffer(ids, dtype=np.uint16)
    np.save("data/val.npy", arr)
