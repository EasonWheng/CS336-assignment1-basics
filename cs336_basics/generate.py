# 实现一个从语言模型解码的函数。建议支持以下特性：
# 为用户提供的 prompt 生成补全（即接收 x1…t，采样补全直到遇到 <|endoftext|>）。
# 允许用户控制最大生成 token 数。
# 给定期望的温度值，在采样前对预测的下一 token 分布施加 softmax 温度缩放。
# 给定用户指定的阈值，做 top-p 采样（[A. Holtzman et al., 2020]，又称核采样）。
import torch
from torch import Tensor
from cs336_basics.model import softmax, TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import load_config, build_model, parse_args


def temperature_scale(x: Tensor, temperature) -> Tensor:
    # 注意 τ → 0 时，v 中最大的元素占据主导，softmax 的输出退化为集中在最大元素上的 one-hot 向量。
    return x / temperature


def temperature_scaled_softmax(x: Tensor, temperature) -> Tensor:
    if temperature < 0:
        raise ValueError("temperature should be non-negative")

    if temperature == 0:
        idx = x.argmax(dim=-1, keepdim=True)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idx, 1)
        return mask

    x = temperature_scale(x, temperature)
    return softmax(x, dim=-1)


def nucleus_sampling(x: Tensor, p_target: float) -> Tensor:
    """
    三维 (batch, seq, vocab) 在调用之前就处理掉了：logits[:, -1, :] 取最后一个位置，温度 softmax 后传入。核采样函数不关心 seq 维。
    :param x:
    :param p_target:
    :return:
    """
    if not 0 <= p_target <= 1:
        raise ValueError("p_target should be between 0 and 1")
    value, _ = x.sort(dim=-1, descending=True)    # 此处sort不会修改原tensor
    cumsum = torch.cumsum(value, dim=-1)
    k = (cumsum < p_target).sum(dim=-1, keepdim=True)
    k = k.clamp(max=x.size(-1) - 1)
    cut_off = value.gather(dim=-1, index=k)
    # masked 就是对输入 x 操作之后的概率分布，可以直接送给torch.multinomial
    masked = torch.where(x >= cut_off, x, torch.zeros_like(x))
    masked = masked / masked.sum(dim=-1, keepdim=True)

    output = torch.multinomial(masked, 1)
    return output


def sampling(x: Tensor, temperature: float, p_nucleus_sampling: float) -> Tensor:
    """
    调用方要保证 x 是最后一个位置的 logits，形状 (B, V)：logits[:, -1, :]

    返回（batch 1）的采样结果
    :param x:
    :param temperature:
    :param p_nucleus_sampling:
    :return:
    """
    tmp = temperature_scaled_softmax(x, temperature)
    return nucleus_sampling(tmp, p_nucleus_sampling)


def decoding(prompt: Tensor, max_token_num: int, context_len, temperature, top_p
             , model: TransformerLM, device, eos_id=256) -> Tensor:
    """
    从语言模型解码的函数。支持以下特性：

    - 为用户提供的 prompt 生成补全（即接收 x1…t，采样补全直到遇到 <|endoftext|>）。
    - 允许用户控制最大生成 token 数。
    - 给定期望的温度值，在采样前对预测的下一 token 分布施加 softmax 温度缩放。
    - 给定用户指定的阈值，做 top-p 采样（[A. Holtzman et al., 2020]，又称核采样）。

    注意：本函数默认用户输入batch == 1
    """
    len_prompt = prompt.size(-1)
    max_len = min(max_token_num + len_prompt, context_len)
    if len_prompt > max_len:
        raise ValueError("len_prompt should be less than context_len")
    if max_token_num < 0:
        raise ValueError("max_token_num should be greater than 0")
    if max_token_num == 0:
        return prompt

    buffer = torch.empty(1, max_len, dtype=torch.long, device=device)
    buffer[:, :len_prompt] = prompt
    pos = len_prompt

    model.eval()
    with torch.no_grad():
        while pos < max_len:
            input_ids = buffer[:, :pos]
            logits = model(input_ids)[:, -1, :]
            next_id = sampling(logits, temperature, p_nucleus_sampling=top_p)
            if next_id.item() == eos_id:
                break
            buffer[:, pos : pos + 1] = next_id
            pos += 1

    return buffer[:, :pos].clone()


def build_tokenizer(vocab_path, merge_path, spectoken):
    return Tokenizer.from_files(vocab_path, merge_path, spectoken)


def handle_user_input(choose_tokenizer, config_path, checkpoint_path, max_token_num, temperature, top_p, device):
    tokenizer_dict = {"tiny": (r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_TinyStoriesV2-GPT4.json",
                               r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_TinyStoriesV2-GPT4.json"),
                      "owt": (r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\vocab_owt.json",
                              r"C:\Users\tomat\Desktop\CSdiy\CS336\CS336_Assignments\assignment1-basics\toknizer_params\merges_owt.json")}
    chosen_tokenizer = tokenizer_dict[choose_tokenizer]
    tokenizer = build_tokenizer(chosen_tokenizer[0], chosen_tokenizer[1], ["<|endoftext|>"])
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    args = parse_args(["--config", config_path])
    config = load_config(config_path, args)
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    def exec_decode(prompt: str) -> str:
        if not prompt:
            return "Input prompt cannot be empty~"
        input_list = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_list], dtype=torch.long, device=device)
        output_ids = decoding(input_ids, max_token_num, config.context_length, temperature, top_p, model, device, eos_id)
        return tokenizer.decode(output_ids[0].tolist())

    return exec_decode
