# 实现一个从语言模型解码的函数。建议支持以下特性：
# 为用户提供的 prompt 生成补全（即接收 x1…t，采样补全直到遇到 <|endoftext|>）。
# 允许用户控制最大生成 token 数。
# 给定期望的温度值，在采样前对预测的下一 token 分布施加 softmax 温度缩放。
# 给定用户指定的阈值，做 top-p 采样（[A. Holtzman et al., 2020]，又称核采样）。
import torch
from torch import Tensor
from cs336_basics.model import softmax


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


