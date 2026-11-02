# AdamW, cosine schedule

import torch
from torch import Tensor
from einops import rearrange, einsum

from cs336_basics.model import count_matmul_FLOPs


class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 betas=(0.9, 0.95),
                 eps=1e-8,
                 weight_decay=0.3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] <= 0 or betas[1] <= 0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid eight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data              # Compute the gradient of the loss
                alpha_t = alpha * (1 - betas[1] ** t) ** 0.5 / (1 - betas[0] ** t)  # Compute adjusted 𝛼 for iteration t
                p.data -= alpha * weight_decay * p.data         # Apply weight decay
                state["m"] = betas[0] * m + (1 - betas[0]) * grad       # Update the first moment estimate
                state["v"] = betas[1] * v + (1 - betas[1]) * grad ** 2  # Update the second moment estimate
                state["t"] = t + 1
                m = state.get("m", 0)
                v = state.get("v", 0)   # 肯定要用最新一次的梯度更新过后的一阶矩和二阶矩去做参数更新！！！
                p.data -= alpha_t * m / (v ** 0.5 + eps)

        return loss


def storage_AdamW(vocab_size: int,
                   context_len: int,
                   num_layers: int,
                   d_model: int,
                   num_heads: int):
    """
    h = storage_AdamW(
          vocab_size=50257,
          context_len=1024,
          num_layers=48,
          d_model=1600,
          num_heads=25,
          )

    print(h(batch_size=1))
    请注意：本函数返回单位为GB
    """
    def h(batch_size):
        return (num_layers * (56 / 3 * batch_size * context_len * d_model + 2 * batch_size * num_heads * context_len ** 2)
                + batch_size * context_len * d_model + batch_size * context_len * vocab_size + batch_size * context_len
                + ((2 * d_model + 12 * d_model ** 2) * num_layers + d_model + 2 * d_model * vocab_size) * 4) * 4 / 1e9
    return h


def FLOPs_AdamW(vocab_size: int,
                   num_layers: int,
                   d_model: int):
    return ((2 * d_model + 12 * d_model ** 2) * num_layers + d_model + 2 * d_model * vocab_size) * 14


def calc_flops(vocab_size: int,
               num_layers: int,
               d_model: int,
               d_ff: int,
               context_len: int,
               batch_size: int):
    """
    print(calc_flops(50257, 48, 1600, 8/3*1600, 1024, 1024) * 400_000 / (495 * 0.5) / 1e12 / 60 /60 /24)
    GPT_2 XL要201.5075449133109天
    :param vocab_size:
    :param num_layers:
    :param d_model:
    :param d_ff:
    :param context_len:
    :param batch_size:
    :return:
    """
    return 3 * count_matmul_FLOPs(d_model, num_layers, d_ff, vocab_size, context_len) * batch_size
