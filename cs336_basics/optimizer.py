# AdamW, cosine schedule

import torch
from torch import Tensor
from einops import rearrange, einsum

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
