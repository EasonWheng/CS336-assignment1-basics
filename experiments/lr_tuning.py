"""手册 4.2.1 的式 (20)：学习率按 α/√(t+1) 衰减的 SGD，用于 learning_rate_tuning。"""

import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def run(lr, steps=10, seed=42):
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []
    for t in range(steps):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        losses.append(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
    return losses


if __name__ == "__main__":
    for lr in (1, 1e1, 1e2, 1e3):
        losses = run(lr)
        print(f"lr={lr:g}")
        for t, value in enumerate(losses):
            print(f"  step {t:2d}  {value:.6e}")
