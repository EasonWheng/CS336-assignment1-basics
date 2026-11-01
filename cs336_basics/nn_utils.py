# softmax, cross_entropy, gradient_clipping

from torch import Tensor
import torch
from einops import rearrange, einsum
from .model import softmax

def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """
    计算交叉熵损失的函数，接收预测 logits（oi）和目标（xi+1），计算交叉熵 ℓi = −log softmax(oi)[xi+1]。

    函数应处理以下事项：

    - 为了数值稳定性减去最大元素。
    - 尽可能把 log 与 exp 相消。
    - 处理额外的批次维度，并返回批次上的平均值。与 3.2 节一致，我们假定类批次维度总在最前面，位于词表大小维度之前。
    :param logits:
    :param target:
    :return:
    """
    tmp = logits - logits.max(dim=-1, keepdim=True).values

    Ui_x = tmp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    sum_exp = torch.sum(torch.exp(tmp), dim=-1, keepdim=True)

    loss = (torch.log(sum_exp) - Ui_x).mean()
    return loss
