# Linear, Embedding, RMSNorm, RoPE, MHA, SwiGLU, Block, LM

import torch, torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np
import os


class Linear(nn.Module):
    """
    构造线性变换模块。该函数应接受以下参数：
    in_features: int：输入的最后一维大小
    out_features: int：输出的最后一维大小
    device: torch.device | None = None：存放参数所在的设备
    dtype: torch.dtype | None = None：参数的数据类型
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0., self.std, -3 * self.std, 3 * self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    构造嵌入模块。该函数应接受以下参数：
    num_embeddings: int：词表大小
    embedding_dim: int：嵌入向量维度，即 dmodel
    device: torch.device | None = None：存放参数所在的设备
    dtype: torch.dtype | None = None：参数的数据类型
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0., 1, -3, 3)  # Embedding：𝒩(μ = 0, σ² = 1)，截断在 [−3, 3]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """构造 RMSNorm 模块。该函数应接受以下参数：
    d_model: int：模型的隐藏维度
    eps: float = 1e-5：用于数值稳定性的 epsilon
    device: torch.device | None = None：存放参数所在的设备
    dtype: torch.dtype | None = None：参数的数据类型
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理形状为 (batch_size, sequence_length, d_model) 的输入张量，返回同形状张量。
        注意：如上所述，执行归一化前记得把输入升到 torch.float32（之后再降回原 dtype）。
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS_x = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        result = x * self.gain / RMS_x

        return result.to(in_type)


def SiLU(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """
    实现 SwiGLU 前馈网络，由 SiLU 激活函数和一个 GLU 组成。
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
