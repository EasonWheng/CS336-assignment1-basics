# Linear, Embedding, RMSNorm, RoPE, MHA, SwiGLU, Block, LM

import torch, torch.nn as nn
from einops import rearrange, einsum
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
        nn.init.trunc_normal_(self.weight, 0., 1, -3, 3) # Embedding：𝒩(μ = 0, σ² = 1)，截断在 [−3, 3]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]




