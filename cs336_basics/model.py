# Linear, Embedding, RMSNorm, SwiGLU, RoPE, MHA, Block, LM

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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理形状为 (batch_size, sequence_length, d_model) 的输入张量，返回同形状张量。
        注意：如上所述，执行归一化前记得把输入升到 torch.float32（之后再降回原 dtype）。
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS_x = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        result = x * self.weight / RMS_x

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


class RotaryPositionalEmbedding(nn.Module):
    """
    构造 RoPE 模块，按需创建缓冲区。
    theta: float：RoPE 的 Θ 值
    d_k: int：查询向量和键向量的维度
    max_seq_len: int：将要输入的最大序列长度
    device: torch.device | None = None：存放缓冲区所在的设备
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super(RotaryPositionalEmbedding, self).__init__()

        positions = torch.arange(max_seq_len, device=device, dtype=torch.long)
        freqs = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)
        angles = einsum(positions, freqs, "max_seq_len, d -> max_seq_len d")
        cos, sin = angles.cos(), angles.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pair_x = rearrange(x, "... (half pair) -> ... half pair", pair=2)
        a = pair_x[..., 0]      # cos
        b = pair_x[..., 1]      # sin
        d_1 = a * self.cos[token_positions] - b * self.sin[token_positions]
        d_2 = a * self.sin[token_positions] + b * self.cos[token_positions]
        result = torch.stack([d_1, d_2], dim=-1)    # out = torch.stack([d_1, d_2], dim=-1)   # (..., seq, half, 2)
        return rearrange(result, "... half pair -> ... (half pair)")


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    tmp = x - x.max(dim=dim, keepdim=True).values
    return torch.exp(tmp) / torch.sum(torch.exp(tmp), dim=dim, keepdim=True)

def scaled_dot_product_attention(query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 mask: Bool[Tensor, " ... queries keys"] | None = None) -> torch.Tensor:
    """
    实现缩放点积注意力函数。实现应能处理形状为 (batch_size, ..., seq_len, d_k) 的键和查询，
    以及形状为 (batch_size, ..., seq_len, d_v) 的值，其中 ... 表示任意数量的其他类批次维度（若提供）。
    实现应返回形状为 (batch_size, ..., seq_len, d_v) 的输出。
    """
    dk = query.size(-1)
    tmp_attn = einsum(query, key, "... n d_k, ... m d_k -> ... n m") / (dk ** 0.5)
    if mask is not None:
        tmp_attn.masked_fill_(~mask, float("-inf"))
    attn_weights = softmax(tmp_attn, dim=-1)
    return einsum(attn_weights, value, "... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 rope: RotaryPositionalEmbedding | None = None, device=None, dtype=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_kv = d_model // num_heads      # dk == dv

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        Q = self.q_proj.forward(in_features)
        K = self.k_proj.forward(in_features)
        V = self.v_proj.forward(in_features)

        Q = rearrange(Q, "... sequence_length (head dkv) -> ... head sequence_length dkv", head=self.num_heads)
        K = rearrange(K, "... sequence_length (head dkv) -> ... head sequence_length dkv", head=self.num_heads)
        V = rearrange(V, "... sequence_length (head dkv) -> ... head sequence_length dkv", head=self.num_heads)

        if self.rope is not None and token_positions is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        mask = torch.tril(torch.ones(in_features.size(-2), in_features.size(-2), dtype=torch.bool, device=in_features.device))
        tmp = rearrange(scaled_dot_product_attention(Q, K, V, mask),
                        "... head sequence_length dkv -> ... sequence_length (head dkv)")
        return self.output_proj.forward(tmp)


class TransformerBlock(nn.Module):
    """
    实现 pre-norm Transformer 块。你的 Transformer 块至少应接受以下参数：
    d_model: int：Transformer 块输入的维度。
    num_heads: int：多头自注意力使用的头数。
    d_ff: int：逐位置前馈内层的维度。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype |None = None):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]):
        out_rsm1 = self.ln1.forward(in_features)
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
        out_attn = self.attn.forward(out_rsm1, token_positions)
        adder1 = in_features + out_attn
        out_rsm2 = self.ln2.forward(adder1)
        out_ffn = self.ffn.forward(out_rsm2)
        fnl_result = adder1 + out_ffn
        return fnl_result


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 rope_theta: float, device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
             for _ in range(num_layers)]
        )
        # THEN ... FINAL RMSNorm(layerNormalize)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # LAST Linear: OUTPUT EMBEDDING
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # 注意：虽然作业手册最后画了 Softmax，但是拼装的时候在最后不要使用softmax

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        embeddings = self.token_embeddings(in_indices)
        for layer in self.layers:
            embeddings = layer.forward(embeddings)
        ln_final_out = self.ln_final.forward(embeddings)
        output_embeddings = self.lm_head.forward(ln_final_out)
        return output_embeddings


# 下面定义的类为消融实验使用
class AblationLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 rope_theta: float, device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 block: type[nn.Module] = TransformerBlock
                 ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [block(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
             for _ in range(num_layers)]
        )
        # THEN ... FINAL RMSNorm(layerNormalize)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # LAST Linear: OUTPUT EMBEDDING
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # 注意：虽然作业手册最后画了 Softmax，但是拼装的时候在最后不要使用softmax

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        embeddings = self.token_embeddings(in_indices)
        for layer in self.layers:
            embeddings = layer.forward(embeddings)
        ln_final_out = self.ln_final.forward(embeddings)
        output_embeddings = self.lm_head.forward(ln_final_out)
        return output_embeddings


class TransformerBlockWithoutRMSNorm(nn.Module):
    """
    Without RMSNorm? Something is seriously wrong!
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype |None = None):
        super().__init__()
        # self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        # self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]):
        out_rsm1 = in_features
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
        out_attn = self.attn.forward(out_rsm1, token_positions)
        adder1 = in_features + out_attn
        out_rsm2 = adder1
        out_ffn = self.ffn.forward(out_rsm2)
        fnl_result = adder1 + out_ffn
        return fnl_result


class TransformerBlockPostNorm(nn.Module):
    """
    Post-Norm 666
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype |None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]):
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
        out_attn = self.attn.forward(in_features, token_positions)
        adder1 = in_features + out_attn
        out_rsm1 = self.ln1.forward(adder1)
        out_ffn = self.ffn.forward(out_rsm1)
        adder2 = out_rsm1 + out_ffn
        return self.ln2(adder2)


class MultiHeadSelfAttentionNoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_model // num_heads      # dk == dv

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        Q = self.q_proj.forward(in_features)
        K = self.k_proj.forward(in_features)
        V = self.v_proj.forward(in_features)

        Q = rearrange(Q, "... sequence_length (head dkv) -> ... head sequence_length dkv", head=self.num_heads)
        K = rearrange(K, "... sequence_length (head dkv) -> ... head sequence_length dkv", head=self.num_heads)
        V = rearrange(V, "... sequence_length (head dkv) -> ... head sequence_length dkv", head=self.num_heads)

        # if self.rope is not None and token_positions is not None:
        #     Q = self.rope.forward(Q, token_positions)
        #     K = self.rope.forward(K, token_positions)
        mask = torch.tril(torch.ones(in_features.size(-2), in_features.size(-2), dtype=torch.bool, device=in_features.device))
        tmp = rearrange(scaled_dot_product_attention(Q, K, V, mask),
                        "... head sequence_length dkv -> ... sequence_length (head dkv)")
        return self.output_proj.forward(tmp)


class TransformerBlockNoPE(nn.Module):
    """
    NoPE! Without Rotary Positional Embedding! No RoPE!
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype |None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        # self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionNoPE(d_model, num_heads, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]):
        out_rsm1 = self.ln1.forward(in_features)
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
        out_attn = self.attn.forward(out_rsm1, token_positions)
        adder1 = in_features + out_attn
        out_rsm2 = self.ln2.forward(adder1)
        out_ffn = self.ffn.forward(out_rsm2)
        fnl_result = adder1 + out_ffn
        return fnl_result


class FFNSiLU(nn.Module):
    """
    实现 SiLU 前馈网络
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)))


class TransformerBlockSiLU(nn.Module):
    """
    注意：FFN只有2个矩阵了！ dff = 4 * dmodel！
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype |None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device=device, dtype=dtype)
        # self.ffn = FFNSiLU(d_model, d_ff, device=device, dtype=dtype)
        self.ffn = FFNSiLU(d_model, 4 * d_model, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]):
        out_rsm1 = self.ln1.forward(in_features)
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
        out_attn = self.attn.forward(out_rsm1, token_positions)
        adder1 = in_features + out_attn
        out_rsm2 = self.ln2.forward(adder1)
        out_ffn = self.ffn(out_rsm2)
        fnl_result = adder1 + out_ffn
        return fnl_result


def build_ablation_model(type: str, config) -> AblationLM:
    model_dict = {
        "layer_norm_ablation": TransformerBlockWithoutRMSNorm,
        "pre_norm_ablation": TransformerBlockPostNorm,
        "NoPE": TransformerBlockNoPE,
        "swiglu_ablation": TransformerBlockSiLU
    }
    block = model_dict[type]
    model = AblationLM(config.d_model,
                       config.num_heads,
                       config.d_ff,
                       config.vocab_size,
                       config.context_length,
                       config.num_layers,
                       config.rope_theta,
                       config.device,
                       block=block)
    return model


def transformer_accounting(d_model, num_layers, d_ff, vocab_size):
    """
    用于计算可训练参数数量，不是计算内存消耗，计算内存消耗请使用下面一个函数！
    :param d_model:
    :param num_layers:
    :param d_ff:
    :param vocab_size:
    :return:
    """
    count_blocks = num_layers * d_model * (2 + 4 * d_model + 3 * d_ff)
    return count_blocks + d_model + 2 * vocab_size * d_model


def transformer_storage_accounting(d_model, num_layers, d_ff, vocab_size):
    """
    单精度（float32）是 32 位：1 位符号、8 位指数、23 位尾数，正好 4 个字节。
    :param d_model:
    :param num_layers:
    :param d_ff:
    :param vocab_size:
    :return:
    """
    params = transformer_accounting(d_model, num_layers, d_ff, vocab_size)
    storage_all = params * 4
    print(f"模型参数量: {params} ({params / 1e9} B), "
          f"模型内存消耗: {storage_all} B ({storage_all / 1e9} GB)")
    return storage_all


def block_matmul_FLOPs(d_model, d_ff, context_length):
    """
    FFN（SwiGLU）消耗最多，在 context_length = 1024 时占单块矩阵乘 FLOPs 的 60.4 %（42.15B / 69.84B），
    其次是QKV三个投影的22.5 %。注意力的分数矩阵与加权按n² 增长，
    序列长度超过约4·d_model（此处约6400token）时会反超FFN。
    """
    WQWKWV_matmul = 6 * context_length * d_model ** 2
    head_matmul = 4 * (context_length ** 2) * d_model
    head_concatenation_matmul = 2 * context_length * (d_model ** 2)
    FFN_SwiGLU_matmul = 6 * context_length * d_model * d_ff
    # all_matmul_block =
    # 2 * context_length * d_model * (4 * d_model + 2 * context_length + 3 * d_ff)
    all_matmul_block = WQWKWV_matmul + head_matmul + head_concatenation_matmul + FFN_SwiGLU_matmul
    print(f"单个block中:\n"
          f"投影: {WQWKWV_matmul}FLOPs, 占比为: {WQWKWV_matmul / all_matmul_block * 100}%\n"
          f"多头注意力计算(不含头拼接): {head_matmul}FLOPs, 占比为: {head_matmul / all_matmul_block * 100}%\n"
          f"head拼接+输出投影: {head_concatenation_matmul}FLOPs, 占比为: {head_concatenation_matmul / all_matmul_block * 100}%\n"
          f"FFN: {FFN_SwiGLU_matmul}FLOPs, 占比为: {FFN_SwiGLU_matmul / all_matmul_block * 100}%\n")
    return all_matmul_block


def count_matmul_FLOPs(d_model, num_layers, d_ff, vocab_size, context_length):
    """
    表 1：GPT-2 small / medium / large（context_length = 1024）
    ┌──────────────────┬─────────────────────┬───────────────────────┬──────────────────────┬──────┐
    │       组件       │ small（12 层，768） │ medium（24 层，1024） │ large（36 层，1280） │ 趋势 │
    ├──────────────────┼─────────────────────┼───────────────────────┼──────────────────────┼──────┤
    │ QKV 投影         │              14.91% │                18.70% │               20.42% │  ↑   │
    ├──────────────────┼─────────────────────┼───────────────────────┼──────────────────────┼──────┤
    │ 分数 + 加权      │              13.25% │                12.46% │               10.89% │  ↓   │
    ├──────────────────┼─────────────────────┼───────────────────────┼──────────────────────┼──────┤
    │ 输出投影         │               4.97% │                 6.23% │                6.81% │  ↑   │
    ├──────────────────┼─────────────────────┼───────────────────────┼──────────────────────┼──────┤
    │ FFN              │              39.76% │                49.86% │               54.46% │  ↑   │
    ├──────────────────┼─────────────────────┼───────────────────────┼──────────────────────┼──────┤
    │ LM head          │              27.10% │                12.75% │                7.42% │  ↓   │
    ├──────────────────┼─────────────────────┼───────────────────────┼──────────────────────┼──────┤
    │ 一次前向总 FLOPs │             2.92e11 │               8.27e11 │              1.78e12 │  ↑   │
    └──────────────────┴─────────────────────┴───────────────────────┴──────────────────────┴──────┘

    表 2 上下文长度的影响（GPT-2 XL，48 层，1600）

    ┌──────────────────┬─────────┬─────────┬──────┐
    │       组件       │  1024   │  16384  │ 趋势 │
    ├──────────────────┼─────────┼─────────┼──────┤
    │ QKV 投影         │  21.47% │   9.04% │  ↓   │
    ├──────────────────┼─────────┼─────────┼──────┤
    │ 分数 + 加权      │   9.16% │  61.73% │  ↑   │
    ├──────────────────┼─────────┼─────────┼──────┤
    │ 输出投影         │   7.16% │   3.01% │  ↓   │
    ├──────────────────┼─────────┼─────────┼──────┤
    │ FFN              │  57.53% │  24.24% │  ↓   │
    ├──────────────────┼─────────┼─────────┼──────┤
    │ LM head          │   4.68% │   1.97% │  ↓   │
    ├──────────────────┼─────────┼─────────┼──────┤
    │ 一次前向总 FLOPs │ 3.52e12 │ 1.34e14 │  ↑   │
    └──────────────────┴─────────┴─────────┴──────┘
    趋势
    模型变宽变深时，四个投影和 FFN 的占比上升，分数矩阵与加权、LM head
    的占比下降。上下文变长时，分数矩阵与加权（连同自注意力整体）的占比大幅上升，其余各项占比都下降。
    """
    one_block_FLOPs = block_matmul_FLOPs(d_model, d_ff, context_length)
    all_FLOPs = one_block_FLOPs * num_layers + 2 * context_length * d_model * vocab_size
    print(f"单个block的matmul计算量为 {one_block_FLOPs} FLOPs\n"
          f"语言模型的matmul总计算量为 {all_FLOPs} FLOPs")
    return all_FLOPs
