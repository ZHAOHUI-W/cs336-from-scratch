"""
nn_utils.py：数值稳定的 softmax 和缩放点积注意力（SDPA，Scaled Dot-Product Attention）。
"""
import torch
import torch.nn as nn
from einops import einsum
import math
import torch.cuda.nvtx as nvtx
def softmax(x,dim):
    """
    数值稳定的 softmax（归一化指数函数）。

    输入：
        x: 任意形状张量
        dim (int): 在哪一维上做 softmax
    输出：与 x 同形状，dim 这一维上的元素之和为 1
    公式：softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
          先减去最大值不改变结果，但能防止 exp 上溢（例如 exp(1000) = inf）。
    注意：如果某一行全是 -inf，max 也是 -inf，相减得到 NaN，这一行输出会是 NaN。
    """
    max_value = torch.max(x,dim=dim,keepdim=True).values  # (..., 1, ...)，keepdim 便于广播
    shifted = x - max_value  # 减最大值后最大项为 0，exp 不会上溢
    exp = torch.exp(shifted)
    total = torch.sum(exp,dim=dim,keepdim=True)

    return exp/total
                      

# 注意: 没有对「整行都被 mask 掉」的情况做特殊处理，那样 softmax 会得到 NaN；因果 mask 下对角线总是 True，所以不会出现。
def scaled_dot_product_attention(Q:torch.Tensor,K,V,mask=None):
    """
    缩放点积注意力。

    输入：
        Q: (..., m, d_k)   m 个查询（query）位置
        K: (..., n, d_k)   n 个键（key）位置
        V: (..., n, d_v)   n 个值（value）
        mask: 可选，布尔张量，形状可广播到 (..., m, n)；True 表示「允许看」，False 表示屏蔽
    输出：(..., m, d_v)
    公式：Attention(Q, K, V) = softmax( Q Kᵀ / √d_k + M ) V
          其中 M 在 mask 为 False 处取 -inf（softmax 后权重为 0），为 True 处取 0。
    """
    with nvtx.range("attn_scores"):
        scores = einsum(Q,K,"... m d_k, ... n d_k -> ... m n") / math.sqrt(Q.shape[-1])  # (..., m, d_k) x (..., n, d_k) -> (..., m, n)，再除以 √d_k
    if mask is not None:
        with nvtx.range("attn_mask"):
            scores = scores.masked_fill(~mask,float('-inf'))  # mask 为 False 的位置填 -inf
        with nvtx.range("attn_softmax"):
            attn_softmax = softmax(scores,dim=-1) # softmax 沿 n 维；(..., m, n) x (..., n, d_v) -> (..., m, d_v) 
        with nvtx.range("attn_final_matual"):
            attn_final_matual = einsum(attn_softmax,V,"... m n, ... n d_v -> ... m d_v")  
    return attn_final_matual