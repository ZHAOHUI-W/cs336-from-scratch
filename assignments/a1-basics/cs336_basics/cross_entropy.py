"""
cross_entropy.py：数值稳定的交叉熵损失（cross-entropy loss）。
"""
import torch
import torch.nn as nn


def cross_entropy(inputs, targets):
    """
    计算平均交叉熵。

    输入：
        inputs: (..., vocab_size) 的 logits o（未经过 softmax），例如 (batch, seq_len, V)
        targets: (...,) 的整数张量，每个位置的正确 token id
    输出：标量张量，所有位置的平均损失
    公式：ℓ = -log softmax(o)[target] = logsumexp(o) - o_target
          logsumexp(o) = m + log Σ_j exp(o_j - m)，m = max_j o_j（减最大值防止 exp 上溢）
          loss = mean(ℓ)，对所有位置取平均
    困惑度（perplexity）= exp(loss)。
    """
    targets = targets.reshape(-1)  # (...,) -> (N,)，N 为总位置数
    inputs = inputs.reshape(-1,inputs.shape[-1])  # (..., V) -> (N, V)
    m = torch.max(inputs,dim=-1,keepdim=True).values  # (N, 1) 每行最大值
    logsumexp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - m),dim=-1))  # (N,)：m + log Σ exp(o - m)
    N = inputs.shape[0]
    correct = inputs[torch.arange(N,device=inputs.device),targets]  # (N,)：取出每行正确类别的 logit o_target

    loss = (logsumexp - correct).mean()  # (N,) 求平均 -> 标量

    return loss



