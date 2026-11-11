"""
embedding.py：词嵌入层 EmbeddingLayer（token id → 向量）。

功能：用 token id 作为行号，从嵌入矩阵中取出对应的行，不做矩阵乘法。
公式：E[token_id] ，即 out[b, s, :] = weight[token_ids[b, s], :]
可学习参数：weight: (num_embeddings, embedding_dim) = (vocab_size, d_model)
初始化：截断正态分布，std = 1，截断在 [-3, 3]
"""
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    词嵌入层。

    构造参数：
        num_embeddings (int): 词表大小 vocab_size
        embedding_dim (int): 每个 token 向量的维度 d_model
        device, dtype: 设备与数据类型
    可学习参数：
        weight: (num_embeddings, embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype))  # (vocab_size, d_model)
        nn.init.trunc_normal_(self.weight, std = 1, a = -3, b=3)  # 截断正态：均值 0、std 1、范围 [-3, 3]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        输入：token_ids: (batch_size, sequence_length)，整数（long）张量
        输出：(batch_size, sequence_length, d_model)
        做法：高级索引（advanced indexing），一次性按 id 取行
        """
        # input: token_ids: (batch_size, sequence_length)
        # output: token_embeddings (batch_size, sequence_length, d_model)
        return self.weight[token_ids]  # weight[(B, S)] -> (B, S, d_model)