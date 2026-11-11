"""
transformer_lm.py：完整的 Transformer 语言模型（LM，Language Model）。

流程：token id -> 词嵌入 -> num_layers 个 Transformer 块 -> 最终 RMSNorm -> lm_head 线性层 -> logits
形状：(B, S) -> (B, S, d_model) -> ... -> (B, S, d_model) -> (B, S, vocab_size)
输出的是未归一化的 logits（还没做 softmax）。
"""
import torch
import torch.nn as nn
from cs336_basics.embedding import EmbeddingLayer
from cs336_basics.transformer import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import LinearNoBias

class TransformerLM(nn.Module):
    """
    Transformer 语言模型。

    构造参数：
        vocab_size (int): 词表大小 V
        context_length (int): 最大上下文长度，用作 RoPE 的 max_seq_len
        d_model (int): 模型维度
        num_layers (int): Transformer 块的层数
        num_heads (int): 注意力头数
        d_ff (int): FFN 中间维度
        rope_theta (float): RoPE 底数 Θ
        device, dtype: 设备与数据类型
    可学习参数形状：
        token_embeddings.weight: (vocab_size, d_model)
        layers[i]: 见 TransformerBlock
        ln_final.weight: (d_model,)
        lm_head.weight: (vocab_size, d_model)
    """
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,device=None,dtype=None):
        super().__init__()

        self.token_embeddings = EmbeddingLayer(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)  # weight: (V, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,max_seq_len=context_length,theta=rope_theta,device=device,dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head = LinearNoBias(d_in=d_model,d_out=vocab_size,device=device,dtype=dtype)  # weight: (V, d_model)，把隐藏向量映射成每个词的分数

    def forward(self,in_indices): #(B, S)
        """
        输入：in_indices: (batch, seq_len)，整数 token id
        输出：logits: (batch, seq_len, vocab_size)
        """
        x = self.token_embeddings(in_indices) #(B,S,d_model)
        for block in self.layers:
            # 注意: 这里没有传 token_positions，所以每个块里的 RoPE 都默认用位置 0..S-1
            x = block(x)  # (B, S, d_model) -> (B, S, d_model)
        x = self.ln_final(x)

        return self.lm_head(x)  # (B, S, d_model) -> (B, S, vocab_size)