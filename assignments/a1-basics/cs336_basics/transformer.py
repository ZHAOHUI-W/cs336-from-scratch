"""
transformer.py：pre-norm（先归一化）Transformer 块。

公式：
      y = x + MHA( RMSNorm(x) )      # 注意力子层 + 残差连接
      z = y + FFN( RMSNorm(y) )      # 前馈子层（SwiGLU）+ 残差连接
其中 MHA 是带 RoPE 的因果多头自注意力（Multi-Head Attention），FFN 是 SwiGLU 前馈网络。
"""
import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.swiglu import SwiGlu



class TransformerBlock(nn.Module):
    """
    一个 Transformer 块。

    构造参数：
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): FFN 中间维度
        max_seq_len (int): RoPE 预计算的最大长度
        theta (float): RoPE 底数 Θ
        device, dtype: 设备与数据类型
    子模块及其参数形状：
        ln1, ln2: RMSNorm，weight (d_model,)
        attn: q/k/v/output_proj 各 (d_model, d_model)
        ffn: w1, w3 (d_ff, d_model)，w2 (d_model, d_ff)
    """
    def __init__(self,d_model, num_heads, d_ff, max_seq_len, theta, device=None,dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model,num_heads=num_heads,theta=theta,max_seq_len=max_seq_len,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ffn = SwiGlu(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)


    def forward(self,x,token_positions=None):
        """
        输入：x: (batch, seq_len, d_model)；token_positions: 可选，传给 RoPE
        输出：(batch, seq_len, d_model)
        """
        y = x + self.attn(self.ln1(x),token_positions)  # 注意力子层 + 残差
        z = y + self.ffn(self.ln2(y))  # 前馈子层 + 残差

        return z
