"""
attention.py：因果多头自注意力（causal multi-head self-attention），可选 RoPE。

公式：
      Q = x W_Qᵀ, K = x W_Kᵀ, V = x W_Vᵀ，然后拆成 h 个头
      head_j = softmax( Q_j K_jᵀ / √d_k + causal_mask ) V_j
      MultiHead(x) = W_O · concat(head_1, ..., head_h)
      因果 mask（causal mask）：位置 i 只能看 j ≤ i 的位置（下三角为 True）。
可学习参数（都无偏置）：
      q_proj/k_proj/v_proj/output_proj 的 weight: (d_model, d_model)
      其中 d_k = d_v = d_model / num_heads
"""
import torch
import torch.nn as nn
from cs336_basics.linear import LinearNoBias
from cs336_basics.rope import RoPE
from cs336_basics.nn_utils import scaled_dot_product_attention
from einops import rearrange
import torch.cuda.nvtx as nvtx
class MultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力。

    构造参数：
        d_model (int): 模型维度
        num_heads (int): 头数 h，d_k = d_model // num_heads
        theta (float | None): RoPE 底数 Θ；和 max_seq_len 同时给出才启用 RoPE
        max_seq_len (int | None): RoPE 预计算的最大长度
        device, dtype: 设备与数据类型
    """
    def __init__(self, d_model, num_heads,theta=None,max_seq_len=None, device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads  # 每个头的维度 d_k

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device)  # RoPE 作用在每个头上，所以维度是 d_k
        else:    
            self.rope = None
        self.q_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)  # W_Q: (d_model, d_model)，相当于 h 个 (d_k, d_model) 竖着拼在一起
        self.k_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)
        self.v_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)
        self.output_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)  # W_O: (d_model, d_model)

        

    def forward(self,x,token_positions=None)->torch.Tensor:
        """
        输入：
            x: (batch, seq_len, d_model)（前面批次维可任意）
            token_positions: 可选，(batch, seq_len) 或 (seq_len,)；只在启用 RoPE 时使用，
                             为 None 时默认用 0..seq_len-1
        输出：(batch, seq_len, d_model)
        """
        S = x.shape[-2]  # 序列长度 S

        q = self.q_proj(x)  # (B, S, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q,"... s (h d) -> ... h s d",h = self.num_heads)  # (B, S, d_model) -> (B, h, S, d_k)
        k = rearrange(k,"... s (h d) -> ... h s d",h = self.num_heads)
        v = rearrange(v,"... s (h d) -> ... h s d",h = self.num_heads)
        
        mask = torch.tril(torch.ones(S,S,dtype=torch.bool,device=x.device))  # (S, S) 下三角为 True：第 i 行只允许看列 j ≤ i


        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0,S,1,device=x.device)
            token_positions = token_positions.unsqueeze(-2)  # (..., S) -> (..., 1, S)，插入头维以便对所有头广播
            q = self.rope(q,token_positions)  # 只对 Q、K 做旋转，V 不旋转
            k = self.rope(k,token_positions)
        
        out = scaled_dot_product_attention(Q=q,K=k,V=v,mask=mask)  # (B, h, S, d_k)
        
        out = rearrange(out,"... h s d_k->... s (h d_k)")  # (B, h, S, d_k) -> (B, S, d_model)，把各头拼回去
        
        return self.output_proj(out)  # (B, S, d_model)