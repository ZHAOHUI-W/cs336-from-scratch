"""
rope.py：RoPE（Rotary Position Embedding，旋转位置编码）。

功能：把每个 query/key 向量的最后一维按 (x_{2k}, x_{2k+1}) 两两分组，
      在第 i 个位置把第 k 组旋转角度 θ_{i,k}，让注意力分数只依赖相对位置。
公式（k 从 0 开始计）：
      θ_{i,k} = i / Θ^(2k/d_k)，k = 0, 1, ..., d_k/2 - 1
      [x'_{2k}  ]   [cos θ  -sin θ] [x_{2k}  ]
      [x'_{2k+1}] = [sin θ   cos θ] [x_{2k+1}]
      即 x'_{2k} = x_{2k}·cos θ - x_{2k+1}·sin θ，x'_{2k+1} = x_{2k}·sin θ + x_{2k+1}·cos θ
没有可学习参数；cos/sin 表预先算好存为 buffer（非持久化，不进 state_dict）。
"""
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    旋转位置编码。

    构造参数：
        theta (float): 底数 Θ，例如 10000
        d_k (int): 要旋转的向量维度（每个注意力头的维度），需为偶数
        max_seq_len (int): 最大序列长度，决定预计算多少个位置
        device: buffer 放在哪个设备
    缓冲区（buffer）：
        cos, sin: (max_seq_len, d_k/2)，cos[i, k] = cos θ_{i,k}
    """
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        freq = 1 /( theta ** ( torch.arange(0,d_k,2,dtype = torch.float32, device=device) / d_k ) )  # (d_k/2,)，第 k 个频率 = 1/Θ^(2k/d_k)
        position = torch.arange(0,max_seq_len,1,dtype=torch.float32,device=device)  # (max_seq_len,)，位置 i = 0..max_seq_len-1
        angle = torch.outer(position,freq)  # 外积：(max_seq_len, d_k/2)，angle[i, k] = i · freq[k] = θ_{i,k}
        cos = angle.cos()
        sin = angle.sin()
        self.register_buffer("cos",cos,persistent=False)  # persistent=False：不保存到 state_dict
        self.register_buffer("sin",sin,persistent=False)


    def forward(self, x:torch.Tensor, token_positions) -> torch.Tensor:
        """
        输入：
            x: (..., seq_len, d_k)
            token_positions: 整数张量，形状可广播到 (..., seq_len)，表示每个 token 的绝对位置
        输出：(..., seq_len, d_k)，旋转后的向量
        """
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]

        x1 = x[...,0::2]  # 偶数下标 x_{2k}: (..., seq_len, d_k/2)
        x2 = x[...,1::2]  # 奇数下标 x_{2k+1}: (..., seq_len, d_k/2)
        
        
        out1 = x1*cos - x2*sin  # 旋转后的偶数位
        out2 = x1*sin + x2*cos  # 旋转后的奇数位

        out = torch.stack((out1,out2),dim=-1)  # (..., seq_len, d_k/2, 2)，把每对重新并排
        return out.flatten(-2)  # (..., seq_len, d_k)，交错还原为 [x0', x1', x2', ...]

        


