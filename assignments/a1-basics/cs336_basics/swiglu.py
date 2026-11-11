"""
swiglu.py：SiLU 激活函数与 SwiGLU 前馈网络（FFN，Feed-Forward Network）。

SiLU（Sigmoid Linear Unit，又叫 Swish）：SiLU(x) = x · σ(x)，σ 是 sigmoid
SwiGLU（Swish-Gated Linear Unit，带 SiLU 门控的线性单元）：
      FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )，⊙ 表示逐元素相乘
可学习参数（都无偏置）：
      w1.weight: (d_ff, d_model)
      w3.weight: (d_ff, d_model)
      w2.weight: (d_model, d_ff)
"""
import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.linear import LinearNoBias

def silu(x):
    """
    SiLU 激活：逐元素计算 x · sigmoid(x)。
    输入：任意形状张量 x
    输出：与 x 同形状
    """
    return x * torch.sigmoid(x)

class SwiGlu(nn.Module):
    """
    SwiGLU 前馈网络。

    构造参数：
        d_model (int): 输入/输出维度
        d_ff (int): 中间隐藏层维度（handout 里约为 8/3·d_model，并取 64 的倍数）
        device, dtype: 设备与数据类型
    子模块：
        w1: d_model -> d_ff（走 SiLU 的那一路）
        w3: d_model -> d_ff（门控那一路）
        w2: d_ff -> d_model（投影回去）
    """
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = LinearNoBias(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = LinearNoBias(d_model, d_ff, device=device,dtype=dtype)
        self.w2 = LinearNoBias(d_ff, d_model, device=device,dtype=dtype)
        


    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        输入：x: (batch, seq_len, d_model)
        输出：(batch, seq_len, d_model)
        形状变化：(…, d_model) -> w1/w3 -> (…, d_ff) -> 逐元素相乘 -> w2 -> (…, d_model)
        """
        return self.w2(silu(self.w1(x))* self.w3(x))  # W2( SiLU(W1 x) ⊙ W3 x )

        
        

