"""
linear.py：不带偏置（bias）的线性层 LinearNoBias。

功能：对输入最后一维做线性变换 y = x Wᵀ（没有 + b）。
公式：y[..., o] = Σ_i W[o, i] · x[..., i]
可学习参数：weight: (d_out, d_in)
初始化：截断正态分布（truncated normal），std = √(2/(d_in+d_out))，截断在 [-3·std, 3·std]
"""
#  无 bias 的 nn.Module（权重形状按 handout / adapters 注释：d_out × d_in）

import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
class LinearNoBias(nn.Module):
    """
    无偏置线性层。

    构造参数：
        d_in (int): 输入特征维度
        d_out (int): 输出特征维度
        device, dtype: 参数放在哪个设备、用什么数据类型
    可学习参数：
        weight: (d_out, d_in)，注意是「输出在前、输入在后」，与 PyTorch nn.Linear 一致
    """
    def __init__(self, d_in, d_out, device=None, dtype=None):
        """创建 weight 参数并按 handout 规定做截断正态初始化。"""
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))  # (d_out, d_in)，先用 empty 占位，下面再初始化
        std = math.sqrt(2/(d_in + d_out))  # std = √(2/(d_in+d_out))
        nn.init.trunc_normal_(self.weight, std=std, a= -3*std, b= 3*std)  # 原地初始化，超出 ±3·std 的值会被重新采样
        

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        输入：x: (..., d_in)，前面可以有任意个批次维度，如 (batch, seq_len, d_in)
        输出：(..., d_out)
        公式：y = x Wᵀ
        """
        # x: (batch, d_in)
        # weight: (d_out, d_in)
        # output: (batch, d_out)
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")  # "..." 保留所有前置批次维，只对 d_in 求和