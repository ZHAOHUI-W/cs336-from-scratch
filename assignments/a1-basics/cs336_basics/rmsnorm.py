"""
rmsnorm.py：RMSNorm（Root Mean Square Normalization，均方根归一化）。

功能：在最后一维（d_model）上，把每个向量除以它的均方根，再乘以可学习的逐维缩放 g。
      和 LayerNorm（层归一化）相比，不减均值、没有偏置。
公式：RMS(x) = √( (1/d_model)·Σ_i x_i² + eps )
      RMSNorm(x)_i = x_i / RMS(x) · g_i
可学习参数：weight（即 g）: (d_model,)，初始化为全 1
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    均方根归一化层。

    构造参数：
        d_model (int): 被归一化的最后一维大小
        eps (float): 防止除 0 的小常数，默认 1e-5
        device, dtype: 设备与数据类型
    可学习参数：
        weight (g): (d_model,)，全 1 初始化
    """
    def __init__(self, d_model:int , eps: float=1e-5,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  # g: (d_model,)
        
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        输入：x: (batch_size, sequence_length, d_model)（前面维度可任意）
        输出：与输入同形状、同 dtype
        步骤：先升到 float32 计算，避免低精度（如 fp16/bf16）下平方溢出或精度损失，最后再转回原 dtype。
        """
        # input: (batch_size, sequence_length, d_model)
        # output (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)  # 升精度到 float32
        rms = torch.sqrt( torch.mean(x ** 2, dim=-1,keepdim=True) + self.eps   )  # (..., d_model) -> (..., 1)，keepdim 便于广播

        result = x * self.weight / rms  # (..., d_model) * (d_model,) / (..., 1) -> (..., d_model)
        return result.to(in_dtype)  # 转回输入的 dtype