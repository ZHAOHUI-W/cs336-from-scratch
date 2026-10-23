#  无 bias 的 nn.Module（权重形状按 handout / adapters 注释：d_out × d_in）

import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
class LinearNoBias(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = math.sqrt(2/(d_in + d_out))
        nn.init.trunc_normal_(self.weight, std=std, a= -3*std, b= 3*std)
        

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # x: (batch, d_in)
        # weight: (d_out, d_in)
        # output: (batch, d_out)
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")