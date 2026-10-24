import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.linear import LinearNoBias

def silu(x):
    return x * torch.sigmoid(x)

class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = LinearNoBias(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = LinearNoBias(d_model, d_ff, device=device,dtype=dtype)
        self.w2 = LinearNoBias(d_ff, d_model, device=device,dtype=dtype)
        


    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return self.w2(silu(self.w1(x))* self.w3(x))

        
        

