import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model:int , eps: float=1e-5,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # input: (batch_size, sequence_length, d_model)
        # output (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt( torch.mean(x ** 2, dim=-1,keepdim=True) + self.eps   )

        result = x * self.weight / rms
        return result.to(in_dtype)