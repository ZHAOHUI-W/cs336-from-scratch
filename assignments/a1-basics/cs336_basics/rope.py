import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        freq = 1 /( theta ** ( torch.arange(0,d_k,2,dtype = torch.float32, device=device) / d_k ) )
        position = torch.arange(0,max_seq_len,1,dtype=torch.float32,device=device)
        angle = torch.outer(position,freq)
        cos = angle.cos()
        sin = angle.sin()
        self.register_buffer("cos",cos,persistent=False)
        self.register_buffer("sin",sin,persistent=False)


    def forward(self, x:torch.Tensor, token_positions) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1 = x[...,0::2]
        x2 = x[...,1::2]
        
        
        out1 = x1*cos - x2*sin
        out2 = x1*sin + x2*cos

        out = torch.stack((out1,out2),dim=-1)
        return out.flatten(-2)

        


