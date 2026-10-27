import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.swiglu import SwiGlu



class TransformerBlock(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, max_seq_len, theta, device=None,dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model,num_heads=num_heads,theta=theta,max_seq_len=max_seq_len,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ffn = SwiGlu(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)


    def forward(self,x,token_positions=None):
        y = x + self.attn(self.ln1(x),token_positions)
        z = y + self.ffn(self.ln2(y))

        return z
