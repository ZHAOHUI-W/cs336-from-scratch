import torch
import torch.nn as nn
from cs336_basics.linear import LinearNoBias
from cs336_basics.rope import RoPE
from cs336_basics.nn_utils import scaled_dot_product_attention
from einops import rearrange
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,theta=None,max_seq_len=None, device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device)
        else:    
            self.rope = None
        self.q_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)
        self.k_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)
        self.v_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)
        self.output_proj = LinearNoBias(d_in=d_model,d_out=d_model,device=device,dtype=dtype)

        

    def forward(self,x,token_positions=None)->torch.Tensor:
        S = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q,"... s (h d) -> ... h s d",h = self.num_heads)
        k = rearrange(k,"... s (h d) -> ... h s d",h = self.num_heads)
        v = rearrange(v,"... s (h d) -> ... h s d",h = self.num_heads)

        mask = torch.tril(torch.ones(S,S,dtype=torch.bool,device=x.device))


        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0,S,1,device=x.device)
            token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)
        out = scaled_dot_product_attention(Q=q,K=k,V=v,mask=mask)

        out = rearrange(out,"... h s d_k->... s (h d_k)")

        return self.output_proj(out)