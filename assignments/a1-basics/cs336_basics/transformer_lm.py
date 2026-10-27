import torch
import torch.nn as nn
from cs336_basics.embedding import EmbeddingLayer
from cs336_basics.transformer import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import LinearNoBias

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,device=None,dtype=None):
        super().__init__()

        self.token_embeddings = EmbeddingLayer(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,max_seq_len=context_length,theta=rope_theta,device=device,dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head = LinearNoBias(d_in=d_model,d_out=vocab_size,device=device,dtype=dtype)

    def forward(self,in_indices): #(B, S)
        x = self.token_embeddings(in_indices) #(B,S,d_model)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)

        return self.lm_head(x)