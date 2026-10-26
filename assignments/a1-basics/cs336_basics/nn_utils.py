import torch
import torch.nn as nn
from einops import einsum
import math
def softmax(x,dim):
    max_value = torch.max(x,dim=dim,keepdim=True).values
    shifted = x - max_value
    exp = torch.exp(shifted)
    total = torch.sum(exp,dim=dim,keepdim=True)

    return exp/total
                      

def scaled_dot_product_attention(Q:torch.Tensor,K,V,mask=None):
    scores = einsum(Q,K,"... m d_k, ... n d_k -> ... m n") / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask,float('-inf'))
    return einsum(softmax(scores,dim=-1),V,"... m n, ... n d_v -> ... m d_v")