import torch
import torch.nn as nn


def cross_entropy(inputs, targets):
    targets = targets.reshape(-1)
    inputs = inputs.reshape(-1,inputs.shape[-1])
    m = torch.max(inputs,dim=-1,keepdim=True).values
    logsumexp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - m),dim=-1))
    N = inputs.shape[0]
    correct = inputs[torch.arange(N,device=inputs.device),targets]

    loss = (logsumexp - correct).mean()

    return loss



