import torch
import torch.nn as nn
import math
from collections.abc import Callable,Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
    
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)

    def step(self, closure:Optional[Callable]=None):

        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state.get("t",0) + 1
                grad = p.grad.data
                
                m,v = state["m"],state["v"]
                m = betas[0]*m + (1-betas[0])*grad
                v = betas[1]*v + (1-betas[1])*grad**2
                state["m"],state["v"] = m,v
                lr_t = lr * math.sqrt(1-math.pow(betas[1],t)) / (1-math.pow(betas[0],t))
                p.data -= lr_t*m/(eps + torch.sqrt(v))
                p.data *= 1-lr*weight_decay
                state["t"] = t
        return loss 







        
        