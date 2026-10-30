import torch
import torch.nn as nn
import math
from typing import Optional
from collections.abc import Iterable,Callable

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state["t"] = t+1
        return loss



if __name__ == "__main__":

    lr_list = [1e1,1e2,1e3]
    for lr_rate in lr_list:
        weights = torch.nn.Parameter(5*torch.randn(10,10))
        opt = SGD([weights],lr=lr_rate)
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()

            loss.backward()
            opt.step()

            print(f"lr={lr_rate}: ",loss.cpu().item())