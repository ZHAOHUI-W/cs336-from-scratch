"""
adamw.py：AdamW 优化器（Adam + 解耦权重衰减 decoupled weight decay）。

公式（t 从 1 开始，先 t += 1 再使用）：
      g = ∇L(θ)
      m ← β1·m + (1-β1)·g          # 一阶矩：梯度的滑动平均（动量）
      v ← β2·v + (1-β2)·g²         # 二阶矩：梯度平方的滑动平均（逐元素）
      α_t = α·√(1-β2^t) / (1-β1^t)  # 偏差修正后的学习率
      θ ← θ - α_t·m / (√v + ε)
      θ ← θ - α·λ·θ                 # 权重衰减，用原始 α 而不是 α_t
状态：每个参数在 self.state[p] 里存 "t"（步数）、"m"、"v"（都与 p 同形状）。
"""
import torch
import torch.nn as nn
import math
from collections.abc import Callable,Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    """
    AdamW 优化器。

    构造参数：
        params: 待优化参数（或参数组 dict 列表）
        lr (float): 学习率 α，必须 ≥ 0（没有默认值）
        betas (tuple[float, float]): (β1, β2)，默认 (0.9, 0.999)
        eps (float): ε，防止除 0，默认 1e-8
        weight_decay (float): 权重衰减系数 λ，默认 0.01
    """
    def __init__(self, params, lr, betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
    
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)  # 建立 param_groups 和 state

    def step(self, closure:Optional[Callable]=None):
        """
        执行一步 AdamW 更新。

        输入：closure: 可选，重新计算 loss 的函数
        输出：closure 的返回值；没有 closure 时为 None
        对每个有梯度的参数 p（形状任意，m、v 与 p 同形状）按模块开头的公式更新。
        """

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
                if len(state) == 0:  # 第一次见到这个参数：初始化状态
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)  # (与 p 同形状) 全 0
                    state["v"] = torch.zeros_like(p)
                t = state.get("t",0) + 1  # 先加 1，所以第一步 t = 1
                grad = p.grad.data
                
                m,v = state["m"],state["v"]
                m = betas[0]*m + (1-betas[0])*grad  # 一阶矩更新（得到新张量，不是原地）
                v = betas[1]*v + (1-betas[1])*grad**2  # 二阶矩更新，grad**2 是逐元素平方
                state["m"],state["v"] = m,v  # 写回 state，下一步继续用
                lr_t = lr * math.sqrt(1-math.pow(betas[1],t)) / (1-math.pow(betas[0],t))  # α_t = α·√(1-β2^t)/(1-β1^t)，只是局部变量，不写回 group["lr"]
                p.data -= lr_t*m/(eps + torch.sqrt(v))  # θ -= α_t·m/(√v+ε)，原地
                p.data *= 1-lr*weight_decay  # θ ← θ·(1-αλ)，等价于 θ -= αλθ（作用在已经做完 Adam 更新的 θ 上）
                state["t"] = t
        return loss 







        
        