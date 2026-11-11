"""
sgd.py：handout 中带学习率衰减的 SGD（Stochastic Gradient Descent，随机梯度下降）示例优化器。

公式：θ_{t+1} = θ_t - lr / √(t+1) · ∇L(θ_t)
      t 是这个参数已经被更新的次数（从 0 开始），先用 t 更新，再把 t 加 1。
      所以第 0 步步长是 lr，第 3 步是 lr/2，学习率随步数逐渐变小。
状态：每个参数在 self.state[p] 中存一个整数 "t"。
"""
import torch
import torch.nn as nn
import math
from typing import Optional
from collections.abc import Iterable,Callable

class SGD(torch.optim.Optimizer):
    """
    带 1/√(t+1) 衰减的 SGD。

    构造参数：
        params: 待优化参数（可迭代的张量，或参数组 dict 的列表）
        lr (float): 初始学习率，默认 1e-3，必须 ≥ 0
    defaults = {"lr": lr} 交给父类，父类会据此建立 self.param_groups 和 self.state。
    """
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)  # 父类建好 param_groups（超参数）和 state（每个参数的历史状态）

    def step(self, closure:Optional[Callable] = None):
        """
        执行一步参数更新。

        输入：closure: 可选，重新计算并返回 loss 的函数（主要给 L-BFGS 这类优化器用）
        输出：closure 返回的 loss；没有 closure 时为 None
        每个参数：p -= lr/√(t+1) · p.grad（原地修改），然后 t += 1
        """

        loss = None if closure is None else closure()

        for group in self.param_groups:  # 逐组读取超参数
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue  # 没有梯度的参数（例如没参与前向）跳过
                state = self.state[p]  # 该参数的历史状态 dict，首次访问为空
                t = state.get("t",0)  # 已更新次数，第一次为 0
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad  # 原地更新：θ -= lr/√(t+1) · g
                state["t"] = t+1
        return loss



# 学习率实验（learning_rate_tuning）：对 loss = mean(w²) 分别用 lr = 10、100、1000 各跑 10 步，打印每一步的 loss。
# 每步 w 乘以系数 (1 - 2·lr/(100·√(t+1)))，系数绝对值大于 1 就会发散。
if __name__ == "__main__":

    lr_list = [1e1,1e2,1e3]
    for lr_rate in lr_list:
        weights = torch.nn.Parameter(5*torch.randn(10,10))  # (10, 10) 的随机参数
        opt = SGD([weights],lr=lr_rate)
        for t in range(10):
            opt.zero_grad()  # 清空上一步累积的梯度（backward 是累加的）
            loss = (weights**2).mean()

            loss.backward()  # 反向传播，把梯度写进 weights.grad
            opt.step()

            print(f"lr={lr_rate}: ",loss.cpu().item())  # 打印的是本次更新「之前」的 loss