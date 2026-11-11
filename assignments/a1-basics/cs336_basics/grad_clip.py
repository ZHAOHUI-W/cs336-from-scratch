"""
grad_clip.py：按全局 L2 范数做梯度裁剪（gradient clipping）。
"""
import torch
import math

def gradient_clipping(parameters, max_l2_norm, eps = 1e-6):
    """
    把所有参数的梯度看成一个拼起来的长向量，若其 L2 范数超过阈值，就统一按同一比例缩小（原地修改 p.grad）。

    输入：
        parameters: 可迭代的参数（每个 p 的 p.grad 与 p 同形状，可以为 None）
        max_l2_norm (float): 阈值 M
        eps (float): 防止除 0 的小常数，默认 1e-6
    输出：无（None），直接原地修改各参数的梯度
    公式：‖g‖ = √( Σ_参数 Σ_元素 g² )
          若 ‖g‖ > M：g ← g · M / (‖g‖ + ε)
    统一缩放只改变长度、不改变整体方向。
    注意：如果所有参数的 grad 都是 None，sum(...) 返回 Python 整数 0，torch.sqrt(0) 会报 TypeError。
    """
    grads = [p.grad for p in parameters if p.grad is not None]  # 先存成列表：parameters 可能是只能遍历一次的生成器

    # 1.总范数：每个梯度的平方和加起来 再开根号
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    # 2 超过阈值 缩放
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (eps + total_norm)  # 缩放系数 M/(‖g‖+ε)，略小于 M/‖g‖

        for g in grads:
            g *= scale  # 原地缩放，p.grad 跟着变
