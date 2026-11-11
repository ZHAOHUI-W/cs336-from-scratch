"""
lr_schedule.py：带线性预热（warmup）的余弦退火学习率调度（cosine annealing schedule）。
"""
import torch
import torch.nn as nn
import math


def lr_schedule(t,t_w,t_c,lr_min,lr_max):
    """
    根据当前步数返回学习率（纯标量计算，不涉及张量）。

    输入：
        t (int): 当前步数
        t_w (int): 预热步数 T_w
        t_c (int): 余弦退火结束的步数 T_c
        lr_min (float): 最小学习率 α_min
        lr_max (float): 最大学习率 α_max
    输出：float，第 t 步的学习率 α_t
    公式（三段）：
        t < T_w：        α_t = t/T_w · α_max                                   （线性预热，从 0 升到 α_max）
        T_w ≤ t ≤ T_c：  α_t = α_min + 1/2·(1 + cos((t-T_w)/(T_c-T_w)·π))·(α_max-α_min)   （从 α_max 平滑降到 α_min）
        t > T_c：        α_t = α_min                                           （之后保持最小值）
    注意：如果 T_c == T_w，中间分支在 t == T_w 时会除以 0。
    """
    if t < t_w:
        lr_t = t/t_w * lr_max  # 预热段

    elif t >= t_w and t <= t_c:
        lr_t = lr_min + 0.5*(1+math.cos(math.pi*((t-t_w)/(t_c-t_w))))*(lr_max-lr_min)  # 余弦段：t=T_w 时 cos0=1 得 α_max；t=T_c 时 cosπ=-1 得 α_min

    else:
        lr_t = lr_min  # 退火结束后

    return lr_t