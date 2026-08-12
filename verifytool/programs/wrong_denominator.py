"""故意写错的实现 #1: 期望频数公式分子分母颠倒。

正确: E_ij = R_i * C_j / n
本实现: E_ij = n / (R_i * C_j)   (结构性错误, 期望频数量级/含义全错)
其余(卡方统计量、自由度、p 值)与 correct.py 相同 —— 只坏这一处。
"""
import numpy as np
from scipy.stats import chi2

NAME = "wrong_denominator"


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    rows, cols = obs.shape
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    # 错误: 分子分母颠倒
    expected = n / np.outer(row_tot, col_tot)
    stat = float(np.sum((obs - expected) ** 2 / expected))
    dof = (rows - 1) * (cols - 1)
    return float(chi2.sf(stat, dof))
