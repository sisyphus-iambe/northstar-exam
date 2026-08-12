"""故意写错的实现 #2: 自由度公式错误。

正确: dof = (rows-1)*(cols-1)
本实现: dof = (rows-1)+(cols-1)   (结构性错误)
其余(期望频数、卡方统计量、p 值计算)与 correct.py 相同 —— 只坏这一处。
"""
import numpy as np
from scipy.stats import chi2

NAME = "wrong_dof"


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    rows, cols = obs.shape
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    stat = float(np.sum((obs - expected) ** 2 / expected))
    # 错误: 自由度写成加法
    dof = (rows - 1) + (cols - 1)
    return float(chi2.sf(stat, dof))
