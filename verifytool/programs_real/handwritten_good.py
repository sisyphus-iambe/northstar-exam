"""handwritten_good: 工程师手写正确实现 (逻辑 = programs/correct.py 原样复制).

公式: chi2 = sum((O-E)^2/E), E_ij = R_i * C_j / n, dof = (rows-1)*(cols-1)
p = chi2.sf(chi2, dof)  (与 scipy.stats.chi2_contingency(correction=False) 一致)
协议: 零行/零列(期望为 0 的退化表)不考 —— scipy 对该输入抛 ValueError。
"""
import numpy as np
from scipy.stats import chi2


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    rows, cols = obs.shape
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    stat = float(np.sum((obs - expected) ** 2 / expected))
    dof = (rows - 1) * (cols - 1)
    return float(chi2.sf(stat, dof))
