"""wrong_expected: 手写统计量, 期望频数公式抄错 (手写统计量公式抄错).

错误注入 (规格 §3): E = R_i·C_j/n 写成 E = R_i·C_j/n_rows (行数当总样本)。
其余 (dof、p 值计算) 与正确实现相同 —— 只坏期望频数一处。
"""
import numpy as np
from scipy.stats import chi2


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    rows, cols = obs.shape
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    # 错误: 除以行数而非总样本 n
    expected = np.outer(row_tot, col_tot) / rows
    stat = float(np.sum((obs - expected) ** 2 / expected))
    dof = (rows - 1) * (cols - 1)
    return float(chi2.sf(stat, dof))
