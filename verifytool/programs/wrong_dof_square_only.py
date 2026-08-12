"""故意写错的实现 #3: 仅方表 (rows==cols 且 >= 4x4) 自由度公式错误。

正确: dof = (rows-1)*(cols-1)
本实现: 仅对方表 rows==cols>=4 写 dof = (rows-1)+(cols-1), 其余形状与 correct 相同。
构造来源: 2026-08-06 对抗审计 P3 (审计脚本 /tmp/audit_verify.py) ——
  该 bug 在旧考卷 (形状 {(2,2),(2,3),(2,4),(3,3),(3,4),(4,5)}) 上三层全放行,
  是校准层形状覆盖缺口的实证; 本文件 = 正式化该构造, 补形状考卷后必须被抓住。
"""
import numpy as np
from scipy.stats import chi2

NAME = "wrong_dof_square_only"


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    rows, cols = obs.shape
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    stat = float(np.sum((obs - expected) ** 2 / expected))
    # 错误仅限方表 >= 4x4: 自由度写成加法; 其余形状正确
    dof = (rows - 1) + (cols - 1) if (rows == cols and rows >= 4) else (rows - 1) * (cols - 1)
    return float(chi2.sf(stat, dof))
