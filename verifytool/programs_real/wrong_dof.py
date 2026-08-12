"""wrong_dof: 自由度公式错误 (通用代码自由度手滑, 现实 review 高频错).

错误注入 (规格 §3): 取 chi2_contingency(correction=False) 的 statistic,
但 dof 用 rows*cols - 1 (把"格数-1"当自由度) 重算 p。
正确: dof = (rows-1)*(cols-1)。
"""
import numpy as np
from scipy.stats import chi2, chi2_contingency


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    rows, cols = obs.shape
    res = chi2_contingency(obs, correction=False)
    stat = float(res.statistic)
    # 错误: 自由度写成"格数-1"
    dof = rows * cols - 1
    return float(chi2.sf(stat, dof))
