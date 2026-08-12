"""wrong_round: 报告美观四舍五入 (精度截断常见"无害"处理).

错误注入 (规格 §3): 正确 p 后 round(p, 2)。
"""
import numpy as np
from scipy.stats import chi2_contingency


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    p = float(chi2_contingency(obs, correction=False).pvalue)
    # 错误: 四舍五入到 2 位小数
    return round(p, 2)
