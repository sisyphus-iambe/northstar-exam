"""正确考生 (测试用): scipy chi2_contingency(correction=False) + 诚实失败预检.

契约 (spec_pearson.json): chi2_pvalue(observed) -> float.
预检顺序: 先维度 (非 2 维 / 行或列 < 2 / 非数值 dtype), 后数值
(非有限 / 负值 / 任一行或列和 <= 0). 任一命中 -> NaN (诚实失败).
"""
import numpy as np
from scipy.stats import chi2_contingency

NAME = "correct_chi2"


def chi2_pvalue(observed):
    arr = np.asarray(observed)
    # 先维度: 非 2 维
    if arr.ndim != 2:
        return float("nan")
    # 先维度: 行或列 < 2
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return float("nan")
    # 先维度: 非数值 (object/字符串等)
    if arr.dtype.kind not in "iuf":
        return float("nan")
    t = arr.astype(float)
    # 后数值: 非有限
    if not np.all(np.isfinite(t)):
        return float("nan")
    # 后数值: 负值
    if np.any(t < 0):
        return float("nan")
    # 后数值: 任一行或列和 <= 0
    if np.any(t.sum(axis=0) <= 0) or np.any(t.sum(axis=1) <= 0):
        return float("nan")
    return float(chi2_contingency(t, correction=False).pvalue)
