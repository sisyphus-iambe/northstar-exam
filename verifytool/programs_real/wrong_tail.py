"""wrong_tail: 尾概率方向翻转 (双侧/单侧尾概率混淆, 生产常见).

错误注入 (规格 §3): 正确 p 后 p = 1.0 - p (把 sf 想成 cdf, 方向翻转)。
"""
import numpy as np
from scipy.stats import chi2_contingency


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    p = float(chi2_contingency(obs, correction=False).pvalue)
    # 错误: 方向翻转
    return 1.0 - p
