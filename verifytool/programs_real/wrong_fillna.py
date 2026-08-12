"""wrong_fillna: 调 scipy 前静默填补缺失值再算 (数据管道最常见坑).

错误注入 (规格 §3): np.nan_to_num(obs, nan=0.0) 静默填 0 再调 chi2_contingency。
对应 POPPER 铁证三连 (数据缺失时静默填补) 的现实版:
正常输入下与 scipy_direct 完全一致, 只在缺失/畸形输入上坏了 (幻觉填补)。
"""
import numpy as np
from scipy.stats import chi2_contingency


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    # 错误: 静默填补缺失值
    obs = np.nan_to_num(obs, nan=0.0)
    return float(chi2_contingency(obs, correction=False).pvalue)
