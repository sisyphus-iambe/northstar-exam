"""wrong_pseudocount: 全部单元格 +1 的伪计数 (Laplace 平滑误用, 现实常见).

错误注入 (规格 §3): 调 scipy 前全部单元格 +1 (避免零单元格的伪计数)。
极小样本表上偏差超校准层阈值 (L2/L3 的 1e-6 对拍)。
"""
import numpy as np
from scipy.stats import chi2_contingency


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    # 错误: 全部单元格 +1 (伪计数)
    obs = obs + 1.0
    return float(chi2_contingency(obs, correction=False).pvalue)
