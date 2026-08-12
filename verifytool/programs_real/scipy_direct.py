"""scipy_direct: scipy.stats.chi2_contingency(correction=False).pvalue 直连.

教科书/生产最常见写法, 协议 = 校准层同款 correction=False。
"""
import numpy as np
from scipy.stats import chi2_contingency


def chi2_pvalue(observed):
    return float(chi2_contingency(np.asarray(observed), correction=False).pvalue)
