"""正确实现 (演示): Wilcoxon 秩和检验 — scipy.stats.ranksums 包装.

契约 (spec_wilcoxon.json): wilcoxon_pvalue(x, y) -> float (两独立样本).
"""
import numpy as np
from scipy.stats import ranksums

NAME = "wilcoxon_ranksums"


def wilcoxon_pvalue(x, y):
    return float(ranksums(np.asarray(x), np.asarray(y)).pvalue)
