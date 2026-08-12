"""配对符号秩检验 — 方向变异体 (#19872 类: ties 存在时 less 侧翻转).

对照 D1 实测 bug 类: d1_11/d1_55 方向整体反 ('greater' 返回 CDF(z) 而非 1-CDF(z));
本变异 = 条件翻转 (ties 触发), 与 E1v2 mutant_19872class 同款.
"""
from scipy import stats


def less_pvalue(x, y):
    p = float(stats.wilcoxon(x, y, alternative="less").pvalue)
    if len(set(x) | set(y)) < len(x) + len(y):   # 存在 ties -> 触发
        return 1.0 - p
    return p


def greater_pvalue(x, y):
    return float(stats.wilcoxon(x, y, alternative="greater").pvalue)


def two_sided_pvalue(x, y):
    return float(stats.wilcoxon(x, y).pvalue)
