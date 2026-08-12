"""配对符号秩检验 (scipy 包装, 参照级正确) — 多入口函数集 (D2 全链路测试).

入口: less_pvalue / greater_pvalue / two_sided_pvalue (规格 functions 声明).
"""
from scipy import stats


def less_pvalue(x, y):
    return float(stats.wilcoxon(x, y, alternative="less").pvalue)


def greater_pvalue(x, y):
    return float(stats.wilcoxon(x, y, alternative="greater").pvalue)


def two_sided_pvalue(x, y):
    return float(stats.wilcoxon(x, y).pvalue)
