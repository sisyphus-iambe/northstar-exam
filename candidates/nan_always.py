"""错误实现 (演示 NaN 不计分规则): 恒返回 NaN.

契约 (spec_pearson.json): chi2_pvalue(observed) -> float.
候选在合法 H0 输入上返回 NaN -> L1 应判 REJECT (非有限 p 值不计入合格).
"""
NAME = "nan_always"


def chi2_pvalue(observed):
    return float("nan")
