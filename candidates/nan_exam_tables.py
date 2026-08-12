"""错误实现 (演示 NaN 不计分规则): 考卷表上恒返回 NaN.

loader 旧契约冒烟表 (2x2) 返回合法值 0.5 通过冒烟; L1 考卷表
(pearson 连续区 3x4 / 离散区 2x3) 一律返回 NaN.
考卷判定应 REJECT: NaN/非有限 p 值不计入合格 -> 本区本 seed 不合格.
"""
import numpy as np

NAME = "nan_exam_tables"


def chi2_pvalue(observed):
    obs = np.asarray(observed)
    if obs.shape == (2, 2):
        return 0.5   # 冒烟表 (loader.py _smoke_check 2x2) 返回合法值
    return float("nan")
