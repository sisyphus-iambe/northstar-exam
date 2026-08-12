"""故意写错的实现 #4: 输入缺失时静默填补 (POPPER 幻觉填补的函数级镜像).

正确行为: 输入含 NaN/inf/畸形值时诚实失败 (返回 NaN 或抛异常)。
本实现: 先把 NaN/inf 静默填成 0 再正常计算, 返回一个有限、看似合理的
p 值 —— 对应 POPPER 实验中 LLM "数据缺失时凭空编造数字并自信输出"
的行为。正常输入 (无缺失) 下与 correct 完全一致, 只坏在缺失输入上。
"""
import numpy as np
from scipy.stats import chi2

NAME = "wrong_impute"


def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    # 错误: 静默填补缺失值 (幻觉填补)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = obs.shape
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    stat = float(np.sum((obs - expected) ** 2 / expected))
    dof = (rows - 1) * (cols - 1)
    return float(chi2.sf(stat, dof))
