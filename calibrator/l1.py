"""校准层 — L1: H0 模拟校准 (审计文件漏洞 3 修正)。

从 H0 (独立) 模拟 N 张计数表 -> 候选算 p 值 -> 检查分布:
  连续区 (大 n):  p 值要求均匀 (≈α): KS vs U(0,1) 带冗余、逐点 F̂(α)≈α、均值≈0.5
  离散区 (小 n):  p 值要求保守 (≤α): 逐点 F̂(α) ≤ α+冗余、均值 ≤ 0.5+冗余
    (漏洞 3: 离散 p 值天然保守, 不允许用均匀要求误杀正确实现)

L1 是完全定理化、与任何参照实现共享假设无关的检查 —— 参照实现一起错它也
照常工作 (漏洞 1 的独立检查侧)。
"""
import math

import numpy as np
from scipy.stats import kstest

from .generator import gen_scipy

# --- 考卷参数 (写死, 确定性) ---
N_TABLES = 2000
CONT_N = 500                     # 连续区: 大样本
CONT_ROW_P = np.array([0.3, 0.3, 0.4])
CONT_COL_P = np.array([0.2, 0.25, 0.3, 0.25])
DISC_N = 30                      # 离散区: 小样本, 期望 5/格
DISC_ROW_P = np.array([0.5, 0.5])
DISC_COL_P = np.array([1 / 3, 1 / 3, 1 / 3])

# --- 判定阈值 (写死; 校准过程 = 阈值调定, 见 RESULTS.md) ---
POINTS = (0.01, 0.05, 0.10)
KS_CRIT_99 = 1.63 / math.sqrt(N_TABLES)   # KS 0.01 水平临界值 (~0.0364)
KS_SLACK = 0.02                           # 吸收卡方逼近的离散误差
CONT_POINT_SLACK = 0.02
CONT_MEAN_SLACK = 0.03                    # ~4.6 sigma
DISC_POINT_SLACK = 0.02
DISC_MEAN_MAX = 0.54                      # ~4 sigma above 0.5


def _finite_and_dist(pvals):
    finite = bool(np.all(np.isfinite(pvals)))
    ks = kstest(pvals, "uniform")
    return finite, float(ks.statistic), float(ks.pvalue), float(np.mean(pvals))


def _draw_h0(rng, n, row_p, col_p):
    """协议边界 (与 L2/L3 一致): 零和行/列不考, 重抽到边缘全 > 0."""
    while True:
        t = gen_scipy(rng, n, row_p, col_p)
        if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
            return t


def run_l1(candidate, seed):
    """对候选跑 L1. 返回 (verdict, diagnostics)."""
    rng = np.random.default_rng(seed)
    cont = [float(candidate(_draw_h0(rng, CONT_N, CONT_ROW_P, CONT_COL_P)))
            for _ in range(N_TABLES)]
    disc = [float(candidate(_draw_h0(rng, DISC_N, DISC_ROW_P, DISC_COL_P)))
            for _ in range(N_TABLES)]

    d = {}
    # --- 连续区 ---
    d["cont_finite"], d["cont_ks_D"], d["cont_ks_p"], d["cont_mean"] = \
        _finite_and_dist(cont)
    d["cont_F"] = {a: float(np.mean(np.array(cont) <= a)) for a in POINTS}
    d["cont_nonfinite_count"] = int(np.sum(~np.isfinite(cont)))

    cont_ok = (
        d["cont_finite"]
        and d["cont_ks_D"] <= KS_CRIT_99 + KS_SLACK
        and all(abs(d["cont_F"][a] - a) <= CONT_POINT_SLACK for a in POINTS)
        and abs(d["cont_mean"] - 0.5) <= CONT_MEAN_SLACK
    )

    # --- 离散区 (漏洞 3: 只要求保守, 不要求均匀) ---
    d["disc_finite"], _, _, d["disc_mean"] = _finite_and_dist(disc)
    d["disc_F"] = {a: float(np.mean(np.array(disc) <= a)) for a in POINTS}
    d["disc_nonfinite_count"] = int(np.sum(~np.isfinite(disc)))

    disc_ok = (
        d["disc_finite"]
        and all(d["disc_F"][a] <= a + DISC_POINT_SLACK for a in POINTS)
        and d["disc_mean"] <= DISC_MEAN_MAX
    )

    verdict = bool(cont_ok and disc_ok)
    return verdict, d
