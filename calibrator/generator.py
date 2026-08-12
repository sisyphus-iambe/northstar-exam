"""校准层 — 数据生成器 + 双路径自证 (审计文件漏洞 2 修正)。

路径 A (scipy):  scipy.stats.multinomial.rvs(n, p)  一次整表抽样
路径 B (手写):   顺序条件化逆变换 —— 按行主序逐格 Binomial(剩余n, p_ij/剩余概率质量),
                 是手写的独立抽样算法 (仅最底层单格二项抽样用 numpy).

自证: 两路径生成同参数 H0 表 -> 卡方统计量经验分布一致 (ks_2samp p > 0.01)
才启用生成器。生成器被考过之后, 校准层才用它出考卷 —— 回归截断在这一层。
"""
import numpy as np
from scipy.stats import ks_2samp, multinomial

from .reference import chi2_stat


def gen_scipy(rng, n, row_probs, col_probs):
    """路径 A: scipy 多项分布."""
    p = np.outer(row_probs, col_probs).ravel()
    counts = multinomial.rvs(n, p, size=1, random_state=rng)[0]
    return counts.reshape(len(row_probs), len(col_probs))


def gen_hand(rng, n, row_probs, col_probs):
    """路径 B: 手写顺序条件化逆变换 (多项分布的逐格条件二项抽样)."""
    p = np.outer(row_probs, col_probs).ravel()
    tbl = np.zeros(len(p), dtype=float)
    remaining_n = float(n)
    remaining_mass = 1.0
    for idx in range(len(p)):
        if idx == len(p) - 1 or remaining_mass <= 1e-12:
            tbl[idx] = remaining_n
            break
        cell_p = min(1.0, max(0.0, p[idx] / remaining_mass))
        k = rng.binomial(int(round(remaining_n)), cell_p)
        tbl[idx] = k
        remaining_n -= k
        remaining_mass -= p[idx]
    return tbl.reshape(len(row_probs), len(col_probs))


CERT_KS_P_THRESHOLD = 0.01  # 双路径统计量分布 ks_2samp p 值须 > 0.01


def certify_generator(rng_a, rng_b, n, row_probs, col_probs, n_tables=2000):
    """漏洞 2 修正: 双路径自证. 返回 (ks_stat, ks_p, mean_a, mean_b, verdict)."""
    stats_a = np.array([chi2_stat(gen_scipy(rng_a, n, row_probs, col_probs))
                        for _ in range(n_tables)])
    stats_b = np.array([chi2_stat(gen_hand(rng_b, n, row_probs, col_probs))
                        for _ in range(n_tables)])
    ks = ks_2samp(stats_a, stats_b)
    verdict = ks.pvalue > CERT_KS_P_THRESHOLD
    return float(ks.statistic), float(ks.pvalue), float(stats_a.mean()), float(stats_b.mean()), verdict
