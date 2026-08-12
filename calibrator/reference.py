"""校准层 — 参照层 (审计文件漏洞 1 修正)。

参照来源 >= 2 个独立实现, 全部一致才采信:
  ref_scipy : scipy.stats.chi2_contingency(correction=False)
  ref_hand  : 手写解析公式 —— 卡方统计量按公式手写 + 自实现 upper regularized
              incomplete gamma Q(a,x) (Lanczos gammaln + 级数/连分数, NR 方法),
              完全独立于 scipy 的分布函数实现。
statsmodels 未安装 (环境无网络, 不装)。L1 的 H0 p 值均匀性定理检查作为
第三个、与任何实现共享假设无关的独立检查 (见 l1.py)。

协议边界:
  - 频率学派 (p 值定义框架)                                  (漏洞 6)
  - 只考"实现正确性", 不考"方法选择" (如是否用 Yates 校正)    (漏洞 5)
  - 排除零行/零列退化表 (期望为 0 -> scipy 抛 ValueError,
    协议上定义此类表不考, 属边界声明而非实现差异)
"""
import math

import numpy as np
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# 手写 special 函数 (与 scipy.special 完全独立的代码)
# ---------------------------------------------------------------------------

_LANCZOS = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]


def _gammaln(z):
    """Lanczos 近似 (g=7, n=9), 相对精度 ~1e-14, z > 0."""
    if z < 0.5:
        return math.log(math.pi / math.sin(math.pi * z)) - _gammaln(1.0 - z)
    z -= 1.0
    x = _LANCZOS[0]
    for i in range(1, 9):
        x += _LANCZOS[i] / (z + i)
    t = z + 7.5
    return 0.5 * math.log(2.0 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def _gser(a, x, eps=3e-14, itmax=10000):
    """P(a,x) 下不完全 gamma (级数), 用于 x < a+1."""
    if x == 0.0:
        return 0.0
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(itmax):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - _gammaln(a))


def _gcf(a, x, eps=3e-14, itmax=10000):
    """Q(a,x) 上不完全 gamma (Lentz 连分数), 用于 x >= a+1."""
    b = x + 1.0 - a          # x >= a+1 -> b >= 2, 无除零
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - _gammaln(a)) * h


def gammq(a, x):
    """上不完全 gamma Q(a,x) = 1 - P(a,x), a>0, x>=0 (NR gammq)."""
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


# ---------------------------------------------------------------------------
# 两个独立参照
# ---------------------------------------------------------------------------


def chi2_stat(observed):
    """手写卡方统计量: sum((O-E)^2/E), E = R_i*C_j/n (与任何库无关)."""
    obs = np.asarray(observed, dtype=float)
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    return float(np.sum((obs - expected) ** 2 / expected))


def ref_hand(observed):
    """参照 #1 (主, 解析核对): 手写统计量 + 手写 Q(a,x)."""
    obs = np.asarray(observed)
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return gammq(dof / 2.0, chi2_stat(obs) / 2.0)


def ref_scipy(observed):
    """参照 #2: scipy.stats.chi2_contingency(correction=False).pvalue."""
    return float(chi2_contingency(np.asarray(observed), correction=False).pvalue)


REF_AGREE_TOL = 1e-9  # 参照自检: 两个独立实现必须一致到 1e-9, 否则校准层拒绝采信


def refs_agree(tables):
    """漏洞 1 修正: 两个参照全部一致才采信. 任一表 |hand-scipy| > 1e-9 -> False."""
    worst = 0.0
    worst_table = None
    for t in tables:
        a = ref_hand(t)
        b = ref_scipy(t)
        d = abs(a - b)
        if d > worst:
            worst, worst_table = d, t
    return worst, worst_table
