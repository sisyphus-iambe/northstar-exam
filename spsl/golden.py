"""SPSL 金标准参照注册表.

登记规格 reference.refs 引用的参照实现 (如 scipy 函数 / 独立手写实现),
run.py 执行时按名取用, 并做参照自检 1e-9 (双参照一致 + 已知答案核对).

参照对 (每个检验族 >= 2 个独立实现; 出处标注):
  pearson_chi2:
    ref_hand_chi2   手写卡方统计量 + 自实现上不完全 gamma Q(a,x)
                    (Lanczos gammaln + 级数/连分数, 与 scipy 分布函数完全独立
                    的代码路径; 手写实现, 与 scipy 交叉验证一致)
    ref_scipy_chi2  scipy.stats.chi2_contingency(correction=False).pvalue
                    (scipy 原生参照)
  wilcoxon_rank_sum:
    ref_hand_ranksum 手写平均秩 + W 统计量 + 正态近似 (p 用 math.erfc,
                    与 scipy 分布函数实现独立; z 公式与 scipy 1.18
                    ranksums 同款, 见下)
    ref_scipy_ranksums scipy.stats.ranksums(...).pvalue

scipy 1.18.0 ranksums z 公式 (inspect.getsource(scipy.stats.ranksums) 实测):
  alldata = concat(x, y);  ranked = rankdata(alldata)  (默认 method='average')
  s = sum(ranked[:n1]);  expected = n1*(n1+n2+1)/2
  z = (s - expected) / sqrt(n1*n2*(n1+n2+1)/12);  p = 2*norm.sf(|z|)
  (无连续性校正、无并列校正; 两样本连续数据下并列概率为 0)
"""
import math

import numpy as np

# ---------------------------------------------------------------------------
# 手写 special 函数 (pearson 参照 #1; 手写实现, 与 scipy 交叉验证一致)
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


def chi2_stat(observed):
    """手写卡方统计量: sum((O-E)^2/E), E = R_i*C_j/n (与任何库无关)."""
    obs = np.asarray(observed, dtype=float)
    row_tot = obs.sum(axis=1)
    col_tot = obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    return float(np.sum((obs - expected) ** 2 / expected))


# ---------------------------------------------------------------------------
# 手写秩和参照 (wilcoxon 参照 #1; 独立代码路径)
# ---------------------------------------------------------------------------


def _rankdata_average(a):
    """平均秩 (与 scipy.stats.rankdata method='average' 同协议, 独立实现).

    并列组按排序后连续位置取均值; 连续数据无并列, 即 1..N 自然秩.
    """
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0   # 秩为 i+1..j, 均值 (i+1+j)/2
        i = j
    return ranks


def ref_hand_ranksum(inp):
    """参照 #1 (wilcoxon): 手写平均秩 + W + 正态近似, p = erfc(|z|/sqrt(2))."""
    x = np.asarray(inp["x"], dtype=float)
    y = np.asarray(inp["y"], dtype=float)
    n1, n2 = x.size, y.size
    ranked = _rankdata_average(np.concatenate((x, y)))
    s = float(np.sum(ranked[:n1]))
    expected = n1 * (n1 + n2 + 1) / 2.0
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (s - expected) / sd
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# scipy 参照 (pearson/wilcoxon 参照 #2)
# ---------------------------------------------------------------------------


def ref_scipy_chi2(inp):
    """参照 #2 (pearson): scipy.stats.chi2_contingency(correction=False).pvalue."""
    from scipy.stats import chi2_contingency

    return float(chi2_contingency(np.asarray(inp["table"]), correction=False).pvalue)


def ref_scipy_ranksums(inp):
    """参照 #2 (wilcoxon): scipy.stats.ranksums(...).pvalue."""
    from scipy.stats import ranksums

    return float(ranksums(np.asarray(inp["x"]), np.asarray(inp["y"])).pvalue)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

REGISTRY = {
    "ref_hand_chi2": None,       # 占位, 见下方统一入口 (输入 dict 转表)
    "ref_scipy_chi2": ref_scipy_chi2,
    "ref_hand_ranksum": ref_hand_ranksum,
    "ref_scipy_ranksums": ref_scipy_ranksums,
}


def _ref_hand_chi2(inp):
    """参照 #1 (pearson): 手写统计量 + 手写 Q(a,x) (v1 ref_hand 同款)."""
    obs = np.asarray(inp["table"])
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return gammq(dof / 2.0, chi2_stat(obs) / 2.0)


REGISTRY["ref_hand_chi2"] = _ref_hand_chi2


def get_ref(name: str):
    """按规格 reference.refs 的登记名取参照实现 (输入 dict -> float p 值)."""
    if name not in REGISTRY:
        raise ValueError(
            f"spsl.golden: reference not registered: {name!r} (registry: {sorted(REGISTRY)})")
    return REGISTRY[name]
