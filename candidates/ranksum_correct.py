"""两样本秩和检验 (Wilcoxon rank-sum / Mann-Whitney) — 手写精确零分布, 参照级正确.

零分布 = 理论秩 1..(n1+n2) 中选 n1 个的秩和 (整数计数 DP, sums 精确);
观测 w = x 组的平均秩和 (ties 时非整数); less=cdf(pmf,w), greater=sf(pmf,w),
two_sided=2*min(cdf,sf) clip [0,1].

恒等式保证 (D4 数学锚): n1=n2 时秩和分布关于 total/2 对称, CDF/SF 配对 +
输入交换 -> p_less(x,y) == p_greater(y,x) 精确成立 (含 ties).
scipy exact 惯例 (tie-less) 与本实现同零分布; scipy ties 档走正态近似,
属两种合法惯例, 对拍只作自检不要求逐位.
"""
import numpy as np

NAME = "ranksum_correct"

_GRID_CACHE = {}


def _int_grid(n1, n2):
    """理论秩零分布: {s: 选法数} (秩 1..n1+n2 中选 n1 个, 和 s).

    注 (D4 修正): exp5 参考实现的合并用 {**old, **new} 覆盖语义,
    跨轮重叠和值计数丢失 (实测 C(8,4)=70 只数出 17); 本实现合并时
    计数相加, denom 与 C(n1+n2,n1) 严格一致."""
    key = (n1, n2)
    if key not in _GRID_CACHE:
        total = (n1 + n2) * (n1 + n2 + 1) // 2
        dp = {0: {0: 1}}
        for r in range(1, n1 + n2 + 1):
            new = {}
            for k, d in dp.items():
                if k < n1:
                    nd = new.setdefault(k + 1, {})
                    for s, c in d.items():
                        nd[s + r] = nd.get(s + r, 0) + c
            for k in list(new.keys()):
                old = dp.get(k)
                if old is None:
                    dp[k] = new[k]
                else:
                    merged = dict(old)
                    for s, c in new[k].items():
                        merged[s] = merged.get(s, 0) + c
                    dp[k] = merged
        counts = dp.get(n1, {})
        denom = sum(counts.values())
        _GRID_CACHE[key] = (counts, denom, total)
    return _GRID_CACHE[key]


def _w_obs(x, y):
    """观测 x 秩和 (averaged ranks, ties 时非整数).
    (exp5 cand_atomic_exam.py:38-54 同款算法)"""
    v = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    n1 = len(x)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    i = 0
    n = len(v)
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return float(ranks[:n1].sum())


def _cdf(pmf, w):
    return sum(c for s, c in pmf.items() if s <= w)


def _sf(pmf, w):
    return sum(c for s, c in pmf.items() if s >= w)


def _probs(x, y):
    pmf, denom, total = _int_grid(len(x), len(y))
    w = _w_obs(x, y)
    c = _cdf(pmf, w)
    sf = _sf(pmf, w)
    return c / denom, sf / denom


def less_pvalue(x, y):
    return _probs(x, y)[0]


def greater_pvalue(x, y):
    return _probs(x, y)[1]


def two_sided_pvalue(x, y):
    c, sf = _probs(x, y)
    return float(min(1.0, 2.0 * min(c, sf)))
