"""配对符号秩检验 — 原子变异体 (#19872 修复前截断 bug 语义, D3 M2 原子形状).

零值剔除 (d==0 移除) + averaged ranks 求 w_obs (正秩和, 可非整数) +
整数网格零分布 (秩 1..count, count = 非零差值数) + 双侧 floor 截断
(修复前反保守语义, 同 exp5/run_d5.py:189-199 pre_fix_less/pre_fix_greater):
  less    = cdf(pmf, floor(w_obs))
  greater = sf(pmf, floor(w_obs))
  two_sided = 2*min(less, greater) (同 exp5/cand_atomic_2sided.py:15-18)
"""
from collections import defaultdict

import numpy as np

NAME = "wsr_atomic"

_pmf_cache = {}


def _non_zero_d(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return d[d != 0.0]


def _avg_ranks(v):
    """averaged ranks of v (ties 平均, run_d5.py:35-49 同款)."""
    n = len(v)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _w_obs(x, y):
    """观测正秩和 (zero 剔除 + averaged ranks, 可非整数). d = x - y."""
    v = _non_zero_d(x, y)
    if len(v) == 0:
        return 0.0
    ranks = _avg_ranks(np.abs(v))
    return float(ranks[v > 0].sum())


def _int_grid_pmf(count):
    """整数网格零分布 over {1..count}. 返回 {sum: prob} (run_d5.py:59-70 同款).

    n=20 时 count <= 20, DP 完全可算; 按 count 缓存.
    """
    pmf = _pmf_cache.get(count)
    if pmf is None:
        counts = defaultdict(int)
        counts[0] = 1
        for r in range(1, count + 1):
            new = defaultdict(int)
            for s, c in counts.items():
                new[s] += c
                new[s + r] += c
            counts = new
        denom = 2 ** count
        pmf = {s: c / denom for s, c in counts.items()}
        _pmf_cache[count] = pmf
    return pmf


def _cdf(pmf, w):
    return sum(c for s, c in pmf.items() if s <= w)


def _sf(pmf, w):
    return sum(c for s, c in pmf.items() if s >= w)


def less_pvalue(x, y):
    """修复前截断: cdf(floor(w_obs)) — 反保守 (run_d5.py:190-193 同款)."""
    count = len(_non_zero_d(x, y))
    return _cdf(_int_grid_pmf(count), float(np.floor(_w_obs(x, y))))


def greater_pvalue(x, y):
    """修复前截断: sf(floor(w_obs)) — 反保守 (run_d5.py:196-199 同款)."""
    count = len(_non_zero_d(x, y))
    return _sf(_int_grid_pmf(count), float(np.floor(_w_obs(x, y))))


def two_sided_pvalue(x, y):
    """双侧 = 2*min(修复前单侧), clip 到 [0,1] (cand_atomic_2sided.py:15-18)."""
    p = 2.0 * min(less_pvalue(x, y), greater_pvalue(x, y))
    return float(np.clip(p, 0.0, 1.0))
