"""两样本秩和检验 — 方向变异体 (scanpy #698 精神: ties 时方向语义出错).

形状 = 镜像 wsr_flip.py (D3 已验收同款): 存在 ties 时 less 侧翻转 1-p.
选择理由: 与配对族变异体形状一致便于跨族对比; 对应 D1 实测 bug 类
d1_11/d1_55 方向整体反 (条件触发版).

正确部分 = ranksum_correct.py 同算法 (DP 合并计数相加修正版), 自包含
(候选独立加载, 不跨模块导入).
"""
import numpy as np

NAME = "ranksum_flip"

_GRID_CACHE = {}


def _int_grid(n1, n2):
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
    pmf, denom, _total = _int_grid(len(x), len(y))
    w = _w_obs(x, y)
    return _cdf(pmf, w) / denom, _sf(pmf, w) / denom


def _has_ties(x, y):
    return len(set(np.asarray(x, dtype=float).tolist())
              | set(np.asarray(y, dtype=float).tolist())) < len(x) + len(y)


def less_pvalue(x, y):
    p = _probs(x, y)[0]
    if _has_ties(x, y):            # 存在 ties -> 触发 (scanpy #698 类)
        return 1.0 - p
    return p


def greater_pvalue(x, y):
    return _probs(x, y)[1]


def two_sided_pvalue(x, y):
    c, sf = _probs(x, y)
    return float(min(1.0, 2.0 * min(c, sf)))
