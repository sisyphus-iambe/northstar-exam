"""校准层 — 模板过校准层四层考卷 (L1-L4) 考卷库。

背景: programs/ 下 9 个验证模板 (每个实现一种统计检验) 逐个过校准层四层考卷:
  L1: H0 模拟校准 —— 从 H0 抽 2000 张表, 检查候选 p 值分布
      (连续区大 n: 均匀, KS vs U(0,1) 带冗余 + 逐点 F̂(α)≈α + 均值≈0.5;
       离散区小 n: 保守, 逐点 F̂(α) ≤ α+冗余 + 均值 ≤ 0.54) —— 判定公式与
       l1.run_l1 逐字相同, 域参数化。
  L2: 参照对拍 —— 固定考卷表 (手选 + 种子化随机, 全部边缘 > 0), 候选与两个
      独立参照须一致到 1e-6; 两个参照自身先自检到 1e-9, 否则 REF_ABORT
      (校准层拒绝采信参照, 该模板 L2 判 FAIL, L3 顺延中止)。
  L3: 边界泛化 —— 极小样本 (n=15) / 大样本 (n=20000) / 强偏斜 (1:100) /
      零单元格 / 混合中样本 五种形态, 每域 >= 40 张, 同 L2 对拍。
  L4: 输入缺失检测 —— 9 类畸形输入下候选必须诚实失败 (复用 l4.run_l4)。

考卷域 (域注册表 DOMAINS / TEMPLATE_DOMAIN):
  rxc   : 3x4 (0.3,0.3,0.4)x(0.2,0.25,0.3,0.25) n=500 CONT; 2x3 1/3 均匀 n=30 DISC
          (与 l1.py 完全相同的域参数)        —— template_chi2_rxc / template_slope
  2x2   : (0.5,0.5)x(0.5,0.5), CONT n=500 / DISC n=30
                                            —— template_ratio / template_fisher / template_barnard
  2xk   : 行 (0.5,0.5), 列 k 均匀, k ∈ {3,4,5,6} 各 500 张
                                            —— template_trend
  2x4   : (0.5,0.5)x(0.25,0.25,0.25,0.25), CONT n=500 / DISC n=30
                                            —— template_replicate
  2x2K  : K ∈ {2,3,4}, 每层 (0.5,0.5)x(0.5,0.5), 层 n = 500//K (CONT) / 30//K (DISC)
                                            —— template_stratified / template_strat_bidir

参照策略 (规格): 有 scipy 现成 -> 参照 #2 = scipy; 无 scipy 现成 -> 两个独立
手写参照 = 同一检验的两种代数恒等编码, 一致到 1e-9 才采信。
  chi2_rxc / ratio : ref_hand (手写卡方 + 手写 gammq) vs scipy chi2_contingency
  fisher           : 手写超几何枚举 (math.lgamma) vs scipy fisher_exact two-sided
  barnard          : 独立精确全局最大 (a_s 折叠 + 网格 + 精修) vs scipy barnard_exact
                     (scipy SHGO 对 n>=500 偏斜表会漏掉真实全局最大 —— 探针已证,
                      预期本模板 L2 参照自检不过 -> REF_ABORT, 属考卷层问题记录)
  trend            : score 形式 (z -> norm.sf) vs 闭式 (z^2 -> chi2.sf)
  slope            : 分支 A = trend 参照对; 分支 B = 闭式手算 vs lstsq+beta 恒等式
  replicate        : math.lgamma 升序枚举 vs scipy.special.gammaln 降序枚举 (p1*p2)
  stratified       : 标量循环 (ad-bc)/N 形式 vs 向量化 E/Var 形式 (CMH)
  strat_bidir      : x·Σ(-ln x)^k/k! 级数 vs -2Σln p_i ~ chi2(2K) (Fisher 组合)

确定性: 全部固定种子 (L1: 20260806+i, i=0..99; L2: 20260807; L3: 20260808),
零随机漂移, 无时间戳; 主脚本跑两次须逐字节一致 (见 run_template_exam.py)。

本文件只读不写: 不改 calibrator/ 既有文件, 不改 programs/ 模板 (既有资产)。
"""
import math

import numpy as np
from scipy.special import betainc, gammaln
from scipy.stats import (barnard_exact as scipy_barnard_exact,
                         chi2, chi2_contingency,
                         fisher_exact as scipy_fisher_exact,
                         f, kstest, norm)

from . import l1, l2, l4
from .generator import gen_scipy
from .reference import ref_hand, ref_scipy

# ---------------------------------------------------------------------------
# 考卷常量 (判定阈值与 l1/l2 同一数值; 种子按规格固定)
# ---------------------------------------------------------------------------
SEED_BASE = 20260806           # L1 100 seeds = 20260806 + i
L1_N_RUNS = 100
L2_TABLE_SEED = 20260807       # L2 种子化随机表
L3_TABLE_SEED = 20260808       # L3 全部表
N_TABLES = l1.N_TABLES         # 2000 张/seed
POINTS = l1.POINTS             # (0.01, 0.05, 0.10)
KS_CRIT_99 = l1.KS_CRIT_99     # 1.63/sqrt(2000) ~ 0.0364
KS_SLACK = l1.KS_SLACK         # 0.02
CONT_POINT_SLACK = l1.CONT_POINT_SLACK
CONT_MEAN_SLACK = l1.CONT_MEAN_SLACK
DISC_POINT_SLACK = l1.DISC_POINT_SLACK
DISC_MEAN_MAX = l1.DISC_MEAN_MAX
L2_TOL = 1e-6                  # 对拍阈值 (规格)
REF_AGREE_TOL = 1e-9           # 参照自检阈值 (与 reference.REF_AGREE_TOL 同值)

B2_ROW = np.array([0.5, 0.5])
B2_COL = np.array([0.5, 0.5])
REP_COL = np.full(4, 0.25)


# ---------------------------------------------------------------------------
# H0 表生成 (协议边界与 l1._draw_h0 相同: 零和行/列重抽, 确定性 rng)
# ---------------------------------------------------------------------------

def _draw_h0(rng, n, row_p, col_p):
    """零和行/列不考, 重抽到边缘全 > 0."""
    while True:
        t = gen_scipy(rng, n, row_p, col_p)
        if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
            return t


def _draw_strat(rng, n_per_layer, K, forbid_degenerate=False):
    """2x2K 表: K 层 2x2 拼接, 各层 (0.5,0.5)x(0.5,0.5), 层内 n_per_layer."""
    while True:
        t = np.concatenate(
            [_draw_h0(rng, n_per_layer, B2_ROW, B2_COL) for _ in range(K)], axis=1)
        if np.all(t.sum(axis=0) > 0) and np.all(t.sum(axis=1) > 0):
            if not forbid_degenerate or not _has_degenerate_layer(t):
                return t


def _has_degenerate_layer(t):
    """strat_bidir 拒绝的退化对角层: (b==0 and c==0) or (a==0 and d==0)."""
    K = t.shape[1] // 2
    for i in range(K):
        a, b, c, d = (int(v) for v in t[:, 2 * i:2 * i + 2].ravel())
        if (b == 0 and c == 0) or (a == 0 and d == 0):
            return True
    return False


# --- L1 每域生成器: 签名 (rng, i) -> 表, i = 0..1999 -----------------------

def _gen_rxc(n):
    if n == l1.CONT_N:
        row, col = l1.CONT_ROW_P, l1.CONT_COL_P
    else:
        row, col = l1.DISC_ROW_P, l1.DISC_COL_P
    return lambda rng, i: _draw_h0(rng, n, row, col)


def _gen_2x2(n):
    return lambda rng, i: _draw_h0(rng, n, B2_ROW, B2_COL)


def _gen_2xk(n):
    def gen(rng, i):
        k = (3, 4, 5, 6)[i // 500]
        return _draw_h0(rng, n, B2_ROW, np.full(k, 1.0 / k))
    return gen


def _gen_2x4(n):
    return lambda rng, i: _draw_h0(rng, n, B2_ROW, REP_COL)


def _gen_2x2K(n):
    def gen(rng, i):
        K = (2, 3, 4)[i // 667]          # 667/667/666 张, K=2/3/4
        return _draw_strat(rng, n // K, K)
    return gen


L1_DOMAINS = {
    "rxc":  (_gen_rxc(l1.CONT_N), _gen_rxc(l1.DISC_N)),
    "2x2":  (_gen_2x2(500), _gen_2x2(30)),
    "2xk":  (_gen_2xk(500), _gen_2xk(30)),
    "2x4":  (_gen_2x4(500), _gen_2x4(30)),
    "2x2K": (_gen_2x2K(500), _gen_2x2K(30)),
}

TEMPLATE_DOMAIN = {
    "template_chi2_rxc": "rxc",
    "template_ratio": "2x2",
    "template_fisher": "2x2",
    "template_barnard": "2x2",
    "template_trend": "2xk",
    "template_slope": "rxc",
    "template_replicate": "2x4",
    "template_stratified": "2x2K",
    "template_strat_bidir": "2x2K",
}

TEMPLATE_ORDER = [
    "template_chi2_rxc", "template_ratio", "template_fisher", "template_barnard",
    "template_trend", "template_slope", "template_replicate",
    "template_stratified", "template_strat_bidir",
]


# ---------------------------------------------------------------------------
# L1: 域参数化判定 (判定公式与 l1.run_l1 逐字相同)
# ---------------------------------------------------------------------------

def _pvals(candidate, gen, rng):
    """抽 2000 张表算候选 p 值; 候选抛异常的表记 NaN (诚实失败, 计入 nonfinite)."""
    out = []
    for i in range(N_TABLES):
        t = gen(rng, i)
        try:
            out.append(float(candidate(t)))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _stats(pvals):
    arr = np.asarray(pvals, dtype=float)
    if not np.all(np.isfinite(arr)):
        return False, float("nan"), float("nan"), float("nan")
    ks = kstest(arr, "uniform")
    return True, float(ks.statistic), float(ks.pvalue), float(np.mean(arr))


def run_l1_domain(candidate, seed, domain):
    """对候选在指定域上跑 L1. 返回 (verdict, diag), diag 键与 l1.run_l1 相同."""
    rng = np.random.default_rng(seed)
    cont_gen, disc_gen = L1_DOMAINS[domain]
    cont = _pvals(candidate, cont_gen, rng)
    disc = _pvals(candidate, disc_gen, rng)

    d = {}
    # --- 连续区 ---
    d["cont_finite"], d["cont_ks_D"], d["cont_ks_p"], d["cont_mean"] = _stats(cont)
    d["cont_F"] = {a: float(np.mean(cont <= a)) for a in POINTS}
    d["cont_nonfinite_count"] = int(np.sum(~np.isfinite(cont)))

    cont_ok = (
        d["cont_finite"]
        and d["cont_ks_D"] <= KS_CRIT_99 + KS_SLACK
        and all(abs(d["cont_F"][a] - a) <= CONT_POINT_SLACK for a in POINTS)
        and abs(d["cont_mean"] - 0.5) <= CONT_MEAN_SLACK
    )
    d["cont_ok"] = bool(cont_ok)

    # --- 离散区 (漏洞 3: 只要求保守, 不要求均匀) ---
    d["disc_finite"], _, _, d["disc_mean"] = _stats(disc)
    d["disc_F"] = {a: float(np.mean(disc <= a)) for a in POINTS}
    d["disc_nonfinite_count"] = int(np.sum(~np.isfinite(disc)))

    disc_ok = (
        d["disc_finite"]
        and all(d["disc_F"][a] <= a + DISC_POINT_SLACK for a in POINTS)
        and d["disc_mean"] <= DISC_MEAN_MAX
    )
    d["disc_ok"] = bool(disc_ok)

    verdict = bool(cont_ok and disc_ok)
    return verdict, d


# ---------------------------------------------------------------------------
# 参照对 — 每模板两个独立实现
# ---------------------------------------------------------------------------

# --- chi2_rxc / ratio: 复用 reference.py 的 ref_hand / ref_scipy -------------
REF_CHI2 = (ref_hand, ref_scipy)


# --- fisher: 手写超几何枚举 vs scipy fisher_exact ----------------------------

def _fisher_enum(obs, lgamma_fn, ascending=True):
    """2x2 双侧 Fisher 精确 p: 固定边缘超几何枚举, 等概率表双向包含
    (p <= p_obs * (1+1e-12), 与 template_fisher 同口径)."""
    a, b, c, d = (int(x) for x in np.asarray(obs).ravel())
    n = a + b + c + d
    r1, r0, c1, c0 = a + b, c + d, a + c, b + d

    def prob(x):
        return math.exp(lgamma_fn(r1 + 1) + lgamma_fn(r0 + 1) + lgamma_fn(c1 + 1)
                        + lgamma_fn(c0 + 1) - lgamma_fn(n + 1) - lgamma_fn(x + 1)
                        - lgamma_fn(r1 - x + 1) - lgamma_fn(c1 - x + 1)
                        - lgamma_fn(r0 - c1 + x + 1))

    p_obs = prob(a)
    lo, hi = max(0, c1 - r0), min(r1, c1)
    xs = range(lo, hi + 1) if ascending else range(hi, lo - 1, -1)
    total = 0.0
    for x in xs:
        p = prob(x)
        if p <= p_obs * (1.0 + 1e-12):
            total += p
    return float(min(1.0, total))


def ref_fisher_math(obs):
    """参照 #1: math.lgamma 升序枚举."""
    return _fisher_enum(obs, math.lgamma, ascending=True)


def ref_fisher_gammaln(obs):
    """备选 #1': scipy.special.gammaln 降序枚举 (供 replicate/strat_bidir 复用)."""
    return _fisher_enum(obs, gammaln, ascending=False)


def ref_fisher_scipy(obs):
    """参照 #2: scipy.stats.fisher_exact two-sided (经验性与枚举约定一致到 1e-15)."""
    return float(scipy_fisher_exact(np.asarray(obs, dtype=int),
                                    alternative="two-sided").pvalue)


REF_FISHER = (ref_fisher_math, ref_fisher_scipy)


# --- barnard: 独立精确全局最大 vs scipy barnard_exact (SHGO) -----------------

_BARNARD_UNIFORM_PTS = 20000      # 均匀网格点数
_BARNARD_LOG_PTS = 10000          # 对数边界网格点数 (每侧)
_BARNARD_LOG_LO, _BARNARD_LOG_HI = 1e-9, 0.02
_BARNARD_GOLDEN_ITERS = 60        # 黄金分割精修迭代数


def _barnard_fold(obs):
    """a_s 折叠: log a_s = log Σ_{(x1,x2)∈S, x1+x2=s} C(c1,x1)C(c2,x2), 逐行向量化.

    S = {|z(x1,x2)| >= |z_obs|}, z 为合并方差 z 统计量 (与 scipy 内部同一口径,
    探针逐点验证一致到 1e-12). 两遍扫描: 第一遍求全局最大 log 组合数 m_g,
    第二遍 exp(lc - m_g) 按 s bincount; count_s == 0 的 bin 逐对角线精确重算
    (防止小 s 边界 bin 相对 m_g 下溢为 0 导致漏峰)."""
    a, b, c, d = (int(x) for x in np.asarray(obs).ravel())
    c1, c2 = a + c, b + d
    n = c1 + c2
    if n == 0 or c1 == 0 or c2 == 0:
        return None, n
    if a * d == b * c:
        return "all", n                     # 观察表完全成比例: S = 全表, T(π) ≡ 1
    x2 = np.arange(c2 + 1, dtype=float)
    p2 = x2 / c2
    p1_obs = a / c1
    p_obs_den = (x2 + a) / n
    var_obs = p_obs_den * (1.0 - p_obs_den) * (1.0 / c1 + 1.0 / c2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_obs = abs((p1_obs - p2) / np.sqrt(var_obs))
    z_obs = z_obs[b]                         # |z(a, b)|
    logc1 = gammaln(c1 + 1) - gammaln(np.arange(c1 + 1) + 1) - gammaln(c1 + 1 - np.arange(c1 + 1))
    logc2 = gammaln(c2 + 1) - gammaln(x2 + 1) - gammaln(c2 + 1 - x2)
    s_row = np.arange(c2 + 1)
    mg = -np.inf
    for x1 in range(c1 + 1):
        p1 = x1 / c1
        p = (x1 + x2) / n
        var = p * (1.0 - p) * (1.0 / c1 + 1.0 / c2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs((p1 - p2) / np.sqrt(var))
        z[p1 == p2] = 0.0
        m = z >= z_obs
        if m.any():
            v = logc1[x1] + logc2[m]
            vmax = float(v.max())
            if vmax > mg:
                mg = vmax
    acc = np.zeros(n + 1)
    for x1 in range(c1 + 1):
        p1 = x1 / c1
        p = (x1 + x2) / n
        var = p * (1.0 - p) * (1.0 / c1 + 1.0 / c2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs((p1 - p2) / np.sqrt(var))
        z[p1 == p2] = 0.0
        m = z >= z_obs
        if m.any():
            s = (s_row[m] + x1)
            w = np.exp(logc1[x1] + logc2[m] - mg)
            acc += np.bincount(s, weights=w, minlength=n + 1)
    loga = np.full(n + 1, -np.inf)
    nz = acc > 0.0
    loga[nz] = mg + np.log(acc[nz])
    # count_s == 0 的 bin 逐对角线精确重算 (向量化反角线; 含下溢与真空 bin 的区分)
    for s in range(n + 1):
        if acc[s] > 0.0:
            continue
        x1lo, x1hi = max(0, s - c2), min(c1, s)
        if x1lo > x1hi:
            continue
        x1v = np.arange(x1lo, x1hi + 1)
        x2v = s - x1v
        p1 = x1v / c1
        p2v = x2v / c2
        p = s / n
        var = p * (1.0 - p) * (1.0 / c1 + 1.0 / c2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs((p1 - p2v) / np.sqrt(var))
        z[p1 == p2v] = 0.0
        m = z >= z_obs
        if m.any():
            lc = logc1[x1v[m]] + logc2[x2v[m]]
            loga[s] = float(lc.max()) + math.log(np.exp(lc - lc.max()).sum())
    return loga, n


def _barnard_logT(pi, loga, n):
    """对数尾部概率 log T(π) = log Σ_s a_s π^s (1−π)^{n−s}."""
    s = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = loga + s * math.log(pi) + (n - s) * math.log1p(-pi)
    vals = lp[np.isfinite(lp)]
    if vals.size == 0:
        return -np.inf
    m = float(vals.max())
    return m + math.log(np.exp(vals - m).sum())


def _barnard_refine(grid, vals, loga, n):
    """从网格采样找全部局部极大 (离散导数 + -> -) + 端点, 黄金分割精修取最大."""
    d1 = np.sign(np.diff(vals))
    down = np.where(np.diff(d1) < 0)[0]      # 局部极大候选索引 (i+1)
    cands = set(down.tolist()) | {down[i] + 1 for i in range(len(down))}
    cands |= {0, len(vals) - 1}
    best_v, best_pi = -np.inf, None
    for k0 in cands:
        lo = grid[max(0, k0 - 1)]
        hi = grid[min(len(grid) - 1, k0 + 1)]
        for _ in range(_BARNARD_GOLDEN_ITERS):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            v1 = _barnard_logT(m1, loga, n)
            v2 = _barnard_logT(m2, loga, n)
            if v1 > v2:
                hi = m2
            else:
                lo = m1
        vm = _barnard_logT((lo + hi) / 2.0, loga, n)
        if vm > best_v:
            best_v, best_pi = vm, (lo + hi) / 2.0
    return best_v, best_pi


def ref_barnard_exact(obs):
    """参照 #1: 独立精确实现 —— 尾部概率对 nuisance π 的全局最大值.

    网格 = 均匀 20000 点 + 对数边界各 10000 点 (覆盖 π -> 0/1 的窄峰),
    全部局部极大 + 端点黄金分割精修. 探针验证: n<=100 与 scipy 一致到 1e-13;
    n>=500 偏斜表 scipy SHGO 漏检真实最大值 (例 [[157,163],[96,84]]:
    scipy 0.3663 vs 本实现 0.5296 at π≈0.0034)."""
    loga, n = _barnard_fold(obs)
    if loga is None:
        return 1.0
    if isinstance(loga, str):
        return 1.0
    grids = [
        np.linspace(1e-7, 1 - 1e-7, _BARNARD_UNIFORM_PTS),
        np.logspace(math.log10(_BARNARD_LOG_LO), math.log10(_BARNARD_LOG_HI),
                    _BARNARD_LOG_PTS),
        1.0 - np.logspace(math.log10(_BARNARD_LOG_LO), math.log10(_BARNARD_LOG_HI),
                          _BARNARD_LOG_PTS)[::-1],
    ]
    best_v, best_pi = -np.inf, None
    for g in grids:
        vals = np.array([_barnard_logT(pi, loga, n) for pi in g])
        v, pi = _barnard_refine(g, vals, loga, n)
        if v > best_v:
            best_v, best_pi = v, pi
    return float(min(1.0, math.exp(best_v)))


def ref_barnard_scipy(obs):
    """参照 #2: scipy.stats.barnard_exact two-sided (SHGO 全局优化)."""
    return float(scipy_barnard_exact(np.asarray(obs, dtype=int),
                                     alternative="two-sided").pvalue)


REF_BARNARD = (ref_barnard_exact, ref_barnard_scipy)


# --- trend: score 形式 vs 闭式 (两种代数恒等编码) -----------------------------

def _trend_common(obs):
    r = obs[0, :]
    nj = obs.sum(axis=0)
    R = float(obs[0, :].sum())
    N = float(obs.sum())
    if R == 0 or R == N:
        raise ValueError("template_trend: 无结果变异 (全为阳性或全为阴性)")
    w = np.arange(obs.shape[1], dtype=float)
    wnj = np.sum(w * nj)
    T = float(np.sum(w * r)) - (R / N) * float(wnj)      # 分开求和路径
    D1 = float(np.sum(w * w * nj))
    D2 = wnj
    D = N * D1 - D2 * D2
    if D <= 0:
        raise ValueError("template_trend: 方差为 0 (仅一个水平有数据)")
    return T, D, R, N


def ref_trend_score(obs):
    """参照 #1: score 形式 —— z = T / sqrt(VarT) -> 双侧 norm.sf."""
    T, D, R, N = _trend_common(obs)
    var_t = (R / N) * (1.0 - R / N) * (D / N)
    z = T / math.sqrt(var_t)
    return float(2.0 * norm.sf(abs(z)))


def ref_trend_closed(obs):
    """参照 #2: 闭式 —— z^2 = N^3 T^2 / (R S D), S = N - R -> chi2.sf(., 1)."""
    T, D, R, N = _trend_common(obs)
    S = N - R
    z2 = N * N * N * T * T / (R * S * D)
    return float(chi2.sf(z2, 1))


REF_TREND = (ref_trend_score, ref_trend_closed)


# --- slope: 分支 A = trend 参照对; 分支 B = 闭式手算 vs lstsq + beta 恒等式 ---

def _slope_ols_ref1(obs):
    """分支 B 参照 #1: 闭式手算 (中心化正交正规方程, np.dot 路径) -> f.sf."""
    r, c = obs.shape
    n = r * c
    y = obs.ravel()
    i_idx = np.arange(n) // c
    j_idx = np.arange(n) % c
    ic = i_idx - i_idx.mean()
    jc = j_idx - j_idx.mean()
    ybar = y.mean()
    syy = float(np.sum(y * y)) - n * ybar * ybar
    if syy <= 0.0:
        return 1.0
    siy = float(ic @ y)
    sjy = float(jc @ y)
    sii = float(ic @ ic)
    sjj = float(jc @ jc)
    b1 = siy / sii
    b2 = sjy / sjj
    rss_full = syy - b1 * siy - b2 * sjy
    if rss_full <= 0.0:
        return 0.0
    num = syy - rss_full
    if num < 0.0:
        num = 0.0
    F = (num / 2.0) / (rss_full / (n - 3))
    return float(f.sf(F, 2.0, n - 3))


def _slope_ols_ref2(obs):
    """分支 B 参照 #2: np.linalg.lstsq 正规方程 + 不完全 beta 恒等式
    p = I_x(d2/2, d1/2), x = d2/(d2 + d1·F) (独立分布路径)."""
    r, c = obs.shape
    n = r * c
    y = obs.ravel().astype(float)
    i_idx = np.arange(n) // c
    j_idx = np.arange(n) % c
    ic = i_idx - i_idx.mean()
    jc = j_idx - j_idx.mean()
    X = np.column_stack([np.ones(n), ic, jc])
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    ybar = y.mean()
    syy = float(np.sum(y * y)) - n * ybar * ybar
    if syy <= 0.0:
        return 1.0
    rss_full = float(res) if np.ndim(res) == 0 else float(np.sum(res))
    if rss_full <= 0.0:
        return 0.0
    num = syy - rss_full
    if num < 0.0:
        num = 0.0
    F = (num / 2.0) / (rss_full / (n - 3))
    d1, d2 = 2.0, float(n - 3)
    x = d2 / (d2 + d1 * F)
    return float(betainc(d2 / 2.0, d1 / 2.0, x))


def ref_slope_a(obs):
    """分支 A (2 行): trend 参照对."""
    if obs.shape[0] == 2:
        return ref_trend_score(obs)
    return _slope_ols_ref1(obs)


def ref_slope_b(obs):
    if obs.shape[0] == 2:
        return ref_trend_closed(obs)
    return _slope_ols_ref2(obs)


REF_SLOPE = (ref_slope_a, ref_slope_b)


# --- replicate: 两期 2x2 Fisher 双侧 p 的乘积 (e-value 组合量) ----------------

def ref_replicate_math(obs):
    """参照 #1: math.lgamma 升序枚举 + p1*p2."""
    obs = np.asarray(obs, dtype=float)
    p1 = _fisher_enum(obs[:, 0:2], math.lgamma, ascending=True)
    p2 = _fisher_enum(obs[:, 2:4], math.lgamma, ascending=True)
    return p1 * p2


def ref_replicate_gammaln(obs):
    """参照 #2: scipy.special.gammaln 降序枚举 + p1*p2."""
    obs = np.asarray(obs, dtype=float)
    p1 = _fisher_enum(obs[:, 0:2], gammaln, ascending=False)
    p2 = _fisher_enum(obs[:, 2:4], gammaln, ascending=False)
    return p1 * p2


REF_REPLICATE = (ref_replicate_math, ref_replicate_gammaln)


# --- stratified: CMH — 标量循环 (ad-bc)/N 形式 vs 向量化 E/Var 形式 ------------

def _strat_common(obs):
    obs = np.asarray(obs, dtype=float)
    K = obs.shape[1] // 2
    blocks = [obs[:, 2 * i:2 * i + 2] for i in range(K)]
    for i, blk in enumerate(blocks):
        if float(blk.sum()) < 2:
            raise ValueError(f"template_stratified: 第 {i} 层 N < 2")
    return blocks


def ref_strat_scalar(obs):
    """参照 #1: 标量循环, num = Σ(ad−bc)/N (与 E 形式代数恒等, 浮点路径不同)."""
    blocks = _strat_common(obs)
    num = den = 0.0
    for blk in blocks:
        a, b, c, d = (float(v) for v in blk.ravel())
        ni = a + b + c + d
        num += (a * d - b * c) / ni
        den += (a + b) * (a + c) * (b + d) * (c + d) / (ni * ni * (ni - 1))
    if den <= 0:
        raise ValueError("template_stratified: ΣVar_i = 0 (各层均无关联信息)")
    x2 = num * num / den
    return float(chi2.sf(x2, 1))


def ref_strat_vector(obs):
    """参照 #2: 向量化 E/Var 形式, numpy pairwise 求和."""
    blocks = _strat_common(obs)
    arr = np.asarray([blk.ravel() for blk in blocks])      # (K, 4)
    a, b, c, d = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    ni = a + b + c + d
    e = (a + b) * (a + c) / ni
    var = e * (b + d) * (c + d) / (ni * (ni - 1))
    num = float(np.sum(a - e))
    den = float(np.sum(var))
    if den <= 0:
        raise ValueError("template_stratified: ΣVar_i = 0 (各层均无关联信息)")
    x2 = num * num / den
    return float(chi2.sf(x2, 1))


REF_STRATIFIED = (ref_strat_scalar, ref_strat_vector)


# --- strat_bidir: 各层 2x2 双侧精确 p 的 Fisher 组合 --------------------------

def _sb_common(obs):
    obs = np.asarray(obs, dtype=float)
    K = obs.shape[1] // 2
    blocks = []
    for i in range(K):
        a, b, c, d = (int(v) for v in obs[:, 2 * i:2 * i + 2].ravel())
        if a + b + c + d < 2:
            raise ValueError(f"template_strat_bidir: 第 {i} 层 N < 2")
        if (b == 0 and c == 0) or (a == 0 and d == 0):
            raise ValueError(
                f"template_strat_bidir: 第 {i} 层总计数全在一条对角 (层内检验无信息)")
        blocks.append((a, b, c, d))
    return blocks


def _sb_series(ps):
    """组合: x·Σ_{k=0}^{K-1} (−ln x)^k / k! (下尾展开)."""
    x = math.prod(ps)
    if x == 0.0:
        return 0.0
    lx = -math.log(x)
    s, term = 0.0, 1.0
    for k in range(len(ps)):
        s += term
        term *= lx / (k + 1)
    return float(min(1.0, x * s))


def ref_sb_math(obs):
    """参照 #1: math.lgamma 升序枚举 + 级数组合."""
    return _sb_series([_fisher_enum([a, b, c, d], math.lgamma, ascending=True)
                       for a, b, c, d in _sb_common(obs)])


def ref_sb_chi2(obs):
    """参照 #2: gammaln 降序枚举 + 恒等式 −2 Σ ln p_i ~ chi2(2K) 的上尾."""
    ps = [_fisher_enum([a, b, c, d], gammaln, ascending=False)
          for a, b, c, d in _sb_common(obs)]
    t = 2.0 * float(np.sum([-math.log(p) for p in ps]))
    return float(chi2.sf(t, 2 * len(ps)))


REF_STRAT_BIDIR = (ref_sb_math, ref_sb_chi2)


REFS = {
    "template_chi2_rxc": REF_CHI2,
    "template_ratio": REF_CHI2,
    "template_fisher": REF_FISHER,
    "template_barnard": REF_BARNARD,
    "template_trend": REF_TREND,
    "template_slope": REF_SLOPE,
    "template_replicate": REF_REPLICATE,
    "template_stratified": REF_STRATIFIED,
    "template_strat_bidir": REF_STRAT_BIDIR,
}


# ---------------------------------------------------------------------------
# L2/L3: 固定考卷表构建 (确定性, 手选 + 种子化随机)
# ---------------------------------------------------------------------------

def _random_2x2(seed, count, ns=(10, 30, 100, 300)):
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(count):
        n = ns[i % len(ns)]
        rp = rng.random(2) + 0.05
        rp /= rp.sum()
        cp = rng.random(2) + 0.05
        cp /= cp.sum()
        for _ in range(200):
            t = rng.multinomial(n, np.outer(rp, cp).ravel()).reshape(2, 2)
            if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
                break
        else:
            raise RuntimeError("_random_2x2: 200 trials 内未抽到边缘全正表")
        tables.append(t)
    return tables


def _random_2xk(seed, count):
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(count):
        k = (3, 4, 5, 6)[i % 4]
        n = (10, 30, 100, 1000, 5000)[i % 5]
        cp = rng.random(k) + 0.05
        cp /= cp.sum()
        tables.append(_draw_h0(rng, n, B2_ROW, cp))
    return tables


def _random_2x4(seed, count):
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(count):
        n = (10, 100, 1000)[i % 3]
        cp = rng.random(4) + 0.05
        cp /= cp.sum()
        tables.append(_draw_h0(rng, n, B2_ROW, cp))
    return tables


def _random_2x2K(seed, K, count):
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(count):
        n = (10, 100)[i % 2]
        tables.append(_draw_strat(rng, n, K, forbid_degenerate=True))
    return tables


HAND_2XK = [
    np.array([[5, 10, 15], [10, 20, 30]]),        # 单调增
    np.array([[15, 10, 5], [5, 10, 15]]),         # 单调减
    np.array([[1, 2, 4, 8], [8, 4, 2, 1]]),       # 2x4 强单调
    np.array([[10, 10, 10], [10, 10, 10]]),       # 全均衡 -> p=1
    np.array([[0, 5, 10], [10, 5, 0]]),           # 零单元格 + 单调减
    np.array([[3, 7, 2, 8], [7, 3, 8, 2]]),       # 波动
    np.array([[100, 0, 5], [0, 100, 5]]),         # 强分离
    np.array([[12, 8, 5], [3, 15, 9]]),           # 一般 2x3
]

HAND_2X4 = [
    np.array([[10, 20, 10, 20], [30, 40, 30, 40]]),   # 两期同表 (强关联两期一致)
    np.array([[100, 0, 100, 0], [0, 100, 0, 100]]),   # 强分离两期
    np.array([[1, 2, 3, 4], [3, 4, 1, 2]]),           # 一般
    np.array([[0, 5, 5, 5], [5, 5, 5, 0]]),           # 零单元格
    np.array([[7, 11, 13, 17], [19, 23, 29, 31]]),    # 一般
    np.array([[50, 50, 50, 50], [50, 50, 50, 50]]),   # 全均衡 -> p1=p2=1
    np.array([[3, 3, 1, 1], [1, 1, 3, 3]]),           # 两期同向弱
    np.array([[2, 8, 8, 2], [8, 2, 2, 8]]),           # 两期反向强
]

HAND_2X2K = {
    2: [
        np.array([[10, 20, 30, 40], [30, 40, 50, 60]]),   # 两层同向一般
        np.array([[7, 3, 4, 6], [3, 7, 6, 4]]),           # 两层反向
        np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),           # 全均衡
        np.array([[0, 5, 3, 2], [5, 1, 2, 3]]),           # 零单元格 (非退化)
    ],
    3: [
        np.array([[10, 20, 30, 40, 5, 15], [30, 40, 50, 60, 10, 20]]),
        np.array([[0, 5, 3, 2, 1, 4], [5, 1, 2, 3, 4, 1]]),
        np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
        np.array([[7, 3, 2, 8, 6, 4], [3, 7, 8, 2, 4, 6]]),
    ],
    4: [
        np.array([[10, 20, 30, 40, 5, 15, 25, 35], [30, 40, 50, 60, 10, 20, 30, 40]]),
        np.array([[0, 5, 3, 2, 4, 1, 2, 3], [5, 1, 2, 3, 1, 4, 3, 2]]),
        np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]),
        np.array([[7, 3, 2, 8, 6, 4, 5, 5], [3, 7, 8, 2, 4, 6, 5, 5]]),
    ],
}


def build_l2_tables_domain(domain):
    """L2 固定考卷表 (确定性, 手选 + 种子化随机, 全部边缘 > 0)."""
    if domain == "rxc":
        return l2.build_l2_tables()                        # 复用 l2.py 的 40 张
    if domain == "2x2":
        hand = [np.asarray(t) for t in l2._HAND_PICKED if np.asarray(t).shape == (2, 2)]
        return hand + _random_2x2(L2_TABLE_SEED, 20)
    if domain == "2xk":
        return list(HAND_2XK) + _random_2xk(L2_TABLE_SEED, 20)
    if domain == "2x4":
        return list(HAND_2X4) + _random_2x4(L2_TABLE_SEED, 8)
    if domain == "2x2K":
        tables = []
        for K in (2, 3, 4):
            tables += list(HAND_2X2K[K])
            tables += _random_2x2K(L2_TABLE_SEED, K, 6)
        return tables
    raise ValueError(f"未知域: {domain}")


# --- L3: 五种形态 (极小 n=15 / 大样本 n=20000 / 强偏斜 / 零单元格 / 混合) ----

_RXC_SHAPES = [(3, 4), (4, 5), (4, 4), (5, 5), (2, 3), (2, 4), (3, 3), (2, 5)]


def _skew_cp(k, base=0.2):
    """强偏斜列概率: cp_j ∝ base^j (base=0.2 -> 相邻列 5 倍递减, 末列 ~1:1000)."""
    raw = np.power(base, np.arange(k, dtype=float))
    return raw / raw.sum()


def _skew_rp(r, base=0.2):
    """强偏斜行概率: rp_i ∝ base^i (保证末行期望 >= ~1, 避免边缘归零死循环)."""
    raw = np.power(base, np.arange(r, dtype=float))
    return raw / raw.sum()


def _draw_form(rng, domain, n, row_p, col_p, shape=None, K=None):
    """按域抽取一张表 (n 为整表总样本; 2x2K 域 n 为每层样本)."""
    if domain == "rxc":
        r, c = shape if shape is not None else (3, 4)
        return _draw_h0(rng, n, row_p, col_p)
    if domain == "2x2":
        return _draw_h0(rng, n, B2_ROW, B2_COL)
    if domain == "2xk":
        k = shape[1] if shape is not None else 4
        return _draw_h0(rng, n, B2_ROW, col_p)
    if domain == "2x4":
        return _draw_h0(rng, n, B2_ROW, col_p)
    if domain == "2x2K":
        k = K if K is not None else 2
        return _draw_strat(rng, n, k, forbid_degenerate=True)
    raise ValueError(f"未知域: {domain}")


def _zero_cell(rng, t):
    """把表内随机一格清零 (L3 零单元格形态); 返回新表."""
    out = t.copy()
    out[rng.integers(0, t.shape[0]), rng.integers(0, t.shape[1])] = 0
    return out


def build_l3_tables_domain(domain):
    """L3 考卷表: 5 形态 x 每域 >= 40 张 (seed 20260808, 确定性)."""
    rng = np.random.default_rng(L3_TABLE_SEED)
    tables = []

    def add(t):
        tables.append(t)

    if domain == "2x2":
        # 形态 1: 极小样本 n=15 (8)
        for _ in range(8):
            add(_draw_h0(rng, 15, B2_ROW, B2_COL))
        # 形态 2: 大样本 n=20000 (8)
        for _ in range(8):
            add(_draw_h0(rng, 20000, B2_ROW, B2_COL))
        # 形态 3: 强偏斜 1:100 (6)
        for _ in range(6):
            n = (2000, 5000)[_ % 2]
            add(_draw_h0(rng, n, np.array([0.99, 0.01]), np.array([0.9, 0.1])))
        # 形态 4: 零单元格 (9)
        for _ in range(9):
            n = (15, 25, 50)[_ % 3]
            for _t in range(200):
                t = _zero_cell(rng, _draw_h0(rng, n, B2_ROW, B2_COL))
                if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
                    break
            else:
                raise RuntimeError("零单元格形态重抽失败")
            add(t)
        # 形态 5: 混合中样本 (11)
        for _ in range(11):
            n = (50, 200, 800)[_ % 3]
            rp = rng.random(2) + 0.05
            rp /= rp.sum()
            cp = rng.random(2) + 0.05
            cp /= cp.sum()
            add(_draw_h0(rng, n, rp, cp))
        return tables

    if domain == "rxc":
        for _ in range(8):
            r, c = _RXC_SHAPES[_ % len(_RXC_SHAPES)]
            rp = rng.random(r) + 0.05
            rp /= rp.sum()
            cp = rng.random(c) + 0.05
            cp /= cp.sum()
            add(_draw_form(rng, "rxc", 15, rp, cp, shape=(r, c)))
        for _ in range(8):
            r, c = _RXC_SHAPES[_ % len(_RXC_SHAPES)]
            rp = rng.random(r) + 0.05
            rp /= rp.sum()
            cp = rng.random(c) + 0.05
            cp /= cp.sum()
            add(_draw_form(rng, "rxc", 20000, rp, cp, shape=(r, c)))
        for _ in range(6):
            r, c = _RXC_SHAPES[(_ + 1) % len(_RXC_SHAPES)]
            n = (1000, 3000)[_ % 2]
            add(_draw_form(rng, "rxc", n, _skew_rp(r), _skew_cp(c), shape=(r, c)))
        for _ in range(9):
            r, c = _RXC_SHAPES[_ % len(_RXC_SHAPES)]
            rp = np.full(r, 1.0 / r)
            cp = np.full(c, 1.0 / c)
            n = (15, 25, 50)[_ % 3]
            for _t in range(200):
                t = _zero_cell(rng, _draw_form(rng, "rxc", n, rp, cp, shape=(r, c)))
                if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
                    break
            else:
                raise RuntimeError("零单元格形态重抽失败")
            add(t)
        for _ in range(11):
            r, c = _RXC_SHAPES[_ % len(_RXC_SHAPES)]
            rp = rng.random(r) + 0.05
            rp /= rp.sum()
            cp = rng.random(c) + 0.05
            cp /= cp.sum()
            n = (50, 200, 800)[_ % 3]
            add(_draw_form(rng, "rxc", n, rp, cp, shape=(r, c)))
        return tables

    if domain == "2xk":
        for _ in range(8):
            k = (3, 4, 5, 6)[_ % 4]
            add(_draw_form(rng, "2xk", 15, None, np.full(k, 1.0 / k), shape=(2, k)))
        for _ in range(8):
            k = (3, 4, 5, 6)[_ % 4]
            add(_draw_form(rng, "2xk", 20000, None, np.full(k, 1.0 / k), shape=(2, k)))
        for _ in range(6):
            k = (3, 4, 5, 6)[(_ + 1) % 4]
            n = (1000, 3000)[_ % 2]
            add(_draw_form(rng, "2xk", n, None, _skew_cp(k), shape=(2, k)))
        for _ in range(9):
            k = (3, 4, 5, 6)[_ % 4]
            n = (15, 25, 50)[_ % 3]
            for _t in range(200):
                t = _zero_cell(rng, _draw_form(rng, "2xk", n, None,
                                               np.full(k, 1.0 / k), shape=(2, k)))
                if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
                    break
            else:
                raise RuntimeError("零单元格形态重抽失败")
            add(t)
        for _ in range(11):
            k = (3, 4, 5, 6)[_ % 4]
            cp = rng.random(k) + 0.05
            cp /= cp.sum()
            n = (50, 200, 800)[_ % 3]
            add(_draw_form(rng, "2xk", n, None, cp, shape=(2, k)))
        return tables

    if domain == "2x4":
        for _ in range(8):
            add(_draw_form(rng, "2x4", 15, None, REP_COL))
        for _ in range(8):
            add(_draw_form(rng, "2x4", 20000, None, REP_COL))
        for _ in range(6):
            n = (1000, 3000)[_ % 2]
            add(_draw_form(rng, "2x4", n, None, np.array([0.55, 0.25, 0.15, 0.05])))
        for _ in range(9):
            n = (15, 25, 50)[_ % 3]
            for _t in range(200):
                t = _zero_cell(rng, _draw_form(rng, "2x4", n, None, REP_COL))
                if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0):
                    break
            else:
                raise RuntimeError("零单元格形态重抽失败")
            add(t)
        for _ in range(11):
            cp = rng.random(4) + 0.05
            cp /= cp.sum()
            n = (50, 200, 800)[_ % 3]
            add(_draw_form(rng, "2x4", n, None, cp))
        return tables

    if domain == "2x2K":
        for _ in range(8):
            K = (2, 3, 4)[_ % 3]
            add(_draw_strat(rng, 15 // K, K, forbid_degenerate=True))
        for _ in range(8):
            K = (2, 3, 4)[_ % 3]
            add(_draw_strat(rng, 20000 // K, K, forbid_degenerate=True))
        for _ in range(6):
            K = (2, 3, 4)[(_ + 1) % 3]
            n = (100, 200)[_ % 2]
            add(_draw_strat(rng, n, K, forbid_degenerate=True))
        for _ in range(9):
            K = (2, 3, 4)[_ % 3]
            n = (8, 12, 25)[_ % 3]
            for _t in range(200):
                t = _zero_cell(rng, _draw_strat(rng, n, K, forbid_degenerate=True))
                if np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0) \
                        and not _has_degenerate_layer(t):
                    break
            else:
                raise RuntimeError("零单元格形态重抽失败")
            add(t)
        for _ in range(11):
            K = (2, 3, 4)[_ % 3]
            n = (50, 200, 500)[_ % 3]
            add(_draw_strat(rng, n, K, forbid_degenerate=True))
        return tables

    raise ValueError(f"未知域: {domain}")


# ---------------------------------------------------------------------------
# L2/L3: 判定 (参照自检 -> 对拍)
# ---------------------------------------------------------------------------

def refs_agree(tables, ref1, ref2):
    """两个参照全部一致才采信 (漏洞 1 修正); 返回最差偏差."""
    worst = 0.0
    for t in tables:
        worst = max(worst, abs(ref1(t) - ref2(t)))
    return worst


def _pair_grade(candidate, ref1, ref2, tables):
    """参照自检 + 对拍; 返回 (verdict, diag)."""
    worst_ref_dev = refs_agree(tables, ref1, ref2)
    if worst_ref_dev > REF_AGREE_TOL:
        return False, {
            "ref_abort": True,
            "worst_ref_dev": worst_ref_dev,
            "n_tables": len(tables),
        }
    devs = []
    for t in tables:
        try:
            cand_p = float(candidate(t))
            d = max(abs(cand_p - ref1(t)), abs(cand_p - ref2(t)))
        except Exception:
            d = float("inf")                    # 候选抛异常 = 对拍失败 (表均为合法输入)
        devs.append(d)
    max_dev = float(max(devs))
    worst_i = int(np.argmax(devs))
    return bool(max_dev <= L2_TOL), {
        "ref_abort": False,
        "worst_ref_dev": worst_ref_dev,
        "max_dev": max_dev,
        "n_tables": len(tables),
        "n_viol": int(sum(d > L2_TOL for d in devs)),
        "worst_table": np.asarray(tables[worst_i]).tolist(),
    }


def run_l2_domain(candidate, ref1, ref2, domain):
    """L2 对拍. L2 REF_ABORT 时 L3 顺延中止 (参照不可信, 无法判 L3)."""
    return _pair_grade(candidate, ref1, ref2, build_l2_tables_domain(domain))


def run_l3_domain(candidate, ref1, ref2, domain):
    """L3 边界泛化对拍 (仅当 L2 参照自检通过时调用)."""
    return _pair_grade(candidate, ref1, ref2, build_l3_tables_domain(domain))


# ---------------------------------------------------------------------------
# L4: 复用 l4.run_l4 (9 类畸形输入诚实失败)
# ---------------------------------------------------------------------------

def run_l4_domain(candidate):
    return l4.run_l4(candidate)
