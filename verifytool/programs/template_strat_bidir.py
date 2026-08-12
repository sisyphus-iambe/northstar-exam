"""模板 #6: 分层双向组合检验 (每层 2x2 双侧精确 + Fisher 组合) — 2x2xK.

输入维序约定: 与 template_stratified 相同: 2 x (2K) 计数表 (K >= 2 层),
  第 i 层 = 列块 [2i, 2i+1], 层 i 的 2x2 = [[a_i, b_i], [c_i, d_i]]:
    行 0 = 暴露, 行 1 = 非暴露;  列 0 = 事件, 列 1 = 非事件.

零假设 H0: 各层内暴露与事件独立 (各层 OR = 1).

机制 (方向无关的组合检验; 与 CMH 相反 —— 方向相反的层内信号各算各的显著):
  1. 每层做 2x2 双侧精确检验 (与 template_fisher 同口径: 固定边缘超几何,
     等概率表双向包含, 1e-12 容差吸收边界浮点噪声), 得层内 p_i ——
     正/负关联均给出小 p_i, 方向不抵消;
  2. Fisher 组合: x = ∏_i p_i, 组合 p = x · Σ_{k=0}^{K-1} (−ln x)^k / k!
     (精确下尾展开, 数学恒等于 −2 Σ_i ln p_i ~ chi2(2K) 的上尾:
      e^{−t/2} Σ (t/2)^k / k!,  t = −2 ln x;  与 scipy.stats.chi2.sf 对照一致);
  3. x = 0 时组合 p = 0 (某层极强分离使 p_i 下溢为 0).

诚实失败 (L4 考法): 非 2 行 / 列数非偶数或 K < 2 / 含 NaN±inf / 含负值 /
  非整数计数 (层内超几何要求整数, 同 template_fisher) /
  任一层 N_i < 2 / 某层总计数全在一条对角 (a_i,d_i 占据或 b_i,c_i 占据,
  层内检验无信息) -> 抛 ValueError.

确定性: 纯枚举 + 对数阶乘表 + 组合级数, 无随机数.
"""
import math

import numpy as np

NAME = "template_strat_bidir"


def _fisher_two_sided(a, b, c, d):
    """2x2 双侧精确 p — 与 template_fisher.chi2_pvalue 同口径 (照抄其计算方式)."""
    n = a + b + c + d
    r1, r0, c1, c0 = a + b, c + d, a + c, b + d
    lf = [0.0] * (n + 1)
    for i in range(1, n + 1):
        lf[i] = lf[i - 1] + math.log(i)

    def prob(x):
        # 配置 (a'=x, b'=r1-x, c'=c1-x, d'=r0-c1+x) 的超几何概率
        return math.exp(lf[r1] + lf[r0] + lf[c1] + lf[c0] - lf[n]
                        - lf[x] - lf[r1 - x] - lf[c1 - x] - lf[r0 - c1 + x])

    p_obs = prob(a)
    total = 0.0
    for x in range(max(0, c1 - r0), min(r1, c1) + 1):
        p = prob(x)
        if p <= p_obs * (1.0 + 1e-12):
            total += p
    return float(min(1.0, total))


def chi2_pvalue(observed):
    """分层双向组合检验: 各层 2x2 双侧 p 的 Fisher 组合 p (0~1)."""
    obs = np.asarray(observed, dtype=float)   # 字符串/None -> 转换失败或产生 nan, 由后续检查拦截
    if obs.ndim != 2 or obs.shape[0] != 2:
        raise ValueError(
            f"template_strat_bidir: 输入必须是 2 行计数表, 实际形状 {obs.shape}")
    if obs.shape[1] % 2 != 0 or obs.shape[1] < 4:
        raise ValueError(
            f"template_strat_bidir: 列数必须为偶数且 >= 4 (K >= 2 层), 实际 {obs.shape[1]}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_strat_bidir: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_strat_bidir: 输入含负值")
    if not np.all(obs == np.round(obs)):
        raise ValueError("template_strat_bidir: 输入含非整数计数 (超几何要求整数)")
    obs = obs.astype(int)
    K = obs.shape[1] // 2
    pis = []
    for i in range(K):
        a, b, c, d = (int(v) for v in obs[:, 2 * i:2 * i + 2].ravel())
        if a + b + c + d < 2:
            raise ValueError(f"template_strat_bidir: 第 {i} 层 N < 2")
        if (b == 0 and c == 0) or (a == 0 and d == 0):
            raise ValueError(
                f"template_strat_bidir: 第 {i} 层总计数全在一条对角 (层内检验无信息)")
        pis.append(_fisher_two_sided(a, b, c, d))

    x = math.prod(pis)
    if x == 0.0:
        return 0.0
    lx = -math.log(x)              # = Σ_i −ln p_i; t = −2 Σ ln p_i = 2·lx
    s, term = 0.0, 1.0
    for k in range(K):             # Σ_{k=0}^{K-1} lx^k / k!
        s += term
        term *= lx / (k + 1)
    return float(min(1.0, x * s))
