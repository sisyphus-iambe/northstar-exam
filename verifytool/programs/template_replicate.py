"""模板 #5: 两期重复验证 (2x2x2) — "一期发现 -> 二期是否持续"金标准模板.

输入维序约定: 2 x 4 计数表, 列块 = 两期:
  第 0-1 列 = 第一期 2x2, 第 2-3 列 = 第二期 2x2;
  每期内部: 行 0 = 暴露, 行 1 = 非暴露;  列 0 = 事件, 列 1 = 非事件.

零假设 H0 (每期): 暴露与事件独立. 两期独立.

每期 p 值: 2x2 Fisher 双侧精确检验 (公式与 template_fisher 相同):
  P(a,b,c,d) = (r1! r0! c1! c0!) / (n! a! b! c! d!)
  双侧 p_i = 所有概率 <= 观察表概率的表之概率和 (对 a' 从 max(0, c1-r0) 到 min(r1, c1) 枚举)

组合证据量 (e-value 组合): 返回 p1 * p2.
  e-value 视角: 若 p1、p2 为独立有效 p 值 (各期 H0 下), 则 e_i = (1/2) p_i^(-1/2)
  是有效 e-value (E[e_i] <= 1, Vovk-Wang p->e 变换), 组合 e = e1*e2 = (1/4)(p1*p2)^(-1/2);
  证据强度随 p1*p2 单调递减, 故 p1*p2 是两期证据的组合量, 数值越小证据越强.
  注意: p1*p2 不是校准 p 值 (H0 下 P(p1*p2 <= x) = x(1 - ln x) > x, 反保守);
  需要校准 p 值请用 Fisher 组合: -2 ln(p1*p2) ~ chi2(4). 本模板按规格返回 e-value 组合量.

用途: 一期发现信号 (p1 小) -> 二期复现 (p2 小) -> 乘积极小 = 信号持续;
  一期假阳性 (p2 大) -> 乘积被稀释 = 不持续.
  方向边界: 两期 p 值均为双侧, 不区分关联方向; 若要求"同向复现",
  调用方须额外核对两期 OR 方向 (可用 template_ratio.odds_ratio).
  反向强信号的两期同样得到小乘积 —— 其语义是"两期均有非独立证据", 而非"同向复现".

诚实失败 (L4 考法): 非 2x4 / 含 NaN±inf / 负值 / 非整数计数 (Fisher 要求整数) /
  任一期全零 (无数据) -> 抛 ValueError.
  零边际期 (n>0) 的 Fisher p = 1.0 (该期无关联信息, 数学良定义), 不算幻觉填补.

确定性: 纯枚举, 无随机数.
"""
import math

import numpy as np

NAME = "template_replicate"


def _fisher_p(a, b, c, d):
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


def per_period_pvalues(observed):
    """返回 (p1, p2): 两期各自的 Fisher 双侧 p 值."""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape != (2, 4):
        raise ValueError(f"template_replicate: 输入必须是 2x4 表 (两期), 实际形状 {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_replicate: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_replicate: 输入含负值")
    if not np.all(obs == np.round(obs)):
        raise ValueError("template_replicate: 输入含非整数计数 (Fisher 要求整数)")
    ps = []
    for i in range(2):
        a, b, c, d = (int(x) for x in obs[:, 2 * i:2 * i + 2].ravel())
        if a + b + c + d == 0:
            raise ValueError(f"template_replicate: 第 {i + 1} 期全零 (无数据)")
        ps.append(_fisher_p(a, b, c, d))
    return tuple(ps)


def chi2_pvalue(observed):
    """两期重复验证: 返回 p1 * p2 (e-value 组合量, 公式见模块 docstring)."""
    p1, p2 = per_period_pvalues(observed)
    return p1 * p2
