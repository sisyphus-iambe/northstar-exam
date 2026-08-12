"""模板 #1: 2x2 计数表 Fisher 精确检验 (双侧) — 信号检测通用家族.

输入维序约定: 2x2 计数表 [[a, b], [c, d]]:
  行 0 = 暴露 (或事件组), 行 1 = 非暴露;  列 0 = 事件, 列 1 = 非事件.
  任何 2x2 二分类计数表均适用; 行/列标签语义由调用方自行定义.

零假设 H0: 行与列独立 (暴露与事件无关联), 即固定边缘下的超几何模型 (OR = 1).

p 值定义 (双侧): 固定行和/列和条件下, 观察表出现的超几何概率
  P(a,b,c,d) = (r1! r0! c1! c0!) / (n! a! b! c! d!),  其中 r1=a+b, r0=c+d, c1=a+c, c0=b+d, n=a+b+c+d
  双侧 p = 所有概率 <= 观察表概率的表 (对 a' 从 max(0, c1-r0) 到 min(r1, c1) 枚举) 的概率之和.
  与 scipy.stats.fisher_exact(table, alternative="two-sided") 对照一致 (偏差 ~1e-15 浮点噪声,
  1e-12 相对容差吸收边界浮点噪声, 等概率表双向包含).

诚实失败 (L4 考法): 非 2x2 / 含 NaN±inf / 负值 / 非整数计数 (超几何要求整数) / 全零表 (n=0)
  -> 抛 ValueError.
  零行和或零列和 (边缘为 0, n>0) 是数学良定义的单配置表, 返回 p=1.0 (数据无关联信息),
  不算幻觉填补.

确定性: 纯枚举 + math.lgamma, 无随机数.
"""
import math

import numpy as np

NAME = "template_fisher"


def chi2_pvalue(observed):
    """2x2 Fisher 双侧精确检验. 输入 2x2 计数表, 返回 float p (0~1)."""
    obs = np.asarray(observed, dtype=float)   # 字符串/None -> 转换失败或产生 nan, 由后续检查拦截
    if obs.ndim != 2 or obs.shape != (2, 2):
        raise ValueError(f"template_fisher: 输入必须是 2x2 表, 实际形状 {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_fisher: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_fisher: 输入含负值")
    if not np.all(obs == np.round(obs)):
        raise ValueError("template_fisher: 输入含非整数计数 (超几何要求整数)")
    a, b, c, d = (int(x) for x in obs.ravel())
    n = a + b + c + d
    if n == 0:
        raise ValueError("template_fisher: 表总和为 0 (无数据)")

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
