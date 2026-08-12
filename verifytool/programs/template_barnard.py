"""模板 #8: 2x2 计数表 Barnard 无条件精确检验 (双侧) — 稀疏表正确工具.

输入维序约定: 同 template_fisher — 行 0 = 暴露 (或事件组), 行 1 = 非暴露;
列 0 = 事件, 列 1 = 非事件. 任何 2x2 二分类计数表均适用.

零假设 H0: 行与列独立 (暴露与事件无关联, OR = 1). 与 Fisher (条件,
固定行/列和) 不同, Barnard 采用双多项式 (无条件) 模型: 行和/列和不固定,
把公共成功概率 p (nuisance 参数) 在 [0,1] 上枚举 (默认 32 点网格), 对每个
p 枚举所有与观察表同样本量的 2x2 表, 求该 p 下 Wald 统计量 (pooled 方差)
的双侧小尾概率; 最终 p = 各 p 上小尾概率的最大值 (无条件小尾, 比条件
Fisher 更灵敏, 是稀疏表/小样本的正确工具).

p 值来源: 直接引用 scipy.stats.barnard_exact(table, alternative="two-sided"),
与本文档口径一致 (确定性枚举, 无随机数).

诚实失败 (L4 考法): 非 2x2 / 含 NaN±inf / 负值 / 非整数计数 / 全零表 (n=0)
  -> 抛 ValueError, 校验顺序与 template_fisher 逐条一致.
  零行和或零列和 (边缘为 0, n>0): 实测 scipy 1.18.0 barnard_exact 对
  [[0,0],[5,5]]/[[3,0],[3,0]]/[[0,0],[0,5]] 等全部返回 p=1.0, 不抛错、
  不返回 nan —— 与 template_fisher 约定一致 (数据无关联信息), 直接透传.
  含 0 的格 (非零边缘): 实测正常返回精确 p (如 [[0,10],[10,10]] -> 0.00593),
  直接透传.

确定性: scipy 枚举实现, 无随机数; 同表重复调用逐位一致 (已实测).

性能实测 (2026-08-06, scipy 1.18.0): 每格 n=1000 (总和 4000) 单次约 4.6s;
  每格 n=200 (总和 800) 亚秒级; M6 级稀疏表瞬时.
"""
import numpy as np
from scipy.stats import barnard_exact

NAME = "template_barnard"


def chi2_pvalue(observed):
    """2x2 Barnard 无条件双侧精确检验. 输入 2x2 计数表, 返回 float p (0~1)."""
    obs = np.asarray(observed, dtype=float)   # 字符串/None -> 转换失败或产生 nan, 由后续检查拦截
    if obs.ndim != 2 or obs.shape != (2, 2):
        raise ValueError(f"template_barnard: 输入必须是 2x2 表, 实际形状 {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_barnard: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_barnard: 输入含负值")
    if not np.all(obs == np.round(obs)):
        raise ValueError("template_barnard: 输入含非整数计数 (计数表要求整数)")
    if float(obs.sum()) == 0.0:
        raise ValueError("template_barnard: 表总和为 0 (无数据)")
    return float(barnard_exact(obs, alternative="two-sided").pvalue)
