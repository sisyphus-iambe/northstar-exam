"""模板 #2: 2x2 比值信号检验 — "信号检测"通用家族.

输入维序约定: 2x2 计数表 [[a, b], [c, d]]:
  行 0 = 暴露, 行 1 = 非暴露;  列 0 = 事件, 列 1 = 非事件.
  暴露/事件语义由调用方定义, 本模板只要求 2x2 二分类计数表.

零假设 H0: 暴露与事件独立 (OR = RR = 1).

公式:
  比值比  OR = (a/b) / (c/d) = ad / (bc)                                  [odds_ratio]
  风险比  RR = (a/(a+b)) / (c/(c+d))                                      [risk_ratio]
  卡方    chi2 = n (ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d)),  dof = 1
          (2x2 表自由度恒为 1; 与 scipy.stats.chi2_contingency(correction=False) 一致,
           校准层协议固定 correction=False, 不用 Yates 校正)
  p 值    p = chi2.sf(chi2, 1)   (双侧)

返回: chi2_pvalue 返回 chi2 (dof=1) 的 p 值 —— 信号显著性主通道.
  OR/RR 是信号强度量, 由模块级函数 odds_ratio / risk_ratio 单独提供;
  比值无定义 (除零: b*c == 0 或分母为 0) 时两个辅助函数抛 ValueError ——
  禁止把未定义比值当有限数使用. chi2_pvalue 不受零单元格影响 (只要边际 > 0, chi2 良定义),
  强分离信号 (如 [[100,0],[0,100]]) 依然可检验 —— 这正是信号检测要抓的形态.

诚实失败 (L4 考法): 非 2x2 / 含 NaN±inf / 负值 / 任一边际 (行和或列和) 为 0
  (chi2 未定义, scipy 同样抛 ValueError) -> 抛 ValueError.

确定性: 纯公式, 无随机数.
"""
import numpy as np
from scipy.stats import chi2

NAME = "template_ratio"


def _cells(observed):
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape != (2, 2):
        raise ValueError(f"template_ratio: 输入必须是 2x2 表, 实际形状 {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_ratio: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_ratio: 输入含负值")
    a, b, c, d = (float(x) for x in obs.ravel())
    if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        raise ValueError("template_ratio: 存在零边际 (行和/列和为 0), chi2 未定义")
    return a, b, c, d


def chi2_pvalue(observed):
    """2x2 比值信号检验: 返回 chi2 (dof=1) 的 p 值."""
    a, b, c, d = _cells(observed)
    n = a + b + c + d
    stat = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    return float(chi2.sf(stat, 1))


def odds_ratio(observed):
    """比值比 OR = ad/(bc). 除零 (b*c == 0) -> ValueError (未定义)."""
    a, b, c, d = _cells(observed)
    if b * c == 0:
        raise ValueError("template_ratio: OR 未定义 (b*c = 0)")
    return float(a * d / (b * c))


def risk_ratio(observed):
    """风险比 RR = (a/(a+b)) / (c/(c+d)). 分母为 0 -> ValueError (未定义)."""
    a, b, c, d = _cells(observed)
    if a + b == 0 or c + d == 0 or c == 0:
        raise ValueError("template_ratio: RR 未定义 (分母为 0)")
    return float((a / (a + b)) / (c / (c + d)))
