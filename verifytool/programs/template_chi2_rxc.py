"""模板 #5: r×c 计数表 Pearson 卡方独立性检验 — 通用多元形态.

输入维序约定: 任意 2D 计数表 (r >= 2 行, c >= 2 列), 行/列语义由调用方定义.
  覆盖 2x2 (与 template_fisher 同输入形状, 但渐近口径) 到任意 r x c.

零假设 H0: 行与列独立 (各格期望 = 行边际 × 列边际 / N).

公式 (Pearson chi2, 无连续性校正):
  期望 E_ij  = 行和_i × 列和_j / N
  X2         = Σ_ij (O_ij − E_ij)² / E_ij
  dof        = (r−1)(c−1)
  p 值       = chi2.sf(X2, dof)   (双侧)
  与 scipy.stats.chi2_contingency(table, correction=False) 对照一致.

诚实失败 (L4 考法): 非 2D / 行或列数 < 2 / 含 NaN±inf / 含负值 /
  任一行和或列和为 0 (边际 0 → 期望 0 → 除零) -> 抛 ValueError.
  零单元格允许 (只要所在行和、列和 > 0, 期望 > 0).

确定性: 纯公式, 无随机数.
"""
import numpy as np
from scipy.stats import chi2

NAME = "template_chi2_rxc"


def chi2_pvalue(observed):
    """r×c Pearson 卡方独立性检验. 输入 2D 计数表, 返回 float p (0~1)."""
    obs = np.asarray(observed, dtype=float)   # 字符串/None -> 转换失败或产生 nan, 由后续检查拦截
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError(
            f"template_chi2_rxc: 输入必须至少 2 行 2 列, 实际形状 {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_chi2_rxc: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_chi2_rxc: 输入含负值")
    row_sum = obs.sum(axis=1)
    col_sum = obs.sum(axis=0)
    if np.any(row_sum == 0) or np.any(col_sum == 0):
        raise ValueError("template_chi2_rxc: 存在全零行或全零列 (边际 0 → 期望 0 → 除零)")
    n = float(obs.sum())
    expected = np.outer(row_sum, col_sum) / n
    x2 = float(np.sum((obs - expected) ** 2 / expected))
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return float(chi2.sf(x2, dof))
