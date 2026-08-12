"""模板 #4: Mantel-Haenszel 分层检验 (2x2xK) — 手写实现.

输入维序约定: 2 x (2K) 计数表 (K >= 2 层), 第 i 层 = 列块 [2i, 2i+1]:
  层 i 的 2x2 表 [[a_i, b_i], [c_i, d_i]]:
    行 0 = 暴露, 行 1 = 非暴露;  列 0 = 事件, 列 1 = 非事件.
  (第三维 (层) 编码为列块, 保持程序接口的 2D 输入约定; K >= 2, 列数必须为偶数.)

零假设 H0: 各层内暴露与事件独立 (共同 OR = 1).

公式 (CMH 检验统计量, 层 i 记 N_i = a_i+b_i+c_i+d_i):
  期望 E_i   = (a_i+b_i)(a_i+c_i) / N_i
  方差 Var_i = (a_i+b_i)(a_i+c_i)(b_i+d_i)(c_i+d_i) / (N_i^2 (N_i-1))
  X2_MH      = [ Σ_i (a_i - E_i) ]^2 / Σ_i Var_i   ~ chi2(1)  (H0 下渐近)
  p 值       = chi2.sf(X2_MH, 1)   (双侧)
  组合比值比 OR_MH = Σ_i (a_i d_i / N_i) / Σ_i (b_i c_i / N_i)     [mantel_haenszel_or]
    (OR_MH 无定义 (分母为 0) 时 mantel_haenszel_or 抛 ValueError, 禁止把未定义比值当有限数;
     chi2_pvalue 不受影响 —— 强分离信号 (各层 b*c 全为 0) 依然可检验.)

诚实失败 (L4 考法): 非 2 行 / 列数非偶数或 K < 2 / 含 NaN±inf / 负值 /
  任一层 N_i < 2 (Var_i 分母 N_i-1 为 0) / ΣVar_i = 0 (各层均无关联信息) -> 抛 ValueError.
  零单元格允许 (只要该层 N_i >= 2); 零边际层贡献 0, 数学良定义.

确定性: 纯公式, 无随机数.
"""
import numpy as np
from scipy.stats import chi2

NAME = "template_stratified"


def _strata(observed):
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape[0] != 2:
        raise ValueError(f"template_stratified: 输入必须是 2 行计数表, 实际形状 {obs.shape}")
    if obs.shape[1] % 2 != 0 or obs.shape[1] < 4:
        raise ValueError(
            f"template_stratified: 列数必须为偶数且 >= 4 (K >= 2 层), 实际 {obs.shape[1]}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_stratified: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_stratified: 输入含负值")
    K = obs.shape[1] // 2
    strata = []
    for i in range(K):
        block = obs[:, 2 * i:2 * i + 2]
        ni = float(block.sum())
        if ni < 2:
            raise ValueError(f"template_stratified: 第 {i} 层 N = {ni} < 2")
        strata.append((float(block[0, 0]), float(block[0, 1]),
                       float(block[1, 0]), float(block[1, 1]), ni))
    return strata


def chi2_pvalue(observed):
    """Mantel-Haenszel 组合检验: 返回 X2_MH ~ chi2(1) 的 p 值."""
    strata = _strata(observed)
    num, den = 0.0, 0.0
    for a, b, c, d, ni in strata:
        num += a - (a + b) * (a + c) / ni
        den += (a + b) * (a + c) * (b + d) * (c + d) / (ni * ni * (ni - 1))
    if den <= 0:
        raise ValueError("template_stratified: ΣVar_i = 0 (各层均无关联信息)")
    x2 = num * num / den
    return float(chi2.sf(x2, 1))


def mantel_haenszel_or(observed):
    """组合比值比 OR_MH = Σ(a_i d_i/N_i) / Σ(b_i c_i/N_i). 分母为 0 -> ValueError (未定义)."""
    strata = _strata(observed)
    num, den = 0.0, 0.0
    for a, b, c, d, ni in strata:
        num += a * d / ni
        den += b * c / ni
    if den == 0:
        raise ValueError("template_stratified: OR_MH 未定义 (Σ b_i c_i / N_i = 0)")
    return float(num / den)
