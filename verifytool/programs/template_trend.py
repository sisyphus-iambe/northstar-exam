"""模板 #3: Cochran-Armitage 趋势检验 (2 x k) — 手写实现.

输入维序约定: 2 x k 计数表 (k >= 3):
  行 0 = 阳性计数 (事件/暴露), 行 1 = 阴性计数 (非事件/非暴露);
  列 j (j = 0..k-1) = 有序水平 (时间或剂量), 默认评分 w_j = j (等距).
  等距评分即可; z 对评分 w -> alpha*w + beta (alpha>0) 的仿射变换不变,
  故任何线性单调评分给出同一检验.

零假设 H0: 各列阳性比例相等 (无线性趋势). 备择 (双侧): 阳性比例随水平线性上升或下降.

公式 (score 统计量, R_j = 行 0 第 j 列计数, N_j = 第 j 列总数, R = ΣR_j, N = ΣN_j, p_hat = R/N):
  T    = Σ_j w_j (R_j - N_j p_hat)
  VarT = p_hat (1 - p_hat) [ Σ_j w_j^2 N_j - (Σ_j w_j N_j)^2 / N ]
  z    = T / sqrt(VarT)  ~ N(0,1) (H0 下渐近)
  双侧 p = 2 * norm.sf(|z|)
  等价闭式 (交叉核对用): z^2 = N^3 T^2 / (R S [N Σ w_j^2 N_j - (Σ w_j N_j)^2]),  S = N - R

诚实失败 (L4 考法): 非 2 行 / 列数 k < 3 (两点不能定义趋势) / 含 NaN±inf / 负值 /
  无结果变异 (p_hat ∈ {0,1}) / 方差为 0 (仅一个水平有数据) -> 抛 ValueError.
  零计数列 (N_j = 0) 允许 (贡献为 0, 数学良定义).

确定性: 纯公式, 无随机数.
"""
import numpy as np
from scipy.stats import norm

NAME = "template_trend"


def chi2_pvalue(observed):
    """Cochran-Armitage 趋势检验: 输入 2 x k 计数表 (k>=3), 返回双侧趋势 p 值."""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape[0] != 2:
        raise ValueError(f"template_trend: 输入必须是 2 行计数表, 实际形状 {obs.shape}")
    if obs.shape[1] < 3:
        raise ValueError(f"template_trend: 列数 k >= 3 (两点不能定义趋势), 实际 {obs.shape[1]}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_trend: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_trend: 输入含负值")
    r = obs[0, :]                     # 各水平阳性计数
    nj = obs.sum(axis=0)              # 各水平总数
    R = float(obs[0, :].sum())
    N = float(obs.sum())
    if R == 0 or R == N:
        raise ValueError("template_trend: 无结果变异 (全为阳性或全为阴性)")
    p_hat = R / N
    w = np.arange(obs.shape[1], dtype=float)
    T = float(np.sum(w * (r - nj * p_hat)))
    VarT = p_hat * (1.0 - p_hat) * float(np.sum(w ** 2 * nj) - (np.sum(w * nj) ** 2) / N)
    if VarT <= 0:
        raise ValueError("template_trend: 方差为 0 (仅一个水平有数据)")
    z = T / np.sqrt(VarT)
    return float(2.0 * norm.sf(abs(z)))
