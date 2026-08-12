"""模板 #7: 斜率型检验 — 计数随有序行/列索引的线性趋势 (加性结构检测).

输入维序约定: 任意 2D 计数表 (r >= 2 行, c >= 2 列), 行/列语义由调用方定义.
  分支 A (2 x k, k >= 2): 行 0 = 事件, 行 1 = 非事件; 列 = 有序水平 (时间/剂量).
     Cochran-Armitage 趋势检验 — 与 template_trend 同口径 (复制其实现,
     对同一 2 x k 表 p 值逐位一致); k = 2 也允许 (两点对比 = 两比例 z 检验,
     template_trend 因 "两点不能定义趋势" 拒绝, 本模板接受).
  分支 B (r x c, r >= 3): 每格 = 事件计数. 加性线性模型 y_ij = b0 + b1·i + b2·j
     (i, j 为格索引, 中心化后互相正交), H0: b1 = b2 = 0 的 F 检验:
       y 展开按行主序 (i = 行索引, j = 列索引),  i_c = i − ī, j_c = j − j̄
       b1 = Σ(i_c·y)/Σ(i_c²),  b2 = Σ(j_c·y)/Σ(j_c²)   (正交正规方程闭式)
       RSS_red = Σy² − n·ȳ²,  RSS_full = RSS_red − b1·Σ(i_c·y) − b2·Σ(j_c·y)
       F = ((RSS_red − RSS_full)/2) / (RSS_full/(rc−3)) ~ F(2, rc−3)
     (OLS 手算, 不引 statsmodels; 与 np.linalg.lstsq 对照一致)

零假设 H0: 计数与行/列索引无线性关联 (分支 A: 各列事件率相等无线性趋势;
  分支 B: b1 = b2 = 0, 即无加性线性结构).

诚实失败 (L4 考法): 非 2D / 行数 < 2 / 列数 < 2 / 含 NaN±inf / 含负值 /
  非整数计数 / 总计数 0 / 分支 A 无结果变异 (p_hat ∈ {0,1}) 或方差为 0
  (仅一个水平有数据) -> 抛 ValueError.
  零计数列 (n_j = 0) 允许 (贡献为 0, 同 template_trend).
  分支 B 全格计数相等 (总变异 0): 零信号, 返回 p = 1.0 (数据无关联信息,
  不是幻觉填补); 完美线性拟合 (RSS_full = 0): 证据无穷, 返回 p = 0.0.

确定性: 纯公式, 无随机数.
"""
import numpy as np
from scipy.stats import f, norm

NAME = "slope"


def _ca_trend_p(obs):
    """Cochran-Armitage 趋势检验 (2 x k, k >= 2) — 与 template_trend 逐位同口径.

    T = Σ w_j (R_j − N_j p_hat),  VarT = p_hat (1 − p_hat) [Σ w_j^2 N_j − (Σ w_j N_j)^2 / N]
    z = T / sqrt(VarT) ~ N(0,1),  双侧 p = 2 * norm.sf(|z|)
    """
    r = obs[0, :]                     # 各水平阳性计数
    nj = obs.sum(axis=0)              # 各水平总数
    R = float(obs[0, :].sum())
    N = float(obs.sum())
    if R == 0 or R == N:
        raise ValueError("template_slope: 无结果变异 (全为阳性或全为阴性)")
    p_hat = R / N
    w = np.arange(obs.shape[1], dtype=float)
    T = float(np.sum(w * (r - nj * p_hat)))
    VarT = p_hat * (1.0 - p_hat) * float(np.sum(w ** 2 * nj) - (np.sum(w * nj) ** 2) / N)
    if VarT <= 0:
        raise ValueError("template_slope: 方差为 0 (仅一个水平有数据)")
    z = T / np.sqrt(VarT)
    return float(2.0 * norm.sf(abs(z)))


def _ols_linear_trend_p(obs):
    """r x c (r >= 3) 加性线性模型 F 检验: y_ij = b0 + b1·i + b2·j, H0: b1 = b2 = 0."""
    r, c = obs.shape
    n = r * c
    y = obs.ravel()                   # 行主序: k = i*c + j
    i_idx = np.arange(n) // c         # 行索引 0..r-1
    j_idx = np.arange(n) % c          # 列索引 0..c-1
    ic = i_idx - i_idx.mean()         # 中心化 -> 与 1 列正交, 且 i_c ⊥ j_c
    jc = j_idx - j_idx.mean()
    ybar = y.mean()
    syy = float(np.sum(y * y)) - n * ybar * ybar      # RSS_red (总变异)
    if syy <= 0.0:
        return 1.0                    # 全格计数相等 (浮点噪声下 <= 0): 零信号, p = 1.0
    siy = float(np.sum(ic * y))
    sjy = float(np.sum(jc * y))
    sii = float(np.sum(ic * ic))
    sjj = float(np.sum(jc * jc))
    b1 = siy / sii
    b2 = sjy / sjj
    rss_full = syy - b1 * siy - b2 * sjy
    if rss_full <= 0.0:
        return 0.0                    # 完美线性拟合 (理论 RSS_full >= 0): 证据无穷, p = 0.0
    num = syy - rss_full              # = b1·siy + b2·sjy >= 0 (拟合解释的变异)
    if num < 0.0:                     # 浮点保护 (理论 num >= 0)
        num = 0.0
    F = (num / 2.0) / (rss_full / (n - 3))
    return float(f.sf(F, 2.0, n - 3))


def chi2_pvalue(observed):
    """斜率型检验: 2 x k 走 CA 趋势 (分支 A), r x c (r>=3) 走加性线性 F 检验 (分支 B)."""
    obs = np.asarray(observed, dtype=float)   # 字符串/None -> 转换失败或产生 nan, 由后续检查拦截
    if obs.ndim != 2 or obs.shape[0] < 2:
        raise ValueError(f"template_slope: 输入必须至少 2 行, 实际形状 {obs.shape}")
    if obs.shape[1] < 2:
        raise ValueError(f"template_slope: 输入必须至少 2 列, 实际形状 {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("template_slope: 输入含 NaN 或 ±inf")
    if np.any(obs < 0):
        raise ValueError("template_slope: 输入含负值")
    if not np.all(obs == np.round(obs)):
        raise ValueError("template_slope: 输入含非整数计数 (计数表要求整数)")
    if float(obs.sum()) == 0.0:
        raise ValueError("template_slope: 表总和为 0 (无数据)")
    if obs.shape[0] == 2:
        return _ca_trend_p(obs)
    return _ols_linear_trend_p(obs)
