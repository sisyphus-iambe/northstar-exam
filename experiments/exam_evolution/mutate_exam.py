"""考题(表)变异操作器 — 考题库进化实验 (2026-08-11).

进化组件①: 变异操作器. 对考卷对拍表做确定性变换, 生成新考题.
9 类操作 = 改边界/退化行为/组合考点 (立项口径), 全部作用于表形态.

设计约束 (与正式判定口径对齐, spsl/run.py):
  - 合法表 (边缘全 > 0): 进 L2/L3 对拍区. 变异后必须保持合法, 否则好程序
    抛异常 (L2 语义 = run 崩溃全拒), 偏离判定失真.
  - 病态表 = 合法但数值敏感 (边缘=1/极大计数/对角集中): 正确实现也可能
    浮点偏离 > 1e-6 — 这是自由线自举漂移的温床 (本实验刻意包含).
  - 确定性: 全部操作纯 numpy 变换 + 固定 rng seed, 无随机漂移 (北极星纪律).

迁移自 AGI Brain V3 (run_v3_cells.py): 继承父代偏好 + 微突变的同构 —
  本模块的 apply_ops(t, ops) = 父题上叠加操作 (遗传), gen_pool 首代从
  种子表池抽基表 (init), 后续代从入选父表继承 (inherit).
"""
import numpy as np

# ---------------------------------------------------------------------------
# 基础操作 (全部纯 numpy, 保持边缘全 > 0)
# ---------------------------------------------------------------------------

def op_scale_up(t, rng, k=None):
    """M9 大样本: 整表 ×k (k 随机 1e3~1e4). 浮点敏感档."""
    k = k or int(10 ** rng.uniform(3, 4))
    return t * k


def op_scale_down(t, rng, k=None):
    """M1 边界缩放: 整表 /k → 小计数表 (k 随机 2~10, 下限 1)."""
    k = k or int(10 ** rng.uniform(0.3, 1))
    return np.maximum(t // k, 1)


def op_flip_rows(t, rng):
    """M2 行序反转 (行顺序敏感错误的探针)."""
    return t[::-1, :]


def op_flip_cols(t, rng):
    """M2 列序反转."""
    return t[:, ::-1]


def op_shrink_row(t, rng):
    """M3 退化边缘: 随机一行压到最小非零 (该行计数全并入另一行)."""
    r, c = t.shape
    i = int(rng.integers(0, r))
    keep = t.sum(axis=1).argmax()
    if i == keep:
        i = (i + 1) % r
    out = t.copy()
    merged = out[i] + out[keep]
    out[i] = np.ones(c, dtype=t.dtype)
    out[keep] = merged - out[i]
    return out


def op_zero_cell(t, rng):
    """M4 零单元格注入: 随机格置 0 (保持行列边缘 > 0, 与 L2 手选表同款形态)."""
    r, c = t.shape
    out = t.copy()
    for _ in range(50):
        i, j = int(rng.integers(0, r)), int(rng.integers(0, c))
        if out[i, j] > 0 and out[i].sum() > 1 and out[:, j].sum() > 1:
            out[i, j] = 0
            return out
    return out  # 无法安全置零则原样返回 (变异失败, 保持合法)


def op_diag_concentrate(t, rng):
    """M5 极端分布: 对角集中 95~99% (强相关表, 病态档)."""
    r, c = t.shape
    d = min(r, c)
    if d < 2:
        return t
    out = np.zeros_like(t)
    frac = rng.uniform(0.95, 0.99)
    total = t.sum()
    diag_total = int(total * frac)
    base = max(1, diag_total // d)
    for i in range(d):
        out[i, i] = base
    out[0, 0] += diag_total - base * d   # 对角余数归 (0,0)
    out[0, 0] += int(total - diag_total) # 非对角总量也归 (0,0)
    # 保证每行每列 > 0 (对角已覆盖 min(r,c) 行/列, 剩余行/列补 1)
    for j in range(c):
        if out[:, j].sum() == 0:
            out[0, j] = 1
    for i in range(r):
        if out[i].sum() == 0:
            out[i, 0] = 1
    return out


def op_binary(t, rng):
    """M7 全 0/1 表 (极小样本, 病态档)."""
    r, c = t.shape
    return np.ones((r, c), dtype=t.dtype)


def op_shape_extend(t, rng):
    """M6 形状变换: 加一行/一列 (全 1 边缘, 保持合法)."""
    r, c = t.shape
    out = np.zeros((r + 1, c + 1), dtype=t.dtype)
    out[:r, :c] = t
    out[:r, c] = 1
    out[r, :] = 1
    return out


def op_combo(t, rng):
    """M8 组合考点: 随机选 2 个其他操作叠加."""
    ops = [op_zero_cell, op_shrink_row, op_flip_rows, op_flip_cols, op_scale_down]
    a, b = rng.choice(len(ops), size=2, replace=False)
    t2 = ops[a](t, rng)
    return ops[b](t2, rng)


OPS = {
    "scale_up": op_scale_up,
    "scale_down": op_scale_down,
    "flip_rows": op_flip_rows,
    "flip_cols": op_flip_cols,
    "shrink_row": op_shrink_row,
    "zero_cell": op_zero_cell,
    "diag_concentrate": op_diag_concentrate,
    "binary": op_binary,
    "shape_extend": op_shape_extend,
    "combo": op_combo,
}


def legal_table(t) -> bool:
    """合法表 = 2D 有限非负整型, 每行每列和 > 0 (L2 断言同款)."""
    a = np.asarray(t)
    return (
        a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] >= 2
        and np.all(np.isfinite(a)) and np.all(a >= 0)
        and np.all(a.sum(axis=1) > 0) and np.all(a.sum(axis=0) > 0)
    )


def apply_ops(t, ops, rng) -> np.ndarray:
    """按序叠加多个操作. 任一操作破坏合法性 -> 回退到该操作前 (诚实失败)."""
    cur = np.array(t)
    applied = []
    for name in ops:
        nxt = OPS[name](cur, rng)
        if legal_table(nxt):
            cur = nxt
            applied.append(name)
    return cur, applied


# 操作名集合 (组合操作内部再展开, 不参与单步选择)
STEP_OPS = [k for k in OPS if k != "combo"]
