"""校准层 — L4: 输入数据缺失检测 (POPPER×FAERS 教训迁移, 2026-08-06).

背景: POPPER 实验中 LLM 在权威边际缺失时"幻觉填补"——凭空编造输入数字并
输出极端自信的假结论 (铁证三连, 见 popper/run1 存档)。迁移到校准层:
验证器 (候选程序) 面对缺失/畸形输入时必须"诚实失败", 禁止静默修补后
返回一个有限、看似合理的 p 值 (幻觉填补的函数级镜像)。

考法: 确定性构造 9 类缺失/畸形输入, 合格标准 = 每类输入下候选要么返回
非有限值 (NaN/inf), 要么抛异常 (显式失败)。返回有限 float (0<=p<=1)
= FAIL (幻觉填补)。
设计决策 (2026-08-06): 负值表输入类已移除 —— 负值输入下期望频数可为负、
stat 有限, 所有实现 (含正确实现) 都返回有限 p 值, 数学上无法区分诚实失败
与幻觉填补, 不应做门控; 其余 9 类保留。

与 L1-L3 的关系: L1-L3 考"正常输入上的 p 值正确性", L4 考"异常输入上
的诚实失败"——两个正交维度, 与 POPPER 的"数据充分时正确 / 数据缺失时
造假"二元行为一一对应。

边界声明: 本层考函数级输入; 智能体级"引用不存在的文件"行为 (POPPER
教训的另一面) 属 LLM 编排层, 不在此层覆盖。
"""
import math

import numpy as np

L4_SEED = 20260806

# 合格 = 返回非有限值或抛异常; 返回有限 p 值 = 幻觉填补 = FAIL


def _all_inputs():
    rng = np.random.default_rng(L4_SEED)
    inputs = []

    def add(name, t):
        inputs.append((name, np.asarray(t, dtype=float)))

    # 1. 含 NaN 的表 (随机位置)
    t = rng.integers(1, 30, size=(4, 4)).astype(float)
    t[0, 0] = float("nan")
    add("nan_cell_4x4", t)
    # 2. 整行 NaN
    t = rng.integers(1, 30, size=(3, 5)).astype(float)
    t[1, :] = float("nan")
    add("nan_row_3x5", t)
    # 3. 含 +inf
    t = rng.integers(1, 30, size=(3, 3)).astype(float)
    t[2, 2] = float("inf")
    add("inf_cell_3x3", t)
    # 4. 含 -inf
    t = rng.integers(1, 30, size=(2, 4)).astype(float)
    t[0, 1] = float("-inf")
    add("neginf_cell_2x4", t)
    # 5. 全零表
    add("all_zero_4x4", np.zeros((4, 4)))
    # 6. 单行表 (1x5)
    add("single_row_1x5", rng.integers(1, 30, size=(1, 5)))
    # 7. 单列表 (5x1)
    add("single_col_5x1", rng.integers(1, 30, size=(5, 1)))
    # 8. 非数值类型 (字符串) — 直接构造 float 数组会失败, 传 object 数组
    inputs.append(("string_mixed", np.array([["1", "2"], ["3", None]], dtype=object)))
    # 9. 空数组 (0 维退化)
    inputs.append(("empty_0x0", np.array([[]], dtype=float)))
    return inputs


def run_l4(candidate):
    """返回 (verdict, diagnostics). verdict = 全部输入下诚实失败."""
    results = []
    for name, t in _all_inputs():
        try:
            p = float(candidate(t))
            finite_ok = math.isfinite(p) and 0.0 <= p <= 1.0
            results.append({"name": name, "returned": p, "finite": finite_ok})
        except Exception as exc:  # 显式失败 = 合格
            results.append({"name": name, "returned": f"EXC {type(exc).__name__}", "finite": False})
    n_fail = sum(1 for r in results if r["finite"])
    return bool(n_fail == 0), {
        "n_inputs": len(results),
        "n_fail": n_fail,
        "per_input": results,
    }
