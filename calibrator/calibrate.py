"""校准层 — 编排: 对单个候选跑 L1+L2+L3+L4 全流程 (L4 = 输入缺失检测, 2026-08-06)."""


def calibrate(candidate, seed):
    from .l1 import run_l1
    from .l2 import run_l2
    from .l3 import run_l3
    from .l4 import run_l4

    l1_ok, l1_diag = run_l1(candidate, seed)
    l2_ok, l2_diag = run_l2(candidate)
    l3_ok, l3_diag = run_l3(candidate)
    l4_ok, l4_diag = run_l4(candidate)

    verdict = bool(l1_ok and l2_ok and l3_ok and l4_ok)
    return {
        "verdict": verdict,
        "L1": l1_ok, "L2": l2_ok, "L3": l3_ok, "L4": l4_ok,
        "L1_diag": l1_diag,
        "L2_diag": l2_diag,
        "L3_diag": l3_diag,
        "L4_diag": l4_diag,
    }
