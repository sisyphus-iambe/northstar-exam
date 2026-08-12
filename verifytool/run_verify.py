"""验证器质检编排 — 四层考卷 (L1-L4) + 能力地图 (9 模板 x L2-L4) + 盲点清单.

只 import calibrator/ 现有模块 (纯组装, 零新发明), 全部固定种子确定性:
  - 四层考卷: calibrator.calibrate.calibrate(func, seed), 默认 3 runs
    seeds = 20260807/20260808/20260809 (规格 §2.2, 同实验 05 seed_base)
  - 层判定协议: 同 run_experiment.py 100 runs 协议按 run 数归一
    (<=10% 拒 = PASS, >=90% 拒 = REJECT, 其余 MIXED; 总判定同阈值,
    出处: experiments/qualcheck_realworld/results.json comparison_rule)
  - 能力地图: 9 模板考卷表 (calibrator/template_exam.py) x 对拍参照
    (calibrator/reference.py 的 ref_hand + ref_scipy, 参照自检 1e-9,
    容差 1e-6 与 l2.L2_TOL 同值) — 规格 §2.3
  - 盲点清单: L4 幻觉填补 / L2/L3 对拍偏差 top5 / 覆盖边界 (规格 §2.4)
"""
import sys
import time
from pathlib import Path

import numpy as np

# 让 `python -m verifytool` 从任意 cwd 都能 import calibrator (calib_exp/calibrator)
_CALIB_EXP = Path(__file__).resolve().parent.parent
if str(_CALIB_EXP) not in sys.path:
    sys.path.insert(0, str(_CALIB_EXP))

from calibrator import template_exam  # noqa: E402
from calibrator import l2 as _l2, l4 as _l4  # noqa: E402
from calibrator.calibrate import calibrate  # noqa: E402  (calibrate 函数)
from calibrator.l2 import build_l2_tables  # noqa: E402
from calibrator.l3 import build_l3_tables  # noqa: E402
from calibrator.reference import REF_AGREE_TOL, ref_hand, ref_scipy  # noqa: E402
from verifytool import VERSION  # noqa: E402

SEED_BASE = 20260807          # 规格 §2.2: seeds = 20260807 + i
DEFAULT_RUNS = 3
L2_TOL = _l2.L2_TOL           # 1e-6, 对拍容差 (l2.py 同值)
PASS_RATIO = 0.10             # 层判定: 拒绝计数占比 <= 10% -> PASS
REJECT_RATIO = 0.90           # 层判定: 拒绝计数占比 >= 90% -> REJECT
TIME_WARN_SECONDS = 120.0     # 规格 §6: 单验证器 < 2 分钟, 超出如实提示

# L4 9 类畸形输入的形态描述 (出处: calibrator/l4.py _all_inputs 逐类对应)
L4_INPUT_SHAPES = {
    "nan_cell_4x4": "4x4 表, 单格 NaN",
    "nan_row_3x5": "3x5 表, 整行 NaN",
    "inf_cell_3x3": "3x3 表, 单格 +inf",
    "neginf_cell_2x4": "2x4 表, 单格 -inf",
    "all_zero_4x4": "4x4 全零表",
    "single_row_1x5": "1x5 单行退化表",
    "single_col_5x1": "5x1 单列退化表",
    "string_mixed": "2x2 混合字符串/None (非数值)",
    "empty_0x0": "0x0 空表",
}


# ---------------------------------------------------------------------------
# 层判定协议 (run 数归一, 同实验 05 comparison_rule)
# ---------------------------------------------------------------------------

def layer_verdict(rej: int, runs: int) -> str:
    """PASS/REJECT/MIXED. 阈值: <=10% 拒 PASS, >=90% 拒 REJECT (规格 §2.2)."""
    if rej <= PASS_RATIO * runs:
        return "PASS"
    if rej >= REJECT_RATIO * runs:
        return "REJECT"
    return "MIXED"


def total_verdict(reject_runs: int, runs: int) -> str:
    """总判定: ACCEPT/REJECT/MIXED (同 L1 阈值, results.json comparison_rule)."""
    if reject_runs <= PASS_RATIO * runs:
        return "ACCEPT"
    if reject_runs >= REJECT_RATIO * runs:
        return "REJECT"
    return "MIXED"


# ---------------------------------------------------------------------------
# 阶段 1: 四层考卷, 多 run 汇总
# ---------------------------------------------------------------------------

def run_four_layers(func, runs: int):
    per_run = []
    for i in range(runs):
        seed = SEED_BASE + i
        try:
            r = calibrate(func, seed)  # calibrator/calibrate.py: L1+L2+L3+L4
            r["seed"] = seed
        except Exception as exc:
            # 候选在合法抽样表上抛异常 -> 该 run 全层记拒 (诚实失败)
            r = {"seed": seed, "crash": True,
                 "error": f"{type(exc).__name__}: {exc}",
                 "verdict": False, "L1": False, "L2": False, "L3": False, "L4": False}
        per_run.append(r)

    counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for r in per_run:
        for layer in counts:
            if not r[layer]:
                counts[layer] += 1
    reject_runs = sum(1 for r in per_run if not r["verdict"])

    layers = {
        layer: {"verdict": layer_verdict(counts[layer], runs),
                "rej": counts[layer], "runs": runs}
        for layer in ("L1", "L2", "L3", "L4")
    }
    return {
        "n_runs": runs,
        "seeds": [SEED_BASE + i for i in range(runs)],
        "reject_runs": reject_runs,
        "layers": layers,
        "total_verdict": total_verdict(reject_runs, runs),
        "per_run": per_run,
        "first": per_run[0],
    }


# ---------------------------------------------------------------------------
# 阶段 2: 能力地图 (9 模板 x L2-L4, 固定考卷)
# ---------------------------------------------------------------------------

def _pair_cell(func, tables: list) -> dict:
    """与 calibrator/template_exam.py _pair_grade 同协议: 参照自检 1e-9,
    候选 vs 两参照 (ref_hand/ref_scipy, 均为卡方参照) 最大偏差 <= 1e-6 = PASS.
    候选抛异常 = 对拍失败 (考卷表均为合法输入)."""
    worst_ref_dev = max(abs(ref_hand(t) - ref_scipy(t)) for t in tables)
    if worst_ref_dev > REF_AGREE_TOL:
        return {"verdict": "ABORT", "ref_abort": True,
                "worst_ref_dev": worst_ref_dev, "n_tables": len(tables)}
    devs = []
    for t in tables:
        try:
            cand_p = float(func(t))
        except Exception:
            cand_p = float("nan")
            devs.append(float("inf"))
            continue
        devs.append(max(abs(cand_p - ref_hand(t)), abs(cand_p - ref_scipy(t))))
    max_dev = float(max(devs))
    return {
        "verdict": "PASS" if max_dev <= L2_TOL else "FAIL",
        "ref_abort": False,
        "worst_ref_dev": worst_ref_dev,
        "max_dev": max_dev,
        "n_tables": len(tables),
        "n_viol": int(sum(d > L2_TOL for d in devs)),
    }


def build_capability_map(func) -> list:
    """9 模板 (TEMPLATE_ORDER, calibrator/template_exam.py) x L2/L3 对拍.
    L4 与模板无关 (同一 9 类畸形输入), 由调用方填入共享结果."""
    rows = []
    for tpl in template_exam.TEMPLATE_ORDER:
        domain = template_exam.TEMPLATE_DOMAIN[tpl]
        rows.append({
            "template": tpl,
            "domain": domain,
            "L2": _pair_cell(func, template_exam.build_l2_tables_domain(domain)),
            "L3": _pair_cell(func, template_exam.build_l3_tables_domain(domain)),
        })
    return rows


# ---------------------------------------------------------------------------
# 阶段 3: 盲点清单 (零 LLM, 全自动)
# ---------------------------------------------------------------------------

def _top_deviating(func, tables: list, top_k: int = 5) -> list:
    """对拍偏差最大的 top_k 张表 (n / 形状 / 偏差值 / 候选 p / 参照 p)."""
    items = []
    for t in tables:
        try:
            cand_p = float(func(t))
        except Exception:
            cand_p = float("nan")
            dev = float("inf")
        else:
            dev = max(abs(cand_p - ref_hand(t)), abs(cand_p - ref_scipy(t)))
        items.append({
            "table": np.asarray(t).tolist(),
            "shape": list(np.asarray(t).shape),
            "n": int(np.asarray(t).sum()),
            "dev": dev,
            "cand_p": cand_p,
            "ref_p": float(ref_hand(t)),
        })
    items.sort(key=lambda x: x["dev"], reverse=True)
    return items[:top_k]


def _covered_shapes() -> list:
    """固定考卷覆盖的形状集合 (全局 L2/L3 + 9 模板各域 L2/L3, 全确定性)."""
    shapes = set()
    for t in build_l2_tables() + build_l3_tables():
        shapes.add(tuple(t.shape))
    for domain in ("rxc", "2x2", "2xk", "2x4", "2x2K"):
        for t in (template_exam.build_l2_tables_domain(domain)
                  + template_exam.build_l3_tables_domain(domain)):
            shapes.add(tuple(np.asarray(t).shape))
    return sorted(shapes)


def build_blind_spots(func, four: dict) -> dict:
    spots = {"l4_hallucination": [], "pair_dev": {}, "coverage": {}}

    # 1) L4 命中: 9 类畸形输入中哪些返回了有限值 (幻觉填补)
    first = _diag_run(four)
    l4_diag = first.get("L4_diag") or {"per_input": []}
    for it in l4_diag["per_input"]:
        if it["finite"]:
            spots["l4_hallucination"].append({
                "name": it["name"],
                "shape": L4_INPUT_SHAPES.get(it["name"], "?"),
                "returned": it["returned"],
            })

    # 2) L2/L3 对拍 FAIL 的表型: 偏差最大的 5 张表
    for layer, builder in (("L2", build_l2_tables), ("L3", build_l3_tables)):
        diag = first.get(f"{layer}_diag")
        if diag is None:
            spots["pair_dev"][layer] = {"ref_abort": False, "n_viol": -1, "top": [],
                                        "note": "候选在合法考卷表上抛异常, 无对拍诊断"}
            continue
        if diag.get("ref_abort"):
            spots["pair_dev"][layer] = {"ref_abort": True,
                                        "worst_ref_dev": diag["worst_ref_dev"]}
        elif diag["n_viol"] > 0:
            spots["pair_dev"][layer] = {
                "ref_abort": False,
                "n_viol": diag["n_viol"],
                "top": _top_deviating(func, builder()),
            }
        else:
            spots["pair_dev"][layer] = {"ref_abort": False, "n_viol": 0, "top": []}

    # 3) 覆盖边界: 固定考卷无法覆盖的形状类别 (工具边界, 同实验 05 诚实标注)
    covered = _covered_shapes()
    max_r = max(s[0] for s in covered)
    max_c = max(s[1] for s in covered)
    spots["coverage"] = {
        "covered_shapes": [list(s) for s in covered],
        "max_rows": max_r,
        "max_cols": max_c,
        "uncovered": [
            f"行数 > {max_r} 或列数 > {max_c} 的大表",
            "1xN / Nx1 单行/单列退化表 (协议排除: scipy 对 dof=0 行为异常, 见 l4.py 边界声明)",
            "样本量 n > 20000 的超大表 (考卷最大 n = 20000)",
            "非整数/连续计数表 (考卷全为整数计数)",
        ],
    }
    return spots


# ---------------------------------------------------------------------------
# 总编排
# ---------------------------------------------------------------------------

def _diag_run(four: dict) -> dict:
    """取首个未崩溃 run 用于诊断 (L2/L3/L4 固定考卷, 未崩溃 run 间确定性一致)."""
    for r in four["per_run"]:
        if "crash" not in r:
            return r
    return four["first"]


def run_verify(func, runs: int = DEFAULT_RUNS) -> dict:
    """对候选验证器跑完整质检. 返回结果 dict (供 report.py 渲染)."""
    t0 = time.monotonic()
    four = run_four_layers(func, runs)
    t1 = time.monotonic()

    cap_map = build_capability_map(func)
    # L4 与模板无关 (同一 9 类畸形输入, l4.py): 共享四层考卷的 L4 结果
    l4_diag = _diag_run(four).get("L4_diag") or {
        "n_inputs": 9, "n_fail": 9, "note": "候选在合法考卷表上抛异常, L4 无法完成"}
    l4_verdict = "PASS" if l4_diag.get("n_fail", 9) == 0 else "REJECT"
    for row in cap_map:
        row["L4"] = {
            "verdict": l4_verdict,
            "n_inputs": l4_diag["n_inputs"],
            "n_fail": l4_diag["n_fail"],
        }
    t2 = time.monotonic()

    blind = build_blind_spots(func, four)
    t3 = time.monotonic()

    elapsed = t3 - t0
    return {
        "tool": "verifytool",
        "version": VERSION,
        "spec": "SPEC_MVP工具形态_2026-08-07.md",
        "four_layers": four,
        "capability_map": cap_map,
        "blind_spots": blind,
        "elapsed_seconds": round(elapsed, 3),
        "stage_seconds": {
            "four_layers": round(t1 - t0, 3),
            "capability_map": round(t2 - t1, 3),
            "blind_spots": round(t3 - t2, 3),
        },
        "time_warn": elapsed > TIME_WARN_SECONDS,
    }
