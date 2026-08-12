"""考题库进化 — 阶段 4: 进化考卷对新 AI 的漏检率评估 (2026-08-11).

评估设计 (口径如实声明):
  - 新 AI = g3/llm_impls/g3_llm_01..20 (DeepSeek v4-flash 生成, 温度轮转
    0.3~1.3 × 5 组, 未参与考题进化 — "未参与进化的新 AI").
  - 真值 = 逐表对拍金标准: 对 L2+L3 全部 116 张表, 与双参照 (ref_hand_chi2 /
    ref_scipy_chi2) 任意表 max|Δp| > 1e-6 = 真坏 (实现与教科书公式不一致).
    该口径与 README 的 LLM 违背率 (14.8%) 同源 (golden 参照).
  - 漏检率 = 真坏 AI 中被四层判定 ACCEPT 的比例.
  - 误杀率 = 真好 AI 中被判定 REJECT 的比例.
  - 三组考卷对比: 原考卷 (基线) / 原+锚定线 top-30 / 原+自由线 top-30.
    进化考卷 = 原 L2/L3 116 表 + 进化 30 表 (追加不替换, 测边际贡献).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from spsl.envelope import load_exam
from spsl.golden import get_ref
from spsl.run import run_four_layers
from verifytool.loader import load_validator

ROOT = Path(__file__).resolve().parents[2]          # northstar_v2/ (本脚本在 experiments/exam_evolution/)
OUT_DIR = Path(__file__).resolve().parent / "out"
EXAM = ROOT / "exams" / "exam_pearson_full.json"
LLM_DIR = ROOT / "experiments" / "g3" / "llm_impls"
TOL = 1e-6


def load_llm_programs() -> tuple[dict, list]:
    """加载 g3 20 份新 AI 程序. 返回 ({name: func}, 加载失败清单)."""
    funcs, failed = {}, []
    for p in sorted(LLM_DIR.glob("g3_llm_*.py")):
        try:
            fn, _ = load_validator(str(p), func_name="chi2_pvalue")
            funcs[p.stem] = fn
        except Exception as exc:
            failed.append({"name": p.stem, "error": f"{type(exc).__name__}: {exc}"})
    return funcs, failed


def table_deviation(fn, t, refs) -> float:
    """与双参照 max|Δp| (抛异常 = inf). 参照契约: {"table": ...} dict."""
    item = {"table": np.asarray(t, dtype=float).tolist()}
    try:
        p = float(fn(np.array(t, dtype=float)))
    except Exception:
        return float("inf")
    return max(abs(p - float(ref(item))) for ref in refs)


def gold_truth(funcs, exam, refs) -> dict:
    """逐表对拍金标准. 返回 {name: {bad, max_dev, n_viol}}."""
    tables = [np.asarray(inp["table"], dtype=float)
              for inp in exam["l2"]["inputs"] + exam["l3"]["inputs"]]
    truth = {}
    for name, fn in funcs.items():
        devs = [table_deviation(fn, t, refs) for t in tables]
        truth[name] = {"bad": any(d > TOL for d in devs),
                       "max_dev": float(max(devs)),
                       "n_viol": int(sum(d > TOL for d in devs)),
                       "n_tables": len(tables)}
    return truth


def build_evolved_exam(exam, tables: list, tag: str) -> dict:
    """原考卷 + 进化表 (追加 L2/L3 inputs). 内存内使用, 不验签 (新考卷非发布卷)."""
    e = copy.deepcopy(exam)
    for layer in ("l2", "l3"):
        sec = e[layer]
        extras = [{"type": "contingency_table", "table": t.tolist()} for t in tables]
        sec["inputs"] = sec["inputs"] + extras
        sec["n_inputs"] = len(sec["inputs"])
        sec["protocol"] = (sec.get("protocol", "") +
                           f" | 进化考题 {tag}: +{len(tables)} 表 "
                           "(exam_evolution 2026-08-11)")
    return e


def main() -> int:
    exam = load_exam(EXAM)
    refs = [get_ref(n) for n in exam["l2"]["refs"]]

    funcs, failed = load_llm_programs()
    truth = gold_truth(funcs, exam, refs)
    bad_ai = sorted(n for n, t in truth.items() if t["bad"])
    good_ai = sorted(n for n, t in truth.items() if not t["bad"])
    print(f"[eval] new AI: {len(funcs)} programs (load failures {len(failed)}, "
          f"failed list: {[f['name'] for f in failed]})")
    print(f"[eval] gold truth: {len(bad_ai)} bad / {len(good_ai)} good "
          f"(per-table pairwise 116 tables, tol={TOL})")
    print(f"[eval] bad list: {bad_ai}")

    evo = json.loads((OUT_DIR / "evolution.json").read_text(encoding="utf-8"))
    tables_a = [np.asarray(s["table"], dtype=float)
                for s in evo["final"]["anchored"]]
    tables_f = [np.asarray(s["table"], dtype=float)
                for s in evo["final"]["free"]]

    exams = {
        "original": exam,
        "anchored": build_evolved_exam(exam, tables_a, "anchored"),
        "free": build_evolved_exam(exam, tables_f, "free"),
    }

    results = {}
    for tag, e in exams.items():
        verdicts = {}
        for name, fn in funcs.items():
            r = run_four_layers(fn, e)
            verdicts[name] = {"total": r["total_verdict"],
                              "reject_runs": r["reject_runs"],
                              "n_runs": r["n_runs"],
                              "L1": r["layers"]["L1"]["verdict"],
                              "L2": r["layers"]["L2"]["verdict"],
                              "L3": r["layers"]["L3"]["verdict"],
                              "L4": r["layers"]["L4"]["verdict"]}
        leaked = [n for n in bad_ai if verdicts[n]["total"] == "ACCEPT"]
        killed = [n for n in good_ai if verdicts[n]["total"] == "REJECT"]
        miss = len(leaked) / max(1, len(bad_ai))
        fp = len(killed) / max(1, len(good_ai))
        results[tag] = {
            "leak_rate": round(miss, 4),
            "false_pos_rate": round(fp, 4),
            "leaked": leaked,
            "killed": killed,
            "verdicts": verdicts,
            "n_inputs": {layer: e[layer]["n_inputs"] for layer in ("l2", "l3")},
        }
        print(f"[eval] {tag:9s}: leak_rate={miss:.3f} ({len(leaked)}/{len(bad_ai)}) "
              f"fp_rate={fp:.3f} ({len(killed)}/{len(good_ai)}) "
              f"| L2={e['l2']['n_inputs']} L3={e['l3']['n_inputs']} tables")

    payload = {
        "gold": truth,
        "failed_load": failed,
        "exam": {"path": str(EXAM), "n_l2": len(exam["l2"]["inputs"]),
                 "n_l3": len(exam["l3"]["inputs"])},
        "results": results,
    }
    (OUT_DIR / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] done -> {OUT_DIR / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
