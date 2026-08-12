"""恒等式判定 — 恒等式考卷 -> 无参照判定 (北极星 v3 级 1, D2).

用法: python3 -m spsl.run_inv <恒等式考卷.json> <候选.py> [--out verdict.json]

判定语义 (round9_A §③, 镜像 spsl/run.py:215-217 阈值):
  每条恒等式: 违规占比 = viol / R
    <= 5%  -> PASS (违规容忍, 与 L1 拒绝占比 <=10% 同哲学, 更严)
    >= 90% -> REJECT
    其余   -> MIXED
  总判定: 任意条目 REJECT -> REJECT; 任意 MIXED -> MIXED; 全 PASS -> PASS.
  (恒等式层并入总判定: 任意层 REJECT 即总 REJECT)

候选加载: 规格 functions 声明函数集 (入口名 -> 候选模块内函数名),
每入口 getattr + callable 检查 (单函数契约的显式扩展, loader.py:35-42 保留为基线).
输出: verdict JSON + payload_md5/self_md5 (verifytool.report.save_json 同款),
四层 detail 含诊断数字 (硬前提 1 同款: viol/R/worst_dev 每条目落盘).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (verifytool/ lives here)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verifytool.report import save_json  # noqa: E402

from spsl import VERSION  # noqa: E402
from spsl.axioms import check_invariant  # noqa: E402
from spsl.compile_inv import load_inv_exam  # noqa: E402

PASS_FRAC = 0.05
REJECT_FRAC = 0.90


def load_entries(cand_path: str | Path, functions: dict) -> dict:
    """加载候选模块, 按规格 functions 解析入口名 -> 可调用对象."""
    p = Path(cand_path)
    spec = importlib.util.spec_from_file_location("cand_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    entries = {}
    for entry, fname in functions.items():
        if not hasattr(mod, fname):
            raise RuntimeError(f"candidate module missing entry function {fname!r} "
                               f"(entry {entry!r})")
        fn = getattr(mod, fname)
        if not callable(fn):
            raise RuntimeError(f"candidate entry {entry!r} -> {fname!r} is not callable")
        entries[entry] = fn
    return entries


def run_invariants(exam: dict, entries: dict) -> dict:
    """对考卷全部恒等式条目送考 -> {invariants: [...], total_verdict}."""
    funcs = exam["functions"]
    results = []
    any_reject = any_mixed = False
    for item in exam["invariants"]:
        a, b = item["pair"]
        fn_a = entries[a]
        # transpose 自配对: 同入口传两遍 (pair = [f, f])
        fn_b = entries[b] if a != b else fn_a
        rng = np.random.default_rng(item["seed"])
        r = check_invariant(item["name"], fn_a, fn_b, rng, item["n"],
                            item["R"], item["condition"], item["tol"])
        frac = r["viol"] / r["R"]
        verdict = ("PASS" if frac <= PASS_FRAC
                   else "REJECT" if frac >= REJECT_FRAC else "MIXED")
        if verdict == "REJECT":
            any_reject = True
        elif verdict == "MIXED":
            any_mixed = True
        results.append({**r, "pair": item["pair"], "tol": item["tol"],
                        "seed": item["seed"], "viol_frac": frac,
                        "verdict": verdict})
    total = "REJECT" if any_reject else "MIXED" if any_mixed else "PASS"
    return {"invariants": results, "total_verdict": total}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.run_inv",
        description="SPSL: invariant exam -> reference-free verdict (Northstar v3 stage 1 D2)")
    parser.add_argument("exam", help="恒等式考卷 JSON (spsl.compile_inv 产物)")
    parser.add_argument("candidate", help="候选实现 .py 路径 (含全部入口函数)")
    parser.add_argument("--out", default=None,
                        help="verdict JSON output path (default cwd/<exam>_inv_verdict_<candidate>.json)")
    args = parser.parse_args(argv)

    exam = load_inv_exam(args.exam)
    entries = load_entries(args.candidate, exam["functions"])
    t0 = time.monotonic()
    res = run_invariants(exam, entries)
    elapsed = round(time.monotonic() - t0, 3)

    out = (Path(args.out) if args.out
           else Path.cwd() / f"{exam['name']}_inv_verdict_{Path(args.candidate).stem}.json")
    payload = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "INV",
        "exam": {"path": str(Path(args.exam).resolve()), "name": exam["name"],
                 "spec_md5": exam["spec_md5"], "content_md5": exam["content_md5"]},
        "candidate": {"path": str(Path(args.candidate).resolve()),
                      "module": Path(args.candidate).stem,
                      "entries": exam["functions"]},
        "command": " ".join(sys.argv),
        "elapsed_seconds": elapsed,
        "invariants": res["invariants"],
        "total_verdict": res["total_verdict"],
    }
    md5s = save_json(payload, out)

    print(f"[spsl] invariant exam {exam['name']} — {Path(args.candidate).stem}, "
          f"elapsed {elapsed} s")
    for r in res["invariants"]:
        print(f"  {r['invariant']:<24} {r['condition']:<20} "
              f"viol {r['viol']}/{r['R']} ({r['viol_frac']:.3%}) "
              f"worst {r['worst_dev']:.3e} -> {r['verdict']}")
    print(f"  total: {res['total_verdict']}")
    print(f"[spsl] JSON -> {out}")
    print(f"[spsl] payload_md5={md5s['payload_md5']} self_md5={md5s['self_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
