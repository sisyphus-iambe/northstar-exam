#!/usr/bin/env python3
"""北极星公开仓库 CI — 批量确定性: 仓库内真实可运行组合, 同参两遍逐字节一致.

判定数字全部来自 spsl.run 既有考官 (statistical 四层 / demo_data / conclusion_anchor),
本脚本只做: 同参两遍 subprocess -> 读回判定 JSON -> 剥离非确定字段
(elapsed_seconds / command / payload_md5 / self_md5, 按判定 JSON 实际结构
逐键 pop, 缺失即无) -> 规范化序列化 (ensure_ascii=False, sort_keys=True,
indent=2) -> 逐字节对比 + md5 一致断言 -> 证据落盘 out/batch_evidence/.

组合 = 仓库内真实存在且可运行:
  statistical:      exams/exam_pearson_full.json + verifytool/programs_real/handwritten_good.py
  inv:              exams/exam_wsr_inv.json + candidates/wsr_flip.py (恒等式层, 主分发)
  inv:              exams/exam_ranksum_inv.json + candidates/ranksum_flip.py (恒等式层, 主分发)
  demo_data:        specs/spec_demo_data.json (无候选, 检测器自身即被测对象)
  conclusion_anchor: specs/spec_conclusion_anchor.json (无候选)
(state_estimator 无仓库内候选, 不入队; exam_wilcoxon_demo.json 不存在)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "out" / "batch_evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)

# 非确定字段 (spsl/run.py:23-24 声明; demo_data/conclusion_anchor 无 elapsed/command)
_NONDET = ("elapsed_seconds", "command", "payload_md5", "self_md5")

PAIRS = [
    {"name": "statistical_pearson", "family": "statistical",
     "exam": "exams/exam_pearson_full.json",
     "candidate": "verifytool/programs_real/handwritten_good.py"},
    {"name": "inv_wsr_flip", "family": "inv",
     "exam": "exams/exam_wsr_inv.json",
     "candidate": "candidates/wsr_flip.py"},
    {"name": "inv_ranksum_flip", "family": "inv",
     "exam": "exams/exam_ranksum_inv.json",
     "candidate": "candidates/ranksum_flip.py"},
    {"name": "demo_data", "family": "demo_data",
     "spec": "specs/spec_demo_data.json", "candidate": None},
    {"name": "conclusion_anchor", "family": "conclusion_anchor",
     "spec": "specs/spec_conclusion_anchor.json", "candidate": None},
]


def run_once(pair: dict, run_tag: str) -> Path:
    """同参单遍: spsl.run subprocess -> 判定 JSON 路径."""
    out = EVIDENCE / f"{pair['name']}_{run_tag}.json"
    if pair["family"] in ("statistical", "inv"):
        cmd = [sys.executable, "-m", "spsl.run",
               str(ROOT / pair["exam"]), str(ROOT / pair["candidate"]),
               "--out", str(out)]
    else:
        cmd = [sys.executable, "-m", "spsl.run",
               str(ROOT / pair["spec"]), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        raise RuntimeError(f"{pair['name']} rc={proc.returncode}: " + " | ".join(tail))
    return out


def strip_nondet(path: Path) -> bytes:
    """读回判定 JSON, 剥离非确定字段, 规范化序列化 -> 字节."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in _NONDET:
        payload.pop(key, None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      indent=2).encode("utf-8")


def main() -> int:
    rows, all_equal = [], True
    for pair in PAIRS:
        p1, p2 = run_once(pair, "run1"), run_once(pair, "run2")
        b1, b2 = strip_nondet(p1), strip_nondet(p2)
        m1, m2 = hashlib.md5(b1).hexdigest(), hashlib.md5(b2).hexdigest()
        same = (b1 == b2) and (m1 == m2)
        all_equal = all_equal and same
        rows.append({"pair": pair["name"], "family": pair["family"],
                     "exam": pair.get("exam"), "spec": pair.get("spec"),
                     "candidate": pair["candidate"],
                     "bytes_identical": bool(b1 == b2),
                     "md5_run1": m1, "md5_run2": m2,
                     "n_bytes": len(b1)})
        print(f"{pair['name']:<20} byte_cmp={'SAME' if same else 'DIFF'} "
              f"md5={m1[:16]}...")

    out = {"tool": "batch_determinism", "layer": "L4-CI",
           "method": "同参两遍 subprocess + 剥离非确定字段 (elapsed_seconds/"
                     "command/payload_md5/self_md5) 后规范化逐字节 cmp + md5 一致断言",
           "pairs": rows, "n_pairs": len(rows),
           "all_equal": bool(all_equal),
           "overall": "PASS" if all_equal else "FAIL"}
    (ROOT / "out" / "batch_determinism.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"overall={'PASS' if all_equal else 'FAIL'} ({len(rows)} 对)")
    print(f"JSON -> {ROOT / 'out' / 'batch_determinism.json'}")
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
