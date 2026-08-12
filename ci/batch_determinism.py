#!/usr/bin/env python3
"""北极星 v3 阶段 4 — CI 批量确定性: 全部规格 x 全部候选, 两遍运行逐字节 cmp + md5.

判定数字必须来自既有考官 (stage2 run_chain.py / stage3 conclusion_anchor.py),
本脚本只做: 同参两遍 subprocess -> 输出文件 read_bytes() 逐字节比较 ->
md5 (hashlib.md5 原始字节) 一致断言 -> 证据落盘.

对: chain 规格 x 2 候选 (正确/故意错误) + conclusion_anchor 规格 (无候选).
考官输出 JSON 无 elapsed/command/时间戳, 随机性全来自规格内种子 ->
同路径同参重跑逐字节一致. 两遍产物各留一份为证据 (逐字节相同).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMINERS = {
    "chain": (ROOT.parent / "northstar_v3_stage2" / "run_chain.py",),
    "conclusion_anchor": (ROOT.parent / "northstar_v3_stage3"
                          / "conclusion_anchor.py",),
}

PAIRS = [
    {"family": "chain", "spec": "exam_chain_listwise.json",
     "candidate": "chain_listwise_correct.py", "name": "chain_correct"},
    {"family": "chain", "spec": "exam_chain_listwise.json",
     "candidate": "chain_constant_wrong.py", "name": "chain_wrong"},
    {"family": "conclusion_anchor", "spec": "anchor_ar1.json",
     "candidate": None, "name": "anchor_ar1"},
]

EVIDENCE = ROOT / "out" / "batch_evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)


def run_once(pair: dict, run_tag: str) -> Path:
    spec = ROOT / "specs" / pair["spec"]
    out = EVIDENCE / f"{pair['name']}_{run_tag}.json"
    script = EXAMINERS[pair["family"]][0]
    cmd = [sys.executable, str(script), str(spec)]
    if pair["candidate"]:
        cmd.append(str(ROOT / "mcp" / "candidates" / pair["candidate"]))
    cmd += ["--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{pair['name']} rc={proc.returncode}: "
                           + proc.stderr.strip().splitlines()[-5:].join("\n"))
    return out


def main() -> int:
    rows, all_equal = [], True
    for pair in PAIRS:
        p1 = run_once(pair, "run1")
        p2 = run_once(pair, "run2")
        b1, b2 = p1.read_bytes(), p2.read_bytes()
        m1, m2 = hashlib.md5(b1).hexdigest(), hashlib.md5(b2).hexdigest()
        same = (b1 == b2) and (m1 == m2)
        all_equal = all_equal and same
        rows.append({"pair": pair["name"], "family": pair["family"],
                     "spec": pair["spec"], "candidate": pair["candidate"],
                     "bytes_identical": bool(b1 == b2),
                     "md5": m1, "md5_run2": m2,
                     "n_bytes": len(b1)})
        print(f"{pair['name']:<12} byte_cmp={'SAME' if same else 'DIFF'} "
              f"md5={m1[:16]}...")

    out = {"tool": "batch_determinism", "layer": "L4-CI",
           "method": "同参两遍 subprocess + read_bytes 逐字节 cmp + "
                     "hashlib.md5(原始字节) 一致断言",
           "pairs": rows, "n_pairs": len(rows),
           "all_md5_equal": bool(all_equal),
           "overall": "PASS" if all_equal else "FAIL"}
    (ROOT / "out" / "batch_determinism.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"overall={'PASS' if all_equal else 'FAIL'} ({len(rows)} 对)")
    print(f"JSON -> {ROOT / 'out' / 'batch_determinism.json'}")
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
