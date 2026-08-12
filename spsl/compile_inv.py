"""恒等式考卷编译 — 规格 invariants 字段 -> 恒等式考卷 JSON (北极星 v3 级 1, D2).

数据流 (round9_A §③): 规格 JSON(invariants) -> schema.validate_spec
  -> 本模块: 种子派生 (est/est_cover.py splitmix64 同款) + 条目冻结
  -> run_inv.py 送考判定.

考卷 JSON = 规格 (含 spec_md5) + invariants 编译条目 + content_md5
(验签与 spsl.envelope 同款: content_md5 复算, spec_md5 复算).

种子确定性: 每条恒等式一个 seed = cell_seed(seed_base, 条目序号)
(seed_base = spec.l1.seed_base, 缺省 20260807; 与 L1 种子体系同源).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spsl import VERSION
from spsl.axioms import INVARIANTS, SAMPLERS
from spsl.envelope import content_md5
from spsl.schema import spec_md5, validate_spec

if sys.version_info >= (3, 11):
    from spsl.schema import merged_l1  # noqa: F401 (种子基座)


def _seed_base(spec: dict) -> int:
    l1 = spec.get("l1") or {}
    return int(l1.get("seed_base", 20260807))


def _cell_seed(spec_seed: int, index: int) -> int:
    """与 est/est_cover.py:41-49 cell_seed 逐位一致 (splitmix64 派生, 格间独立)."""
    x = spec_seed * 0x100000001B3 + index
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return int((x ^ (x >> 31)) & 0xFFFFFFFF)


def build_inv_exam(spec: dict) -> dict:
    """规格 -> 恒等式考卷 dict (条目冻结 + 种子派生)."""
    spec = validate_spec(spec)
    if "invariants" not in spec:
        raise ValueError("invariant exam requires spec to declare invariants field "
                         "(北极星 v3 级 1 D2)")
    seed_base = _seed_base(spec)
    entries = []
    for i, item in enumerate(spec["invariants"]):
        name = item["name"]
        if name not in INVARIANTS:
            raise ValueError(f"invariant {name!r} not in knowledge base "
                             f"{list(INVARIANTS)} (入库硬条款: 必须有实测档案)")
        if item["condition"] not in SAMPLERS:
            raise ValueError(f"sampling profile {item['condition']!r} not in profile table "
                             f"{list(SAMPLERS)}")
        entries.append({
            "name": name,
            "pair": list(item["pair"]),
            "condition": item["condition"],
            "tol": float(item["tol"]),
            "R": int(item["n_inputs"]),
            "n": int(item.get("n", 20)),          # E1v2 惯例 n=20
            "seed": _cell_seed(seed_base, i),
        })
    exam = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "INV",
        "name": spec["name"],
        "family": spec["family"],
        "spec": spec,
        "spec_md5": spec_md5(spec),
        "functions": dict(spec.get("functions", {})),
        "function": spec["function"],
        "invariants": entries,
    }
    exam["content_md5"] = content_md5(exam)
    return exam


def load_inv_exam(path: str | Path) -> dict:
    """加载恒等式考卷 JSON 并验签 (spec_md5 + content_md5 独立复算)."""
    p = Path(path)
    exam = json.loads(p.read_text(encoding="utf-8"))
    if exam.get("tool") != "spsl" or exam.get("layer") != "INV":
        raise ValueError(f"{p} is not an spsl invariant exam JSON "
                         f"(tool={exam.get('tool')!r}, layer={exam.get('layer')!r})")
    for field in ("name", "spec", "spec_md5", "invariants", "functions"):
        if field not in exam:
            raise ValueError(f"{p} missing exam field: {field}")
    if spec_md5(exam["spec"]) != exam["spec_md5"]:
        raise ValueError(f"{p}: embedded spec spec_md5 mismatch (file modified?)")
    if content_md5(exam) != exam["content_md5"]:
        raise ValueError(f"{p}: content_md5 mismatch (file modified?)")
    return exam


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.compile_inv",
        description="SPSL: spec JSON -> invariant exam JSON (Northstar v3 stage 1 D2)")
    parser.add_argument("spec", help="规格 JSON 路径 (含 functions + invariants)")
    parser.add_argument("--out", required=True, help="输出恒等式考卷 JSON 路径")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    exam = build_inv_exam(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(exam, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[spsl] invariant exam compiled: {args.spec} -> {args.out}")
    print(f"  family: {exam['name']} ({exam['family']})")
    print(f"  functions: {list(exam['functions'])}")
    for e in exam["invariants"]:
        print(f"  invariant {e['name']}: {e['pair']} {e['condition']} "
              f"R={e['R']} n={e['n']} tol={e['tol']} seed={e['seed']}")
    print(f"  spec_md5={exam['spec_md5']}  content_md5={exam['content_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
