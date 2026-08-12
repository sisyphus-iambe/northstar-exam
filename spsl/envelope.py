"""SPSL 包络 — 四层考卷包.

用法: python3 -m spsl.envelope <规格.json> --out <考卷.json>
      (编译路径依赖官方编译模块, 公开仓库不含; 考卷由官方编译分发)

考卷 JSON = 规格 (含 spec_md5) + l1/l2/l3/l4 四层编译产物 + content_md5.
md5 规范与编译层一致 (compile_l1.content_md5 同款):
  content_md5 = md5(规范化 JSON, 去掉 content_md5 自身字段)
  spec_md5    = md5(规范化规格 JSON, schema.spec_md5)
判据修改 = 改规格 -> 新 spec_md5 -> 新 content_md5, 旧 md5 留档.
"""
import argparse
import hashlib
import json
from pathlib import Path

from spsl import VERSION
from spsl.schema import spec_md5, validate_spec


def content_md5(exam: dict) -> str:
    """输出内容 md5: 规范化序列化 (ensure_ascii=False, sort_keys=True),
    不含 content_md5 字段本身 (compile_l1.content_md5 同款)."""
    body = {k: v for k, v in exam.items() if k != "content_md5"}
    return hashlib.md5(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_full_exam(spec: dict) -> dict:
    """规格 -> 完整四层考卷 dict (l1 冻结 + l2/l3/l4 编译产物).

    编译模块 (compile_l1-4) 惰性加载: 送考路径 (load_exam) 不依赖它们,
    仅编译路径需要 (公开仓库不含编译模块, 编译考卷由官方完成).
    """
    from spsl.compile_l1 import build_exam as build_l1
    from spsl.compile_l2 import build_exam as build_l2
    from spsl.compile_l3 import build_exam as build_l3
    from spsl.compile_l4 import build_exam as build_l4

    spec = validate_spec(spec)
    if "reference" not in spec:
        raise ValueError(
            "a full four-layer exam requires the spec to declare reference "
            "(reference sources + self-check protocol; required for full four-layer exams)")
    exam = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "FULL",
        "name": spec["name"],
        "family": spec["family"],
        "spec": spec,
        "spec_md5": spec_md5(spec),
        "l1": build_l1(spec)["l1"],   # 冻结的 L1 配置
        "l2": build_l2(spec),
        "l3": build_l3(spec),
        "l4": build_l4(spec),
    }
    exam["content_md5"] = content_md5(exam)
    return exam


def load_exam(path: str | Path) -> dict:
    """加载完整考卷 JSON 并验签 (spec_md5 + content_md5 独立复算, 失败即报)."""
    p = Path(path)
    exam = json.loads(p.read_text(encoding="utf-8"))
    if exam.get("tool") != "spsl" or exam.get("layer") != "FULL":
        raise ValueError(f"{p} is not an spsl full four-layer exam JSON "
                         f"(tool={exam.get('tool')!r}, layer={exam.get('layer')!r})")
    for field in ("name", "spec", "spec_md5", "l1", "l2", "l3", "l4"):
        if field not in exam:
            raise ValueError(f"{p} missing exam field: {field}")
    if spec_md5(exam["spec"]) != exam["spec_md5"]:
        raise ValueError(f"{p}: embedded spec spec_md5 recomputation mismatch (file modified?)")
    if content_md5(exam) != exam["content_md5"]:
        raise ValueError(f"{p}: content_md5 recomputation mismatch (file modified?)")
    return exam


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.envelope",
        description="SPSL: spec JSON -> full four-layer exam JSON")
    parser.add_argument("spec", help="spec JSON path")
    parser.add_argument("--out", required=True, help="output full exam JSON path")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    exam = build_full_exam(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(exam, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[spsl] full four-layer exam compiled: {args.spec} -> {args.out}")
    print(f"  family: {exam['name']} ({exam['family']})  function: {spec['function']}")
    print(f"  L1: seeds {exam['l1']['seeds'][0]}..{exam['l1']['seeds'][-1]}  "
          f"n_tables={exam['l1']['n_tables']}")
    print(f"  L2: {exam['l2']['n_inputs']} inputs  refs {exam['l2']['refs']}  "
          f"match tol={exam['l2']['tol']}")
    print(f"  L3: {exam['l3']['n_inputs']} inputs (5 forms)")
    print(f"  L4: {exam['l4']['n_inputs']} malformed-input classes")
    print(f"  spec_md5={exam['spec_md5']}")
    print(f"  content_md5={exam['content_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
