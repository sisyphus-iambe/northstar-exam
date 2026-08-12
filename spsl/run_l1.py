"""SPSL L1 考卷运行 — 编译考卷 JSON + 候选验证器 -> L1 判定 JSON.

用法: python3 -m spsl.run_l1 <exam.json> <validator.py> [--out verdict.json]

加载链: 考卷 JSON 内嵌规格 (spec.function) -> spsl.run.load_validator
按函数名解绑加载候选 (v1 旧契约默认不变).
判定 JSON 落盘用 spsl.run.save_json (payload_md5/self_md5),
全部确定性字段, 无时间戳 -> 重跑逐字节一致.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from spsl import VERSION  # noqa: E402
from spsl.l1 import run_l1_exam  # noqa: E402
from spsl.run import load_validator, save_json  # noqa: E402
from spsl.schema import spec_md5  # noqa: E402


def _content_md5(exam: dict) -> str:
    """考卷内容 md5: 规范化序列化, 不含 content_md5 字段本身."""
    body = {k: v for k, v in exam.items() if k != "content_md5"}
    return hashlib.md5(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_exam(path) -> dict:
    """加载 L1 考卷 JSON 并验签 (spec_md5 + content_md5 独立复算, 失败即报)."""
    p = Path(path)
    exam = json.loads(p.read_text(encoding="utf-8"))
    if exam.get("tool") != "spsl" or exam.get("layer") != "L1":
        raise ValueError(f"{p} is not an spsl L1 exam JSON "
                         f"(tool={exam.get('tool')!r}, layer={exam.get('layer')!r})")
    for field in ("name", "spec", "spec_md5", "l1"):
        if field not in exam:
            raise ValueError(f"{p} missing exam field: {field}")
    if spec_md5(exam["spec"]) != exam["spec_md5"]:
        raise ValueError(f"{p}: embedded spec spec_md5 recomputation mismatch (file modified?)")
    if _content_md5(exam) != exam["content_md5"]:
        raise ValueError(f"{p}: content_md5 recomputation mismatch (file modified?)")
    return exam


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.run_l1",
        description="SPSL: run a compiled L1 exam against a candidate validator")
    parser.add_argument("exam", help="compiled exam JSON (L1 layer artifact)")
    parser.add_argument("validator", help="candidate validator .py path")
    parser.add_argument("--out", default=None,
                        help="verdict JSON output path (default cwd/<exam-name>_verdict_<validator-name>.json)")
    args = parser.parse_args(argv)

    exam = load_exam(args.exam)  # 验签: spec_md5 + content_md5
    try:
        func, mod_name = load_validator(args.validator,
                                        func_name=exam["spec"]["function"])
    except RuntimeError as exc:
        print(f"[spsl] load failed: {exc}", file=sys.stderr)
        return 2

    result = run_l1_exam(func, exam)
    out = (Path(args.out) if args.out
           else Path.cwd() / f"{exam['name']}_verdict_{mod_name}.json")
    payload = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "L1",
        "exam": {"path": str(Path(args.exam).resolve()), "name": exam["name"],
                 "spec_md5": exam["spec_md5"], "content_md5": exam["content_md5"]},
        "validator": {"path": str(Path(args.validator).resolve()),
                      "basename": mod_name,
                      "function": exam["spec"]["function"]},
        "rule": exam["l1"]["protocol"],
        "l1": result,
    }
    md5s = save_json(payload, out)

    print(f"[spsl] exam {exam['name']} — {mod_name}.{exam['spec']['function']}, "
          f"{result['n_seeds']} seeds ({result['seeds'][0]}..{result['seeds'][-1]})")
    print(f"  L1: {result['verdict']} (rejected {result['n_rej']}/{result['n_seeds']}; "
          f"cont zone {result['n_cont_rej']}, disc zone {result['n_disc_rej']})")
    if result["verdict"] != "PASS":
        d0 = result["first_seed_diag"]
        print(f"  first-seed diag: cont_finite={d0['cont_finite']} "
              f"disc_finite={d0['disc_finite']} "
              f"cont_nonfinite={d0['cont_nonfinite_count']} "
              f"disc_nonfinite={d0['disc_nonfinite_count']}")
        for key in ("cont_note", "disc_note", "cont_first_exception",
                    "disc_first_exception"):
            if d0.get(key):
                print(f"  {d0[key]}")
    print(f"[spsl] JSON -> {out}")
    print(f"[spsl] payload_md5={md5s['payload_md5']} self_md5={md5s['self_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
