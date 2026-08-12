"""SPSL L2 批量执行 — N 个候选/输入 -> 汇总 JSON.

语义: 汇总 = 逐条判定集合的并集。对每个 (名称, 候选, 考卷) 条目独立执行
spsl.run.run_four_layers, 逐条判定 JSON 落盘 (out_dir/), 汇总 = 逐条判定
集合的确定性拼接 (判定内容仅由条目与考卷决定, 无时序/并发依赖 ->
顺序与并发结果逐字节一致)。

确定性 (G3 判据): 汇总 JSON 不含 elapsed 类字段 (spsl.run 唯一非确定 =
elapsed_seconds 及派生 md5, run.py:16-18 声明), 同输入两次运行
md5 逐字节一致。

用法:
  # 单候选批量: 一个 .py 候选跑演示考卷
  python3 -m gateway.batch --python-candidate my_chi2.py \
      --exam exams/exam_pearson_demo.json \
      --out out/g3_batch_cands.json
  # 协议候选 (任意语言实现, 经 gateway.protocol 子进程桥接)
  python3 -m gateway.batch --cmd "node my_impl.js" \
      --exam exams/exam_pearson_demo.json \
      --out out/g3_batch_node.json --out-dir out/g3_batch_node_runs/
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from gateway import VERSION  # noqa: E402
from gateway.protocol import ProtocolCandidate  # noqa: E402
from spsl.envelope import load_exam  # noqa: E402
from spsl.run import run_four_layers  # noqa: E402


def _deterministic_digest(payload: dict) -> str:
    """确定性 md5: 汇总/判定 JSON 去掉时序字段后的指纹 (G3 证据)."""
    body = {k: v for k, v in payload.items()
            if k not in ("elapsed_seconds", "payload_md5", "self_md5")}
    return hashlib.md5(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def judge(func, exam: dict, seeds: list[int] | None = None) -> dict:
    """对单个候选执行完整四层考卷 -> 判定 dict (seeds 为 None 用考卷自带).

    返回与 spsl.run.run_four_layers 同构; 汇总层再附加名称/确定性指纹。
    """
    if seeds is not None:
        exam = _with_seeds(exam, seeds)
    return run_four_layers(func, exam)


def _with_seeds(exam: dict, seeds: list[int]) -> dict:
    """考卷派生: 仅替换 l1.seeds (运行时投影, 不重签名, 报告里如实声明)."""
    out = dict(exam)
    out["l1"] = dict(exam["l1"])
    out["l1"]["seeds"] = list(seeds)
    return out


def run_batch(entries, exam: dict, out_dir: Path | None = None,
              summary_meta: dict | None = None) -> dict:
    """批量执行: entries = [(name, func), ...] 或 [(name, func, seeds), ...].

    (name, func) 用考卷自带 seeds; (name, func, seeds) 用派生种子做单条目判定
    (N=20 演示语义: 20 条目 = 20 个单 seed 判定)。
    逐条判定落盘 out_dir/<name>.json; 返回汇总 dict (确定性字段, 无时序):
      {tool, version, gate, exam{name,spec_md5,content_md5}, n_entries,
       per_entry: [{name, n_runs, reject_runs, total_verdict,
                    layers{L1..L4 判定}, seed_verdicts[bool...]}],
       aggregate: {n_entries, n_accept, n_reject, n_mixed,
                   by_layer: {L1: PASS/REJECT/MIXED 计数...}},
       summary_md5, summary_self_md5}
    """
    per_entry = []
    for entry in entries:
        name, func = entry[0], entry[1]
        seeds = entry[2] if len(entry) > 2 else None
        four = judge(func, exam, seeds)
        per_entry.append({
            "name": name,
            "n_runs": four["n_runs"],
            "reject_runs": four["reject_runs"],
            "total_verdict": four["total_verdict"],
            "layers": {k: v["verdict"] for k, v in four["layers"].items()},
            "seed_verdicts": [bool(r["verdict"]) for r in four["per_run"]],
            "md5": _deterministic_digest(four),
        })
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}.json").write_text(
                json.dumps(four, ensure_ascii=False, indent=2),
                encoding="utf-8")

    by_layer = {}
    for e in per_entry:
        for layer, v in e["layers"].items():
            by_layer.setdefault(layer, {}).setdefault(v, 0)
            by_layer[layer][v] += 1
    summary = {
        "tool": "spsl-gateway",
        "version": VERSION,
        "gate": "G3",
        "exam": {"name": exam["name"], "spec_md5": exam["spec_md5"],
                 "content_md5": exam["content_md5"]},
        "n_entries": len(per_entry),
        "per_entry": per_entry,
        "aggregate": {
            "n_entries": len(per_entry),
            "n_accept": sum(1 for e in per_entry
                            if e["total_verdict"] == "ACCEPT"),
            "n_reject": sum(1 for e in per_entry
                            if e["total_verdict"] == "REJECT"),
            "n_mixed": sum(1 for e in per_entry
                           if e["total_verdict"] == "MIXED"),
            "by_layer": by_layer,
        },
        "note": "summary = union of per-entry verdict sets (each entry independently "
                "runs the full four-layer exam, no ordering dependency); deterministic "
                "fields exclude elapsed/md5-derived fields.",
    }
    if summary_meta:
        summary.update(summary_meta)
    summary["summary_md5"] = _deterministic_digest(summary)
    return summary


def _make_func(candidate: str) -> tuple:
    """候选参数 -> 可调用对象: --python-candidate 走 load_validator,
    --cmd 走 ProtocolCandidate (跨语言)."""
    if candidate.startswith("node ") or candidate.endswith(".js"):
        cmd = candidate.split() if candidate.startswith("node ") \
            else ["node", candidate]
        return ProtocolCandidate(cmd)
    from spsl.run import load_validator

    fn, mod = load_validator(candidate, func_name="chi2_pvalue")
    return fn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m gateway.batch",
        description="SPSL L2 batch: N candidates/inputs -> summary = union of per-entry verdicts")
    parser.add_argument("--exam", required=True, help="full four-layer exam JSON")
    parser.add_argument("--cmd", default=None,
                        help="protocol candidate command (e.g. 'node my_impl.js')")
    parser.add_argument("--python-candidate", default=None, help=".py candidate path")
    parser.add_argument("--candidate-dir", default=None,
                        help="directory: each .py candidate in it runs the exam once")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seed list; a single integer N = take N derived seeds "
                             "(only with --cmd/--python-candidate)")
    parser.add_argument("--out", required=True, help="summary JSON output")
    parser.add_argument("--out-dir", default=None, help="per-entry verdict JSON directory")
    args = parser.parse_args(argv)

    exam = load_exam(args.exam)
    cands = []
    if args.cmd:
        cands.append((Path(args.cmd.split()[-1]).stem, _make_func(args.cmd)))
    if args.python_candidate:
        cands.append((Path(args.python_candidate).stem,
                      _make_func(args.python_candidate)))
    if args.candidate_dir:
        for p in sorted(Path(args.candidate_dir).glob("*.py")):
            cands.append((p.stem, _make_func(str(p))))

    if args.seeds:
        seeds = ([int(s) for s in args.seeds.split(",")]
                 if "," in args.seeds else list(
                     range(20260810, 20260810 + int(args.seeds))))
        # 单候选 -> 每个种子一次单 seed 判定 (N=20 演示语义)
        entries = [(f"{n}_seed{s}", f, [s]) for n, f in cands for s in seeds]
    else:
        entries = cands

    # 关闭用不上的协议候选 (全部条目共用同一候选实例, 逐个判定, 结束时统一关闭)。
    used = set()
    for entry in entries:
        used.add(id(entry[1]))
    summary = run_batch(entries, exam,
                        out_dir=Path(args.out_dir) if args.out_dir else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in used:
        if isinstance(f, ProtocolCandidate):
            f.close()

    print(f"[gateway.batch] {len(entries)} verdict entries -> {args.out}")
    for e in summary["per_entry"][:30]:
        print(f"  {e['name']:<24} {e['total_verdict']:<7} "
              f"rejected {e['reject_runs']}/{e['n_runs']}")
    a = summary["aggregate"]
    print(f"[gateway.batch] ACCEPT {a['n_accept']} / REJECT {a['n_reject']} "
          f"/ MIXED {a['n_mixed']}")
    print(f"[gateway.batch] summary_md5={summary['summary_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
