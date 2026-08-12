"""CLI 入口: python3 -m verifytool <validator.py> [--runs N] [--out PATH]

主命令: 输入一个 .py 验证器 (暴露 chi2_pvalue(observed) -> float),
输出 HTML 报告 (四层判定 + 能力地图 + 盲点清单) + JSON (md5 双字段).
规格: SPEC_MVP工具形态_2026-08-07.md.

插件扩展子命令 (规格: SPEC_插件扩展_2026-08-07.md; 无子命令 → 原主命令, 行为不变):
  python3 -m verifytool templates [list|info <名>]        # ② 模板库入口
  python3 -m verifytool exam <模板名> [--runs N]          # ② 考任意模板 (复用主管道)
  python3 -m verifytool construct --rule <rule.json> <数据csv>  # ③ 构造器 (离线规则执行)
  python3 -m verifytool prune [--world synthetic|--cands <p.json>] [--budget B]  # ④ 剪枝调度
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from verifytool import VERSION
from verifytool.loader import load_validator
from verifytool.report import render_html, save_json
from verifytool.run_verify import DEFAULT_RUNS, run_verify

_SUBCOMMANDS = ("templates", "exam", "construct", "prune")


def _dispatch(argv: list) -> int:
    """子命令分发 (argv 首 token = 子命令名)."""
    cmd, rest = argv[0], argv[1:]
    if cmd == "templates":
        from verifytool.templates import cmd_templates
        return cmd_templates(rest)
    if cmd == "exam":
        from verifytool.templates import cmd_exam
        return cmd_exam(rest)
    if cmd == "construct":
        from verifytool.constructor import cmd_construct
        return cmd_construct(rest)
    if cmd == "prune":
        from verifytool.prune import cmd_prune
        return cmd_prune(rest)
    print(f"[verifytool] unknown subcommand: {cmd!r}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # 子命令分发 (无子命令 → 原主命令, 逐字节不变)
    if argv and argv[0] in _SUBCOMMANDS:
        return _dispatch(argv)

    parser = argparse.ArgumentParser(
        prog="python3 -m verifytool",
        description="Validator QC CLI: takes a .py validator exposing chi2_pvalue(observed) -> float, "
                    "outputs a four-layer exam calibration report (HTML + JSON). Pure program, zero LLM, zero network.",
    )
    parser.add_argument("validator", help="验证器 .py 文件路径")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="L1 sampling runs (default %d, seeds = 20260807 + i)" % DEFAULT_RUNS)
    parser.add_argument("--out", default=None,
                        help="HTML output path (default cwd/verifytool_report_<validator>.html; "
                             "JSON = same-name .json)")
    args = parser.parse_args(argv)

    if args.runs < 1:
        print("[verifytool] --runs must be >= 1", file=sys.stderr)
        return 2

    # ---------- 加载 ----------
    try:
        func, mod_name = load_validator(args.validator)
    except RuntimeError as exc:
        print(f"[verifytool] load failed: {exc}", file=sys.stderr)
        return 2

    # ---------- 输出路径 ----------
    out_html = (Path(args.out) if args.out
                else Path.cwd() / f"verifytool_report_{mod_name}.html")
    out_html = out_html.resolve()
    out_json = out_html.with_suffix(".json")

    # ---------- 质检 ----------
    result = run_verify(func, runs=args.runs)

    # ---------- 落盘 (JSON 先, 含 md5 双字段) ----------
    payload = dict(result)
    payload["validator"] = {"path": str(Path(args.validator).resolve()),
                            "basename": mod_name}
    payload["command"] = " ".join(sys.argv)
    md5s = save_json(payload, out_json)
    meta = {
        "validator_path": payload["validator"]["path"],
        "validator_name": mod_name,
        "n_runs": args.runs,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_path": str(out_html),
        "payload_md5": md5s["payload_md5"],
        "self_md5": md5s["self_md5"],
    }
    out_html.write_text(render_html(result, meta), encoding="utf-8")

    # ---------- stdout 摘要 ----------
    four = result["four_layers"]
    print(f"[verifytool] {mod_name} — {args.runs} runs (seeds "
          f"{four['seeds'][0]}..{four['seeds'][-1]}), elapsed {result['elapsed_seconds']} s")
    for layer in ("L1", "L2", "L3", "L4"):
        lv = four["layers"][layer]
        print(f"  {layer}: {lv['verdict']:<6} (rej {lv['rej']}/{lv['runs']})")
    print(f"  total: {four['total_verdict']} (reject {four['reject_runs']}/{four['n_runs']})")
    if result.get("time_warn"):
        print("[verifytool] warning: past the 2-minute guardrail; the candidate validator itself is slow "
              "(L1 3 runs x 4000 tables, slow candidates scale linearly)")
    print(f"[verifytool] HTML -> {out_html}")
    print(f"[verifytool] JSON -> {out_json}")
    print(f"[verifytool] payload_md5={md5s['payload_md5']} "
          f"self_md5={md5s['self_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
