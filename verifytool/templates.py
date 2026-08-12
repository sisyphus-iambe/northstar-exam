"""模板库统一入口 (②): 显式注册表 + list/info + exam 编排.

规格: SPEC_插件扩展_2026-08-07.md §2.

- TEMPLATE_REGISTRY: 显式注册表 (不目录扫描), 22 条目 =
  programs/ 9 考卷模板 + 5 错误对照 + programs_real/ 8 演示样.
  每条目 {name, path, kind, desc}: name = 统一名 (模板去 template_ 前缀,
  演示样加 demo_ 前缀避免与 programs/ 重名 — wrong_dof 两处都有),
  desc = docstring 首行 (ast 解析读取, 不执行模块, 与文件保持同步).
- load_template(name): importlib 加载 + 契约检查 (复用 loader.py).
- exam <模板名>: 加载后走 run_verify.run_verify 全管道 (四层+能力地图+盲点),
  输出 HTML+JSON (与主命令同格式, 文件名前缀 exam_<模板名>).

语义说明 (如实): exam 考的是模板的"实现正确性"; 模板库自校准档案
(calibrator/template_exam.py, 9 域 L1-L4, 9 模板全 FAIL — 见
experiments/template_exam/results_template_exam.json) 是另一套口径:
那是"考卷对模板是否够严"的档案 (barnard 类慢速考卷 100 seeds), 本命令
是"单模板单验证器四层判定"的快照 — 两者不可互相替代, 工具输出注明差异.
"""
import argparse
import ast
import sys
from pathlib import Path

from verifytool.loader import load_validator

_REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root
PROGRAMS = _REPO_ROOT / "verifytool" / "programs"
PROGRAMS_REAL = _REPO_ROOT / "verifytool" / "programs_real"


def _doc_first_line(path: Path) -> str:
    """docstring 首行 (ast 只解析不执行, 与文件内容保持同步, 确定性)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
    except Exception:
        return ""
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    return first


def _entry(name: str, path: Path, kind: str) -> dict:
    return {"name": name, "path": path, "kind": kind, "desc": _doc_first_line(path)}


# 9 考卷模板 (programs/template_*.py, 统一名去掉 template_ 前缀)
_TEMPLATE_FILES = [
    "template_barnard", "template_chi2_rxc", "template_fisher", "template_ratio",
    "template_replicate", "template_slope", "template_strat_bidir",
    "template_stratified", "template_trend",
]
# 5 错误对照 (programs/, correct.py 为正确实现基线, 其余为故意写错)
_CONTROL_FILES = ["correct", "wrong_denominator", "wrong_dof",
                  "wrong_dof_square_only", "wrong_impute"]
# 8 演示样 (experiments/qualcheck_realworld/programs_real/, 实验 05 真实程序;
# 统一名加 demo_ 前缀避免与 programs/ 重名)
_DEMO_FILES = ["handwritten_good", "scipy_direct", "wrong_dof", "wrong_expected",
               "wrong_fillna", "wrong_pseudocount", "wrong_round", "wrong_tail"]

TEMPLATE_REGISTRY = (
    [_entry(f, PROGRAMS / f"{f}.py", "exam template") for f in _TEMPLATE_FILES]
    + [_entry(f, PROGRAMS / f"{f}.py", "control") for f in _CONTROL_FILES]
    + [_entry(f"demo_{f}", PROGRAMS_REAL / f"{f}.py", "demo") for f in _DEMO_FILES]
)
NAME_TO_ENTRY = {e["name"]: e for e in TEMPLATE_REGISTRY}


# 模板库自校准档案差异说明 (写进 exam 输出, 出处: experiments/template_exam/)
CALIBRATION_ARCHIVE_NOTE = (
    "Note the differing gauges: this report = a snapshot of one template judged on the "
    "generic four-layer exam (calibrator/calibrate.py), testing template 'implementation "
    "correctness'. The template-library self-calibration archive (calibrator/template_exam.py "
    "9 domains, L1-L4 slow exams, 100 seeds) uses a different gauge - that archive flags all "
    "9 templates FAIL (5 rejected 100/100 + 4 boundary rejects, small-sample fluctuation of "
    "F-hat(0.10) past the 0.12 threshold in the L1 discrete region; source "
    "experiments/template_exam/results_template_exam.json + BARNARD_ROOTCAUSE.md), concluding "
    "'the exams themselves need per-domain subdivision'. The two gauges cannot substitute "
    "for each other."
)


def list_templates() -> None:
    """输出表格 (name/kind/desc)."""
    print("[verifytool templates] template registry — %d entries (9 exam templates + 5 error controls + 8 demos)"
          % len(TEMPLATE_REGISTRY))
    rels = [str(e["path"].relative_to(_REPO_ROOT)) for e in TEMPLATE_REGISTRY]
    w_path = max(len(r) for r in rels)
    print(f"{'name':<24}{'kind':<14}{'path':<{w_path}}desc")
    for e, rel in zip(TEMPLATE_REGISTRY, rels):
        print(f"{e['name']:<24}{e['kind']:<14}{rel:<{w_path}}{e['desc']}")
    print("\nCommon contract: chi2_pvalue(observed) -> float (any r x c contingency table, all margins positive).")
    print("Usage: python3 -m verifytool exam <template> [--runs N]   "
          "(template = the name column above)")


def info_template(name: str) -> int:
    e = NAME_TO_ENTRY.get(name)
    if e is None:
        print(f"[verifytool templates] unknown template: {name!r} (available: "
              f"{', '.join(NAME_TO_ENTRY)} )", file=sys.stderr)
        return 2
    print(f"name : {e['name']}")
    print(f"kind : {e['kind']}")
    print(f"path : {e['path']}")
    print(f"desc : {e['desc']}")
    return 0


def load_template(name: str):
    """按注册名加载模板: 返回 (chi2_pvalue 可调用对象, 模块名). 复用 loader 契约检查."""
    e = NAME_TO_ENTRY.get(name)
    if e is None:
        raise KeyError(
            f"unknown template: {name!r} (available: {', '.join(NAME_TO_ENTRY)})")
    return load_validator(e["path"])


def cmd_exam(argv) -> int:
    """python3 -m verifytool exam <模板名> [--runs N] [--out PATH]."""
    parser = argparse.ArgumentParser(
        prog="python3 -m verifytool exam",
        description="Exam any template (reuses the main pipeline: four layers + capability map + blind spots). Outputs exam_<template>.html/.json")
    parser.add_argument("template", help="template name (see templates list)")
    parser.add_argument("--runs", type=int, default=3,
                        help="L1 sampling runs (default 3, seeds = 20260807 + i)")
    parser.add_argument("--out", default=None, help="HTML output path (default cwd/exam_<template>.html)")
    args = parser.parse_args(argv)
    if args.runs < 1:
        print("[verifytool exam] --runs must be >= 1", file=sys.stderr)
        return 2

    from verifytool.report import render_exam_html, save_json
    from verifytool.run_verify import DEFAULT_RUNS, run_verify

    try:
        func, mod_name = load_template(args.template)
    except (KeyError, RuntimeError) as exc:
        print(f"[verifytool exam] load failed: {exc}", file=sys.stderr)
        return 2

    entry = NAME_TO_ENTRY[args.template]
    out_html = (Path(args.out) if args.out
                else Path.cwd() / f"exam_{args.template}.html")
    out_html = out_html.resolve()
    out_json = out_html.with_suffix(".json")

    result = run_verify(func, runs=args.runs)
    payload = dict(result)
    payload["command"] = "exam"
    payload["exam"] = {
        "template": args.template,
        "path": str(entry["path"].resolve()),
        "kind": entry["kind"],
        "calibration_note": CALIBRATION_ARCHIVE_NOTE,
    }
    md5s = save_json(payload, out_json)
    meta = {
        "validator_path": str(entry["path"].resolve()),
        "validator_name": args.template,
        "n_runs": args.runs,
        "run_at": None,  # 渲染时填充 (report 模板需 str; 见下方)
        "out_path": str(out_html),
        "payload_md5": md5s["payload_md5"],
        "self_md5": md5s["self_md5"],
    }
    from datetime import datetime
    meta["run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_html.write_text(render_exam_html(result, meta,
                                         note=CALIBRATION_ARCHIVE_NOTE),
                        encoding="utf-8")

    four = result["four_layers"]
    print(f"[verifytool exam] {args.template} — {args.runs} runs (seeds "
          f"{four['seeds'][0]}..{four['seeds'][-1]}), elapsed {result['elapsed_seconds']} s")
    for layer in ("L1", "L2", "L3", "L4"):
        lv = four["layers"][layer]
        print(f"  {layer}: {lv['verdict']:<6} (rej {lv['rej']}/{lv['runs']})")
    print(f"  total: {four['total_verdict']} (reject {four['reject_runs']}/{four['n_runs']})")
    print(f"[verifytool exam] HTML -> {out_html}")
    print(f"[verifytool exam] JSON -> {out_json}")
    print(f"[verifytool exam] payload_md5={md5s['payload_md5']} "
          f"self_md5={md5s['self_md5']}")
    print(f"[verifytool exam] calibration note: {CALIBRATION_ARCHIVE_NOTE}")
    return 0


def cmd_templates(argv) -> int:
    """python3 -m verifytool templates [list|info <名>] (默认 list)."""
    if argv and argv[0] == "list":
        list_templates()
        return 0
    if argv and argv[0] == "info":
        if len(argv) < 2:
            print("[verifytool templates] info requires a template name: "
                  "python3 -m verifytool templates info <name>", file=sys.stderr)
            return 2
        return info_template(argv[1])
    if argv:
        print(f"[verifytool templates] unknown subcommand: {argv[0]!r} "
              f"(available: list, info <name>)", file=sys.stderr)
        return 2
    list_templates()
    return 0


if __name__ == "__main__":
    sys.exit(cmd_templates(sys.argv[1:]))
