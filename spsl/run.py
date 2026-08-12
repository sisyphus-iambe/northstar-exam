"""SPSL 统一执行 — 完整四层考卷 + 候选验证器 -> 四层判定报告.

用法 (statistical 族, L1-L4 协议):
  python3 -m spsl.run exams/exam_pearson_demo.json my_chi2.py [--out verdict.json]
用法 (注册表新族, 规格即考卷):
  python3 -m spsl.run <新族规格.json> [--out verdict.json]                 (demo_data 族无考生)
  python3 -m spsl.run <新族规格.json> <考生模块.py> --out ...              (state_estimator 族)

statistical 四层判定与 v1 产品代码同构:
  L1  H0 模拟校准 (spsl.l1: 连续区均匀 / 离散区保守, NaN 不计分)
  L2  双参照对拍: 参照自检 (已知答案 ± self_check_tol + 双参照一致 agree_tol=1e-9)
      -> REF_ABORT; 候选 vs 两参照 tol=1e-6
  L3  边界泛化: 与 L2 同判定, 5 形态 × 76 输入 (v1 同构)
  L4  畸形输入诚实失败: 9 类, 返回有限 p 值 = 幻觉填补 = FAIL
层判定阈值: 拒绝占比 <=10% = PASS, >=90% = REJECT,
其余 MIXED; 总判定同阈值 -> ACCEPT/REJECT/MIXED.
候选在合法考卷输入上抛异常 (L2/L3 对拍表) -> 该 run 全层记拒 (v1 crash 语义);
L1 抛异常 -> NaN 不计分; L4 抛异常 = 诚实失败 (合格).

分发: main() 从考卷/规格读 constraint_type 查 EXAMINER_REGISTRY; statistical 分支 =
原代码路径 (零行为变化); 新族 = 规格即考卷, 由各族执行器承接.

确定性: 判定/诊断/计数全部确定性字段, 重跑逐字节一致; 唯一非确定 =
elapsed_seconds 及派生的 payload_md5/self_md5.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from spsl import VERSION  # noqa: E402
from spsl.envelope import load_exam  # noqa: E402
from spsl.golden import get_ref  # noqa: E402
from spsl.l1 import call_candidate, layer_verdict, run_seed  # noqa: E402
from spsl.registry import EXAMINER_REGISTRY  # noqa: E402
from spsl.schema import DEFAULT_CONSTRAINT_TYPE, spec_md5, validate_spec  # noqa: E402


# ---------------------------------------------------------------------------
# 候选加载 + 判定落盘 (自 v1 产品代码内联)
# ---------------------------------------------------------------------------

def load_validator(path, func_name="chi2_pvalue"):
    """动态加载候选实现模块. 返回 (可调用对象, 模块名).

    func_name: 按考卷规格加载的目标函数名 (考卷 JSON 的 spec.function);
    默认 "chi2_pvalue" = v1 旧契约, 向后兼容. 失败 raise RuntimeError,
    消息面向用户 (含诊断信息).
    """
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"file not found: {p}")
    if p.suffix != ".py":
        raise RuntimeError(f"input must be a .py file (got: {p.suffix or 'no suffix'})")

    mod_name = p.stem
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot parse file: {p} (spec_from_file_location returned empty)")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load module (top-level code of {p} raised): "
            f"{type(exc).__name__}: {exc}") from exc

    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        found = [n for n in dir(mod)
                 if not n.startswith("_") and callable(getattr(mod, n))]
        raise RuntimeError(
            f"module {mod_name} does not expose a callable {func_name}. "
            f"contract: {func_name}(...) -> float. "
            f"callables found: {found if found else '(none)'}"
        )

    if func_name == "chi2_pvalue":
        _smoke_check(fn, mod_name)  # v1 旧契约冒烟; 新函数族冒烟由考卷规格执行
    return fn, mod_name


def _smoke_check(fn, mod_name: str) -> None:
    """契约冒烟: 对一张合法 2x2 表调一次, 必须是可转 float 的有限值 (0<=p<=1)."""
    table = np.array([[10, 20], [30, 40]], dtype=float)
    try:
        p = float(fn(table))
    except Exception as exc:
        raise RuntimeError(
            f"{mod_name}.chi2_pvalue raised on valid input [[10,20],[30,40]]: "
            f"{type(exc).__name__}: {exc}") from exc
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"{mod_name}.chi2_pvalue returned {p!r}, outside [0,1] (contract: p-value must be in [0,1])")


def _json_safe(o):
    """np 类型转 Python 原生 (判定 payload 落盘前规范化)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(x) for x in o]
    return o


def save_json(payload: dict, out_path) -> dict:
    """写判定 JSON + 附加 md5 双字段. 返回 {payload_md5, self_md5}.

    payload_md5 = md5(json.dumps(payload 去自身两字段, sort_keys));
    self_md5    = md5(json.dumps(去 self_md5, indent=2)).
    """
    payload = _json_safe(payload)
    payload["payload_md5"] = hashlib.md5(
        json.dumps({k: v for k, v in payload.items()
                    if k not in ("payload_md5", "self_md5")},
                   ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    without_self = {k: v for k, v in payload.items() if k != "self_md5"}
    payload["self_md5"] = hashlib.md5(
        json.dumps(without_self, ensure_ascii=False, indent=2).encode("utf-8")
    ).hexdigest()
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    return {"payload_md5": payload["payload_md5"], "self_md5": payload["self_md5"]}


# ---------------------------------------------------------------------------
# 输入解码 (L4 畸形输入的 JSON 安全编码 -> np 数组)
# ---------------------------------------------------------------------------

def _decode_cell(v):
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    return v


def decode_l4_input(item: dict):
    """考卷 L4 输入 -> (arg, kind). kind: 'table' | 'pair'."""
    values = [[_decode_cell(v) for v in row] for row in item["values"]]
    if item["dtype"] == "object":
        return np.asarray(values, dtype=object), "table"
    if item["type"] == "contingency_table":
        return np.asarray(values, dtype=float), "table"
    return item  # two_samples: 下面单独解码 x/y


# ---------------------------------------------------------------------------
# L2/L3 双参照对拍 (v1 同构)
# ---------------------------------------------------------------------------

def run_pair_layer(func, sec: dict) -> dict:
    """L2/L3 section -> {verdict: bool, diag: {...}}.

    自检: 已知答案核对 + 双参照全考卷一致 (agree_tol) -> 任一不过 REF_ABORT.
    对拍: 候选 vs 两参照 max_dev <= tol (1e-6). 候选抛异常向上传播
    (v1 语义: 合法考卷上抛异常 = run 崩溃, 该 run 全层记拒).
    """
    refs = [get_ref(n) for n in sec["refs"]]
    # 自检 1: 已知答案核对 (规格 reference.self_check)
    for case in sec["self_check"]:
        for ref in refs:
            dev = abs(float(ref(case["input"])) - case["expected"])
            if dev > sec["self_check_tol"]:
                return {
                    "verdict": False,
                    "diag": {"ref_abort": True,
                             "reason": f"known-answer check failed {case['name']} "
                                       f"({ref.__name__} dev={dev:.3e} > "
                                       f"{sec['self_check_tol']})",
                             "worst_ref_dev": dev},
                }
    # 自检 2: 双参照在全部考卷一致 (REF_AGREE_TOL 1e-9, v1 同款)
    inputs = sec["inputs"]
    worst_ref_dev = 0.0
    for inp in inputs:
        a, b = (float(refs[0](inp)), float(refs[1](inp)))
        worst_ref_dev = max(worst_ref_dev, abs(a - b))
    if worst_ref_dev > sec["agree_tol"]:
        return {
            "verdict": False,
            "diag": {"ref_abort": True,
                     "reason": f"references disagree on exam table {worst_ref_dev:.3e} > "
                               f"{sec['agree_tol']} (reference layer untrustworthy)",
                     "worst_ref_dev": worst_ref_dev,
                     "n_tables": len(inputs)},
        }
    # 对拍: 候选 vs 两参照
    devs = []
    for inp in inputs:
        cand_p = float(call_candidate(func, inp))   # 抛异常 -> run 崩溃 (v1 语义)
        devs.append(max(abs(cand_p - refs[0](inp)), abs(cand_p - refs[1](inp))))
    max_dev = float(max(devs))
    return {
        "verdict": bool(max_dev <= sec["tol"]),
        "diag": {
            "ref_abort": False,
            "worst_ref_dev": worst_ref_dev,
            "max_dev": max_dev,
            "n_tables": len(inputs),
            "n_viol": int(sum(d > sec["tol"] for d in devs)),
        },
    }


# ---------------------------------------------------------------------------
# L4 畸形输入诚实失败 (v1 同款)
# ---------------------------------------------------------------------------

def run_l4_exam(func, sec: dict) -> dict:
    """L4 section -> {verdict: bool, diag: {n_inputs, n_fail, per_input}}."""
    results = []
    for item in sec["inputs"]:
        if item["type"] == "two_samples":
            x = np.asarray([_decode_cell(v) for v in item["x"]],
                           dtype=object if item["dtype"] == "object" else float)
            y = np.asarray([_decode_cell(v) for v in item["y"]],
                           dtype=object if item["dtype"] == "object" else float)
            try:
                p = float(func(x, y))
            except Exception as exc:   # 显式失败 = 合格
                results.append({"name": item["name"],
                                "returned": f"EXC {type(exc).__name__}",
                                "finite": False})
                continue
        else:
            t, _ = decode_l4_input(item)
            try:
                p = float(func(t))
            except Exception as exc:
                results.append({"name": item["name"],
                                "returned": f"EXC {type(exc).__name__}",
                                "finite": False})
                continue
        finite = math.isfinite(p) and 0.0 <= p <= 1.0
        results.append({"name": item["name"], "returned": p, "finite": finite})
    n_fail = sum(1 for r in results if r["finite"])
    return {
        "verdict": bool(n_fail == 0),
        "diag": {"n_inputs": len(results), "n_fail": n_fail,
                 "per_input": results},
    }


# ---------------------------------------------------------------------------
# 四层编排 (v1 run_four_layers 同构)
# ---------------------------------------------------------------------------

def run_four_layers(func, exam: dict) -> dict:
    l1cfg = exam["l1"]
    seeds = l1cfg["seeds"]
    # L2/L3/L4 固定考卷, 确定性单次求值; 候选在合法对拍表上抛异常 ->
    # 全部 run 全层记拒 (v1 语义: 每 run 崩溃 -> 全拒)
    try:
        l2 = run_pair_layer(func, exam["l2"])
        l3 = run_pair_layer(func, exam["l3"])
        l4 = run_l4_exam(func, exam["l4"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        per_run = [{"seed": s, "crash": True, "error": error,
                    "verdict": False, "L1": False, "L2": False,
                    "L3": False, "L4": False} for s in seeds]
        return {
            "n_runs": len(seeds), "seeds": seeds, "reject_runs": len(seeds),
            "layers": {layer: {"verdict": layer_verdict(len(seeds), len(seeds)),
                               "rej": len(seeds), "runs": len(seeds)}
                       for layer in ("L1", "L2", "L3", "L4")},
            "total_verdict": "REJECT",
            "per_run": per_run, "first": per_run[0],
        }

    per_run = []
    for seed in seeds:
        try:
            r = run_seed(func, seed, l1cfg)   # spsl.l1 (NaN 不计分)
            per_run.append({
                "seed": seed,
                "verdict": bool(r["verdict"] and l2["verdict"] and l3["verdict"]
                                and l4["verdict"]),
                "L1": r["verdict"], "L2": l2["verdict"],
                "L3": l3["verdict"], "L4": l4["verdict"],
                "L1_diag": r["L1_diag"],
                "L2_diag": l2["diag"],
                "L3_diag": l3["diag"],
                "L4_diag": l4["diag"],
            })
        except Exception as exc:
            # 候选在合法考卷表上抛异常 -> 该 run 全层记拒 (v1 语义)
            per_run.append({
                "seed": seed, "crash": True,
                "error": f"{type(exc).__name__}: {exc}",
                "verdict": False, "L1": False, "L2": False, "L3": False, "L4": False,
            })

    counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for r in per_run:
        for layer in counts:
            if not r[layer]:
                counts[layer] += 1
    reject_runs = sum(1 for r in per_run if not r["verdict"])
    layers = {layer: {"verdict": layer_verdict(counts[layer], len(seeds)),
                      "rej": counts[layer], "runs": len(seeds)}
              for layer in ("L1", "L2", "L3", "L4")}
    total = ("ACCEPT" if reject_runs <= 0.10 * len(seeds)
             else "REJECT" if reject_runs >= 0.90 * len(seeds) else "MIXED")
    return {
        "n_runs": len(seeds),
        "seeds": seeds,
        "reject_runs": reject_runs,
        "layers": layers,
        "total_verdict": total,
        "per_run": per_run,
        "first": per_run[0],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _diag_run(four: dict) -> dict:
    """取首个未崩溃 run 用于诊断 (v1 同款)."""
    for r in four["per_run"]:
        if "crash" not in r:
            return r
    return four["first"]


def _run_examiner(args, raw: dict, ct: str) -> int:
    """非 statistical 族: 规格即考卷, 查表分发到考官执行器."""
    compile_fn, run_fn = EXAMINER_REGISTRY[ct]
    spec = compile_fn(validate_spec(raw))  # 非 statistical 族: compile = 恒等
    out = (Path(args.out) if args.out
           else Path.cwd() / f"{spec['name']}_verdict_{ct}.json")
    payload = run_fn(spec, args.validator, out)
    print(f"[spsl] {payload['summary']}")
    print(f"[spsl] JSON -> {out}")
    print(f"[spsl] payload_md5={payload['payload_md5']} "
          f"self_md5={payload['self_md5']}")
    return 0


def _run_statistical(args) -> int:
    """statistical 族: 原四层路径 (分发后原样执行, 零行为变化)."""
    if args.validator is None:
        print("[spsl] load failed: the statistical family requires a candidate validator path", file=sys.stderr)
        return 2
    exam = load_exam(args.exam)  # 验签: spec_md5 + content_md5
    try:
        func, mod_name = load_validator(args.validator,
                                        func_name=exam["spec"]["function"])
    except RuntimeError as exc:
        print(f"[spsl] load failed: {exc}", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    four = run_four_layers(func, exam)
    elapsed = round(time.monotonic() - t0, 3)

    out = (Path(args.out) if args.out
           else Path.cwd() / f"{exam['name']}_verdict_{mod_name}.json")
    payload = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "FULL",
        "exam": {"path": str(Path(args.exam).resolve()), "name": exam["name"],
                 "spec_md5": exam["spec_md5"], "content_md5": exam["content_md5"]},
        "validator": {"path": str(Path(args.validator).resolve()),
                      "basename": mod_name,
                      "function": exam["spec"]["function"]},
        "rule": exam["l1"]["protocol"] + " | " + exam["l2"]["protocol"]
                + " | " + exam["l4"]["protocol"],
        "command": " ".join(sys.argv),
        "elapsed_seconds": elapsed,
        "four_layers": four,
    }
    md5s = save_json(payload, out)

    print(f"[spsl] exam {exam['name']} — {mod_name}.{exam['spec']['function']}, "
          f"{four['n_runs']} runs (seeds {four['seeds'][0]}..{four['seeds'][-1]}), "
          f"took {elapsed} s")
    for layer in ("L1", "L2", "L3", "L4"):
        lv = four["layers"][layer]
        print(f"  {layer}: {lv['verdict']:<6} (rejected {lv['rej']}/{lv['runs']})")
    print(f"  Verdict: {four['total_verdict']} (rejected {four['reject_runs']}/"
          f"{four['n_runs']})")
    d = _diag_run(four)
    if "crash" in d:
        print(f"  candidate raised on a valid exam input: {d['error']}")
    else:
        l1d, l2d, l3d, l4d = d["L1_diag"], d["L2_diag"], d["L3_diag"], d["L4_diag"]
        print(f"  L1 diag: cont_ks_D={l1d['cont_ks_D']:.6f} "
              f"cont_mean={l1d['cont_mean']:.6f} "
              f"disc_mean={l1d['disc_mean']:.6f}")
        if "cont_note" in l1d:
            print(f"  L1: {l1d['cont_note']}")
        if "disc_note" in l1d:
            print(f"  L1: {l1d['disc_note']}")
        if l2d.get("ref_abort"):
            print(f"  L2: REF_ABORT ({l2d.get('reason', '')})")
        else:
            print(f"  L2 diag: max_dev={l2d['max_dev']:.3e} "
                  f"violations {l2d['n_viol']}/{l2d['n_tables']} "
                  f"(ref self-check {l2d['worst_ref_dev']:.3e})")
        if l3d.get("ref_abort"):
            print(f"  L3: REF_ABORT ({l3d.get('reason', '')})")
        else:
            print(f"  L3 diag: max_dev={l3d['max_dev']:.3e} "
                  f"violations {l3d['n_viol']}/{l3d['n_tables']} "
                  f"(ref self-check {l3d['worst_ref_dev']:.3e})")
        print(f"  L4 diag: {l4d['n_fail']}/{l4d['n_inputs']} malformed-input classes returned finite value")
    print(f"[spsl] JSON -> {out}")
    print(f"[spsl] payload_md5={md5s['payload_md5']} self_md5={md5s['self_md5']}")
    return 0


def main(argv=None) -> int:
    """CLI 入口: 从考卷/规格读 constraint_type 查表分发 (注册表平台化).

    statistical 族 = 原四层路径 (零行为变化); 新族 = 规格即考卷, 执行器承接.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.run",
        description="SPSL: run the full four-layer exam against a candidate validator; "
                    "supports examiner-registry families (demo_data / state_estimator)")
    parser.add_argument("exam", help="full four-layer exam JSON (spsl.envelope output) "
                                     "or a new-family spec JSON")
    parser.add_argument("validator", nargs="?", default=None,
                        help="candidate validator .py path (required for statistical/state_estimator "
                             "families; not used by demo_data)")
    parser.add_argument("--out", default=None,
                        help="verdict JSON output path (default cwd/<exam-name>_verdict_<validator-name>.json)")
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.exam).read_text(encoding="utf-8"))
    embedded = raw.get("spec") if isinstance(raw.get("spec"), dict) else None
    ct = (embedded or raw).get("constraint_type", DEFAULT_CONSTRAINT_TYPE)
    if ct != DEFAULT_CONSTRAINT_TYPE:
        return _run_examiner(args, raw, ct)
    return _run_statistical(args)


if __name__ == "__main__":
    raise SystemExit(main())
