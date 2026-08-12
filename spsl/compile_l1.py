"""SPSL L1 编译 — 规格 JSON -> L1 H0 模拟校准考卷 JSON (北极星 v2, D1).

用法: python3 -m spsl.compile_l1 spec.json --out exam.json

考卷 JSON = 规格 (含 spec_md5) + 冻结的 L1 参数/阈值/种子 + content_md5.
阈值与公式冻结自 calibrator/l1.py:19-34 (v1 同款), 执行时零自由度;
判据修改 = 改规格 -> 新 spec_md5 -> 新 content_md5 (北极星v2.md §五.3).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from spsl import VERSION
from spsl.schema import merged_l1, spec_md5, validate_spec


def content_md5(exam: dict) -> str:
    """输出内容 md5: 规范化序列化, 不含 md5 字段本身."""
    body = {k: v for k, v in exam.items() if k != "content_md5"}
    return hashlib.md5(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_exam(spec: dict) -> dict:
    """规格 -> 冻结的 L1 考卷 dict."""
    spec = validate_spec(spec)
    l1 = merged_l1(spec)
    nd = spec["null_dist"]
    n = l1["n_tables"]

    def zone_cfg(zone: str) -> dict:
        z = dict(nd[zone])
        z["type"] = nd["type"]
        return z

    thresholds = {
        "ks_crit_99": 1.63 / math.sqrt(n),                 # l1.py:29
        "ks_slack": l1["ks_slack"],                        # l1.py:30 (0.02)
        "ks_crit_99_plus_slack": 1.63 / math.sqrt(n) + l1["ks_slack"],
        "cont_point_slack": l1["cont_point_slack"],        # l1.py:31 (0.02)
        "cont_mean_slack": l1["cont_mean_slack"],          # l1.py:32 (0.03)
        "disc_point_slack": l1["disc_point_slack"],        # l1.py:33 (0.02)
        "disc_mean_max": l1["disc_mean_max"],              # l1.py:34 (0.54)
    }
    exam = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "L1",
        "name": spec["name"],
        "family": spec["family"],
        "spec": spec,
        "spec_md5": spec_md5(spec),
        "l1": {
            "generator": nd["type"],
            "n_tables": n,
            "cont": zone_cfg("cont"),
            "disc": zone_cfg("disc"),
            "points": l1["points"],
            "thresholds": thresholds,
            "seeds": [l1["seed_base"] + i for i in range(l1["runs"])],
            "protocol": (
                "H0 重抽: 计数表重抽到边缘全 > 0 (calibrator/l1.py:43-48); "
                "两样本 = 连续分布独立抽样. "
                "NaN/非有限 p 值不计入合格 -> 本区本 seed 不合格 (方案 D1; "
                "v1 同: calibrator/l1.py:66-82 cont_finite/disc_finite). "
                "判定: 拒绝计数占比 <=10% = PASS, >=90% = REJECT, 其余 MIXED "
                "(run_verify.py:36-37/58-64)."),
        },
    }
    exam["content_md5"] = content_md5(exam)
    return exam


def load_exam(path: str | Path) -> dict:
    """加载编译考卷 JSON 并验签 (spec_md5 + content_md5 独立复算, 失败即报)."""
    p = Path(path)
    exam = json.loads(p.read_text(encoding="utf-8"))
    if exam.get("tool") != "spsl" or exam.get("layer") != "L1":
        raise ValueError(f"{p} is not an spsl L1 exam JSON "
                         f"(tool={exam.get('tool')!r}, layer={exam.get('layer')!r})")
    for field in ("name", "spec", "spec_md5", "l1"):
        if field not in exam:
            raise ValueError(f"{p} missing exam field: {field}")
    if spec_md5(exam["spec"]) != exam["spec_md5"]:
        raise ValueError(f"{p}: embedded spec spec_md5 mismatch (file modified?)")
    if content_md5(exam) != exam["content_md5"]:
        raise ValueError(f"{p}: content_md5 mismatch (file modified?)")
    return exam


def write_exam(exam: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(exam, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.compile_l1",
        description="SPSL: spec JSON -> L1 H0 simulation calibration exam JSON (Northstar v2 D1)")
    parser.add_argument("spec", help="规格 JSON 路径")
    parser.add_argument("--out", required=True, help="输出考卷 JSON 路径")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    exam = build_exam(spec)
    write_exam(exam, Path(args.out))
    l1 = exam["l1"]
    print(f"[spsl] compiled: {args.spec} -> {args.out}")
    print(f"  family: {exam['name']} ({exam['family']})  function: {spec['function']}")
    print(f"  L1: n_tables={l1['n_tables']}, cont {l1['cont']}, disc {l1['disc']}")
    print(f"  seeds: {l1['seeds'][0]}..{l1['seeds'][-1]}  "
          f"阈值: ks≤{l1['thresholds']['ks_crit_99_plus_slack']:.4f} "
          f"cont±{l1['thresholds']['cont_point_slack']} "
          f"disc≤+{l1['thresholds']['disc_point_slack']}/mean≤{l1['thresholds']['disc_mean_max']}")
    print(f"  spec_md5={exam['spec_md5']}")
    print(f"  content_md5={exam['content_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
