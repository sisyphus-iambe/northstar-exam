"""SPSL L4 编译 — 规格 JSON -> L4 畸形输入诚实失败考卷 JSON (北极星 v2, D2).

用法: python3 -m spsl.compile_l4 spec.json --out l4_section.json

协议 (北极星v2.md v1 基线 + D2): 确定性构造 9 类畸形/缺失输入, 合格标准 =
每类下候选要么返回非有限值 (NaN/inf), 要么抛异常 (显式失败);
返回有限 float (0<=p<=1) = 幻觉填补 = FAIL (v1 calibrator/l4.py 同款).

9 类一一对应 v1 (calibrator/l4.py:31-64, 逐字同款, L4_SEED=20260806):
  contingency_table: nan_cell_4x4 / nan_row_3x5 / inf_cell_3x3 /
    neginf_cell_2x4 / all_zero_4x4 / single_row_1x5 / single_col_5x1 /
    string_mixed (object 数组) / empty_0x0
  two_samples (语义一一对应): nan_in_x / nan_all_y / inf_in_x /
    neginf_in_y / both_empty (统计量无定义) / singleton_x / singleton_y /
    string_mixed_x / empty_x

JSON 编码: 非有限单元格存字符串标记 "nan"/"inf"/"-inf", 运行时解码
(run.py decode_l4_input); string_mixed 类 dtype="object" 保持字符串/None.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from spsl.schema import validate_spec

L4_SEED = 20260806               # v1 同款 (calibrator/l4.py:26)


def _encode_cell(v):
    """非有限 -> 字符串标记; 其余 -> 原生数值 (JSON 安全)."""
    if isinstance(v, float) and not math.isfinite(v):
        if math.isnan(v):
            return "nan"
        return "inf" if v > 0 else "-inf"
    return v


def _encode(arr):
    """np 数组 -> JSON 安全嵌套列表 (含 nan/inf 标记)."""
    return [[_encode_cell(v) for v in row] for row in arr.tolist()]


# ---------------------------------------------------------------------------
# contingency_table 9 类 (逐字同款 calibrator/l4.py:31-64)
# ---------------------------------------------------------------------------


def _chi2_l4_inputs():
    rng = np.random.default_rng(L4_SEED)
    inputs = []

    def add(name, t):
        inputs.append({"name": name, "type": "contingency_table",
                       "dtype": "float", "values": _encode(t)})

    # 1. 含 NaN 的表 (随机位置)
    t = rng.integers(1, 30, size=(4, 4)).astype(float)
    t[0, 0] = float("nan")
    add("nan_cell_4x4", t)
    # 2. 整行 NaN
    t = rng.integers(1, 30, size=(3, 5)).astype(float)
    t[1, :] = float("nan")
    add("nan_row_3x5", t)
    # 3. 含 +inf
    t = rng.integers(1, 30, size=(3, 3)).astype(float)
    t[2, 2] = float("inf")
    add("inf_cell_3x3", t)
    # 4. 含 -inf
    t = rng.integers(1, 30, size=(2, 4)).astype(float)
    t[0, 1] = float("-inf")
    add("neginf_cell_2x4", t)
    # 5. 全零表
    add("all_zero_4x4", np.zeros((4, 4)))
    # 6. 单行表 (1x5)
    add("single_row_1x5", rng.integers(1, 30, size=(1, 5)))
    # 7. 单列表 (5x1)
    add("single_col_5x1", rng.integers(1, 30, size=(5, 1)))
    # 8. 非数值类型 (字符串) — object 数组
    inputs.append({"name": "string_mixed", "type": "contingency_table",
                   "dtype": "object", "values": [["1", "2"], ["3", None]]})
    # 9. 空数组 (0 维退化; v1 同款 np.array([[]], dtype=float) -> (1,0))
    inputs.append({"name": "empty_0x0", "type": "contingency_table",
                   "dtype": "float", "values": [[]]})
    assert len(inputs) == 9
    return inputs


# ---------------------------------------------------------------------------
# two_samples 9 类 (语义一一对应 v1)
# ---------------------------------------------------------------------------


def _wilcoxon_l4_inputs():
    rng = np.random.default_rng(L4_SEED)
    x = rng.uniform(1.0, 9.0, 6)
    y = rng.uniform(1.0, 9.0, 8)

    def pair(name, xv, yv, dtype="float"):
        return {"name": name, "type": "two_samples", "dtype": dtype,
                "x": [_encode_cell(v) for v in xv],
                "y": [_encode_cell(v) for v in yv]}

    inputs = []
    # 1. x 含单格 NaN (对应 nan_cell_4x4)
    x1 = x.copy(); x1[2] = float("nan")
    inputs.append(pair("nan_in_x", x1, y))
    # 2. 整向量 NaN (对应 nan_row_3x5)
    inputs.append(pair("nan_all_y", x, [float("nan")] * len(y)))
    # 3. x 含 +inf (对应 inf_cell_3x3)
    x3 = x.copy(); x3[1] = float("inf")
    inputs.append(pair("inf_in_x", x3, y))
    # 4. y 含 -inf (对应 neginf_cell_2x4)
    y4 = y.copy(); y4[0] = float("-inf")
    inputs.append(pair("neginf_in_y", x, y4))
    # 5. 双空 (统计量无定义; 对应 all_zero_4x4)
    inputs.append(pair("both_empty", [], []))
    # 6. 单元素样本 (退化边界; 对应 single_row_1x5)
    inputs.append(pair("singleton_x", [1.0], y))
    # 7. 单元素样本 (退化边界; 对应 single_col_5x1)
    inputs.append(pair("singleton_y", x, [1.0]))
    # 8. 非数值混合 (object; 对应 string_mixed)
    inputs.append(pair("string_mixed_x", ["1", None, 3.0], y, dtype="object"))
    # 9. 空样本 (对应 empty_0x0)
    inputs.append(pair("empty_x", [], y))
    assert len(inputs) == 9
    return inputs


# ---------------------------------------------------------------------------
# 编译
# ---------------------------------------------------------------------------

_PROTOCOL = (
    "9 classes; on classes where the reference itself fails (exception/non-finite), "
    "returning a finite p-value = hallucination filling = FAIL; classes where the "
    "reference returns a finite value are not judged (diagnostic ref_dev only).")


def build_exam(spec: dict) -> dict:
    """规格 -> L4 考卷 section dict (嵌入 9 类畸形输入 + 协议)."""
    spec = validate_spec(spec)
    itype = spec["inputs"]["type"]
    if itype == "contingency_table":
        inputs = _chi2_l4_inputs()
    elif itype == "two_samples":
        inputs = _wilcoxon_l4_inputs()
    else:
        raise ValueError(f"compile_l4: unsupported input type {itype!r}")
    return {
        "layer": "L4",
        "n_inputs": len(inputs),
        "inputs": inputs,
        "protocol": _PROTOCOL,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.compile_l4",
        description="SPSL: spec JSON -> L4 malformed-input honest-failure exam JSON (Northstar v2 D2)")
    parser.add_argument("spec", help="规格 JSON 路径")
    parser.add_argument("--out", required=True, help="输出考卷 section JSON 路径")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    sec = build_exam(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sec, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[spsl] L4 compiled: {args.spec} -> {args.out}")
    print(f"  family: {spec['name']}  inputs: {spec['inputs']['type']}  "
          f"9 malformed-input classes: {[i['name'] for i in sec['inputs']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
