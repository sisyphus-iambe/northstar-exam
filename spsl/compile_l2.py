"""SPSL L2 编译 — 规格 JSON -> L2 双参照对拍考卷 JSON (北极星 v2, D2).

用法: python3 -m spsl.compile_l2 spec.json --out l2_section.json

协议 (北极星v2.md v1 基线 + D2):
  - 双参照: 规格 reference.refs 登记的两个独立参照实现 (spsl.golden 注册表).
  - 参照自检 (先于对拍): (1) 已知答案核对 — 每个参照在 reference.self_check
    输入上须给出 expected (± self_check_tol); (2) 双参照在全部考卷输入上
    一致到 agree_tol (1e-9, v1 REF_AGREE_TOL = calibrator/reference.py:118).
    任一不过 -> REF_ABORT (参照层不可信, 中止并如实报告; v1 l2.py:64-71 同款).
  - 对拍: 候选 p 值与两个参照都一致到 tol = 1e-6 (v1 L2_TOL, calibrator/l2.py:11).
  - 考卷 (contingency_table): 与 v1 完全同款 40 张表 (calibrator/l2.py:14-58:
    12 手选 + 28 种子化随机, L2_TABLE_SEED=20260806, 重抽到边缘全 > 0)
    —— 位级一致, 保证锚点对照 (verifytool_report_correct.json L2_diag).
  - 考卷 (two_samples): 同协议的两样本对拍考卷 (12 手选 + 28 种子化随机,
    同 seed; n1/n2 >= 2, 有限值, 全部合法输入).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spsl.schema import validate_spec

L2_TABLE_SEED = 20260806          # v1 同款 (calibrator/l2.py:12)
PAIR_TOL = 1e-6                   # 对拍容差 (v1 L2_TOL, calibrator/l2.py:11)

# ---------------------------------------------------------------------------
# contingency_table 考卷 (逐字同款 calibrator/l2.py:14-58)
# ---------------------------------------------------------------------------

_HAND_PICKED = [
    np.array([[10, 20], [30, 40]]),
    np.array([[1, 2], [3, 4]]),            # 极小样本
    np.array([[0, 5], [5, 0]]),            # 零单元格, 边缘>0
    np.array([[100, 0], [0, 100]]),        # 零单元格
    np.array([[3, 3], [3, 3]]),            # 完全均衡 -> chi2=0, p=1
    np.array([[1, 1], [1, 1]]),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),   # 3x3 稀疏
    np.array([[1000, 1000], [1000, 1000]]),         # 大样本均衡
    np.array([[2, 98], [98, 2]]),           # 强相关
    np.array([[5, 10, 15], [10, 20, 30]]),  # 成比例 -> p=1
    np.array([[7, 11], [13, 17]]),
    np.array([[12, 8, 5], [3, 15, 9], [4, 2, 20]]),
]


def _random_tables(seed, count, shapes, ns):
    """确定性随机考卷; 与 L3.draw_pos 相同协议: 零和行/列重抽 (确定性 rng)."""
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(count):
        r, c = shapes[i % len(shapes)]
        n = ns[i % len(ns)]
        rp = rng.random(r) + 0.05
        rp /= rp.sum()
        cp = rng.random(c) + 0.05
        cp /= cp.sum()
        p = np.outer(rp, cp)
        for _trial in range(200):
            tbl = rng.multinomial(n, p.ravel()).reshape(r, c)
            if np.all(tbl.sum(axis=1) > 0) and np.all(tbl.sum(axis=0) > 0):
                break
        else:
            raise RuntimeError("_random_tables: no all-positive-margin table within 200 trials")
        tables.append(tbl)
    return tables


def _chi2_l2_inputs():
    seeded = _random_tables(
        L2_TABLE_SEED, count=28,
        shapes=[(2, 2), (3, 3), (2, 4), (4, 5), (2, 3), (4, 4), (5, 5), (3, 5)],
        ns=[10, 30, 100, 1000, 5000, 1000, 2000, 3000],
    )
    tables = _HAND_PICKED + seeded
    assert len(tables) == 40
    for t in tables:
        assert np.all(t.sum(axis=1) > 0) and np.all(t.sum(axis=0) > 0)
    return [{"type": "contingency_table", "table": t.tolist()} for t in tables]


# ---------------------------------------------------------------------------
# two_samples 考卷 (同协议, 两样本形态)
# ---------------------------------------------------------------------------

_WILCOXON_HAND_PICKED = [
    ([1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]),      # 完全分离
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),                           # 小样本分离
    ([0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35]),                   # 交错
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),                           # 同分布 -> p=1
    ([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),  # 并列全等
    ([1.0, 2.0], [3.0, 4.0]),                                      # 最小样本
    ([0.5] * 5, [1.5] * 5),                                        # 常数组
    ([-3.0, -2.0, -1.0], [1.0, 2.0, 3.0]),                         # 负值
    ([0.0, 0.0, 0.0], [1e-12, 2e-12, 3e-12]),                      # 极小值
    ([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
     [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),                # 交替
    ([100.0, 200.0], [100.5, 200.5]),                              # 大偏移近邻
    ([0.5, 0.5], [0.5, 0.5]),                                      # 全等 -> p=1
]

_WILCOXON_L2_SIZES = [(5, 5), (8, 10), (10, 15), (20, 25),
                      (10, 30), (15, 15), (25, 25), (12, 18)]


def _wilcoxon_l2_inputs():
    rng = np.random.default_rng(L2_TABLE_SEED)
    inputs = [{"type": "two_samples", "x": list(x), "y": list(y)}
              for x, y in _WILCOXON_HAND_PICKED]
    for i in range(28):
        n1, n2 = _WILCOXON_L2_SIZES[i % len(_WILCOXON_L2_SIZES)]
        inputs.append({"type": "two_samples",
                       "x": rng.uniform(0.0, 1.0, n1).tolist(),
                       "y": rng.uniform(0.0, 1.0, n2).tolist()})
    assert len(inputs) == 40
    return inputs


# ---------------------------------------------------------------------------
# 编译
# ---------------------------------------------------------------------------

_PROTOCOL = (
    "双参照: 规格 reference.refs 登记的两个独立实现 (spsl.golden 注册表). "
    "参照自检: 已知答案核对 (±self_check_tol) + 双参照在全部考卷一致到 agree_tol "
    "(1e-9, v1 REF_AGREE_TOL = calibrator/reference.py:118); 任一不过 -> "
    "REF_ABORT (参照层不可信, 中止, 如实报告; v1 calibrator/l2.py:64-71 同款). "
    "对拍: 候选 p 值与两参照都一致到 1e-6 (v1 L2_TOL, calibrator/l2.py:11), "
    "否则 FAIL. 考卷全部为合法输入 (计数表边缘全 > 0; 两样本 n1/n2 >= 2 有限值).")


def build_exam(spec: dict) -> dict:
    """规格 -> L2 考卷 section dict (嵌入全部考卷输入 + 参照/容差配置)."""
    spec = validate_spec(spec)
    itype = spec["inputs"]["type"]
    if itype == "contingency_table":
        inputs = _chi2_l2_inputs()
    elif itype == "two_samples":
        inputs = _wilcoxon_l2_inputs()
    else:
        raise ValueError(f"compile_l2: unsupported input type {itype!r}")
    ref = spec["reference"]
    return {
        "layer": "L2",
        "n_inputs": len(inputs),
        "inputs": inputs,
        "refs": list(ref["refs"]),
        "agree_tol": ref["agree_tol"],
        "self_check": ref["self_check"],
        "self_check_tol": ref["self_check_tol"],
        "tol": PAIR_TOL,
        "protocol": _PROTOCOL,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.compile_l2",
        description="SPSL: spec JSON -> L2 dual-reference pairwise exam JSON (Northstar v2 D2)")
    parser.add_argument("spec", help="规格 JSON 路径")
    parser.add_argument("--out", required=True, help="输出考卷 section JSON 路径")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    sec = build_exam(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sec, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[spsl] L2 compiled: {args.spec} -> {args.out}")
    print(f"  family: {spec['name']}  inputs: {spec['inputs']['type']}  "
          f"exam {sec['n_inputs']} inputs")
    print(f"  refs: {sec['refs']}  self-check agree tol={sec['agree_tol']}  "
          f"known answers {len(sec['self_check'])}  pairwise tol={sec['tol']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
