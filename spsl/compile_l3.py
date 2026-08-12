"""SPSL L3 编译 — 规格 JSON -> L3 边界泛化考卷 JSON (北极星 v2, D2).

用法: python3 -m spsl.compile_l3 spec.json --out l3_section.json

协议 (北极星v2.md v1 基线 + D2): 与 L2 相同判定 (双参照自检 1e-9 +
REF_ABORT; 候选 vs 两参照 1e-6), 考卷侧重边界泛化:
  5 形态 × 76 表 (v1 L3 = 5 形态 76 表, calibrator/l3.py:13-91 同款):
    contingency_table (逐字同款 v1):
      形态 1 (15): 极小样本 n=15 (2x2/2x3)
      形态 2 (16): 大样本 n=20000 (3x4/4x5/4x4/5x5)
      形态 3 (10): 强偏斜边缘 (比例 1:100 量级)
      形态 4 (15): 零单元格 (10 随机 + 5 手设, 边缘仍 > 0)
      形态 5 (20): 中样本混合形状
    two_samples (同协议两样本形态, 同样 5 形态 × 76):
      形态 1 (15): 极小样本 n1,n2 ∈ {2..5}
      形态 2 (16): 大样本 n1,n2 ∈ {300..500}
      形态 3 (10): 强不均衡 (10 vs 5000 / 5 vs 2000)
      形态 4 (15): 边界手设 (同分布/完全分离/常数组/交替) 5 + 随机极小 10
      形态 5 (20): 中样本混合
  全部考卷合法输入 (计数表边缘全 > 0; 两样本 n >= 2 有限值).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spsl.schema import validate_spec

L3_TABLE_SEED = 20260806          # v1 同款 (calibrator/l3.py:12)
PAIR_TOL = 1e-6                   # 对拍容差 (v1 L3_TOL, calibrator/l3.py:11)
L3_N_REGS = 76

# ---------------------------------------------------------------------------
# contingency_table 考卷 (逐字同款 calibrator/l3.py:16-91)
# ---------------------------------------------------------------------------


def _chi2_l3_inputs():
    rng = np.random.default_rng(L3_TABLE_SEED)
    tables = []

    def add(t):
        tables.append(np.asarray(t, dtype=float))

    def draw_pos(r, c, n, rp, cp):
        """协议边界: 零和行/列不考, 重抽到边缘全 > 0 (确定性 rng)."""
        for _trial in range(200):
            tbl = rng.multinomial(n, np.outer(rp, cp).ravel()).reshape(r, c)
            if np.all(tbl.sum(axis=1) > 0) and np.all(tbl.sum(axis=0) > 0):
                return tbl
        raise RuntimeError("draw_pos: no all-positive-margin table within 200 trials")

    # --- 形态 1: 极小样本 n=15 ---
    for _ in range(15):
        r, c = (2, 2) if _ % 2 == 0 else (2, 3)
        rp = rng.random(r) + 0.1
        rp /= rp.sum()
        cp = rng.random(c) + 0.1
        cp /= cp.sum()
        add(draw_pos(r, c, 15, rp, cp))

    # --- 形态 2: 大样本 n=20000 (含 4x4/5x5 方表: 审计 P3 形状覆盖缺口) ---
    for _ in range(16):
        r, c = [(3, 4), (4, 5), (4, 4), (5, 5)][_ % 4]
        rp = rng.random(r) + 0.05
        rp /= rp.sum()
        cp = rng.random(c) + 0.05
        cp /= cp.sum()
        add(draw_pos(r, c, 20000, rp, cp))

    # --- 形态 3: 强偏斜边缘 (比例 1:100 量级) ---
    skewed_rows = [np.array([0.98, 0.02]), np.array([0.95, 0.04, 0.01])]
    skewed_cols = [np.array([0.97, 0.02, 0.01]), np.array([0.9, 0.09, 0.01])]
    for _ in range(10):
        rp = skewed_rows[_ % 2]
        cp = skewed_cols[_ % 2]
        n = [1000, 300][_ % 2]
        add(draw_pos(len(rp), len(cp), n, rp, cp))

    # --- 形态 4: 零单元格 (观测 0, 边缘>0) ---
    for _ in range(10):
        r, c = (2, 3) if _ % 2 == 0 else (3, 3)
        rp = rng.random(r) + 0.1
        rp /= rp.sum()
        cp = rng.random(c) + 0.1
        cp /= cp.sum()
        tbl = draw_pos(r, c, 25, rp, cp)
        # 强制至少一个观测 0 (边缘仍 >0)
        if np.all(tbl > 0):
            idx = np.unravel_index(int(rng.integers(0, tbl.size)), tbl.shape)
            tbl[idx] = 0
        add(tbl)
    # 手设零单元格表
    add([[0, 8], [3, 9]])
    add([[12, 0], [4, 11]])
    add([[0, 1, 30], [2, 4, 60], [1, 3, 15]])
    add([[0, 0, 20], [1, 2, 40]])
    add([[25, 0, 0, 7], [3, 10, 12, 9], [5, 6, 2, 4]])

    # --- 形态 5: 中样本混合形状 (含 4x4/5x5/3x5: 审计 P3 形状覆盖缺口) ---
    for _ in range(20):
        r, c = [(2, 2), (3, 3), (2, 4), (4, 5), (4, 4), (5, 5), (3, 5)][_ % 7]
        rp = rng.random(r) + 0.2
        rp /= rp.sum()
        cp = rng.random(c) + 0.2
        cp /= cp.sum()
        add(draw_pos(r, c, [50, 200, 800][_ % 3], rp, cp))

    assert len(tables) == L3_N_REGS, f"L3 table count {len(tables)} != {L3_N_REGS}"
    # 协议边界: 全部考卷边缘 > 0 (零和行/列 -> scipy 抛 ValueError, 不考)
    for t in tables:
        assert np.all(np.asarray(t).sum(axis=1) > 0) and np.all(np.asarray(t).sum(axis=0) > 0)
    return [{"type": "contingency_table", "table": t.tolist()} for t in tables]


# ---------------------------------------------------------------------------
# two_samples 考卷 (同协议, 5 形态 × 76)
# ---------------------------------------------------------------------------

_TINY = [(2, 2), (2, 3), (3, 3), (3, 4), (2, 5)]
_BIG = [(300, 400), (400, 500), (400, 400), (500, 500)]
_MID = [(15, 15), (20, 20), (25, 25), (30, 40), (25, 45), (40, 40), (20, 50)]
_L3_HAND_BOUNDARY = [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),      # 同分布 -> p=1
    ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),     # 完全分离
    ([0.5] * 10, [1.5] * 10),                # 常数组 (全并列)
    ([1, 2, 3], [1, 2, 3]),                  # 同分布小样本
    ([1, 3, 5], [2, 4, 6]),                  # 交替
]


def _wilcoxon_l3_inputs():
    rng = np.random.default_rng(L3_TABLE_SEED)
    inputs = []

    def add(x, y):
        inputs.append({"type": "two_samples",
                       "x": [float(v) for v in x],
                       "y": [float(v) for v in y]})

    # --- 形态 1: 极小样本 (15) ---
    for i in range(15):
        n1, n2 = _TINY[i % len(_TINY)]
        add(rng.uniform(0.0, 1.0, n1), rng.uniform(0.0, 1.0, n2))

    # --- 形态 2: 大样本 (16) ---
    for i in range(16):
        n1, n2 = _BIG[i % len(_BIG)]
        add(rng.uniform(0.0, 1.0, n1), rng.uniform(0.0, 1.0, n2))

    # --- 形态 3: 强不均衡 (10) ---
    for i in range(10):
        n1, n2 = (10, 5000) if i % 2 == 0 else (5, 2000)
        add(rng.uniform(0.0, 1.0, n1), rng.uniform(0.0, 1.0, n2))

    # --- 形态 4: 边界手设 5 + 随机极小 10 (15) ---
    for x, y in _L3_HAND_BOUNDARY:
        add(x, y)
    for i in range(10):
        n1, n2 = _TINY[i % len(_TINY)]
        add(rng.uniform(0.0, 1.0, n1), rng.uniform(0.0, 1.0, n2))

    # --- 形态 5: 中样本混合 (20) ---
    for i in range(20):
        n1, n2 = _MID[i % len(_MID)]
        scale = [50, 200, 800][i % 3]
        n = max(2, round(scale * n1 / (n1 + n2)))
        m = max(2, scale - n)
        add(rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, m))

    assert len(inputs) == L3_N_REGS
    return inputs


# ---------------------------------------------------------------------------
# 编译
# ---------------------------------------------------------------------------

_PROTOCOL = (
    "与 L2 相同判定 (双参照自检 1e-9 + REF_ABORT; 候选 vs 两参照 1e-6), "
    "考卷侧重边界泛化: 5 形态 × 76 输入 (v1 L3 = 5 形态 76 表, "
    "calibrator/l3.py:13-91 同款): 极小样本 / 大样本 / 强偏斜边缘 (两样本: "
    "强不均衡) / 零单元格 (两样本: 边界手设) / 中样本混合. 全部合法输入.")


def build_exam(spec: dict) -> dict:
    """规格 -> L3 考卷 section dict (嵌入 76 个考卷输入 + 参照/容差配置)."""
    spec = validate_spec(spec)
    itype = spec["inputs"]["type"]
    if itype == "contingency_table":
        inputs = _chi2_l3_inputs()
    elif itype == "two_samples":
        inputs = _wilcoxon_l3_inputs()
    else:
        raise ValueError(f"compile_l3: unsupported input type {itype!r}")
    ref = spec["reference"]
    return {
        "layer": "L3",
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
        prog="python3 -m spsl.compile_l3",
        description="SPSL: spec JSON -> L3 boundary-generalization exam JSON (Northstar v2 D2)")
    parser.add_argument("spec", help="规格 JSON 路径")
    parser.add_argument("--out", required=True, help="输出考卷 section JSON 路径")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    sec = build_exam(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sec, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[spsl] L3 compiled: {args.spec} -> {args.out}")
    print(f"  family: {spec['name']}  inputs: {spec['inputs']['type']}  "
          f"exam {sec['n_inputs']} inputs (5 shapes)")
    print(f"  refs: {sec['refs']}  self-check agree tol={sec['agree_tol']}  "
          f"pairwise tol={sec['tol']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
