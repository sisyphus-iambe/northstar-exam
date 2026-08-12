#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 r1_faers_seed100.csv: 实验 9a (bootstrap_monotone) R1 规则测试数据.

数据来源 (只读引用, 不复制):
  /Users/nister/Desktop/北极星/FAERS/results/counts.pkl   Counter[(year,drug,rxn)] -> n
  /Users/nister/Desktop/北极星/FAERS/results/signals.pkl  (S24, S25, metas)
  (experiments/bootstrap_monotone/run_bootstrap_monotone.py 的 load 对象, 不 import
   judge1_common — 它要求 MEMGRAPH_API_KEY 且写入 FAERS 目录日志)

CSV 格式 (verifytool construct 输入): pair,year,a,b,c,d,label
  每对 (S24 全池 12699 对) x 两年各一行 2x2 四格表:
    a = 该药该事件计数, b = 该药其他事件, c = 其他药该事件, d = 其余
    (分母 = 整年总数, 与 features_for_pairs / judge1_common 四格表公式逐位同款)
    label = 1 当且仅当 对 ∈ S25 (每对恒定, 两行同值)

确定性: 无随机数, 两次运行逐字节一致 (同 Python 版本同 numpy).
用法: python3 gen_r1_faers_csv.py [--out r1_faers_seed100.csv]
输出大小: ~12699 对 x 2 年 = 25398 行.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

FAERS_RESULTS = Path("/Users/nister/Desktop/北极星/FAERS/results")


def features_cells(c: dict, year: int, pairs: list) -> dict:
    """对每对计算 (a, b, c, d) — features_for_pairs 同款四格表 (judge1 公式), 逐位一致.
    b = 该药整年总数 - a, c = 该事件整年总数 - a, d = 整年总数 - b - c - a."""
    yc = {(d, r): n for (y, d, r), n in c.items() if y == year}
    dtot = {}
    etot = {}
    for (d, r), n in yc.items():
        dtot[d] = dtot.get(d, 0) + n
        etot[r] = etot.get(r, 0) + n
    grand = sum(yc.values())
    out = {}
    for p in pairs:
        a = yc.get(p, 0)
        b = dtot.get(p[0], 0) - a
        cc = etot.get(p[1], 0) - a
        d = grand - b - cc - a
        out[p] = (a, b, cc, d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 r1_faers_seed100.csv (构造器演示数据)")
    ap.add_argument("--out", default="r1_faers_seed100.csv")
    ap.add_argument("--faers-results", default=str(FAERS_RESULTS))
    args = ap.parse_args()

    c, _rows = pickle.load(open(Path(args.faers_results) / "counts.pkl", "rb"))
    s24, s25, _metas = pickle.load(open(Path(args.faers_results) / "signals.pkl", "rb"))

    pairs = sorted(s24)                       # make_split 的采样池 (sorted, 同实验)
    cells24 = features_cells(c, 2024, pairs)
    cells25 = features_cells(c, 2025, pairs)

    import csv

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "year", "a", "b", "c", "d", "label"])
        for p in pairs:
            label = 1 if p in s25 else 0
            pair_s = f"{p[0]}|{p[1]}"
            a24, b24, c24, d24 = cells24[p]
            a25, b25, c25, d25 = cells25[p]
            w.writerow([pair_s, 2024, a24, b24, c24, d24, label])
            w.writerow([pair_s, 2025, a25, b25, c25, d25, label])
    print(f"written: {out.resolve()} ({len(pairs)} 对 x 2 年 = {2 * len(pairs)} 行)")
    print(f"|S24|={len(s24)} |S25|={len(s25)} 正类率={sum(1 for p in pairs if p in s25) / len(pairs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
