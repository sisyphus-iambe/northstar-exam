"""校准层 — L2: 参照对拍 (审计文件漏洞 1 修正)。

固定考卷 (确定性, 无随机): 手选经典表 + 种子化随机表, 全部边缘 > 0。
候选 p 值与两个独立参照 (手写解析公式 + scipy) 都须一致到 1e-6。
先自检: 两个参照自身在全部考卷上一致到 1e-9, 否则校准层拒绝采信 (abort)。
"""
import numpy as np

from .reference import ref_hand, ref_scipy

L2_TOL = 1e-6          # 对拍阈值 (规格)
L2_TABLE_SEED = 20260806

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
            raise RuntimeError("_random_tables: 200 trials 内未抽到边缘全正表")
        tables.append(tbl)
    return tables


def build_l2_tables():
    seeded = _random_tables(
        L2_TABLE_SEED, count=28,
        shapes=[(2, 2), (3, 3), (2, 4), (4, 5), (2, 3), (4, 4), (5, 5), (3, 5)],
        ns=[10, 30, 100, 1000, 5000, 1000, 2000, 3000],
    )
    return _HAND_PICKED + seeded


def run_l2(candidate):
    """返回 (verdict, diagnostics). verdict 仅在参照自检通过且全表一致时 True."""
    tables = build_l2_tables()
    # 参照自检: 两个独立参照必须一致, 否则校准层自身不可信 (漏洞 1)
    worst_ref_dev, bad_table = refs_agree_l2(tables)
    if worst_ref_dev > 1e-9:
        return False, {
            "ref_abort": True,
            "worst_ref_dev": worst_ref_dev,
            "n_tables": len(tables),
        }
    max_dev = 0.0
    devs = []
    for t in tables:
        cand_p = float(candidate(t))
        dev = max(abs(cand_p - ref_hand(t)), abs(cand_p - ref_scipy(t)))
        devs.append(dev)
        max_dev = max(max_dev, dev)
    return bool(max_dev <= L2_TOL), {
        "ref_abort": False,
        "worst_ref_dev": worst_ref_dev,
        "max_dev": max_dev,
        "n_tables": len(tables),
        "n_viol": int(sum(d > L2_TOL for d in devs)),
    }


def refs_agree_l2(tables):
    worst = 0.0
    for t in tables:
        worst = max(worst, abs(ref_hand(t) - ref_scipy(t)))
    return worst, None
