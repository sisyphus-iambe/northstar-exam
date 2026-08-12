"""校准层 — L3: 边界泛化 (审计文件漏洞 4 修正: L3 只测形态泛化).

不同样本量 / 不同边缘分布形状 / 零单元格 (观测 0, 边缘>0) / 大表。
与 L2 相同: 与两个独立参照一致到 1e-6; 固定考卷 (确定性, 无随机)。
零行/零列退化表排除 (scipy 抛 ValueError, 协议边界, 见 reference.py)。
"""
import numpy as np

from .reference import ref_hand, ref_scipy

L3_TOL = 1e-6
L3_TABLE_SEED = 20260806
L3_N_REGS = 76


def build_l3_tables():
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
        raise RuntimeError("draw_pos: 200 trials 内未抽到边缘全正表")

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
    return tables


def run_l3(candidate):
    """返回 (verdict, diagnostics)."""
    tables = build_l3_tables()
    worst_ref_dev = max(abs(ref_hand(t) - ref_scipy(t)) for t in tables)
    if worst_ref_dev > 1e-9:
        return False, {"ref_abort": True, "worst_ref_dev": worst_ref_dev,
                       "n_tables": len(tables)}
    devs = []
    for t in tables:
        cand_p = float(candidate(t))
        devs.append(max(abs(cand_p - ref_hand(t)), abs(cand_p - ref_scipy(t))))
    max_dev = max(devs)
    return bool(max_dev <= L3_TOL), {
        "ref_abort": False,
        "worst_ref_dev": worst_ref_dev,
        "max_dev": max_dev,
        "n_tables": len(tables),
        "n_viol": int(sum(d > L3_TOL for d in devs)),
    }
