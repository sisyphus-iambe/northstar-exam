"""剪枝调度 (④): 三臂调度 FULL/MAP_CRIT/ONESHOT + Fisher 合并 + BH 判定.

规格: SPEC_插件扩展_2026-08-07.md §4.

纯函数逐行复制自 experiments/design_prune/design_prune_v7c.py (出处标注,
必须逐位一致 — 那是 v6/v7 锚点红线的依据): gen_seed / gen_m1_table /
gen_m7_table / gen_m6_table / cand_form / GEN / topup_full_order /
combine_k / topup_crit_recursive / topup_crit_recursive_mech /
topup_oneshot / bh_correct / bh_stats / bh_method_a / run_arm / arm_stats.

合成世界 (gen_seed): 200 候选 = 70 超弱 M1 d0.1_n50 + 70 中弱 M1 d0.2_n50
+ 30 盲点 M7 5x5_add0.05 + 30 盲点 M6 n10_d0.15; 每候选真概率 0.5;
SLOTS=8 (每候选 8 张预生成表, 第 k 次加投用 slot k). 检验 = template_fisher
(Fisher 精确, programs/ 只读加载).

三臂 (判据口径与 v7/4h 一致, BH alpha=0.05, m=候选数):
  FULL     = 全部候选首检 (slot 0) + 剩余 (B-N) 次随机加投 (独立 rng 流
             default_rng(seed*7919+BUDGET).permutation(N_CAND)[:B-N], slot 1)
  MAP_CRIT = 140 可验证首检 + 递归临界救援 (topup_crit_recursive: 每轮池 =
             未检出候选按组合 p 升序, 取 min(剩余, len(池)) 全部加投, 每加投
             立即更新组合 p; 盲点永不加投)
  ONESHOT  = 140 首检 + 单轮: 未检出按首检 p 升序每候选最多加投 1 次,
             取前 min(B-140, len(池)) 个, 剩余预算不用 (n_tests <= B)

输入: --world synthetic (默认, 复现 v7 口径) 或 --cands <p.json>
  p.json: {"p_first": [N 候选首检 p], "p_slots": [[slot1 p...], [slot2 p...], ...],
           "truth": [0/1...] 可选}
  cands 模式 = 全部候选可验证 (都有 p), 槽位有限; 无 truth 时不产出真/假检出
  (只给 n_detected 与轨迹, 如实).
输出: 三臂真检出/fp 对比 + 调度轨迹, JSON+HTML.

语义说明 (如实): 本子命令 = 剪枝调度的通用执行器; 4h 结论"部分成立"照录
(方向同向, 零 fp 未复制 — finance_weaksig/ 真实数据 FULL 71真/4fp vs
MAP_CRIT 73真/5fp, 差异 +2 真/+1 fp, 无显著性检验), 每份输出引用出处.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from verifytool import VERSION

# 让 `python -m verifytool` 从任意 cwd 都能 import 包内模块
_CALIB_EXP = Path(__file__).resolve().parent.parent
if str(_CALIB_EXP) not in sys.path:
    sys.path.insert(0, str(_CALIB_EXP))

# ===========================================================================
# v7c 复制段 (逐行一致; 出处: experiments/design_prune/design_prune_v7c.py)
# ===========================================================================

# ---------------- 世界参数 (v5 规格固定, 与 v4/v3 逐字一致) ----------------
SEED_BASE = 20260806
ALPHA = 0.05
N_CAND = 200                 # 候选/seed
N_WEAK = 70                  # 超弱档 M1 d0.1_n50 (coverage2 blind_spots.M1[0]: fisher=0.20)
N_MED = 70                   # 中弱档 M1 d0.2_n50 (coverage2 blind_spots.M1[1]: fisher=0.76)
N_VER = N_WEAK + N_MED       # 140 可验证形态候选 (M1 两档)
N_BLIND_M7 = 30              # 盲点 M7 加性 5x5
N_BLIND_M6 = 30              # 盲点 M6 稀疏 n10_d0.15 (coverage2 blind_spots.M6[1]: fisher=0.08)
N_SKIP = 60                  # 世界参数保留 (与 v5 一致; v6 不用 — 加投次数 = BUDGET - 首检数)
SLOTS = 8                    # v6 改动 1: 槽位扩到 8 (v5 为 2)

P_TRUE = 0.5                 # 每候选真概率 (独立)
N_CELL_WEAK = 50             # 超弱档每格样本数 (M1 d0.1_n50)
DELTA_WEAK = 0.1             # 超弱档 M1 真效应
N_CELL_MED = 50              # 中弱档每格样本数 (M1 d0.2_n50)
DELTA_MED = 0.2              # 中弱档 M1 真效应
N_CELL_M6 = 10               # M6 n10_d0.15 每格样本数 (coverage2 configs.M6.n10_d0.15)
DELTA_M6 = 0.15              # M6 真效应
P_BASE = 0.3                 # 对照/零效应风险 (考卷 M1/M6/M9 同款)
M7_RISK_BASE = 0.25          # M7 加性风险基底 (考卷 gen_m7: 0.25 + 0.05i + 0.05j)
M7_STEP = 0.05
M7_N_CELL = 40               # M7 每格样本数 (考卷 gen_m7)

# 强度分层 (SMART 加投顺序; v6 不用 SMART 但保留): 超弱档先, 中弱档后
TIERS = ((0, N_WEAK), (N_WEAK, N_VER))   # 超弱 0..69, 中弱 70..139


# ---------------- 形态生成器 (考卷同款, 与 v5 逐行一致) ----------------

def gen_m1_table(rng, true: bool, n_cell: int, delta: float) -> np.ndarray:
    """M1 风险差异 (考卷 gen_m1 同款): 每格 n_cell 样本, 暴露 p=0.3+delta; 原生 2x2."""
    d = delta if true else 0.0
    t = np.array([rng.multinomial(2 * n_cell, [P_BASE + d, 1 - P_BASE - d]),
                  rng.multinomial(2 * n_cell, [P_BASE, 1 - P_BASE])], dtype=float)
    return t


def gen_m7_table(rng, true: bool) -> np.ndarray:
    """M7 加性 5x5 (考卷 gen_m7 同款): 风险 0.25+0.05i+0.05j, 每格 binomial(40,·).
    适配器 = 行{2,3,4}vs{0,1} 合并后端点对比 (col4 vs col0) → 2x2, 与考卷 template_fisher 输入一致.
    假=零效应: 同生成器 delta→0, 全表平风险 0.25."""
    risks = M7_RISK_BASE + (M7_STEP if true else 0.0) * np.arange(5)[:, None] \
        + (M7_STEP if true else 0.0) * np.arange(5)[None, :]
    t = np.array([[rng.binomial(M7_N_CELL, risks[i, j]) for j in range(5)]
                  for i in range(5)], dtype=float)
    pooled = np.array([t[0] + t[1], t[2] + t[3] + t[4]], dtype=float)   # 2x5
    ext = np.array([[pooled[0, 4], pooled[1, 4]], [pooled[0, 0], pooled[1, 0]]],
                   dtype=float)                                          # 2x2 端点对比
    return ext


def gen_m6_table(rng, true: bool) -> np.ndarray:
    """M6 稀疏 n10_d0.15 (考卷 gen_m6 同款): 每格 10 样本, 真=暴露 p=0.3+0.15=0.45; 原生 2x2."""
    d = DELTA_M6 if true else 0.0
    t = np.array([rng.multinomial(2 * N_CELL_M6, [P_BASE + d, 1 - P_BASE - d]),
                  rng.multinomial(2 * N_CELL_M6, [P_BASE, 1 - P_BASE])], dtype=float)
    return t


# 候选布局: 0..69 = 超弱 M1 d0.1_n50, 70..139 = 中弱 M1 d0.2_n50, 140..169 = 盲点 M7,
# 170..199 = 盲点 M6 (固定, 确定性)
def cand_form(c: int) -> str:
    if c < N_WEAK:
        return "WEAK"
    if c < N_VER:
        return "MED"
    if c < N_VER + N_BLIND_M7:
        return "M7"
    return "M6"


GEN = {
    "WEAK": lambda rng, true: gen_m1_table(rng, true, N_CELL_WEAK, DELTA_WEAK),
    "MED": lambda rng, true: gen_m1_table(rng, true, N_CELL_MED, DELTA_MED),
    "M7": gen_m7_table,
    "M6": gen_m6_table,
}


def gen_seed(seed: int):
    """生成一个 seed 的完整世界: 真值向量、RAND 跳过集、MAP_RAND/RAND 加投集、全部 (候选,槽位) 表.
    单 rng 流, 固定抽取顺序 (truth → rand_skip → topup_map_rand → topup_rand → 表), 确定性
    (与 v5 逐行一致 → 预算 200 档锚点). v6 改动 1: 表 = for c: for slot in range(SLOTS),
    slot 0/1 先生成, 生成顺序/内容与 v5 (for slot in (0,1)) 逐位一致; slot 2..7 为新增
    (消耗后续 rng). FULL 的随机加投顺序不在此 (独立 rng 流, 见 topup_full_order)."""
    rng = np.random.default_rng(seed)
    truth = rng.random(N_CAND) < P_TRUE                       # 1. 每候选真/假 (独立 p=0.5)
    rand_skip = rng.permutation(N_CAND)[:N_SKIP]              # 2. RAND 不验集 (60/200)
    skip_set = set(rand_skip.tolist())
    topup_map_rand = rng.permutation(N_VER)[:N_SKIP]          # 3. MAP_RAND 加投 (60/140 可验证)
    # 4. RAND 加投: 已首检候选 = 全部 200 减跳过集 (恒 140 个), 均匀随机不重复 (v3 规格明示,
    #    可落在盲点候选上; 与 v2 的仅限可验证不同)
    first_tested = [int(i) for i in range(N_CAND) if int(i) not in skip_set]
    topup_rand = rng.permutation(first_tested)[:N_SKIP]
    tables = {}                                               # 5. 表: (cand, slot) → 2x2
    # v6 槽位生成顺序关键决策: 字面嵌套循环 for c: for slot in range(SLOTS) 会使候选 c>=1
    # 的 slot 0/1 抽取位于候选 c-1 的 slot 2..7 之后 (新增槽位消耗后续 rng), 与 v5 的
    # slot 0/1 流 (for c: for slot in (0,1)) 不一致 → 200 档锚点必然断裂 (实测候选 0 一致,
    # 候选 1 起全部分歧). 因此 slot 0/1 先按 v5 原流全部生成 (逐位一致), slot 2..7 随后生成
    # (消耗 v5 流之后的 rng = 规格中"slot 2..7 是新增(消耗后续 rng)").
    for c in range(N_CAND):
        gen = GEN[cand_form(c)]
        for slot in (0, 1):                                   # 与 v5 逐位一致 (锚点)
            tables[(c, slot)] = gen(rng, bool(truth[c]))
    for c in range(N_CAND):
        gen = GEN[cand_form(c)]
        for slot in range(2, SLOTS):                          # 新增槽位 (v5 流之后)
            tables[(c, slot)] = gen(rng, bool(truth[c]))
    return {"truth": truth, "rand_skip": rand_skip,
            "topup_map_rand": topup_map_rand, "topup_rand": topup_rand, "tables": tables}


def topup_full_order(seed: int, budget: int) -> np.ndarray:
    """FULL 臂 (budget > 200 时) 的随机加投顺序: 独立 rng 流 default_rng(seed*7919 + budget)
    生成 permutation(N_CAND)[:budget-200], 与主 rng 无关 (不影响表生成 → 锚点), 确定性.
    均匀随机不重复: 预算 200 时为空; 预算 400 时 = 全池 200 各 +1 次 (每候选恰好 2 检, slot 1)."""
    n_topup = budget - N_CAND
    if n_topup <= 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(seed * 7919 + budget)
    return rng.permutation(N_CAND)[:n_topup]


def smart_topup(p_first: dict) -> list:
    """SMART 加投集 (v5 保留, v6 不用): 按强度分层 (超弱档先, 中弱档后), 层内按首检 p 值降序
    (最不显著优先). 稳定排序 (Python sorted 稳定, 等 p 保持候选升序). 盲点永不加投.
    本规格预算 N_SKIP=60 < 超弱档 70 个, 全部落在超弱档, 中弱档 0 加投."""
    topup: list[int] = []
    remaining = N_SKIP
    for start, end in TIERS:
        cands = list(range(start, end))
        ordered = sorted(cands, key=lambda c: p_first[c], reverse=True)
        take = min(remaining, len(ordered))
        topup.extend(ordered[:take])
        remaining -= take
        if remaining == 0:
            break
    return topup


def combine_k(ps: list[float]) -> float:
    """候选判定 p (v6 改动 3, Fisher 组合, 任意 k>=1 个 p): p 数 = k, 统计量 = -2*Σ ln p,
    自由度 2k, 返回 chi2.sf(stat, 2k). k=1 = 直判 (返回 p 本身, 与 v5 combine len==1 分支
    一致 — 实测 chi2.sf(-2 ln p, 2) 与 p 仅差 ~1 ulp 非位级相等, 直判保证锚点位级精确);
    k=2 与 v5 combine 一致 (chi2.sf(stat, 4)).
    p=0.0 → -2*ln(0)=inf → sf(inf)=0.0, 数学极限正确, 无需钳制."""
    if len(ps) == 1:
        return ps[0]
    stat = -2.0 * sum(np.log(p) for p in ps)
    return float(chi2.sf(stat, 2 * len(ps)))


def topup_crit_recursive(p_cur: dict[int, float], n_topup: int, next_p) -> tuple[list[int], int]:
    """递归临界救援 (v6 改动 3): 每轮池 = 未检出候选 (p_cur[c] >= ALPHA) 按 p_cur 升序
    (稳定排序, 等 p 保持候选升序), 取 min(剩余, len(池)) 个全部加投; 每加投一次立即更新
    p_cur[c] = combine_k(该候选全部 p) (加投后 p_cur 更新, 可能救回候选), 剩余递减;
    池空或剩余=0 停止. 盲点永不加投 (池只含 0..139).
    纯函数: 不消耗 rng, 不参与 rng 流 (新检验 p 值来自 next_p 回调, 读取预生成表).
    返回 (加投候选列表, 轮数): 列表顺序 = 实际加投顺序 (用于机制明细);
    轮数 = 实际执行了加投的轮次数 (用于 mechanism_detail.n_rounds_used)."""
    p_all: dict[int, list[float]] = {c: [p_cur[c]] for c in p_cur}
    topup: list[int] = []
    remaining = n_topup
    rounds = 0
    while remaining > 0:
        pool = sorted([c for c in p_cur if p_cur[c] >= ALPHA], key=lambda c: p_cur[c])
        if not pool:
            break
        take = min(remaining, len(pool))
        for c in pool[:take]:
            p_all[c].append(next_p(c))
            p_cur[c] = combine_k(p_all[c])
            topup.append(c)
        remaining -= take
        rounds += 1
    return topup, rounds


def topup_crit_recursive_mech(p_cur: dict[int, float], n_topup: int, next_p,
                              truth: np.ndarray) -> tuple[list[int], int, list[dict]]:
    """topup_crit_recursive 的机制明细版 (v7c): 行为与 topup_crit_recursive 完全一致
    (同一轮池定义/排序/更新/停止), 额外逐轮记录池构成 (n_pool/n_true_pool/n_false_pool/
    n_topup), 供 mechanism_detail.recur.per_round 聚合.
    返回 (加投候选列表, 轮数, per_round 明细列表)."""
    p_all: dict[int, list[float]] = {c: [p_cur[c]] for c in p_cur}
    topup: list[int] = []
    remaining = n_topup
    rounds = 0
    per_round: list[dict] = []
    while remaining > 0:
        pool = sorted([c for c in p_cur if p_cur[c] >= ALPHA], key=lambda c: p_cur[c])
        if not pool:
            break
        take = min(remaining, len(pool))
        n_t = int(sum(1 for c in pool if truth[c]))
        per_round.append({"round": rounds, "n_pool": len(pool), "n_true_pool": n_t,
                          "n_false_pool": len(pool) - n_t,
                          "false_ratio": float((len(pool) - n_t) / len(pool)),
                          "n_topup": take})
        for c in pool[:take]:
            p_all[c].append(next_p(c))
            p_cur[c] = combine_k(p_all[c])
            topup.append(c)
        remaining -= take
        rounds += 1
    return topup, rounds, per_round


def topup_oneshot(p_first: dict[int, float], n_topup: int, next_p) -> list[int]:
    """ONESHOT 单轮加投 (v7c 新增臂): 未检出候选 (首检 p >= ALPHA) 按 p 升序
    (稳定排序, 等 p 保持候选升序), 每候选最多加投 1 次, 取前 min(n_topup, len(池)) 个;
    剩余预算不用. 盲点永不在池 (池只含 0..139).
    纯函数: 不消耗 rng, 不参与 rng 流 (新检验 p 值来自 next_p 回调, 读取预生成表).
    返回加投候选列表 (实际加投顺序 = p 升序)."""
    pool = sorted([c for c in p_first if p_first[c] >= ALPHA], key=lambda c: p_first[c])
    take = min(n_topup, len(pool))
    topup = pool[:take]
    for c in topup:
        next_p(c)
    return topup


def run_arm(w, fisher, arm: str, budget: int, topup_order) -> dict:
    """跑一臂 (预算档 budget): 返回 {候选 → p 值列表}. 每次检验独立重采样(来自预生成表),
    Fisher 精确检验.
    FULL: 200 候选各 1 次首检 (slot 0) + topup_order (B-200 个, 各 slot 1 一次).
    MAP_CRIT: 140 可验证首检 (slot 0) + 递归临界救援加投 (topup_crit_recursive);
      候选第 k 次加投用 slot k (1-based), 超过 SLOTS 断言失败 (理论上不会); 盲点永不加投."""
    pvals: dict[int, list[float]] = {}
    if arm == "FULL":
        for c in range(N_CAND):
            pvals[c] = [fisher.chi2_pvalue(w["tables"][(c, 0)])]
        for c in topup_order:                                # 加投: 该候选 slot 1 (不重复)
            pvals[c].append(fisher.chi2_pvalue(w["tables"][(c, 1)]))
    elif arm == "MAP_CRIT":
        for c in range(N_VER):                               # 盲点 60 候选 0 次检验
            pvals[c] = [fisher.chi2_pvalue(w["tables"][(c, 0)])]
        p_cur = {c: pvals[c][0] for c in range(N_VER)}
        slots = {c: 1 for c in range(N_VER)}                 # 每候选下一加投槽位 (1-based)

        def next_p(c: int) -> float:
            s = slots[c]
            assert s < SLOTS, f"MAP_CRIT: 候选 {c} 加投超过 {SLOTS-1} 次 (槽位 {s})"
            slots[c] += 1
            v = fisher.chi2_pvalue(w["tables"][(c, s)])
            pvals[c].append(v)
            return v

        topup_crit_recursive(p_cur, budget - N_VER, next_p)  # 递归临界救援 (纯函数)
    elif arm == "ONESHOT":
        for c in range(N_VER):                               # 盲点 60 候选 0 次检验
            pvals[c] = [fisher.chi2_pvalue(w["tables"][(c, 0)])]
        p_cur = {c: pvals[c][0] for c in range(N_VER)}
        slots = {c: 1 for c in range(N_VER)}                 # 每候选下一加投槽位 (1-based)

        def next_p(c: int) -> float:
            s = slots[c]
            assert s < SLOTS, f"ONESHOT: 候选 {c} 加投超过 {SLOTS-1} 次 (槽位 {s})"
            slots[c] += 1
            v = fisher.chi2_pvalue(w["tables"][(c, s)])
            pvals[c].append(v)
            return v

        topup_oneshot(p_cur, budget - N_VER, next_p)  # 单轮: 未检出按 p 升序各 +1 次 (剩余预算不用)
    else:
        raise ValueError(f"v7c 未知臂: {arm}")
    return pvals


def arm_stats(w, pvals, truth: np.ndarray) -> dict:
    """按候选真值/形态汇总一臂的检出统计 (判定 = combine_k(全部检验 p) < ALPHA; 0 检验=不检出)."""
    n_tests = sum(len(ps) for ps in pvals.values())
    true_det = false_det = ver_true_det = blind_true_det = blind_false_det = 0
    for c in range(N_CAND):
        det = combine_k(pvals[c]) < ALPHA if pvals.get(c) else False
        if truth[c]:
            if det:
                true_det += 1
                if c < N_VER:
                    ver_true_det += 1
                else:
                    blind_true_det += 1
        elif det:
            false_det += 1
            if c >= N_VER:
                blind_false_det += 1
    return {"true_det": true_det, "false_det": false_det,
            "verifiable_true_det": ver_true_det, "blind_true_det": blind_true_det,
            "blind_false_det": blind_false_det, "n_tests": n_tests}


def bh_correct(p_final: dict[int, float], m: int, alpha: float = 0.05) -> dict[int, float]:
    """标准 Benjamini-Hochberg (与 v5 逐行一致): 返回每候选的 q 值 (保序回填).
    输入 p_final = {候选 -> 最终 p};缺失候选视为 p=1.0 补全 (调用方决定 m 口径).
    实现: 排序, q_(i) = min(1.0, p_(i) * m / rank), 从最大 rank 到最小做累积 min, 再回填到候选字典.
    纯函数: 不消耗 rng, 确定性."""
    full = dict(p_final)                                   # 不改变入参 (不可变风格)
    for i in range(m):                                     # 缺失候选 (0..m-1) 以 p=1.0 补全
        if i not in full:
            full[i] = 1.0
    items = sorted(full.items(), key=lambda kv: kv[1])     # 稳定排序: 等 p 保持候选升序 (入参按候选升序)
    n = len(items)                                         # 补全后恒 == m (调用方口径决定池子)
    q_map: dict[int, float] = {}
    running = 1.0
    for rank in range(n, 0, -1):                           # 从最大 rank 往最小做累积 min
        cand, p = items[rank - 1]
        qk = min(1.0, p * m / rank)
        running = min(running, qk)
        q_map[cand] = running
    return q_map


def bh_stats(truth: np.ndarray, q_map: dict[int, float], pool) -> dict:
    """按候选真值/形态汇总 BH 判定 (q<ALPHA=检出) 的检出统计 (与 v5 逐行一致).
    pool = 参与判定的候选集 (口径 A: 全 200). 池外候选不检出、不计入."""
    true_det = false_det = ver_true_det = blind_true_det = 0
    for c in pool:
        if q_map[c] >= ALPHA:
            continue
        if truth[c]:
            true_det += 1
            if c < N_VER:
                ver_true_det += 1
            else:
                blind_true_det += 1
        else:
            false_det += 1
    return {"true_det": true_det, "false_det": false_det,
            "verifiable_true_det": ver_true_det, "blind_true_det": blind_true_det}


def bh_method_a(pvals: dict, truth: np.ndarray) -> dict:
    """BH 判定 (口径 A 只, v6 改动 4): 全 200 候选入池, 未检验候选最终 p=1.0, m=200;
    每候选最终 p = combine_k(该候选全部检验 p). 返回 {"stats": 检出统计,
    "final_p": 全部 200 候选最终 p (未验=1.0)}. 纯函数: 不消耗 rng, 不参与 rng 流."""
    final_p = {c: (combine_k(pvals[c]) if pvals.get(c) else 1.0) for c in range(N_CAND)}
    q_map = bh_correct(final_p, N_CAND)
    return {"stats": bh_stats(truth, q_map, range(N_CAND)), "final_p": final_p}

# ===========================================================================
# v7c 复制段结束
# ===========================================================================

ARMS = ("MAP_CRIT", "ONESHOT", "FULL")


def _fisher():
    """programs/template_fisher.py 只读加载 (v7c load_module 同款, 返回模块对象 —
    run_arm 以 fisher.chi2_pvalue 调用, 保持 v7c 复制段逐行一致)."""
    import importlib.util

    path = _CALIB_EXP / "programs" / "template_fisher.py"
    spec = importlib.util.spec_from_file_location("template_fisher", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# cands 模式 (用户候选 p 值数组; 全部候选可验证, 槽位有限)
# ---------------------------------------------------------------------------

def run_schedule_cands(p_first: list, p_slots: list, truth: list, budget: int,
                       arm: str, seed: int) -> dict:
    """cands 模式三臂调度: p_first = N 个首检 p; p_slots[k] = 第 k 次加投的 N 个 p
    (k=0 为 slot 1). 全部候选首检. 语义与合成模式逐条对应:
      FULL = 首检全部 + topup_full_order 随机加投 (slot 1, 每候选最多 1 次)
      MAP_CRIT = 首检全部 + 递归临界救援 (每候选最多 len(p_slots) 次加投;
                 候选槽位用尽后不再入池)
      ONESHOT = 首检全部 + 单轮加投 (slot 1, 每候选最多 1 次)
    返回 {候选: p 值列表}."""
    n = len(p_first)
    pvals = {c: [p_first[c]] for c in range(n)}
    if arm == "FULL":
        # cands 模式 FULL 加投: 候选数 = n (非合成世界 N_CAND=200),
        # 独立 rng 流与 v7c 同款种子公式 (seed*7919+budget), 确定性
        n_topup = budget - n
        if n_topup > 0:
            rng = np.random.default_rng(seed * 7919 + budget)
            order = rng.permutation(n)[:n_topup]
        else:
            order = []
        for c in [int(c) for c in order]:
            pvals[c].append(p_slots[0][c] if p_slots else 1.0)
    elif arm in ("MAP_CRIT", "ONESHOT"):
        p_cur = {c: p_first[c] for c in range(n)}
        used = {c: 0 for c in range(n)}

        def next_p(c: int) -> float:
            k = used[c]
            used[c] += 1
            v = p_slots[k][c] if k < len(p_slots) else 1.0
            pvals[c].append(v)
            return v

        if arm == "MAP_CRIT":
            # 同 topup_crit_recursive 语义, 但池 = 未检出 且 有剩余槽位 (cands 槽位有限)
            p_all = {c: [p_cur[c]] for c in p_cur}
            remaining = budget - n
            while remaining > 0:
                pool = sorted([c for c in p_cur
                               if p_cur[c] >= ALPHA and used[c] < len(p_slots)],
                              key=lambda c: p_cur[c])
                if not pool:
                    break
                take = min(remaining, len(pool))
                for c in pool[:take]:
                    p_all[c].append(next_p(c))
                    p_cur[c] = combine_k(p_all[c])
                remaining -= take
        else:
            topup_oneshot(p_cur, budget - n, next_p)
    else:
        raise ValueError(f"未知臂: {arm}")
    return pvals


def _cands_stats(pvals: dict, truth) -> dict:
    """cands 模式统计: 有 truth 时给真/假检出, 无 truth 时只给 n_detected (BH q<0.05)."""
    n = len(pvals)
    final_p = {c: combine_k(pvals[c]) for c in range(n)}
    q_map = bh_correct(final_p, n)
    n_det = sum(1 for c in range(n) if q_map[c] < ALPHA)
    n_tests = sum(len(ps) for ps in pvals.values())
    out = {"n_detected": n_det, "n_tests": n_tests}
    if truth is not None:
        true_det = sum(1 for c in range(n) if q_map[c] < ALPHA and truth[c] == 1)
        false_det = sum(1 for c in range(n) if q_map[c] < ALPHA and truth[c] == 0)
        out["true_det"] = true_det
        out["false_det"] = false_det
    return out


# ---------------------------------------------------------------------------
# 合成世界模式编排
# ---------------------------------------------------------------------------

def run_prune_synthetic(budget: int, n_seeds: int, fisher) -> dict:
    """合成世界三臂 (v7c 同款): 每 seed 共享世界表, 三臂各自调度, BH 口径 A."""
    seeds = [SEED_BASE + i for i in range(n_seeds)]
    per_seed = []
    totals = {a: {"true_det": 0, "false_det": 0, "blind_true_det": 0,
                  "verifiable_true_det": 0, "n_tests": 0} for a in ARMS}
    n_true_total = n_false_total = 0
    recur_rounds_used = 0
    recur_per_round = {}
    oneshot_topup = 0
    full_topup_total = 0
    for seed in seeds:
        w = gen_seed(seed)
        truth = w["truth"]
        n_true_total += int(truth.sum())
        n_false_total += int((~truth).sum())
        full_order = topup_full_order(seed, budget)
        pvals = {}
        for arm in ARMS:
            pvals[arm] = run_arm(w, fisher, arm, budget,
                                 full_order if arm == "FULL" else None)
        entry = {"seed": seed, "arms": {}}
        for arm in ARMS:
            st = bh_method_a(pvals[arm], truth)["stats"]
            nt = sum(len(ps) for ps in pvals[arm].values())
            entry["arms"][arm] = {"true_det": st["true_det"],
                                  "false_det": st["false_det"],
                                  "blind_true_det": st["blind_true_det"],
                                  "verifiable_true_det": st["verifiable_true_det"],
                                  "n_tests": nt}
            for k in totals[arm]:
                totals[arm][k] += st[k] if k != "n_tests" else nt
        per_seed.append(entry)
        # 机制明细 (重跑确定性纯函数, 与臂运行校验)
        p_first = {c: pvals["MAP_CRIT"][c][0] for c in range(N_VER)}
        slots = {c: 1 for c in range(N_VER)}

        def next_p_mech(c: int) -> float:
            s = slots[c]
            assert s < SLOTS
            slots[c] += 1
            return fisher.chi2_pvalue(w["tables"][(c, s)])

        topup, rounds, per_round = topup_crit_recursive_mech(
            dict(p_first), budget - N_VER, next_p_mech, truth)
        assert N_VER + len(topup) == sum(len(ps) for ps in pvals["MAP_CRIT"].values())
        recur_rounds_used += rounds
        for pr in per_round:
            r = pr["round"]
            acc = recur_per_round.get(r)
            if acc is None:
                acc = [0, 0, 0, 0]
                recur_per_round[r] = acc
            acc[0] += pr["n_pool"]
            acc[1] += pr["n_true_pool"]
            acc[2] += pr["n_false_pool"]
            acc[3] += pr["n_topup"]
        p_first_on = {c: pvals["ONESHOT"][c][0] for c in range(N_VER)}
        slots_on = {c: 1 for c in range(N_VER)}

        def next_p_on(c: int) -> float:
            s = slots_on[c]
            assert s < SLOTS
            slots_on[c] += 1
            return fisher.chi2_pvalue(w["tables"][(c, s)])

        on_topup = topup_oneshot(dict(p_first_on), budget - N_VER, next_p_on)
        assert N_VER + len(on_topup) == sum(len(ps) for ps in pvals["ONESHOT"].values())
        oneshot_topup += len(on_topup)
        full_topup_total += int(len(full_order))

    per_round_list = []
    for r in sorted(recur_per_round.keys()):
        n_pool, n_true_pool, n_false_pool, n_topup = recur_per_round[r]
        per_round_list.append({"round": int(r), "n_pool": n_pool,
                               "n_true_pool": n_true_pool, "n_false_pool": n_false_pool,
                               "false_ratio": float(n_false_pool / n_pool) if n_pool else 0.0,
                               "n_topup": n_topup})
    aggregate = {}
    for arm in ARMS:
        t = totals[arm]
        aggregate[arm] = {
            "true_det_mean": t["true_det"] / n_seeds,
            "false_det_mean": t["false_det"] / n_seeds,
            "blind_true_det_mean": t["blind_true_det"] / n_seeds,
            "verifiable_true_det_mean": t["verifiable_true_det"] / n_seeds,
            "fp_rate": t["false_det"] / n_false_total,
            "recall": t["true_det"] / n_true_total,
            "n_tests_total": int(t["n_tests"]),
        }
    return {
        "mode": "synthetic",
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "trajectory": {
            "MAP_CRIT": {"n_rounds_used": recur_rounds_used,
                         "per_round": per_round_list},
            "ONESHOT": {"n_topup_done": oneshot_topup},
            "FULL": {"n_topup_total": full_topup_total},
        },
    }


# ---------------------------------------------------------------------------
# cands 模式编排
# ---------------------------------------------------------------------------

def run_prune_cands(p_first: list, p_slots: list, truth, budget: int,
                    n_seeds: int, seed_base: int) -> dict:
    """用户候选 p 模式: p_first = N 首检 p; p_slots[k] = 第 k 次加投 p 列表.
    三臂各跑 n_seeds 次 (仅 FULL 的随机加投顺序随 seed 变; MAP_CRIT/ONESHOT 确定性,
    与 seed 无关 — 如实呈现)."""
    n = len(p_first)
    seeds = [seed_base + i for i in range(n_seeds)]
    per_seed = []
    totals = {a: {"n_detected": 0, "n_tests": 0} for a in ARMS}
    if truth is not None:
        for a in totals:
            totals[a]["true_det"] = 0
            totals[a]["false_det"] = 0
    for seed in seeds:
        entry = {"seed": seed, "arms": {}}
        for arm in ARMS:
            pvals = run_schedule_cands(p_first, p_slots, truth, budget, arm, seed)
            st = _cands_stats(pvals, truth)
            entry["arms"][arm] = dict(st)
            for k in st:
                totals[arm][k] = totals[arm].get(k, 0) + st[k]
        per_seed.append(entry)
    aggregate = {}
    for arm in ARMS:
        agg = {"n_detected_mean": totals[arm]["n_detected"] / n_seeds,
               "n_tests_total": int(totals[arm]["n_tests"])}
        if truth is not None:
            agg["true_det_mean"] = totals[arm]["true_det"] / n_seeds
            agg["false_det_mean"] = totals[arm]["false_det"] / n_seeds
        aggregate[arm] = agg
    return {
        "mode": "cands",
        "n_cands": n,
        "n_slots": len(p_slots),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "trajectory": {"note": "cands 模式轨迹: FULL 加投顺序随 seed 变 (独立 rng 流); "
                               "MAP_CRIT/ONESHOT 为确定性调度 (与 seed 无关)."},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_prune(argv) -> int:
    """python3 -m verifytool prune [--world synthetic|--cands <p.json>]
    [--budget B] [--n-seeds N] [--out PATH]."""
    import time

    from verifytool.report import render_prune_html, save_json

    parser = argparse.ArgumentParser(
        prog="python3 -m verifytool prune",
        description="剪枝调度 (④): 三臂 FULL/MAP_CRIT/ONESHOT + Fisher 合并 + BH 判定. "
                    "合成世界 (v7c 同款) 或用户候选 p 值数组.")
    parser.add_argument("--world", default="synthetic", choices=["synthetic", "cands"],
                        help="synthetic = v7c 合成世界 (默认); cands = --cands <p.json>")
    parser.add_argument("--cands", default=None, help="用户候选 p.json: "
                        "{p_first: [...], p_slots: [[...], ...], truth: [0/1...] 可选}")
    parser.add_argument("--budget", type=int, default=N_CAND,
                        help=f"预算 B 检验/seed (默认 {N_CAND} = 候选数)")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="评估 seed 数 (默认 20; seeds = 20260806+i; 与 v7c 的 50 种子集同源)")
    parser.add_argument("--out", default=None, help="HTML 输出路径 (默认 cwd/prune_report.html)")
    args = parser.parse_args(argv)

    if args.world == "cands":
        if not args.cands:
            print("[verifytool prune] --world cands 需要 --cands <p.json>", file=sys.stderr)
            return 2
        cands_path = Path(args.cands)
        if not cands_path.exists():
            print(f"[verifytool prune] 候选 p.json 不存在: {cands_path}", file=sys.stderr)
            return 2
        try:
            cands = json.load(open(cands_path, encoding="utf-8"))
        except OSError as exc:
            print(f"[verifytool prune] 读取候选 p.json 失败: {exc}", file=sys.stderr)
            return 2
        p_first = cands.get("p_first")
        if not isinstance(p_first, list) or not p_first \
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           for x in p_first):
            print("[verifytool prune] p.json 须含 p_first (数字数组)", file=sys.stderr)
            return 2
        p_slots = cands.get("p_slots", [])
        for lst in p_slots:
            if not isinstance(lst, list) or len(lst) != len(p_first) \
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                               for x in lst):
                print("[verifytool prune] p.json 的 p_slots 每层须为与 p_first 等长的数字数组",
                      file=sys.stderr)
                return 2
        truth = cands.get("truth")
        if truth is not None:
            if not isinstance(truth, list) or len(truth) != len(p_first) \
                    or not all(x in (0, 1) for x in truth):
                print("[verifytool prune] p.json 的 truth 须为与 p_first 等长的 0/1 数组",
                      file=sys.stderr)
                return 2
    if args.budget < 1:
        print("[verifytool prune] --budget 必须 >= 1", file=sys.stderr)
        return 2
    if args.n_seeds < 1:
        print("[verifytool prune] --n-seeds 必须 >= 1", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    if args.world == "cands":
        body = run_prune_cands([float(x) for x in p_first],
                               [[float(x) for x in lst] for lst in p_slots],
                               truth, args.budget, args.n_seeds, SEED_BASE)
        # 预算断言: FULL/MAP_CRIT n_tests <= n_seeds*B 恒成立 (cands 槽位有限)
        for arm in ("FULL", "MAP_CRIT"):
            nt = body["aggregate"][arm]["n_tests_total"]
            assert nt <= args.n_seeds * args.budget, f"{arm} 检验数超预算"
        world_desc = f"cands ({cands_path.name})"
    else:
        fisher = _fisher()
        body = run_prune_synthetic(args.budget, args.n_seeds, fisher)
        assert body["aggregate"]["MAP_CRIT"]["n_tests_total"] == args.n_seeds * args.budget
        assert body["aggregate"]["FULL"]["n_tests_total"] == args.n_seeds * args.budget
        assert body["aggregate"]["ONESHOT"]["n_tests_total"] <= args.n_seeds * args.budget
        world_desc = "synthetic (v7c 同款合成世界)"

    body["budget"] = args.budget
    body["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    body["honesty_notes"] = [
        "剪枝调度通用执行器: 纯组装 v7c 已验证纯函数 (design_prune_v7c.py 逐行复制), 不产生新实验结果.",
        "4h 结论'部分成立'照录: 真实数据 finance_weaksig/ (θ=3%/PRR1.25 档, 92 候选 73 金标) "
        "FULL 71真/4fp vs MAP_CRIT 73真/5fp — 方向同向 (+2真), 零 fp 未复制 (+1 fp); "
        "单实例无显著性检验, 加投 p 自相关, 解读受限 (出处 results.json conclusion).",
        "合成世界 v7c 参考: B300 RECUR 29.74/FULL 29.72/ONESHOT 27.66, "
        "B400 RECUR 31.28/FULL 36.98/ONESHOT 27.66 (results_v7c.json aggregate); "
        "预算稀缺 (B=200) 时临界救援集中可验证弱档, 真检出 >= FULL (实测 27.16 vs 20.36, 50 seeds).",
        "cands 模式: 无 truth 时只输出 n_detected 与调度轨迹, 无真/假检出 (如实).",
    ]

    out_html = (Path(args.out) if args.out
                else Path.cwd() / "prune_report.html")
    out_html = out_html.resolve()
    out_json = out_html.with_suffix(".json")

    body["tool"] = "verifytool"
    body["version"] = VERSION
    body["spec"] = "SPEC_插件扩展_2026-08-07.md"
    body["subcommand"] = "prune"
    payload = dict(body)
    payload["command"] = " ".join(sys.argv[1:])
    md5s = save_json(payload, out_json)
    meta = {
        "world": world_desc,
        "budget": args.budget,
        "n_seeds": args.n_seeds,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_path": str(out_html),
        "payload_md5": md5s["payload_md5"],
        "self_md5": md5s["self_md5"],
    }
    out_html.write_text(render_prune_html(body, meta), encoding="utf-8")

    agg = body["aggregate"]
    print(f"[verifytool prune] 世界: {world_desc} | 预算 {args.budget} | "
          f"{args.n_seeds} seeds (seeds {body['seeds'][0]}..{body['seeds'][-1]}), "
          f"耗时 {body['elapsed_seconds']} s")
    for arm in ARMS:
        a = agg[arm]
        if "true_det_mean" in a and "recall" in a:
            print(f"  {arm:9s}: 真检出 {a['true_det_mean']:.3f}/seed "
                  f"(recall {a['recall']:.4f}) | 假阳性 {a['false_det_mean']:.3f}/seed "
                  f"(fp_rate {a['fp_rate']:.4f}) | 检验 {a['n_tests_total']}")
        elif "true_det_mean" in a:
            print(f"  {arm:9s}: 真检出 {a['true_det_mean']:.3f}/seed | "
                  f"假阳性 {a['false_det_mean']:.3f}/seed | 检验 {a['n_tests_total']}")
        else:
            print(f"  {arm:9s}: n_detected {a['n_detected_mean']:.3f}/seed | "
                  f"检验 {a['n_tests_total']} (无 truth, 无真/假检出)")
    if "true_det_mean" in agg["MAP_CRIT"] and "true_det_mean" in agg["FULL"]:
        d = agg["MAP_CRIT"]["true_det_mean"] - agg["FULL"]["true_det_mean"]
        print(f"  MAP_CRIT - FULL 真检出: {d:+.3f}/seed "
              f"({'CRIT >= FULL (预算稀缺方向成立)' if d >= 0 else 'CRIT < FULL (方向反转)'})")
    print(f"[verifytool prune] HTML -> {out_html}")
    print(f"[verifytool prune] JSON -> {out_json}")
    print(f"[verifytool prune] payload_md5={md5s['payload_md5']} "
          f"self_md5={md5s['self_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(cmd_prune(sys.argv[1:]))
