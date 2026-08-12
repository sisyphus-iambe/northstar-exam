"""考题库进化引擎 — 进化组件②③④ (2026-08-11).

进化组件② 选择压力:
  锚定线 = 区分度评分 (好 AI 过 / 坏 AI 挂, 真值已知):
    score = 坏集偏离率 × 好集放行率  (双向正确才算区分)
  自由线 = 自举评分 (无真值, 判定器自我证实):
    score = 全体考生偏离率  (偏离参照 = 错, 抓得越多越好 — 不区分好坏)
  漂移 = 自由线选出的考题对好考生的误抓率 > 锚定线 (杀敌一千自损八百).

进化组件③ 跨代继承: 第 N+1 代变异基表 = 第 N 代入选表 (父题上叠加操作,
  同 V3 create_cell inherit_prefs 微突变).

进化组件④ 世代循环: 3 代 × 100 题/代, 每代选 top-30 (两线独立).

判定口径: 表级偏离 = max(|p - ref_hand|, |p - ref_scipy|) > tol (1e-6),
  与正式 L2/L3 对拍语义逐字一致 (spsl/run.py run_pair_layer); 合法表上
  抛异常 = 偏离 (L2 崩溃全拒语义). 参照 = spsl.golden.get_ref.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from spsl.envelope import load_exam
from spsl.golden import get_ref
from spsl.run import run_four_layers
from verifytool.loader import load_validator

from mutate_exam import STEP_OPS, apply_ops, legal_table

ROOT = Path(__file__).resolve().parents[2]          # northstar_v2/ (本脚本在 experiments/exam_evolution/)
EXAM = ROOT / "exams" / "exam_pearson_full.json"
TOL = 1e-6                                          # L2 对拍容差 (compile_l2.py)
POOL_SIZE = 100                                     # 每代变异体数
TOP_K = 30                                          # 每代入选题数
N_GEN = 3                                           # 世代数
SEED = 20260811                                     # 实验种子 (确定性)

# 22 模板注册表 (verifytool/templates.py 同款清单, 含路径与真值分组)
TEMPLATES = {
    # 9 考卷模板 (programs/)
    "template_chi2_rxc": ("programs", None), "template_ratio": ("programs", None),
    "template_fisher": ("programs", None), "template_barnard": ("programs", None),
    "template_trend": ("programs", None), "template_slope": ("programs", None),
    "template_replicate": ("programs", None),
    "template_stratified": ("programs", None),
    "template_strat_bidir": ("programs", None),
    # 5 错误对照 (programs/)
    "correct": ("programs", "good"), "wrong_denominator": ("programs", "bad"),
    "wrong_dof": ("programs", "bad"),
    "wrong_dof_square_only": ("programs", "bad"), "wrong_impute": ("programs", "bad"),
    # 8 演示样 (programs_real/)
    "demo_handwritten_good": ("real", "good"), "demo_scipy_direct": ("real", "bad"),
    "demo_wrong_dof": ("real", "bad"), "demo_wrong_expected": ("real", "bad"),
    "demo_wrong_fillna": ("real", "bad"), "demo_wrong_pseudocount": ("real", "bad"),
    "demo_wrong_round": ("real", "bad"), "demo_wrong_tail": ("real", "bad"),
}

# 非 pearson 算法模板 (正确实现但算法族不同, 在 pearson 考卷下 L2/L3 天然 REJECT
# — 不适合当本考卷的考生, 记录并排除出评分)
NON_PEARSON = {"template_ratio", "template_fisher", "template_barnard",
               "template_trend", "template_slope", "template_replicate",
               "template_stratified", "template_strat_bidir"}


def template_path(name: str) -> Path:
    kind, _ = TEMPLATES[name]
    if kind == "programs":
        return ROOT / "verifytool" / "programs" / f"{name}.py"
    # demo 统一名带前缀, 磁盘文件名不带 (verifytool/templates.py 同款语义)
    return (ROOT / "verifytool" / "programs_real"
            / f"{name.removeprefix('demo_')}.py")


def load_all_templates():
    """加载 22 模板. 返回 {name: func} (加载失败记录但不中断)."""
    funcs, failed = {}, []
    for name in TEMPLATES:
        try:
            fn, _ = load_validator(str(template_path(name)), func_name="chi2_pvalue")
            funcs[name] = fn
        except Exception as exc:
            failed.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
    return funcs, failed


def run_baseline(funcs, exam) -> dict:
    """阶段 0: 22 模板四层判定 -> 真值分组."""
    rows = []
    for name, fn in funcs.items():
        r = run_four_layers(fn, exam)
        rows.append({"name": name, "total_verdict": r["total_verdict"],
                     "reject_runs": r["reject_runs"],
                     "L1": r["layers"]["L1"]["verdict"],
                     "L2": r["layers"]["L2"]["verdict"],
                     "L3": r["layers"]["L3"]["verdict"],
                     "L4": r["layers"]["L4"]["verdict"]})
    good = {r["name"] for r in rows
            if r["total_verdict"] == "ACCEPT" and r["name"] not in NON_PEARSON}
    bad = {r["name"] for r in rows
           if r["total_verdict"] == "REJECT" and r["name"] not in NON_PEARSON}
    excluded = sorted(NON_PEARSON)
    return {"rows": rows, "good": sorted(good), "bad": sorted(bad),
            "excluded": excluded}


def table_deviation(fn, t, refs) -> float:
    """表级偏离: 与双参照的 max|Δp| (抛异常 = inf).

    参照契约 (spsl/golden.py _ref_hand_chi2): 输入是 {"table": ...} dict
    (与 run_pair_layer 传 sec["inputs"] 原样 item 同款), 不是裸数组.
    """
    item = {"table": np.asarray(t, dtype=float).tolist()}
    try:
        p = float(fn(np.array(t, dtype=float)))
    except Exception:
        return float("inf")
    return max(abs(p - float(ref(item))) for ref in refs)


def score_table(t, funcs, good, bad, refs) -> dict:
    """双线评分. 返回 {anchored, free, good_fp, bad_tp} 及各考生偏离明细.

    考生集 = 适格考生 (good ∪ bad, 全为卡方公式实现, 快). 非 pearson 族
    模板 (barnard/fisher/trend... 枚举式实现, 每表百万级组合) 不参与评分 —
    它们在 pearson 考卷下天然偏离且单次 p 计算极慢 (实测致评分阶段卡死).
    """
    cand = [n for n in funcs if n in good or n in bad]
    devs = {}
    for name in cand:
        devs[name] = table_deviation(funcs[name], t, refs)
    bad_tp = sum(1 for n in bad if devs[n] > TOL) / max(1, len(bad))
    good_fp = sum(1 for n in good if devs[n] > TOL) / max(1, len(good))
    free = sum(1 for n in cand if devs[n] > TOL) / max(1, len(cand))
    return {
        "anchored": bad_tp * (1 - good_fp),
        "free": free,
        "bad_tp": bad_tp,
        "good_fp": good_fp,
        "devs": {n: round(devs[n], 9) for n in devs},
    }


def gen_pool(parents, rng, n=POOL_SIZE) -> list:
    """生成变异池. parents=None = 首代 (从考卷种子表池抽基表);
    parents=入选表 = 继承代 (在父题上叠加操作, 可叠加多步)."""
    pool = []
    for i in range(n):
        if parents is None:
            base_idx = int(rng.integers(0, len(SEED_TABLES)))
            base = SEED_TABLES[base_idx]
            lineage = ["seed"]
            n_ops = int(rng.integers(1, 3))
        else:
            parent = parents[int(rng.integers(0, len(parents)))]
            base = parent["table"]
            lineage = parent["lineage"] + [
                f"g{parent.get('gen', 0)}:{parent['seed_idx']}"]
            n_ops = int(rng.integers(1, 3))
        ops = [STEP_OPS[int(rng.integers(0, len(STEP_OPS)))] for _ in range(n_ops)]
        t, applied = apply_ops(base, ops, rng)
        if not legal_table(t):
            continue  # 变异失败 (诚实丢弃, 不计入池)
        pool.append({"table": np.asarray(t, dtype=float),
                     "ops": applied, "lineage": lineage})
        if len(pool) >= n:
            break
    return pool


def select_top(pool_scores, key, k=TOP_K) -> list:
    """按 score 取 top-k (tie-break: 偏离稳定性, 确定性排序)."""
    ranked = sorted(pool_scores, key=lambda x: (-x["score"][key], x["seed_idx"]))
    return ranked[:k]


def evolve(funcs, good, bad, exam) -> dict:
    """主循环: 3 代 × 两线. 返回全轨迹."""
    global SEED_TABLES
    # 种子表池 = L2 40 表 + L3 76 表 (现成考卷, 已验证可判)
    seed_tables = [np.asarray(inp["table"], dtype=float)
                   for inp in exam["l2"]["inputs"] + exam["l3"]["inputs"]]
    SEED_TABLES = seed_tables

    refs = [get_ref(n) for n in exam["l2"]["refs"]]
    rng = np.random.default_rng(SEED)

    # 首代
    pool = gen_pool(None, rng)
    scored = []
    for i, item in enumerate(pool):
        s = score_table(item["table"], funcs, good, bad, refs)
        scored.append({"seed_idx": i, "table": item["table"].tolist(),
                       "ops": item["ops"], "lineage": item["lineage"],
                       "score": s})
    anchored = select_top(scored, "anchored")
    free = select_top(scored, "free")
    gen0 = {
        "pool_size": len(pool),
        "anchored": [summarize(s) for s in anchored],
        "free": [summarize(s) for s in free],
        "metrics": {
            "anchored": {"bad_tp": avg(s["score"]["bad_tp"] for s in anchored),
                         "good_fp": avg(s["score"]["good_fp"] for s in anchored)},
            "free": {"bad_tp": avg(s["score"]["bad_tp"] for s in free),
                     "good_fp": avg(s["score"]["good_fp"] for s in free)},
        },
    }

    # 继承代
    generations = [gen0]
    parents_a, parents_f = anchored, free
    for g in range(1, N_GEN):
        pool_a = gen_pool(parents_a, rng)
        pool_f = gen_pool(parents_f, rng)
        scored_a = []
        for i, item in enumerate(pool_a):
            s = score_table(item["table"], funcs, good, bad, refs)
            scored_a.append({"seed_idx": i, "table": item["table"].tolist(),
                             "ops": item["ops"], "lineage": item["lineage"],
                             "gen": g, "score": s})
        scored_f = []
        for i, item in enumerate(pool_f):
            s = score_table(item["table"], funcs, good, bad, refs)
            scored_f.append({"seed_idx": i, "table": item["table"].tolist(),
                             "ops": item["ops"], "lineage": item["lineage"],
                             "gen": g, "score": s})
        anchored = select_top(scored_a, "anchored")
        free = select_top(scored_f, "free")
        generations.append({
            "gen": g, "pool_size": len(pool_a),
            "anchored": [summarize(s) for s in anchored],
            "free": [summarize(s) for s in free],
            "metrics": {
                "anchored": {"bad_tp": avg(s["score"]["bad_tp"] for s in anchored),
                             "good_fp": avg(s["score"]["good_fp"] for s in anchored)},
                "free": {"bad_tp": avg(s["score"]["bad_tp"] for s in free),
                         "good_fp": avg(s["score"]["good_fp"] for s in free)},
            },
        })
        parents_a, parents_f = anchored, free

    return {"generations": generations,
            "final": {"anchored": [summarize(s) for s in parents_a],
                      "free": [summarize(s) for s in parents_f]}}


def summarize(s: dict) -> dict:
    """入选题目摘要 (表值序列化, 供落盘)."""
    return {"idx": s["seed_idx"], "ops": s["ops"], "lineage": s["lineage"],
            "table": s["table"], "bad_tp": s["score"]["bad_tp"],
            "good_fp": s["score"]["good_fp"],
            "anchored_score": s["score"]["anchored"],
            "free_score": s["score"]["free"]}


def avg(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main() -> int:
    exam = load_exam(EXAM)
    funcs, failed = load_all_templates()
    baseline = run_baseline(funcs, exam)
    good = baseline["good"]
    bad = baseline["bad"]

    # 真值覆盖检查: 预期的 good/bad 是否都被判定命中 (校准诚实性)
    expect_good = {n for n, (_, t) in TEMPLATES.items() if t == "good"}
    expect_bad = {n for n, (_, t) in TEMPLATES.items() if t == "bad"}
    baseline["expect_good_missed"] = sorted(expect_good - set(good))
    baseline["expect_bad_missed"] = sorted(expect_bad - set(bad))

    print(f"[evolve] baseline: {len(good)} good / {len(bad)} bad / "
          f"{len(baseline['excluded'])} ineligible (non-pearson) / {len(failed)} load failures")
    print(f"[evolve] expected good missed: {baseline['expect_good_missed']} | "
          f"expected bad missed: {baseline['expect_bad_missed']}")

    traj = evolve(funcs, good, bad, exam)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "evolution.json").write_text(
        json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
    for i, g in enumerate(traj["generations"]):
        m = g["metrics"]
        print(f"[evolve] gen{i} anchored: bad_tp={m['anchored']['bad_tp']} "
              f"fp={m['anchored']['good_fp']} | "
              f"free: bad_tp={m['free']['bad_tp']} fp={m['free']['good_fp']}")
    print(f"[evolve] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
