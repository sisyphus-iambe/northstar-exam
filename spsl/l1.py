"""SPSL — L1 考卷引擎: H0 生成 + 计分.

判定公式与阈值 (v1 同款):
  连续区 (大 n): p 值须均匀 — KS vs U(0,1) (1.63/√N + slack 0.02),
                逐点 F̂(α) ≈ α (± 0.02), 均值 ≈ 0.5 (± 0.03)
  离散区 (小 n): p 值须保守 — 逐点 F̂(α) ≤ α + 0.02, 均值 ≤ 0.54
NaN 不计分规则: 候选在合法 H0 输入上返回 NaN/inf 或抛异常, 该 p 值
  视为非有限, 不计入合格 —— 任一非有限 -> 本区本 seed 不合格.
  (v1 同款门控: cont_finite/disc_finite)
"""
import math

import numpy as np
from scipy.stats import kstest, multinomial


def draw_h0(rng, gen: dict) -> dict:
    """按 null_dist 生成器配置抽一张 H0 输入 (确定性 rng).
    协议: 计数表重抽到边缘全 > 0;
          两样本 = 连续分布独立抽样 (无并列)."""
    t = gen["type"]
    if t == "multinomial_independence":
        n = int(gen["n"])
        row_p = np.asarray(gen["row_p"], dtype=float)
        col_p = np.asarray(gen["col_p"], dtype=float)
        p = np.outer(row_p, col_p).ravel()
        while True:
            tbl = multinomial.rvs(n, p, size=1, random_state=rng)[0]
            tbl = tbl.reshape(len(row_p), len(col_p))
            if np.all(tbl.sum(axis=1) > 0) and np.all(tbl.sum(axis=0) > 0):
                return {"type": "contingency_table", "table": tbl}
    if t == "iid_two_samples":
        lo, hi = float(gen["lo"]), float(gen["hi"])
        return {"type": "two_samples",
                "x": rng.uniform(lo, hi, int(gen["n1"])),
                "y": rng.uniform(lo, hi, int(gen["n2"]))}
    raise ValueError(f"unknown null_dist.type: {t!r} (schema.SUPPORTED_GENERATORS)")


def call_candidate(fn, inp: dict) -> float:
    """按输入结构调用候选 (inputs.type 决定调用约定)."""
    if inp["type"] == "contingency_table":
        return float(fn(inp["table"]))
    if inp["type"] == "two_samples":
        return float(fn(inp["x"], inp["y"]))
    raise ValueError(f"unknown input type: {inp['type']!r}")


def collect_pvals(fn, rng, gen: dict, n_tables: int):
    """抽 n_tables 张 H0 输入 -> 候选 p 值序列.
    候选抛异常 -> 记为非有限 (NaN 不计分规则), 记录首个异常消息."""
    pvals, n_exc, first_exc = [], 0, None
    for _ in range(n_tables):
        inp = draw_h0(rng, gen)
        try:
            pvals.append(call_candidate(fn, inp))
        except Exception as exc:
            pvals.append(float("nan"))
            n_exc += 1
            if first_exc is None:
                first_exc = f"{type(exc).__name__}: {exc}"
    return pvals, n_exc, first_exc


def check_zone(pvals, zone: str, thresholds: dict, points: list) -> dict:
    """L1 区计分.
    返回 diag dict, 含 ok 布尔. NaN/非有限 p 值 -> 不计入合格 -> 本区不合格."""
    arr = np.asarray(pvals, dtype=float)
    n_nonfinite = int(np.sum(~np.isfinite(arr)))
    d = {"nonfinite_count": n_nonfinite, "ok": False}
    if n_nonfinite > 0:
        # NaN/非有限 p 值不计入合格, 本区不合格 (v1 同: l1.py cont_finite/disc_finite)
        d["note"] = (f"{n_nonfinite} p-values non-finite (NaN/inf), not scored -> zone fails")
        return d
    F = {str(a): float(np.mean(arr <= a)) for a in points}
    mean = float(np.mean(arr))
    if zone == "cont":
        ks = kstest(arr, "uniform")
        d.update({"ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
                  "mean": mean, "F": F})
        ok = (ks.statistic <= thresholds["ks_crit_99_plus_slack"]
              and all(abs(F[str(a)] - a) <= thresholds["cont_point_slack"]
                      for a in points)
              and abs(mean - 0.5) <= thresholds["cont_mean_slack"])
    else:  # disc
        d.update({"mean": mean, "F": F})
        ok = (all(F[str(a)] <= a + thresholds["disc_point_slack"]
                  for a in points)
              and mean <= thresholds["disc_mean_max"])
    d["ok"] = bool(ok)
    return d


def run_seed(fn, seed: int, l1cfg: dict) -> dict:
    """单个 seed 的 L1 判定. 返回 {seed, verdict, L1_diag}, diag 字段名
    与 v1 输出一致 (供报告渲染复用)."""
    rng = np.random.default_rng(seed)
    cont_p, n_exc_c, first_c = collect_pvals(fn, rng, l1cfg["cont"],
                                             l1cfg["n_tables"])
    disc_p, n_exc_d, first_d = collect_pvals(fn, rng, l1cfg["disc"],
                                             l1cfg["n_tables"])
    cont_d = check_zone(cont_p, "cont", l1cfg["thresholds"], l1cfg["points"])
    disc_d = check_zone(disc_p, "disc", l1cfg["thresholds"], l1cfg["points"])
    diag = {
        "cont_finite": cont_d["nonfinite_count"] == 0,
        "cont_ks_D": cont_d.get("ks_D"),
        "cont_ks_p": cont_d.get("ks_p"),
        "cont_mean": cont_d.get("mean"),
        "cont_F": cont_d.get("F"),
        "cont_nonfinite_count": cont_d["nonfinite_count"],
        "cont_ok": cont_d["ok"],
        "disc_finite": disc_d["nonfinite_count"] == 0,
        "disc_mean": disc_d.get("mean"),
        "disc_F": disc_d.get("F"),
        "disc_nonfinite_count": disc_d["nonfinite_count"],
        "disc_ok": disc_d["ok"],
    }
    if "note" in cont_d:
        diag["cont_note"] = cont_d["note"]
    if "note" in disc_d:
        diag["disc_note"] = disc_d["note"]
    if n_exc_c:
        diag["cont_exception_count"] = n_exc_c
        diag["cont_first_exception"] = first_c
    if n_exc_d:
        diag["disc_exception_count"] = n_exc_d
        diag["disc_first_exception"] = first_d
    return {"seed": seed, "verdict": bool(cont_d["ok"] and disc_d["ok"]),
            "L1_diag": diag}


PASS_RATIO = 0.10    # v1 层判定协议: 拒绝计数占比 <= 10% -> PASS
REJECT_RATIO = 0.90  # v1 层判定协议: >= 90% -> REJECT


def layer_verdict(n_rej: int, runs: int) -> str:
    """PASS/REJECT/MIXED (v1 层判定协议同款)."""
    if n_rej <= PASS_RATIO * runs:
        return "PASS"
    if n_rej >= REJECT_RATIO * runs:
        return "REJECT"
    return "MIXED"


def run_l1_exam(func, exam: dict) -> dict:
    """对候选执行编译考卷的全部 seed. 返回 L1 判定 dict
    (per_seed 结构同模板考卷产物)."""
    l1cfg = exam["l1"]
    seeds = l1cfg["seeds"]
    per = [run_seed(func, s, l1cfg) for s in seeds]
    n_rej = sum(1 for r in per if not r["verdict"])
    n_cont_rej = sum(1 for r in per if not r["L1_diag"]["cont_ok"])
    n_disc_rej = sum(1 for r in per if not r["L1_diag"]["disc_ok"])
    return {
        "layer": "L1",
        "exam_name": exam["name"],
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_rej": n_rej,
        "n_cont_rej": n_cont_rej,
        "n_disc_rej": n_disc_rej,
        "verdict": layer_verdict(n_rej, len(seeds)),
        "per_seed": per,
        "first_seed_diag": per[0]["L1_diag"],
    }
