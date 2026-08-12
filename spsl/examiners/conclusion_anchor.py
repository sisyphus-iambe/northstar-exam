#!/usr/bin/env python3
"""结论锚执行器 — conclusion_anchor 族 (北极星 v3 阶段 3 级 3 结论锚 -> 阶段 1 考官注册表).

机制 (判据③: 数据集路径/切片/阈值全从规格 JSON 读, 执行器零硬编码数据路径,
除数据集/设计/方法注册表键):
  规格 -> 校验 (validate_anchor_spec, 自 domain_schema.py 合并, 包内自包含,
  零 experiments 依赖) -> 加载数据集 (注册表: parquet_robotdata 读取器 /
  generator_ar1 / generator_normal) -> 逐 chain 按设计 (window 相邻窗口 /
  resample iid 重抽样) 抽样, method=pred_ci_correct 计区间, N_SIMS 次蒙特卡洛
  算 cov_hat (记录 first_sample) -> 按 verdict_rule 判定 -> 与 known_conclusion
  比对 -> 总判定 (全链一致 + 零崩溃零非有限).

确定性: 每格 rng = default_rng(cell_seed(seed_base, cell_index)), cell_seed =
splitmix64(seed_base * 0x100000001B3 + index) & 0xFFFFFFFF (est.est_cover 同款);
verdict JSON 不含 elapsed/command/时间戳/绝对路径, 同命令重跑逐字节一致
(save_json 双 md5).

判定规则出处: 窗口链 threshold_lt(T_minus_2SE) = g4 design_c_precheck 同款
(cov_hat < T-2SE 即 FAIL); iid 链 correct_coverage = 北极星v2.md §二 G4 行
(cov_hat >= T-2SE 且 |cov_hat - cov_hat_control| <= 3SE); AR(1) threshold_lt(T)
= expD 同款固定阈值.

run_fn 契约 (考官注册表): run_spec(spec, validator_path, out_path), validator_path
忽略 (无考生, 检测器自身即被测对象, 与 demo_data 族一致); 返回 payload 含
summary / payload_md5 / self_md5 (save_json 双 md5, 显式挂回调用方 dict).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import t as _t

_REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root (verifytool/ lives here)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verifytool.report import save_json  # noqa: E402

from spsl import VERSION  # noqa: E402
from spsl.schema import spec_md5  # noqa: E402


# ---- 规格 schema (domain_schema.py 合并: 校验 + 指纹; 算法逐字保留) ----

REQUIRED_TOP = ("name", "family", "layer", "description",
                "datasets", "designs", "chains", "thresholds")
DATASET_KINDS = ("parquet_robotdata", "generator_ar1", "generator_normal")
DESIGN_TYPES = ("window", "resample")
METHODS = ("pred_ci_correct",)
RULE_TYPES = ("threshold_lt", "correct_coverage")
THRESHOLD_KEYS = ("T", "T_minus_2SE", "SE")
AR1_TARGETS = ("adjacent", "independent")


def splitmix64(x: int) -> int:
    """SplitMix64 标量混合 (est.est_cover.splitmix64 同款, 逐位一致)."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def cell_seed(spec_seed: int, index: int) -> int:
    """格种子派生: splitmix64(seed_base * 0x100000001B3 + index) & 0xFFFFFFFF
    (est.est_cover.cell_seed 同款, 逐位一致)."""
    return int(splitmix64(spec_seed * 0x100000001B3 + index) & 0xFFFFFFFF)


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"结论锚规格校验失败 [{path}]: {msg}")


def _check(cond: bool, path: str, msg: str) -> None:
    if not cond:
        raise _err(path, msg)


def _require_dict(x, path: str) -> None:
    _check(isinstance(x, dict), path, "必须是 JSON 对象")


def _require_str(x, path: str) -> None:
    _check(isinstance(x, str) and bool(x), path, "必须是非空字符串")


def _validate_datasets(datasets) -> None:
    _require_dict(datasets, "datasets")
    _check(len(datasets) >= 1, "datasets", "必须声明 >= 1 个数据集")
    for key, ds in datasets.items():
        _require_dict(ds, f"datasets.{key}")
        _check(ds.get("kind") in DATASET_KINDS, f"datasets.{key}.kind",
               f"允许 kind ∈ {DATASET_KINDS}")
        if ds["kind"] == "parquet_robotdata":
            _require_str(ds.get("root"), f"datasets.{key}.root")
            cols = ds.get("columns")
            _require_dict(cols, f"datasets.{key}.columns")
            _require_str(cols.get("observation"), f"datasets.{key}.columns.observation")
            dims = cols.get("dimensions")
            _check(isinstance(dims, list) and len(dims) >= 1
                   and all(isinstance(d, int) and d >= 0 for d in dims),
                   f"datasets.{key}.columns.dimensions", "必须是正整数列表")
            _check(ds.get("track") == "per_file", f"datasets.{key}.track",
                   "当前支持 track == 'per_file' (每文件一轨迹)")
        else:
            _require_dict(ds.get("params"), f"datasets.{key}.params")


def _validate_designs(designs) -> None:
    _require_dict(designs, "designs")
    for key, ds in designs.items():
        _require_dict(ds, f"designs.{key}")
        _check(ds.get("type") in DESIGN_TYPES, f"designs.{key}.type",
               f"允许类型 ∈ {DESIGN_TYPES}")
        _require_dict(ds.get("params"), f"designs.{key}.params")


def _validate_seed(seed, path: str) -> None:
    _require_dict(seed, path)
    for k in ("seed_base", "cell_index"):
        _check(isinstance(seed.get(k), int) and seed[k] >= 0,
               f"{path}.{k}", "必须是非负整数")


def _validate_rule(rule, path: str) -> None:
    _require_dict(rule, path)
    _check(rule.get("type") in RULE_TYPES, f"{path}.type",
           f"允许规则 ∈ {RULE_TYPES}")
    if rule["type"] == "threshold_lt":
        _check(rule.get("threshold") in THRESHOLD_KEYS, f"{path}.threshold",
               f"允许阈值 ∈ {THRESHOLD_KEYS}")


def _validate_chain(chain, datasets, designs) -> None:
    _require_dict(chain, "chains[]")
    _require_str(chain.get("name"), "chains[].name")
    _check(chain.get("dataset") in datasets, "chains[].dataset",
           f"数据集 {chain.get('dataset')!r} 未在 datasets 声明")
    _require_dict(chain.get("dataset_params"), "chains[].dataset_params")
    dp = chain["dataset_params"]
    kind = datasets[chain["dataset"]]["kind"]
    if kind == "parquet_robotdata":
        _check(isinstance(dp.get("dimension"), int) and dp["dimension"] >= 0,
               "chains[].dataset_params.dimension", "必须是非负整数")
        _check(isinstance(dp.get("history_n"), int) and dp["history_n"] >= 2,
               "chains[].dataset_params.history_n", "必须 >= 2")
    elif kind == "generator_ar1":
        rho, n, target = dp.get("rho"), dp.get("n"), dp.get("target")
        _check(isinstance(rho, (int, float)) and 0 <= rho < 1,
               "chains[].dataset_params.rho", "必须在 [0, 1)")
        _check(isinstance(n, int) and n >= 2, "chains[].dataset_params.n", "必须 >= 2")
        _check(target in AR1_TARGETS, "chains[].dataset_params.target",
               f"允许目标 ∈ {AR1_TARGETS}")
    design = chain.get("design")
    _check(design is None or design in designs, "chains[].design",
           f"设计 {design!r} 未在 designs 声明")
    _check(chain.get("method") in METHODS, "chains[].method",
           f"允许方法 ∈ {METHODS}")
    _require_str(chain.get("target"), "chains[].target")
    _check(isinstance(chain.get("n_sims"), int) and chain["n_sims"] >= 100,
           "chains[].n_sims", "必须 >= 100")
    _validate_seed(chain.get("seed"), "chains[].seed")
    ctl = chain.get("control")
    if ctl is not None:
        _require_dict(ctl, "chains[].control")
        _check(ctl.get("dataset") in datasets, "chains[].control.dataset",
               f"对照数据集 {ctl.get('dataset')!r} 未在 datasets 声明")
        _check(datasets[ctl["dataset"]]["kind"] == "generator_normal",
               "chains[].control.dataset", "对照必须是 generator_normal")
        _validate_seed(ctl.get("seed"), "chains[].control.seed")
    _validate_rule(chain.get("verdict_rule"), "chains[].verdict_rule")
    if chain["verdict_rule"]["type"] == "correct_coverage":
        _check(ctl is not None, "chains[].verdict_rule",
               "correct_coverage 规则必须声明 control")
    _check(chain.get("known_conclusion") in ("PASS", "FAIL"),
           "chains[].known_conclusion", "必须是 'PASS' 或 'FAIL'")


def _validate_thresholds(thr) -> None:
    _require_dict(thr, "thresholds")
    _check(isinstance(thr.get("T"), (int, float)), "thresholds.T", "必须是数值")
    nom = thr.get("nominal")
    _check(isinstance(nom, (int, float)) and 0 < nom < 1,
           "thresholds.nominal", "必须在 (0, 1)")
    for k in ("correct_rule", "defect_rule"):
        _require_str(thr.get(k), f"thresholds.{k}")
    for k in ("defect_cov_max", "separation_min"):
        v = thr.get(k)
        _check(v is None or isinstance(v, (int, float)),
               f"thresholds.{k}", "必须是数值或 null (无缺陷判据的规格可空)")


def validate_anchor_spec(spec: dict) -> dict:
    """结论锚规格校验 (非法用例一律抛 ValueError)."""
    _require_dict(spec, "spec")
    for key in REQUIRED_TOP:
        _check(key in spec, key, "必填字段缺失")
    _require_str(spec["name"], "name")
    _check(spec["family"] == "conclusion_anchor", "family",
           "结论锚规格 family 必须 == 'conclusion_anchor'")
    _check(spec["layer"] == "L3", "layer", "结论锚规格 layer 必须 == 'L3'")
    _require_str(spec.get("description", "结论锚规格"), "description")
    _validate_datasets(spec["datasets"])
    _validate_designs(spec["designs"])
    _check(isinstance(spec["chains"], list) and len(spec["chains"]) >= 1,
           "chains", "必须声明 >= 1 条链")
    names = [c.get("name") for c in spec["chains"]]
    _check(len(names) == len(set(names)), "chains[].name", "链名必须唯一")
    for c in spec["chains"]:
        _validate_chain(c, spec["datasets"], spec["designs"])
    _validate_thresholds(spec["thresholds"])
    return spec


# ---- 方法注册表 (规格声明 method 键 -> 实现) ----

def pred_ci_correct(xs) -> tuple:
    """正确 t 预测区间 (est_mutants.pred_ci_correct 同式: mean ± t_{0.975,n-1}*s*sqrt(1+1/n))."""
    xs = np.asarray(xs, dtype=float)
    n = xs.size
    m = float(xs.mean())
    s = float(xs.std(ddof=1))
    h = float(_t.ppf(0.975, n - 1)) * s * np.sqrt(1 + 1 / n)
    return m - h, m + h


METHODS = {"pred_ci_correct": pred_ci_correct}


# ---- 数据集注册表 (kind 键 -> 读取器 / 生成器; 路径/切片/阈值全在规格) ----

def read_robotdata(ds: dict) -> dict:
    """parquet_robotdata: 流式逐文件读 observation.state 单列, 只保留规格声明维度.
    返回 pop{维度: ndarray} / offsets (每 episode 起始行偏移) / n_total / n_files.
    与 g4 run_robotdata_cover.load_population 同码同 draw 顺序 (维值逐位一致)."""
    dims = list(ds["columns"]["dimensions"])
    root = Path(ds["root"]).expanduser()
    files = sorted(root.glob("*.parquet"))
    pops = {d: [] for d in dims}
    offsets, n_total = [], 0
    for f in files:
        a = np.stack(
            pq.read_table(f, columns=[ds["columns"]["observation"]])
            .to_pandas()[ds["columns"]["observation"]].to_numpy()
        ).astype(np.float64)
        offsets.append(n_total)
        n_total += a.shape[0]
        for d in dims:
            pops[d].append(a[:, d])
    pops = {d: np.concatenate(v) for d, v in pops.items()}
    return {"kind": "parquet_robotdata", "pop": pops, "offsets": offsets,
            "n_total": n_total, "n_files": len(files)}


def gen_ar1(rng, params: dict):
    """generator_ar1: x_t = rho*x_{t-1} + eps, eps ~ N(0,1-rho^2), 乘 sqrt(1-rho^2)
    使边际 N(0,1); target=adjacent -> y = rho*x_n + eps_next, independent -> y ~ N(0,1)
    (expD run_ar1.py 同码同 draw 顺序)."""
    rho, n, target = float(params["rho"]), int(params["n"]), params["target"]
    eps = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    x = x * np.sqrt(1.0 - rho * rho)
    if target == "adjacent":
        y = float(rho * x[-1] + rng.normal(0.0, np.sqrt(1.0 - rho * rho)))
    else:
        y = float(rng.normal(0.0, 1.0))
    return x, y


def gen_normal(rng, params: dict):
    """generator_normal: 合成正态对照 xs ~ N(0,1,n), y ~ N(0,1) 独立 (g4 cov_hat_synthetic
    同码同 draw 顺序)."""
    n = int(params["n"])
    return rng.normal(0.0, 1.0, n), float(rng.normal(0.0, 1.0))


DATASETS = {"parquet_robotdata": read_robotdata, "generator_ar1": gen_ar1,
            "generator_normal": gen_normal}


# ---- 切片设计 (规格 design 键 -> 抽样器; 参数全在 dataset_params) ----

def sample_window(rng, pop, offsets, n_total, n_files, dp: dict):
    """window: 拒绝采样保证窗口完整 (episode 行数 > n), 同 episode 内取相邻 n 行,
    目标 = 紧邻下一行 (g4 design_c_precheck 同码同 draw 顺序)."""
    n = int(dp["history_n"])
    while True:
        ep = int(rng.integers(0, n_files))
        rows = offsets[ep + 1] - offsets[ep] if ep + 1 < len(offsets) else n_total - offsets[ep]
        if rows > n:
            break
    s = offsets[ep] + int(rng.integers(0, rows - n))
    return pop[s:s + n], float(pop[s + n])


def sample_resample(rng, pop, offsets, n_total, n_files, dp: dict):
    """resample: iid 行重抽样 (with replacement), 目标 = 独立 iid 新行
    (g4 cov_hat 同码同 draw 顺序: xs = size=n 一次抽取, y = 单行抽取).
    offsets/n_files 不使用 (与 sample_window 统一签名)."""
    n = int(dp["history_n"])
    xs = pop[rng.integers(0, n_total, size=n)]
    y = float(pop[rng.integers(0, n_total)])
    return xs, y


DESIGNS = {"window": sample_window, "resample": sample_resample}


def _se(thr: dict, n_sims: int) -> float:
    """SE 由规格 nominal + n_sims 派生: sqrt(nominal*(1-nominal)/n_sims)."""
    return float(np.sqrt(thr["nominal"] * (1.0 - thr["nominal"]) / n_sims))


def eval_rule(rule: dict, cov_hat: float, cov_control, thr: dict, se: float) -> str:
    """规格 verdict_rule -> verdict ('PASS'|'FAIL')."""
    rtype = rule["type"]
    if rtype == "threshold_lt":
        key = rule["threshold"]
        vals = {"T": thr["T"], "SE": se, "T_minus_2SE": thr["T"] - 2.0 * se}
        return "FAIL" if cov_hat < vals[key] else "PASS"
    if rtype == "correct_coverage":
        lo = thr["T"] - 2.0 * se
        ok = (cov_hat >= lo) and (abs(cov_hat - cov_control) <= 3.0 * se)
        return "PASS" if ok else "FAIL"
    raise ValueError(f"unknown verdict_rule type: {rtype}")


def run_chain(chain: dict, data: dict, thr: dict) -> dict:
    """单链: N_SIMS 蒙特卡洛 -> cov_hat (+可选对照 cov_hat_control) -> verdict."""
    dp = chain["dataset_params"]
    ds = data[chain["dataset"]]
    n_sims = int(chain["n_sims"])
    rng = np.random.default_rng(cell_seed(chain["seed"]["seed_base"],
                                          chain["seed"]["cell_index"]))
    method = METHODS[chain["method"]]

    inside, first, crashes, nonfinite = 0, None, 0, 0
    for _ in range(n_sims):
        try:
            if ds["kind"] == "parquet_robotdata":
                pop = ds["pop"][int(dp["dimension"])]
                sampler = DESIGNS[chain["design"]]
                xs, y = sampler(rng, pop, ds["offsets"], ds["n_total"], ds["n_files"], dp)
            else:
                xs, y = DATASETS[ds["kind"]](rng, dp)
            lo, hi = method(xs)
            if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(y)):
                nonfinite += 1
                continue
            hit = bool(lo <= y <= hi)
            inside += int(hit)
            if first is None:
                first = [float(lo), float(hi), float(y), hit]
        except Exception:
            crashes += 1
    cov_hat = inside / n_sims

    cov_control = None
    if chain.get("control") is not None:
        ctl = chain["control"]
        c_rng = np.random.default_rng(cell_seed(ctl["seed"]["seed_base"],
                                                ctl["seed"]["cell_index"]))
        c_inside = 0
        for _ in range(n_sims):
            xs, y = gen_normal(c_rng, {"n": int(dp["history_n"])})
            lo, hi = method(xs)
            c_inside += int(lo <= y <= hi)
        cov_control = c_inside / n_sims

    se = _se(thr, n_sims)
    verdict = eval_rule(chain["verdict_rule"], cov_hat, cov_control, thr, se)
    match = verdict == chain["known_conclusion"]
    out = {
        "name": chain["name"],
        "dataset": chain["dataset"],
        "dataset_params": dp,
        "design": chain.get("design"),
        "target": chain["target"],
        "n_sims": n_sims,
        "seed_base": chain["seed"]["seed_base"],
        "cell_index": chain["seed"]["cell_index"],
        "cov_hat": round(cov_hat, 6),
        "cov_hat_control": round(cov_control, 6) if cov_control is not None else None,
        "first_sample": first,
        "crash_count": crashes,
        "nonfinite_count": nonfinite,
        "verdict": verdict,
        "known_conclusion": chain["known_conclusion"],
        "match": bool(match),
        "verdict_rule": chain["verdict_rule"],
    }
    if chain["verdict_rule"]["type"] == "threshold_lt":
        key = chain["verdict_rule"]["threshold"]
        out["diagnostics"] = {"threshold": {"T": thr["T"], "SE": round(se, 6),
                                            "T_minus_2SE": round(thr["T"] - 2 * se, 6)},
                              "selected": key,
                              "rule_note": "cov_hat < 阈值 -> FAIL"}
    else:
        lo = thr["T"] - 2 * se
        out["diagnostics"] = {"T_minus_2SE": round(lo, 6),
                              "diff_to_control": round(abs(cov_hat - cov_control), 6),
                              "pass_ge_Tm2SE": bool(cov_hat >= lo),
                              "pass_le_3SE": bool(abs(cov_hat - cov_control) <= 3 * se),
                              "3SE": round(3 * se, 6),
                              "rule_note": "cov_hat >= T-2SE 且 |cov_hat-cov_hat_control| <= 3SE -> PASS"}
    return out


def _run_spec(spec: dict) -> dict:
    """规格 -> 数据加载 -> 全链 -> 总判定 (数字只来自实际运行)."""
    validate_anchor_spec(spec)
    data = {}
    meta = {}
    for key, ds in spec["datasets"].items():
        if ds["kind"] == "parquet_robotdata":
            data[key] = read_robotdata(ds)
            meta[key] = {"kind": ds["kind"], "root": ds["root"],
                         "n_files": data[key]["n_files"],
                         "n_total_rows": data[key]["n_total"]}
        else:
            data[key] = {"kind": ds["kind"], "gen": DATASETS[ds["kind"]]}
            meta[key] = {"kind": ds["kind"]}
    chains = [run_chain(c, data, spec["thresholds"]) for c in spec["chains"]]
    n_match = sum(1 for c in chains if c["match"])
    crash_total = sum(c["crash_count"] for c in chains)
    nonfinite_total = sum(c["nonfinite_count"] for c in chains)
    ok = (n_match == len(chains)) and crash_total == 0 and nonfinite_total == 0
    return {"data": meta, "chains": chains,
            "overall": {"n_chains": len(chains), "n_match": n_match,
                        "crash_total": crash_total, "nonfinite_total": nonfinite_total,
                        "verdict": "PASS" if ok else "FAIL"}}


def run_spec(spec: dict, validator_path=None, out_path=None) -> dict:
    """run_fn 契约 (考官注册表): 规格 -> 校验 -> 全链 -> payload.

    validator_path 忽略 (无考生, 检测器自身即被测对象, 与 demo_data 族一致);
    返回 payload 含 summary / payload_md5 / self_md5 (save_json 双 md5, 显式挂回).
    """
    res = _run_spec(spec)
    n_sims0 = spec["chains"][0]["n_sims"]
    thr = spec["thresholds"]
    se = _se(thr, n_sims0) if thr.get("se_formula") else None

    payload = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "L3",
        "family": spec["family"],
        "spec": {"name": spec["name"], "spec_md5": spec_md5(spec)},
        "thresholds": {"T": thr["T"], "nominal": thr["nominal"],
                       "se": round(se, 6) if se is not None else None,
                       "se_formula": thr.get("se_formula"),
                       "correct_rule": thr.get("correct_rule"),
                       "defect_rule": thr.get("defect_rule")},
        **res,
    }
    o = res["overall"]
    payload["summary"] = (f"全链 {o['n_match']}/{o['n_chains']} 与 known_conclusion "
                          f"一致, crash={o['crash_total']} "
                          f"nonfinite={o['nonfinite_total']} -> {o['verdict']}")

    out = Path(out_path) if out_path else Path.cwd() / f"{spec['name']}_verdict.json"
    md5s = save_json(payload, out)
    # save_json 不就地改写调用方 dict, 显式挂回 md5 字段 (与落盘文件逐字段一致)
    payload["payload_md5"] = md5s["payload_md5"]
    payload["self_md5"] = md5s["self_md5"]

    print(f"[conclusion_anchor] spec {spec['name']} "
          f"(spec_md5={payload['spec']['spec_md5'][:12]}...)")
    for c in res["chains"]:
        ctl = f" control={c['cov_hat_control']:.4f}" if c["cov_hat_control"] is not None else ""
        print(f"  {c['name']:<22} cov_hat={c['cov_hat']:.4f}{ctl} "
              f"-> {c['verdict']} (known={c['known_conclusion']}, "
              f"{'一致' if c['match'] else '不一致!'})")
    print(f"[conclusion_anchor] {payload['summary']}")
    print(f"[conclusion_anchor] JSON -> {out}  payload_md5={payload['payload_md5']} "
          f"self_md5={payload['self_md5']}")
    return payload


if __name__ == "__main__":
    raise SystemExit(0)
