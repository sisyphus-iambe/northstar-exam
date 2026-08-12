#!/usr/bin/env python3
"""E1 脏样本检测执行器 — demo_data 族 (北极星阶段 1 考官注册表).

判据 (第五轮讨论_2026-08-10 轮 8/10 B 上锁): 注入检出 >= 54/60 且 干净误杀 <= 1/30.
语义照搬 /tmp/northstar_v5_verify/seat1/e1_signal_detect.py (判据锁定, 零改动):
  谓词: NaN / 关节限位 / 帧间跳变 >= jump_thresh (确定性, 零 LLM)
  限位: 注入前全量干净数据冻结 min/max x (1±tol), 向外扩展 (软限位)
  注入: 3 类 x n_per_type, 类序 nan -> limit -> jump, rng 确定性, 类内奇偶定方向/符号
  R1 双轨 (任务书改动清单 7): 官方 G1 关节限位入 spec.reference_limits 对照档
  (见 specs/spec_demo_data.json 与 第五轮讨论_2026-08-10/阶段1_限位规格搜索.md),
  只存档不参与检测; 检测限位 = 数据驱动软限位 (limit_source 字段记录为决策).

确定性: verdict 不含 elapsed/command/时间戳; 同命令重跑逐字节一致 (save_json 双 md5).
执行器零硬编码数据路径: 根路径/列名/阈值/种子/注入数全由规格声明.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root (verifytool/ lives here)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verifytool.report import save_json  # noqa: E402

from spsl import VERSION  # noqa: E402
from spsl.schema import spec_md5  # noqa: E402

REQUIRED_SPEC = ("root", "observation_col", "seed", "tol", "jump_thresh",
                 "limit_factor", "n_fp", "n_per_type", "criteria")
REQUIRED_CRITERIA = ("min_detected", "max_fp")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"spsl demo_data spec validation failed: {msg}")


def validate_spec(spec: dict) -> None:
    """demo_data 族规格字段校验 (fail fast, 带字段名)."""
    for f in REQUIRED_SPEC:
        _check(f in spec, f"缺少 {f}")
    _check(isinstance(spec["seed"], int) and spec["seed"] >= 0, "seed 必须是非负整数")
    for key in ("tol", "jump_thresh", "limit_factor"):
        _check(isinstance(spec[key], (int, float)) and spec[key] > 0,
               f"{key} 必须是正数值")
    for key in ("n_fp", "n_per_type"):
        _check(isinstance(spec[key], int) and spec[key] > 0, f"{key} 必须是正整数")
    c = spec["criteria"]
    _check(isinstance(c, dict), "criteria 必须是对象")
    for f in REQUIRED_CRITERIA:
        _check(f in c and isinstance(c[f], int), f"criteria.{f} 必须是整数")


def run_spec(spec: dict, validator_path=None, out_path=None) -> dict:
    """规格 -> 加载数据 -> 冻结限位 -> 误杀/注入检测 -> 判定 (全部确定性).

    数据引用 robotdata 路径, 数据本身不入库; 规格即考卷, 无候选考生 (validator_path
    不用, 检测器自身即被测对象, 与 v5 席 1 E1 一致).
    """
    validate_spec(spec)
    root = Path(spec["root"]).expanduser()
    _check(root.is_dir(), f"root 不存在: {root}")
    seed, tol = int(spec["seed"]), float(spec["tol"])
    jump_thresh, limit_factor = float(spec["jump_thresh"]), float(spec["limit_factor"])
    n_fp, n_per_type = int(spec["n_fp"]), int(spec["n_per_type"])
    col = spec["observation_col"]
    min_detected, max_fp = int(spec["criteria"]["min_detected"]), \
        int(spec["criteria"]["max_fp"])

    files = sorted(f for f in root.iterdir() if f.suffix == ".parquet")
    if "expected_n_files" in spec:
        _check(len(files) == int(spec["expected_n_files"]),
               f"文件数 {len(files)} != 规格声明 expected_n_files "
               f"{int(spec['expected_n_files'])}")

    def load_state(p: Path):
        df = pd.read_parquet(p, columns=[col])
        return np.asarray(df[col].tolist(), dtype=np.float64)

    states = [load_state(f) for f in files]
    if "dims" in spec:
        for i, st in enumerate(states):
            _check(st.shape[1] == int(spec["dims"]),
                   f"episode {i} 维数 {st.shape[1]} != 规格声明 dims {int(spec['dims'])}")

    # 1) 冻结限位 (注入前, 全量干净统计; e1_signal_detect.py 同款)
    S = np.concatenate(states, axis=0)
    gmin = S.min(axis=0)
    gmax = S.max(axis=0)
    lo = gmin - tol * np.abs(gmin)          # 容差向外扩展, 防负最小值收缩
    hi = gmax + tol * np.abs(gmax)

    # 2) 干净数据统计 (报告用)
    clean_max_jump = 0.0
    for st in states:
        d = np.abs(np.diff(st, axis=0))
        if d.size:
            clean_max_jump = max(clean_max_jump, float(d.max()))
    clean_nan = int(np.isnan(S).sum())

    def pred_nan(st):   return bool(np.isnan(st).any())
    def pred_limit(st): return bool(((st < lo) | (st > hi)).any())
    def pred_jump(st):
        if len(st) < 2:
            return False
        return bool((np.abs(np.diff(st, axis=0)) >= jump_thresh).any())

    # 3) 误杀: 前 n_fp 条干净轨迹 (e1_signal_detect.py: 注入轨迹池 = 同一前 n_fp)
    fp_eps = [i for i in range(n_fp)
              if pred_nan(states[i]) or pred_limit(states[i]) or pred_jump(states[i])]
    fp = len(fp_eps)

    # 4) 注入 (3 类 x n_per_type, rng 确定性; 类序/奇偶方向/符号与 seat1 逐位一致)
    rng = random.Random(seed)
    dims = states[0].shape[1]
    cases = []
    for k in range(n_per_type):            # NaN
        ep = rng.randrange(n_fp); n = len(states[ep])
        cases.append({"id": k + 1, "type": "nan", "episode": ep,
                      "frame": rng.randrange(n), "dim": rng.randrange(dims)})
    for k in range(n_per_type):            # 关节限位 (limit_factor x 极值, 越过冻结界)
        ep = rng.randrange(n_fp); n = len(states[ep])
        cases.append({"id": n_per_type + k + 1, "type": "limit", "episode": ep,
                      "frame": rng.randrange(n), "dim": rng.randrange(dims),
                      "direction": "up" if k % 2 == 0 else "down"})
    for k in range(n_per_type):            # 帧间跳变 >= jump_thresh
        ep = rng.randrange(n_fp); n = len(states[ep])
        cases.append({"id": 2 * n_per_type + k + 1, "type": "jump", "episode": ep,
                      "frame": rng.randrange(1, n), "dim": rng.randrange(dims),
                      "sign": 1.0 if k % 2 == 0 else -1.0})

    def apply_case(c):
        st = states[c["episode"]].copy()
        f, d = c["frame"], c["dim"]
        if c["type"] == "nan":
            st[f, d] = np.nan
        elif c["type"] == "limit":
            # limit_factor x 极值幅度语义 (预注册): 任意符号/任意幅度必越冻结界
            if c["direction"] == "up":
                st[f, d] = gmax[d] + (limit_factor - 1.0) * abs(gmax[d])
            else:
                st[f, d] = gmin[d] - (limit_factor - 1.0) * abs(gmin[d])
        else:
            st[f, d] = st[f - 1, d] + c["sign"] * jump_thresh
        return st

    PRED = {"nan": pred_nan, "limit": pred_limit, "jump": pred_jump}
    per_type = {t: {"injected": 0, "detected": 0, "missed_ids": []} for t in PRED}
    detected_ids = []
    for c in cases:
        st = apply_case(c)
        hit = PRED[c["type"]](st)
        any_hit = pred_nan(st) or pred_limit(st) or pred_jump(st)
        per_type[c["type"]]["injected"] += 1
        if hit:
            per_type[c["type"]]["detected"] += 1
            detected_ids.append(c["id"])
        else:
            per_type[c["type"]]["missed_ids"].append(c["id"])
        c["expected_predicate_hit"] = bool(hit)
        c["any_predicate_hit"] = bool(any_hit)

    total = sum(v["injected"] for v in per_type.values())
    detected = sum(v["detected"] for v in per_type.values())
    pass_criterion = detected >= min_detected and fp <= max_fp

    payload = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "demo_data",
        "family": spec["family"],
        "spec": {"path": str(root), "name": spec["name"],
                 "spec_md5": spec_md5(spec)},
        "data": {"root": str(root), "n_files": len(files),
                 "n_total_rows": int(S.shape[0]), "dims": int(S.shape[1])},
        "results": {
            "per_type": per_type,
            "total_injected": total,
            "total_detected": detected,
            "fp_clean_episodes": fp,
            "fp_episode_ids": fp_eps,
            "pass": pass_criterion,
            "criterion": f"detected>={min_detected}/{total} and fp<={max_fp}/{n_fp}",
        },
        "limit_source": spec.get("limit_source", "data_driven_frozen"),
        "reference_limits": spec.get("reference_limits"),
        "clean_stats": {
            "nan_total": clean_nan,
            "max_jump_clean": float(clean_max_jump),
            "global_min_first5": [float(x) for x in gmin[:5]],
            "global_max_first5": [float(x) for x in gmax[:5]],
        },
        "bounds_frozen": {
            "lo_first5": [float(x) for x in lo[:5]],
            "hi_first5": [float(x) for x in hi[:5]],
        },
        "predicates": ["nan", f"limit(lo/hi frozen x(1±{tol}))",
                       f"jump>={jump_thresh}"],
        "injections": cases,
        "summary": (f"detected={detected}/{total} fp={fp}/{n_fp} "
                    f"-> {'PASS' if pass_criterion else 'FAIL'}"),
    }
    out = Path(out_path) if out_path else Path.cwd() / f"{spec['name']}_verdict.json"
    md5s = save_json(payload, out)
    # save_json 不就地改写调用方 dict, 显式挂回 md5 字段 (与落盘文件逐字段一致)
    payload["payload_md5"] = md5s["payload_md5"]
    payload["self_md5"] = md5s["self_md5"]
    print(f"[demo_data] {payload['summary']}  (spec_md5={payload['spec']['spec_md5'][:12]}...)")
    print(f"[demo_data] JSON -> {out}  payload_md5={payload['payload_md5']} "
          f"self_md5={payload['self_md5']}")
    return payload


if __name__ == "__main__":
    raise SystemExit(0)
