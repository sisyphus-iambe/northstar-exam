#!/usr/bin/env python3
"""E3 旋转恒等式考卷执行器 — state_estimator 族 (北极星阶段 1 考官注册表).

判据 (第五轮讨论_2026-08-10 轮 8/9/10 B 锁定):
  正确 0 违例 PASS / 变异 >= 90% REJECT / 误杀 0/60.
语义照搬 /tmp/northstar_v5_verify/seat2/exam_rotation.py (P1-P5, 判据锁定, 零改动):
  P1 范数 / P2 正交 / P3 det / P4 M->q->M' 循环 / P5 q->R->q' 循环
  ±q 符号歧义预注册: P5 按 min(||q2-q||, ||q2+q||) 判等; R_to_q 符号归一 w>=0
  变异体 = 范数失约 (归一化除 2), 避 D3 "整体互换" 构造性偏 0 坑.
内置变异体作考卷灵敏度自检 (同 reference.self_check 角色):
  M1 q_to_R 范数失约 / M2 R_to_q 范数失约 (exam_rotation.py 同款)
  M3 全局翻转 (a_exam_rotation.py 同款: rotvec 链, 逆旋转输入; 检测率 54/60 由
  0 与 pi 精确边界用例的旋转对称性决定, 确定性)
考生接口: 双函数模块 (importlib 加载, 规格声明 functions 列表).

确定性: 固定种子, verdict 不含 elapsed/command/时间戳, 同命令重跑逐字节一致.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root (verifytool/ lives here)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verifytool.report import save_json  # noqa: E402

from spsl import VERSION  # noqa: E402
from spsl.schema import spec_md5  # noqa: E402

from scipy.spatial.transform import Rotation as SciRot  # noqa: E402

REQUIRED_SPEC = ("functions", "seed", "tol", "n_m", "n_q", "criteria")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"spsl state_estimator spec validation failed: {msg}")


def validate_spec(spec: dict) -> None:
    """state_estimator 族规格字段校验 (fail fast, 带字段名)."""
    for f in REQUIRED_SPEC:
        _check(f in spec, f"缺少 {f}")
    fns = spec["functions"]
    _check(isinstance(fns, list) and len(fns) == 2
           and all(isinstance(x, str) and x for x in fns),
           "functions 必须是非空字符串列表 (双函数模块)")
    _check(isinstance(spec["seed"], int) and spec["seed"] >= 0, "seed 必须是非负整数")
    _check(isinstance(spec["tol"], (int, float)) and 0 < spec["tol"] <= 1e-3,
           "tol 必须是 (0, 1e-3] 正容差")
    for key in ("n_m", "n_q"):
        _check(isinstance(spec[key], int) and spec[key] > 0, f"{key} 必须是正整数")


# ---- 正确参照实现 (exam_rotation.py 同款, scipy 交叉验证背书 6.4e-16) --------

def q_to_R(q):
    """四元数 [x,y,z,w] -> 旋转矩阵 (单位输入假定)."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def R_to_q(R):
    """旋转矩阵 -> 四元数 [x,y,z,w] (Shepperd 鲁棒法, 符号归一 w>=0)."""
    R = np.asarray(R, dtype=float)
    q = np.zeros(4)
    t = np.trace(R)
    if t > 0.0:
        s = 2.0 * np.sqrt(max(0.0, t + 1.0))
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
            s = 2.0 * np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            q[0] = 0.25 * s
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] >= R[2, 2]:
            s = 2.0 * np.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            q[1] = 0.25 * s
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            q[2] = 0.25 * s
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
    if q[3] < 0.0:
        q = -q
    return q


# ---- 内置变异体 (考卷灵敏度自检, exam_rotation.py / a_exam_rotation.py 同款) ---

def M1_q_to_R(q):
    """变异 1: q_to_R 范数失约 — 四元数除 2 后走公式."""
    return q_to_R(np.asarray(q, dtype=float) / 2.0)


def M2_R_to_q(R):
    """变异 2: R_to_q 范数失约 — R_to_q 结果除 2."""
    return R_to_q(R) / 2.0


def rot_dist(r1, r2):
    """旋转距离 (弧度), 经矩阵迹比较 -> 天然免疫 ±q 符号歧义与 2pi 卷绕
    (a_exam_rotation.py 同款)."""
    R1 = SciRot.from_rotvec(np.asarray(r1, float)).as_matrix()
    R2 = SciRot.from_rotvec(np.asarray(r2, float)).as_matrix()
    c = float(np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(c))


def gen_rotvec_cases(n: int, seed: int):
    """M3 考卷用例 (a_exam_rotation.py 同款): 30 固定边界 (0/1e-4/0.3/pi/2pi 邻域
    x 3 轴, 含精确 0 与 pi) + 30 随机 (角混合: 40% 小角 / 40% 中角 / 20% pi 邻域).
    精确 0 与 pi 用例 6 个对全局翻转不可辨 -> M3 检出 = n-6."""
    rng = np.random.default_rng(seed)
    axes = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    angs = [0.0, 1e-4, 0.3, np.pi / 2 - 1e-3, np.pi / 2, np.pi / 2 + 1e-3,
            np.pi - 1e-3, np.pi, np.pi + 1e-3, 2 * np.pi - 1e-3]
    fixed = [tuple((a * np.asarray(ax, float)).tolist()) for ax in axes for a in angs]
    rand = []
    while len(fixed) + len(rand) < n:
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        u = rng.random()
        if u < 0.4:
            ang = rng.uniform(0.0, np.pi / 2)
        elif u < 0.8:
            ang = rng.uniform(np.pi / 2, np.pi)
        else:
            ang = rng.uniform(np.pi - 0.05, np.pi + 0.05)
        rand.append(tuple((ax * ang).tolist()))
    return fixed + rand


def self_check_m3(seed: int, tol: float) -> dict:
    """M3 全局翻转自检: 实现输出逆旋转 (R(-r) 系) 的 q/r_hat, 循环闭包 P3 抓
    (a_exam_rotation.py 同款; 预期 54/60, 边界 6 例 = 精确 0 与 pi)."""
    cases = gen_rotvec_cases(60, seed)
    flipped = 0
    for r in cases:
        rv = np.asarray(r, float)
        Rm = SciRot.from_rotvec(-rv).as_matrix()
        q = SciRot.from_matrix(Rm).as_quat()
        r_hat = SciRot.from_quat(q).as_rotvec()
        if rot_dist(r_hat, rv) > tol:
            flipped += 1
    return {"violations": flipped, "n": len(cases),
            "note": "M3 全局翻转 (a_exam_rotation.py 同款, rotvec 链, 逆旋转输入)"}


def exam(cand_q_to_R, cand_R_to_q, cases, tol: float):
    """P1-P5 考卷 (exam_rotation.py 同款): 返回 (case 违例数, 各谓词违例计数)."""
    vp = {"P1_norm_q": 0, "P2_ortho": 0, "P3_det": 0, "P4_cycle_M": 0, "P5_cycle_q": 0}
    violated = 0
    for c in cases:
        fail = False
        if c["kind"] == "M":
            q = cand_R_to_q(c["M"])
            if abs(np.linalg.norm(q) - 1.0) > tol:
                vp["P1_norm_q"] += 1; fail = True
            R2 = cand_q_to_R(q)
            if np.linalg.norm(R2.T @ R2 - np.eye(3), ord="fro") > tol:
                vp["P2_ortho"] += 1; fail = True
            if abs(np.linalg.det(R2) - 1.0) > tol:
                vp["P3_det"] += 1; fail = True
            if np.linalg.norm(R2 - c["M"], ord="fro") > tol:
                vp["P4_cycle_M"] += 1; fail = True
        else:
            R = cand_q_to_R(c["q"])
            if np.linalg.norm(R.T @ R - np.eye(3), ord="fro") > tol:
                vp["P2_ortho"] += 1; fail = True
            if abs(np.linalg.det(R) - 1.0) > tol:
                vp["P3_det"] += 1; fail = True
            q2 = cand_R_to_q(R)
            if min(np.linalg.norm(q2 - c["q"]), np.linalg.norm(q2 + c["q"])) > tol:
                vp["P5_cycle_q"] += 1; fail = True
        if fail:
            violated += 1
    return violated, vp


def load_candidate(validator_path, functions: list):
    """考生双函数模块 importlib 加载 (规格声明 functions 列表)."""
    p = Path(validator_path)
    if not p.is_file():
        raise FileNotFoundError(f"candidate module not found: {p}")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = []
    for name in functions:
        if not hasattr(mod, name):
            raise AttributeError(f"candidate module {p} missing function {name}")
        fns.append(getattr(mod, name))
    return mod.__name__, fns


def run_spec(spec: dict, validator_path=None, out_path=None) -> dict:
    """规格 -> 生成考卷用例 -> 考生过 P1-P5 -> 判定 + 变异自检 (确定性)."""
    validate_spec(spec)
    if validator_path is None:
        raise ValueError("state_estimator family requires a candidate module path (validator)")
    seed, tol = int(spec["seed"]), float(spec["tol"])
    n_m, n_q = int(spec["n_m"]), int(spec["n_q"])
    n = n_m + n_q

    Ms = SciRot.random(n_m, random_state=seed).as_matrix()
    qs = SciRot.random(n_q, random_state=seed + 1).as_quat()  # [x,y,z,w]
    cases = ([{"kind": "M", "M": M} for M in Ms]
             + [{"kind": "q", "q": q} for q in qs])
    inputs_md5 = hashlib.md5(json.dumps(
        {"Ms": Ms.tolist(), "qs": qs.tolist()}, sort_keys=True).encode()).hexdigest()

    cand_name, (cand_q_to_R, cand_R_to_q) = load_candidate(validator_path,
                                                           spec["functions"])
    v, vp = exam(cand_q_to_R, cand_R_to_q, cases, tol)
    verdict = "PASS" if v == 0 else ("REJECT" if v / n >= 0.90 else "FAIL_UNDER90")

    # 变异自检 (reference.self_check 角色): 考卷灵敏度证明
    m1_v, m1_vp = exam(M1_q_to_R, R_to_q, cases, tol)
    m2_v, m2_vp = exam(q_to_R, M2_R_to_q, cases, tol)
    m3 = self_check_m3(seed, tol)
    reject_min = int(0.90 * n)
    self_check = {
        "M1_q_to_R_norm_half": {"violations": m1_v, "rate": round(m1_v / n, 6),
                                "verdict": "REJECT" if m1_v >= reject_min else "MISS",
                                "per_predicate": m1_vp},
        "M2_R_to_q_norm_half": {"violations": m2_v, "rate": round(m2_v / n, 6),
                                "verdict": "REJECT" if m2_v >= reject_min else "MISS",
                                "per_predicate": m2_vp},
        "M3_global_flip": {"violations": m3["violations"], "n": m3["n"],
                           "rate": round(m3["violations"] / m3["n"], 6),
                           "verdict": "REJECT" if m3["violations"] >= reject_min
                           else "MISS",
                           "note": m3["note"]},
    }

    # 交叉验证 (仅日志): 正确实现 vs scipy 双向 (exam_rotation.py 同款)
    max_err_mat, max_err_quat = 0.0, 0.0
    for M in Ms:
        R2 = SciRot.from_quat(R_to_q(M)).as_matrix()
        max_err_mat = max(max_err_mat, np.linalg.norm(R2 - M, ord="fro"))
    for q in qs:
        q2 = SciRot.from_matrix(q_to_R(q)).as_quat()
        max_err_quat = max(max_err_quat, min(np.linalg.norm(q2 - q),
                                             np.linalg.norm(q2 + q)))
    crosscheck_scipy = {"max_err_matrix_cycle": round(float(max_err_mat), 12),
                        "max_err_quat_cycle": round(float(max_err_quat), 12)}

    payload = {
        "tool": "spsl",
        "version": VERSION,
        "layer": "state_estimator",
        "family": spec["family"],
        "spec": {"path": str(Path(validator_path).resolve()), "name": spec["name"],
                 "spec_md5": spec_md5(spec)},
        "candidate": {"path": str(Path(validator_path).resolve()),
                      "module": cand_name, "functions": spec["functions"]},
        "exam": {"seed": seed, "tol": tol, "n_cases": n, "n_m": n_m, "n_q": n_q,
                 "inputs_md5": inputs_md5},
        "result": {"violations": v, "rate": round(v / n, 6), "verdict": verdict,
                   "per_predicate": vp,
                   "criterion": "0 violations -> PASS; >=90% -> REJECT"},
        "self_check": self_check,
        "crosscheck_scipy": crosscheck_scipy,
        "summary": (f"{cand_name} violations={v}/{n} -> {verdict} "
                    f"(self_check M1={self_check['M1_q_to_R_norm_half']['verdict']} "
                    f"M2={self_check['M2_R_to_q_norm_half']['verdict']} "
                    f"M3={self_check['M3_global_flip']['verdict']})"),
    }
    out = Path(out_path) if out_path else Path.cwd() / f"{spec['name']}_verdict.json"
    md5s = save_json(payload, out)
    # save_json 不就地改写调用方 dict, 显式挂回 md5 字段 (与落盘文件逐字段一致)
    payload["payload_md5"] = md5s["payload_md5"]
    payload["self_md5"] = md5s["self_md5"]
    print(f"[state_estimator] {payload['summary']}  "
          f"(spec_md5={payload['spec']['spec_md5'][:12]}...)")
    print(f"[state_estimator] JSON -> {out}  payload_md5={payload['payload_md5']} "
          f"self_md5={payload['self_md5']}")
    return payload


if __name__ == "__main__":
    raise SystemExit(0)
