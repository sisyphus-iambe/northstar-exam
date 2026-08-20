"""SPSL 形状判定器 — 11 维输出形状特征 IRLS logistic (北极星 v3 自进化, 2026-08-20).

协议照搬已验证实验 (不许改进, 与实验逐字一致):
- 11 维特征 features(): experiments/gate_learn_2026-08-19/run_gatelearn.py:144-165
- 判定器 IRLS (iters=20, reg=1e-6, _guard_step 有限性守卫):
  experiments/selfevolve_2026-08-20/run_selfevolve.py:43-60 logreg_fit
- 推理 predict_proba: run_selfevolve.py:63-65 (阈值 0.5, 正类 = bad)
- AUC = Mann-Whitney (run_selfevolve.py:68-72); conf (run_selfevolve.py:74-81)
- 预处理 (中位数填充 NaN/inf/|x|>1e100 + 训练池统计量标准化): run_selfevolve.py:116-125
- md5 纪律: digest = sha256(json.dumps(sort_keys, ensure_ascii=False)) (run_selfevolve.py:84-85)

本模块 = 纯库 (无 CLI). 自进化管线 (档案/入口防污染/外部裁决门/审计) 见 spsl/selfevolve.py.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

from spsl import VERSION

N_TABLES = 116                      # 考卷 L2/L3 正常表数 (run_gatelearn.py:33)
N_L4 = 9                            # L4 畸形输入数 (run_gatelearn.py:34)
N_TOTAL = N_TABLES + N_L4           # 125 全部调用 (run_gatelearn.py:35)
TOL = 1e-6                          # 对拍容差 (run_gatelearn.py:26)
THRESHOLD = 0.5                     # 判定阈值: prob > 0.5 -> bad
IRLS_ITERS = 20                     # IRLS 迭代上限 (run_selfevolve.py:43)
IRLS_REG = 1e-6                     # 正则 (run_selfevolve.py:43)
_ABS_MAX = 1e100                    # |x| > 1e100 视为坏值 (run_selfevolve.py:121)

FEAT_FIELDS = ["mean", "std", "median", "min", "max", "frac_nan",
               "frac_lt_001", "frac_gt_099", "ks_unif", "n_unique", "frac_l4_nan"]


def features(vec, l4_nan_count):
    """11 维特征, 纯输出形状 (run_gatelearn.py:144-165 逐行照搬).
    frac_nan = (116 中 NaN/异常 + 9 中诚实失败) / 125 全部调用占比"""
    ok = np.array([v for v in vec if not math.isnan(v)], dtype=float)
    n_nan = sum(1 for v in vec if math.isnan(v))
    d = {"frac_nan": (n_nan + l4_nan_count) / N_TOTAL}
    if len(ok) == 0:
        for k in ["mean", "std", "median", "min", "max", "frac_lt_001",
                  "frac_gt_099", "ks_unif", "n_unique"]:
            d[k] = float("nan")
    else:
        d["mean"] = float(ok.mean())
        d["std"] = float(ok.std())
        d["median"] = float(np.median(ok))
        d["min"] = float(ok.min())
        d["max"] = float(ok.max())
        d["frac_lt_001"] = float((ok < 0.01).mean())
        d["frac_gt_099"] = float((ok > 0.99).mean())
        d["ks_unif"] = float(stats.kstest(ok, "uniform").statistic)
        d["n_unique"] = float(len(np.unique(ok)))
    d["frac_l4_nan"] = l4_nan_count / N_L4
    return d


# ---------------------------------------------------------------------------
# IRLS logistic (run_selfevolve.py:36-81 照搬)
# ---------------------------------------------------------------------------

def _guard_step(w, step):
    w_new = w - step
    if np.all(np.isfinite(w_new)):
        return w_new, True
    return w, False


def logreg_fit(Xs, y, iters=IRLS_ITERS, reg=IRLS_REG):
    """IRLS logistic 含截距 (run_selfevolve.py:43-60 照搬)."""
    n, d = Xs.shape
    w = np.zeros(d + 1)
    X1 = np.hstack([Xs, np.ones((n, 1))])
    for _ in range(iters):
        z = np.clip(X1 @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        S = np.diag(p * (1 - p) + reg)
        try:
            H = X1.T @ S @ X1 + reg * np.eye(d + 1)
            g = X1.T @ (p - y)
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return w
        w, ok = _guard_step(w, step)
        if not ok:
            break
    return w


def predict_proba(Xs, w):
    z = np.clip(Xs @ w[:-1] + w[-1], -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def auc(y, p):
    """Mann-Whitney AUC (run_selfevolve.py:68-72 照搬). 缺任一类别 -> NaN."""
    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return stats.mannwhitneyu(p[y == 1], p[y == 0]).statistic / (n_pos * n_neg)


def conf(y, p):
    """混淆 (run_selfevolve.py:74-81 照搬; 阈值 0.5)."""
    pred = (p > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def digest(obj):
    """md5 规范 (run_selfevolve.py:84-85 照搬)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 预处理 (run_selfevolve.py:113-124 口径) 与推理
# ---------------------------------------------------------------------------

def median_fill(X, med):
    """坏值 (NaN/inf/|x|>1e100) -> 列中位数 (run_selfevolve.py:118-122 口径)."""
    Xf = X.copy()
    for j, m in enumerate(med):
        bad = np.isnan(Xf[:, j]) | np.isinf(Xf[:, j]) | (np.abs(Xf[:, j]) > _ABS_MAX)
        Xf[:, j] = np.where(bad, m, Xf[:, j])
    return Xf


def standardize(X, mu, sd):
    """训练池统计量标准化 (analyze_gatelearn.py:56-58 口径). sd==0 除 1."""
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def cold_start_fit(samples):
    """冷启动: 训练子集 (role=='train', = g3_p1 14 份) 统计量 + 权重.

    返回 (w, med, mu, sd). w 与 gatelearn_result.json result.weight 逐位相等
    (实现时实测 max diff = 0.0); 统计量冻结于训练子集 (防泄漏), 更新永不重算.
    """
    rows = [s for s in samples if s.get("role") == "train"]
    if not rows:
        raise ValueError("冷启动需要训练子集 (role=='train')")
    X = np.array([[s["feat"][f] for f in FEAT_FIELDS] for s in rows], dtype=float)
    y = np.array([s["label"] for s in rows], dtype=int)
    med = np.nanmedian(X, axis=0)
    Xf = median_fill(X, med)
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
    Xs = standardize(Xf, mu, sd)
    return logreg_fit(Xs, y), med, mu, sd


def judge(feat, weights):
    """单样本判定: 特征 dict -> bool (True = bad). 阈值 0.5, 正类 = bad.

    超大 Python int (float 不可表示) -> ValueError (审计发现, 2026-08-20).
    """
    try:
        x = np.array([[feat[f] for f in FEAT_FIELDS]], dtype=float)
    except OverflowError:
        raise ValueError("feat 含 float 不可表示的数值 (超大整数)")
    xf = median_fill(x, np.array(weights["med"]))
    xs = standardize(xf, np.array(weights["mu"]), np.array(weights["sd"]))
    p = predict_proba(xs, np.array(weights["w"]))[0]
    return bool(p > THRESHOLD)


# ---------------------------------------------------------------------------
# 权重持久化 (JSON + payload_md5/self_md5 验签, 同实验落盘规范)
# ---------------------------------------------------------------------------

def save_weights(path, *, w, med, mu, sd, meta):
    """权重 JSON 落盘 (w + 标准化统计量 + meta + 双 md5), 返回 payload."""
    payload = {
        "tool": "spsl", "layer": "shape-judger", "version": VERSION,
        "feat_fields": FEAT_FIELDS,
        "params": {"iters": IRLS_ITERS, "reg": IRLS_REG, "threshold": THRESHOLD},
        "w": [float(x) for x in w],
        "med": [float(x) for x in med],
        "mu": [float(x) for x in mu],
        "sd": [float(x) for x in sd],
        "std_from": meta.get("std_from"),
        "meta": meta,
    }
    payload["payload_md5"] = digest(payload)
    payload["self_md5"] = digest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")      # 原子写 (tmp + os.replace)
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, path)
    return payload


def load_weights(path):
    """加载权重并验签 (payload_md5/self_md5 独立复算, 失败即报)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("tool") != "spsl" or payload.get("layer") != "shape-judger":
        raise ValueError(f"{path} 不是 shape-judger 权重 JSON "
                         f"(tool={payload.get('tool')!r}, layer={payload.get('layer')!r})")
    # 落盘规范 (实验同款): payload_md5 = digest(无任何 md5), self_md5 = digest(含 payload_md5)
    core = {k: v for k, v in payload.items() if k not in ("payload_md5", "self_md5")}
    if digest(core) != payload.get("payload_md5"):
        raise ValueError(f"{path}: payload_md5 复算不一致 (文件被改动?)")
    signed = dict(core)
    signed["payload_md5"] = payload["payload_md5"]
    if digest(signed) != payload.get("self_md5"):
        raise ValueError(f"{path}: self_md5 复算不一致 (文件被改动?)")
    return payload
