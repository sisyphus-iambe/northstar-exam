"""shape_judger 测试 — 11 维特征公式 / IRLS 判定器 / 权重持久化.

协议出处: run_gatelearn.py:144-165 (特征), run_selfevolve.py:43-81 (IRLS/AUC/conf),
run_selfevolve.py:113-124 (预处理), spsl/shape_judger.py.
关键断言: 冻结权重复现 (冷启动权重与 gatelearn_result.json result.weight 逐位相等).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from spsl import shape_judger as sj

NSV2 = Path(__file__).resolve().parents[1]
DATA = NSV2 / "experiments" / "gate_learn_2026-08-19" / "gatelearn_data.json"
RESULT = NSV2 / "experiments" / "gate_learn_2026-08-19" / "gatelearn_result.json"


def test_features_math():
    """手造 116 值向量 -> 11 字段逐一对照公式 (run_gatelearn.py:144-165)."""
    rng = np.random.default_rng(7)
    vec = list(rng.uniform(0.0, 1.0, sj.N_TABLES))
    vec[10] = float("nan")           # 1 个 NaN
    l4 = 3.0
    f = sj.features(vec, l4_nan_count=l4)
    ok = np.array([v for v in vec if not np.isnan(v)], dtype=float)
    assert f["mean"] == float(ok.mean())
    assert f["std"] == float(ok.std())
    assert f["median"] == float(np.median(ok))
    assert f["min"] == float(ok.min())
    assert f["max"] == float(ok.max())
    assert f["frac_nan"] == (1 + l4) / sj.N_TOTAL
    assert f["frac_lt_001"] == float((ok < 0.01).mean())
    assert f["frac_gt_099"] == float((ok > 0.99).mean())
    assert f["ks_unif"] == float(__import__("scipy").stats.kstest(ok, "uniform").statistic)
    assert f["n_unique"] == float(len(np.unique(ok)))
    assert f["frac_l4_nan"] == l4 / sj.N_L4
    assert set(f.keys()) == set(sj.FEAT_FIELDS)


def test_features_all_nan_branch():
    """全 NaN 向量 -> mean..n_unique 全 NaN (run_gatelearn.py:151-155)."""
    vec = [float("nan")] * sj.N_TABLES
    f = sj.features(vec, l4_nan_count=9.0)
    for k in ["mean", "std", "median", "min", "max", "frac_lt_001",
              "frac_gt_099", "ks_unif", "n_unique"]:
        assert np.isnan(f[k]), k
    assert f["frac_nan"] == 1.0
    assert f["frac_l4_nan"] == 1.0


def _frozen_check():
    """读实验数据, 返回 (w, frozen)."""
    d = json.loads(DATA.read_text(encoding="utf-8"))
    frozen = json.loads(RESULT.read_text(encoding="utf-8"))["result"]["weight"]
    samples = [s for s in d["exam"]["samples"] if s.get("load_ok")]
    w, _, _, _ = sj.cold_start_fit(samples)
    return np.asarray(w, dtype=float), np.asarray(frozen, dtype=float)


def test_reproduces_frozen_weight():
    """冷启动权重复现冻结权重 (端口正确性关键).

    同环境 (实验环境 homebrew python + numpy 2.4.4) 逐位相等 maxdiff=0.0;
    跨 BLAS/numpy 微版本有 ULP 级差异 (实测 1.7e-11), 故此处容差断言,
    逐位断言见 test_reproduces_frozen_weight_strict.
    """
    a, b = _frozen_check()
    assert len(a) == len(b) == 12
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9), f"maxdiff={np.abs(a - b).max()!r}"


def test_reproduces_frozen_weight_strict():
    """逐位复现冻结权重 (仅实验环境: numpy 2.4.4, maxdiff=0.0)."""
    if np.__version__ != "2.4.4":
        import pytest
        pytest.skip(f"严格逐位复现需实验环境 numpy 2.4.4 (当前 {np.__version__})")
    a, b = _frozen_check()
    assert np.array_equal(a, b)


def test_separable_and_threshold():
    """可分数据 -> conf 全对; p=0.5 边界判 0 (conf 用 p > 0.5)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = np.array([1] * 20 + [0] * 20)
    X[:20, 0] += 5.0
    X[20:, 0] -= 5.0
    w = sj.logreg_fit(X, y)
    p = sj.predict_proba(X, w)
    c = sj.conf(y, p)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (20, 0, 0, 20)
    # 边界: p == 0.5 判 0
    assert sj.conf(np.array([0, 1]), np.array([0.5, 0.5]))["tp"] == 0
    assert sj.conf(np.array([0, 1]), np.array([0.5, 0.5]))["fp"] == 0


def test_auc_mannwhitney():
    """Mann-Whitney AUC: 完美排序 = 1.0, 逆序 = 0.0, 缺类 = NaN."""
    y = np.array([1, 1, 0, 0])
    assert sj.auc(y, np.array([1.0, 0.9, 0.1, 0.0])) == 1.0
    assert sj.auc(y, np.array([0.0, 0.1, 0.9, 1.0])) == 0.0
    assert np.isnan(sj.auc(np.array([1, 1]), np.array([0.5, 0.5])))


def test_median_fill_standardize():
    """坏值 (NaN/inf/1e298) -> 中位数; sd==0 除 1."""
    X = np.array([[1.0, 5.0], [float("nan"), 5.0], [float("inf"), 5.0],
                  [-float("inf"), 5.0], [1e298, 5.0]])
    med = np.array([1.0, 5.0])
    Xf = sj.median_fill(X, med)
    assert np.array_equal(Xf, np.tile([1.0, 5.0], (5, 1)))
    Xs = sj.standardize(Xf, np.array([1.0, 5.0]), np.array([0.0, 0.0]))
    assert np.array_equal(Xs, np.zeros((5, 2)))


def test_weights_roundtrip_and_tamper(tmp_path):
    """权重往返逐位一致; 篡改 w -> load 报错."""
    meta = {"n_seen": 120, "trained_on": "g3_p1", "n_bad": 66, "n_good": 54}
    w = [80.5, 19.9, -45.3, 8.0, -13.2, -6.5, -54.6, -57.5, 89.0, 35.6, 0.3, 9.8]
    p = sj.save_weights(tmp_path / "w.json", w=w, med=[0.5] * 11, mu=[0.4] * 11,
                        sd=[0.3] * 11, meta=meta)
    got = sj.load_weights(tmp_path / "w.json")
    assert got["w"] == p["w"] == w
    assert got["meta"] == meta
    assert got["std_from"] is None
    # 篡改 w -> 验签失败
    raw = json.loads((tmp_path / "w.json").read_text())
    raw["w"][0] += 1.0
    (tmp_path / "w.json").write_text(json.dumps(raw))
    try:
        sj.load_weights(tmp_path / "w.json")
        raise AssertionError("篡改后应报错")
    except ValueError:
        pass


def test_digest_deterministic():
    """digest: 键序无关, 值变则变."""
    a = {"x": 1, "y": [2.0, 3.0]}
    b = {"y": [2.0, 3.0], "x": 1}
    assert sj.digest(a) == sj.digest(b)
    assert sj.digest(a) != sj.digest({"x": 2, "y": [2.0, 3.0]})


# ---------------------------------------------------------------------------
# 审计覆盖 (2026-08-20): judge()/load_weights 错误层/超大 int
# ---------------------------------------------------------------------------

def _judge_weights():
    return {"w": [1.0] * 11 + [0.0], "med": [0.5] * 11,
            "mu": [0.5] * 11, "sd": [1.0] * 11}


def test_judge_threshold():
    """judge(): 特征低分 -> good (False), 高分 -> bad (True). 阈值 0.5.

    全 0 特征: standardize -> -0.5, 得分 = 11x(-0.5)x1 = -5.5 -> p < 0.5.
    全 1 特征: standardize -> +0.5, 得分 = +5.5 -> p > 0.5.
    """
    weights = _judge_weights()
    assert sj.judge({f: 0.0 for f in sj.FEAT_FIELDS}, weights) is False
    assert sj.judge({f: 1.0 for f in sj.FEAT_FIELDS}, weights) is True


def test_judge_nan_feature_ok():
    """NaN 特征 (合法垃圾特征) 判定不崩 (median_fill 兜底)."""
    feat = {f: 0.5 for f in sj.FEAT_FIELDS}
    feat["mean"] = float("nan")
    assert isinstance(sj.judge(feat, _judge_weights()), bool)


def test_judge_huge_int_rejected():
    """超大 int (float 不可表示) -> ValueError (审计 A4 配套)."""
    feat = {f: 0.5 for f in sj.FEAT_FIELDS}
    feat["mean"] = 10 ** 400
    with pytest.raises(ValueError, match="不可表示"):
        sj.judge(feat, _judge_weights())


def test_load_weights_wrong_layer(tmp_path):
    """错误 tool/layer -> 明确报错 (审计 B3: load_weights 层校验零覆盖)."""
    (tmp_path / "w.json").write_text(json.dumps(
        {"tool": "spsl", "layer": "other", "payload_md5": "x", "self_md5": "y"}))
    with pytest.raises(ValueError, match="不是 shape-judger"):
        sj.load_weights(tmp_path / "w.json")
