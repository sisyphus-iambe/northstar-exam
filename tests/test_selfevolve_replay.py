"""replay 复现测试 — 与 selfevolve_2026-08-20/selfevolve_data.json 逐字一致.

实验环境 (homebrew python + numpy 2.4.4) 下 payload_md5 == c05b91b6...7555d1e
(实测命中, 见 test_replay_payload_md5_strict); 跨 numpy 微版本/BLAS 有 ULP 级
浮点差异 (实测 ~1.7e-11: 测试审计 2026-08-20 实测 homebrew 2.4.4 vs Framework
2.4.3 冻结权重 maxdiff), 数字断言用容差, 布尔/整数断言跨环境稳定.
字段级对比 (test_replay_field_level_matches_experiment) 任意环境可跑, 是严格
md5 测试 (numpy 2.4.4 专属) 的环境彩票盲区兜底.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from spsl import selfevolve as se

NSV2 = Path(__file__).resolve().parents[1]
DATA = NSV2 / "experiments" / "gate_learn_2026-08-19" / "gatelearn_data.json"
EXP_DATA = NSV2 / "experiments" / "selfevolve_2026-08-20" / "selfevolve_data.json"
EXPECTED_MD5 = "c05b91b64b7c0d908e37eb4a414b53b4ac622ddcc4373466fd4c1f9927555d1e"


def _replay(out):
    """replay 直读 gatelearn_data.json (审计 A3), 不依赖 state."""
    return se.cmd_replay(argparse.Namespace(out=str(out), data=str(DATA)))


@pytest.fixture(scope="module")
def replay_payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "replay.json"
    assert _replay(out) == 0
    return json.loads(out.read_text())


def test_replay_verdict_matches_experiment(replay_payload):
    """M1/M2/M3/M4 数字与实验 selfevolve_data.json 一致 (实验实测值)."""
    v = replay_payload["verdict"]
    assert v["main"] == "FAIL" and v["aux"] == "FAIL" and v["verdict"] == "PARTIAL"
    m1 = v["M1_selfevolve_ge_static_plus_005"]
    assert m1["pass"] is True
    assert m1["I_auc"] == pytest.approx(0.8818181818181818, abs=1e-12)
    assert m1["G_auc"] == pytest.approx(0.9727272727272728, abs=1e-12)
    assert m1["S_auc"] == pytest.approx(0.7893048128342246, abs=1e-12)
    m2 = v["M2_no_forget_b1"]
    assert m2["pass"] is False
    assert m2["I_b1_t5"] == pytest.approx(0.8636363636363636, abs=1e-12)
    assert m2["I_b1_t1"] == m2["G_b1_t1"] == m2["G_b1_t5"] == 1.0
    m3 = v["M3_no_fp_growth"]
    assert m3["pass"] is True
    assert m3["I_fp"] == m3["G_fp"] == m3["S_fp"] == 0
    m4 = v["M4_pollution_gate"]
    assert m4["pass"] is False
    assert m4["Gp_auc"] == pytest.approx(0.8467857142857143, abs=1e-12)
    assert m4["Ip_auc"] == pytest.approx(0.9503571428571429, abs=1e-12)
    assert m4["Gp_fp"] == 3 and m4["Ip_fp"] == 4


def test_replay_gate_rejects_t5_and_freezes(replay_payload):
    """场景 A: 门在 t5 拒 (0.86885 vs 0.97541) -> G 权重冻结在 t4,
    A_G_t5 != A_I_t5 (I 在 t5 学进去了, G 没有)."""
    g = replay_payload["result"]["scene_A"]["G"]
    assert g["rejections"] == 1
    gates = g["gates"]
    assert [x["ok"] for x in gates] == [True, True, True, False]
    t5 = gates[-1]
    assert t5["t"] == 5
    assert t5["auc_cand"] == pytest.approx(0.8688524590163934, abs=1e-12)
    assert t5["auc_cur"] == pytest.approx(0.9754098360655737, abs=1e-12)
    assert t5["fp_cand"] == t5["fp_cur"] == 0
    w = replay_payload["result"]["weights"]
    assert w["A_G_t5"] != w["A_I_t5"]


def test_replay_scene_p_gate_records(replay_payload):
    """场景 P (污染 t3 前 5 bad 翻 0): 门拒 t3/t4/t5 共 3 次 —
    污染批在 t3 就触发门 (fp_cand 6 > 0)."""
    g = replay_payload["result"]["scene_P"]["G"]
    assert g["rejections"] == 3
    assert [x["ok"] for x in g["gates"]] == [True, False, False, False]
    t3 = g["gates"][1]
    assert t3["t"] == 3
    assert t3["auc_cand"] == pytest.approx(0.7045454545454546, abs=1e-12)
    assert t3["fp_cand"] == 6 and t3["fp_cur"] == 0


def test_replay_exam_batches_and_pollute(replay_payload):
    """批次划分/字面量字段与实验逐字一致."""
    ex = replay_payload["exam"]
    assert ex["src"] == "gatelearn_data.json" and ex["n"] == 120 and ex["n_feat"] == 11
    assert ex["final_mask"] == "B2..B5(106)"
    assert ex["preprocess"] == "median_fill+std 仅用 B1 统计量"
    assert ex["pollute"] == {"n": 5, "names": ["vg:vg_01_b1", "vg:vg_01_b2",
                                               "vg:vg_01_b3", "vg:vg_01_b4",
                                               "vg:vg_02_b3"]}
    b = ex["batches"]
    assert [b[str(k)]["n"] for k in range(1, 6)] == [14, 19, 35, 36, 16]
    assert (b["1"]["bad"], b["1"]["good"]) == (11, 3)


def test_replay_double_run_byte_identical(tmp_path):
    """双跑逐字节一致 (确定性纪律)."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert _replay(a) == 0 and _replay(b) == 0
    assert a.read_bytes() == b.read_bytes()


def test_replay_payload_md5_strict(tmp_path):
    """逐字 md5 复现实验 (仅实验环境 numpy 2.4.4, 实测命中).

    生产环境 (homebrew python + numpy 2.4.4) 下必须 == c05b91b6...7555d1e.
    """
    if np.__version__ != "2.4.4":
        pytest.skip(f"严格 md5 需实验环境 numpy 2.4.4 (当前 {np.__version__})")
    out = tmp_path / "replay.json"
    assert _replay(out) == 0
    p = json.loads(out.read_text())
    assert p["payload_md5"] == EXPECTED_MD5
    assert p["self_md5"] == se.sj.digest({k: v for k, v in p.items()
                                          if k not in ("payload_md5", "self_md5")})


def test_replay_field_level_matches_experiment(replay_payload):
    """字段级对比实验 selfevolve_data.json (任意环境可跑, 容差 1e-9).

    覆盖 strict md5 测试 (numpy 2.4.4 专属) 的盲区: 环境彩票下仍守数字契约
    (审计 B2: V10 突变在 pytest 环境可逃逸严格断言, 本测试堵住).
    """
    exp = json.loads(EXP_DATA.read_text())
    v, ev = replay_payload["verdict"], exp["verdict"]
    assert v["main"] == ev["main"] and v["aux"] == ev["aux"]
    assert v["verdict"] == ev["verdict"]
    for k in ("M1_selfevolve_ge_static_plus_005", "M2_no_forget_b1"):
        assert v[k]["pass"] == ev[k]["pass"]
        for fk in ("I_auc", "G_auc", "S_auc") if k.endswith("005") else ("I_b1_t5", "I_b1_t1",
                                                                         "G_b1_t5", "G_b1_t1"):
            assert v[k][fk] == pytest.approx(ev[k][fk], abs=1e-9)
    assert v["M3_no_fp_growth"]["I_fp"] == ev["M3_no_fp_growth"]["I_fp"] == 0
    assert v["M4_pollution_gate"]["pass"] == ev["M4_pollution_gate"]["pass"]
    assert (v["M4_pollution_gate"]["Gp_fp"] == ev["M4_pollution_gate"]["Gp_fp"]
            and v["M4_pollution_gate"]["Ip_fp"] == ev["M4_pollution_gate"]["Ip_fp"])
    # weights 容差 1e-3: 增量轨迹 (I/G) 是 5 次嵌套拟合, 跨环境误差累积放大
    # (实测 numpy 2.4.3: A_I_t5 maxdiff 7e-5, A_G_t5 maxdiff 2.8e-4; A_S_t5 1.7e-11);
    # 判据数字 (AUC) 实测零漂移, 由上方断言独立守住.
    for wk in ("A_S_t5", "A_I_t5", "A_G_t5", "P_I_t5", "P_G_t5"):
        w, ew = replay_payload["result"]["weights"][wk], exp["result"]["weights"][wk]
        assert len(w) == len(ew) == 12
        assert all(x == pytest.approx(y, abs=1e-3) for x, y in zip(w, ew))
    for scene in ("scene_A", "scene_P"):
        g, eg = replay_payload["result"][scene]["G"], exp["result"][scene]["G"]
        assert g["rejections"] == eg["rejections"]
        for a, b in zip(g["gates"], eg["gates"]):
            assert a["t"] == b["t"] and a["ok"] == b["ok"]
            assert a["auc_cand"] == pytest.approx(b["auc_cand"], abs=1e-9)
            assert a["auc_cur"] == pytest.approx(b["auc_cur"], abs=1e-9)
            assert a["fp_cand"] == b["fp_cand"] and a["fp_cur"] == b["fp_cur"]
    for arm in ("S", "I", "G"):
        p, ep = (replay_payload["result"]["scene_A"][arm]["per_t"],
                 exp["result"]["scene_A"][arm]["per_t"])
        for t in "12345":
            assert p[t]["new_auc"] == pytest.approx(ep[t]["new_auc"], abs=1e-9)
            for fk in range(1, int(t) + 1):
                assert p[t]["forget_auc"][str(fk)] == pytest.approx(
                    ep[t]["forget_auc"][str(fk)], abs=1e-9)
            c, ec = p[t]["new_conf"], ep[t]["new_conf"]
            for f in ("tp", "fp", "fn", "tn"):
                assert c[f] == ec[f]
    ex, ee = replay_payload["exam"], exp["exam"]
    assert ex["src"] == ee["src"] and ex["n"] == ee["n"] == 120
    assert ex["batches"] == ee["batches"]
    assert ex["pollute"] == ee["pollute"]
    assert ex["final_mask"] == ee["final_mask"]
    assert ex["preprocess"] == ee["preprocess"]
