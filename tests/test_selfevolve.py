"""selfevolve 管线测试 — 冷启动 / 验签防篡改 / 入口防污染 / 门裁决.

协议出处: run_selfevolve.py:141-154 (gate_check), 北极星v3.md 自进化节
(①增量学习 ②入口防污染 ③门控防退化). 复现实验见 test_selfevolve_replay.py.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from spsl import selfevolve as se
from spsl import shape_judger as sj

NSV2 = Path(__file__).resolve().parents[1]
DATA = NSV2 / "experiments" / "gate_learn_2026-08-19" / "gatelearn_data.json"
RESULT = NSV2 / "experiments" / "gate_learn_2026-08-19" / "gatelearn_result.json"


# ---------------------------------------------------------------------------
# 入口防污染 (北极星v3.md ②): 每条全过才收, 各拒绝原因逐条断言
# ---------------------------------------------------------------------------

def _entry(label=1, **over):
    e = {
        "name": "g3:g3_llm_01", "pool": "g3", "stem": "g3_llm_01",
        "load_ok": True, "label_ok": True,
        "label": label, "n_err": 3, "n_l4_err": 2,
        "feat": {f: 0.5 for f in sj.FEAT_FIELDS},
        "evidence": {"ref": se.EVIDENCE_REF, "tol": 1e-6,
                     "n_viol": 3, "pvec_md5": "a" * 64},
    }
    e["feat"]["frac_l4_nan"] = 2 / 9
    if "feat" in over and isinstance(over["feat"], dict):
        e["feat"].update(over.pop("feat"))
    e.update(over)
    return e


@pytest.mark.parametrize("over,reason", [
    ({"name": "no_colon"}, "name 结构不符"),
    ({"name": "evil:bad01", "pool": "evil", "stem": "bad01"}, "白名单外池"),
    ({"pool": "vg"}, "name 与 pool 不一致"),
    ({"stem": "other"}, "name 与 stem 不一致"),
    ({"load_ok": False}, "load_ok 非真"),
    ({"label_ok": False}, "label_ok 非真"),
    ({"label": True}, "label 非 int{0,1}"),
    ({"label": 2}, "label 非 int{0,1}"),
    ({"label": 0.0}, "label 非 int{0,1}"),
    ({"n_err": -1}, "n_err 非法"),
    ({"n_err": 1.5}, "n_err 非法"),
    ({"n_l4_err": -1}, "n_l4_err 非法"),
    ({"label": 0, "n_err": 1}, "label=0 但 n_err>0"),
    ({"feat": None}, "feat 缺失"),
    ({"feat": {"mean": "x"}}, "feat.mean 非数值"),
    ({"feat": {"frac_l4_nan": 0.5}}, "frac_l4_nan 与 n_l4_err 不一致"),
    ({"evidence": None}, "evidence 缺失"),
    ({"evidence": {"ref": "other", "tol": 1e-6, "n_viol": 3, "pvec_md5": "a" * 64}},
     "evidence.ref 非对拍来源"),
    ({"evidence": {"ref": se.EVIDENCE_REF, "tol": 1e-5, "n_viol": 3, "pvec_md5": "a" * 64}},
     "evidence.tol 非 1e-6"),
    ({"evidence": {"ref": se.EVIDENCE_REF, "tol": 1e-6, "n_viol": -1, "pvec_md5": "a" * 64}},
     "evidence.n_viol 非法"),
    ({"evidence": {"ref": se.EVIDENCE_REF, "tol": 1e-6, "n_viol": 0, "pvec_md5": "a" * 64}},
     "n_viol 与 label 矛盾"),
    ({"evidence": {"ref": se.EVIDENCE_REF, "tol": 1e-6, "n_viol": 3, "pvec_md5": "xyz"}},
     "pvec_md5 非 64 位 hex"),
])
def test_entry_reject_reasons(over, reason):
    e = _entry(**over)
    acc, rej = se.filter_batch([e], [])
    assert acc == []
    assert rej == [{"name": e["name"], "reason": reason}]


def test_entry_accept_and_dup():
    e = _entry()
    acc, rej = se.filter_batch([e, _entry(), _entry(name="g3:g3_llm_02", stem="g3_llm_02")],
                               ["g3:g3_llm_02"])
    assert len(acc) == 1            # 批内重复 + 档案已有都拒
    assert [r["reason"] for r in rej] == ["duplicate_name", "duplicate_name"]


def test_entry_label0_ok():
    """label=0 合法样本: n_err=0, n_viol=0, 全过."""
    e = _entry(label=0, n_err=0)
    e["evidence"]["n_viol"] = 0
    acc, rej = se.filter_batch([e], [])
    assert len(acc) == 1 and rej == []


def test_entry_nan_feat_allowed():
    """NaN/inf 特征 = 合法垃圾特征 (真实 bad 样本有 NaN), 不拒."""
    e = _entry()
    e["feat"]["mean"] = float("nan")
    e["feat"]["std"] = float("inf")
    acc, rej = se.filter_batch([e], [])
    assert len(acc) == 1 and rej == []


# ---------------------------------------------------------------------------
# gate_check 门逻辑 (run_selfevolve.py:141-154 照搬)
# ---------------------------------------------------------------------------

def test_gate_check_accepts_good_candidate():
    """候选在旧账上不降 AUC 且 fp 不增 -> 过."""
    rng = np.random.default_rng(11)
    Xs = rng.normal(size=(40, 3))
    y = np.array([1] * 20 + [0] * 20)
    Xs[:20, 0] += 3.0
    Xs[20:, 0] -= 3.0
    w_cur = sj.logreg_fit(Xs, y)
    w_cand = sj.logreg_fit(np.vstack([Xs, [5.0, 0, 0], [-5.0, 0, 0]]),
                           np.append(y, [1, 0]))
    ok, a_c, a_u, fp_c, fp_u = se.gate_check(Xs, y, np.ones(40, bool), w_cand, w_cur)
    assert ok and fp_c == 0 and fp_u == 0 and a_c >= a_u - se.GATE_AUC_TOL


def test_gate_check_rejects_bad_candidate():
    """候选把旧 bad 全翻 0 (遗忘) -> 旧账 AUC 大降 -> 拒."""
    rng = np.random.default_rng(12)
    Xs = rng.normal(size=(40, 3))
    y = np.array([1] * 20 + [0] * 20)
    Xs[:20, 0] += 3.0
    Xs[20:, 0] -= 3.0
    w_cur = sj.logreg_fit(Xs, y)
    # 逆标签训练 = 完全遗忘的候选
    w_bad = sj.logreg_fit(Xs, 1 - y)
    ok, a_c, a_u, fp_c, fp_u = se.gate_check(Xs, y, np.ones(40, bool), w_bad, w_cur)
    assert not ok
    assert a_c < a_u - se.GATE_AUC_TOL


def test_gate_check_empty_mask_passes():
    """空 mask / 缺类别 -> 通过 (无比较对象, run_selfevolve.py:145)."""
    Xs = np.zeros((10, 3))
    y = np.zeros(10, int)
    w = sj.logreg_fit(Xs, y)
    ok, a_c, a_u, fp_c, fp_u = se.gate_check(Xs, y, np.zeros(10, bool), w, w)
    assert ok and np.isnan(a_c) and np.isnan(a_u)


# ---------------------------------------------------------------------------
# 冷启动 init (tmp state) + 验签防篡改 + 二次 init
# ---------------------------------------------------------------------------

def _init_args(state):
    return argparse.Namespace(state=str(state), data=str(DATA),
                              result=str(RESULT), force=False)


def test_init(tmp_path):
    assert se.cmd_init(_init_args(tmp_path)) == 0
    archive, weights, audit = se.load_state(tmp_path)
    assert archive["exam"]["n_entries"] == 120
    assert archive["exam"]["n_bad"] == 66 and archive["exam"]["n_good"] == 54
    assert [s["seq"] for s in archive["samples"]] == list(range(1, 121))
    assert archive["samples"][0]["evidence"]["origin"] == "coldstart"
    assert weights["meta"]["n_seen"] == 120
    assert audit["events"][0]["kind"] == "init"
    assert audit["events"][0]["frozen_weight_maxdiff"] <= se.FROZEN_MATCH_TOL
    assert weights["meta"]["std_from"] == "g3_p1 冷启动训练池(14份, 冻结不重算)"
    # 冻结权重复现 (容差, 跨环境 ULP; 逐位见 shape_judger strict 测试)
    frozen = json.loads(RESULT.read_text())["result"]["weight"]
    assert np.allclose(weights["w"], frozen, rtol=1e-9, atol=1e-9)


def test_init_verifies_source_files(tmp_path):
    """源文件被篡改 -> init 拒绝 (三重信任之一: 源文件 self_md5 验签)."""
    tampered = tmp_path / "tampered_data.json"
    tampered.write_text(json.dumps({"tool": "spsl", "layer": "gate-learn",
                                    "exam": {"samples": []}}))
    with pytest.raises(ValueError):
        se.cmd_init(argparse.Namespace(state=str(tmp_path / "s"), data=str(tampered),
                                       result=str(RESULT), force=False))


def test_init_second_time_fails_and_force_rebuilds(tmp_path):
    assert se.cmd_init(_init_args(tmp_path)) == 0
    assert se.cmd_init(_init_args(tmp_path)) == 1   # 已存在, 无 --force
    before = (tmp_path / "archive.json").read_bytes()
    assert se.cmd_init(argparse.Namespace(state=str(tmp_path), data=str(DATA),
                                          result=str(RESULT), force=True)) == 0
    assert (tmp_path / "archive.json").read_bytes() == before   # 重建逐字节一致


def test_tamper_detected(tmp_path):
    """篡改 archive/audit/weights -> load_state 报错."""
    assert se.cmd_init(_init_args(tmp_path)) == 0
    for name, field in [("archive.json", "samples"), ("audit.json", "events"),
                        ("weights.json", "w")]:
        d = json.loads((tmp_path / name).read_text())
        if field == "samples":
            d[field][0]["label"] = 1 - d[field][0]["label"]
        elif field == "events":
            d[field][0]["kind"] = "hacked"
        else:
            d[field][0] += 1.0
        (tmp_path / name).write_text(json.dumps(d))
        with pytest.raises(ValueError):
            se.load_state(tmp_path)


def test_update_all_rejected_no_change(tmp_path):
    """入口全败 -> 权重不变, audit 逐条记 entry_reject."""
    assert se.cmd_init(_init_args(tmp_path)) == 0
    _, weights0, _ = se.load_state(tmp_path)
    batch = tmp_path / "bad_batch.json"
    batch.write_text(json.dumps({"samples": [{"name": "evil:p1", "pool": "evil",
                                              "stem": "p1", "load_ok": True},
                                             {"name": "g3:dup", "pool": "g3",
                                              "stem": "dup", "load_ok": True,
                                              "label_ok": True, "label": 1, "n_err": 1,
                                              "n_l4_err": 1, "feat": {},
                                              "evidence": {"ref": se.EVIDENCE_REF,
                                                           "tol": 1e-6, "n_viol": 1,
                                                           "pvec_md5": "a" * 64}}]}))
    assert se.cmd_update(argparse.Namespace(batch=str(batch), state=str(tmp_path))) == 0
    archive, weights, audit = se.load_state(tmp_path)
    assert archive["exam"]["n_entries"] == 120            # 未动
    assert weights["self_md5"] == weights0["self_md5"]    # 权重不变
    kinds = [e["kind"] for e in audit["events"]]
    assert kinds[1:] == ["entry_reject", "entry_reject"]


def _clone(src, new_stem, label, data_samples):
    """从 gatelearn 真实样本克隆出新样本 (补证据链, name 保留白名单池前缀)."""
    c = dict(src)
    c["name"] = f"{src['pool']}:{new_stem}"
    c["stem"] = new_stem
    c["label"] = label
    c["n_err"] = 0 if label == 0 else c["n_err"]
    c["evidence"] = {"ref": se.EVIDENCE_REF, "tol": 1e-6,
                     "n_viol": 0 if label == 0 else c["n_err"], "pvec_md5": "a" * 64}
    return c


@pytest.fixture()
def gatelearn_samples():
    d = json.loads(DATA.read_text())
    return [s for s in d["exam"]["samples"] if s.get("load_ok")]


def _update(tmp_path, samples):
    bf = tmp_path / "batch.json"
    bf.write_text(json.dumps({"samples": samples}))
    return se.cmd_update(argparse.Namespace(batch=str(bf), state=str(tmp_path)))


def test_update_rejects_fp_growth(tmp_path, gatelearn_samples):
    """6 份新 bad -> 4 份 L4-only (n_err=0) 被入口拒, 2 份过门检 -> 旧账 fp 0->10 超门 -> 拒.

    数字 (numpy 2.4.4): auc_cand=0.88468 auc_cur=0.82492 fp_cand=10 fp_cur=0.
    入口拦截 = M4 教训落地 (污染进不了训练集): 6 份中 4 份 label=1 但 n_viol=0
    矛盾 (L4-only bad 克隆) -> entry_reject; 剩余 2 份候选被门拒.
    """
    assert se.cmd_init(_init_args(tmp_path)) == 0
    _, weights0, _ = se.load_state(tmp_path)
    batch = [_clone(s, f"clA{i:02d}", 1, gatelearn_samples)
             for i, s in enumerate([s for s in gatelearn_samples if s["label"] == 1][:6])]
    assert _update(tmp_path, batch) == 0
    archive, weights, audit = se.load_state(tmp_path)
    assert archive["exam"]["n_entries"] == 120
    assert weights["self_md5"] == weights0["self_md5"]
    rej = [e for e in audit["events"] if e["kind"] == "reject"]
    assert len(rej) == 1
    assert rej[0]["n_pending"] == 2          # 6 份中只有 2 份过入口
    entry_rejs = [e for e in audit["events"] if e["kind"] == "entry_reject"]
    assert len(entry_rejs) == 4
    assert {r["reason"] for r in entry_rejs} == {"n_viol 与 label 矛盾"}
    g = rej[0]["gate"]
    assert not (g["auc_cand"] >= g["auc_cur"] - se.GATE_AUC_TOL and g["fp_cand"] <= g["fp_cur"])
    assert g["fp_cand"] == 10 and g["fp_cur"] == 0
    assert g["auc_cand"] == pytest.approx(0.8846801346801347, abs=1e-9)
    assert g["auc_cur"] == pytest.approx(0.8249158249158249, abs=1e-9)


def test_update_accept_good_batch(tmp_path, gatelearn_samples):
    """8 份新 good -> 门放行: 档案追加, 权重更新, audit accept (独立场景实测数字).

    数字 (numpy 2.4.4): auc_cand=0.93939 auc_cur=0.82492 fp 0->0.
    """
    assert se.cmd_init(_init_args(tmp_path)) == 0
    _, weights0, _ = se.load_state(tmp_path)
    batch = [_clone(s, f"clC{i:02d}", 0, gatelearn_samples)
             for i, s in enumerate([s for s in gatelearn_samples if s["label"] == 0][:8])]
    assert _update(tmp_path, batch) == 0
    archive, weights, audit = se.load_state(tmp_path)
    assert archive["exam"]["n_entries"] == 128
    assert weights["meta"]["n_seen"] == 128
    assert weights["self_md5"] != weights0["self_md5"]
    acc = [e for e in audit["events"] if e["kind"] == "accept"]
    assert len(acc) == 1 and acc[0]["n_new"] == 8
    g = acc[0]["gate"]
    assert g["fp_cand"] == g["fp_cur"] == 0
    assert g["auc_cand"] == pytest.approx(0.9393939393939394, abs=1e-9)
    assert g["auc_cur"] == pytest.approx(0.8249158249158249, abs=1e-9)


def test_update_polluted_flip_passes_gate(tmp_path, gatelearn_samples):
    """fn 方向语义污染 (20 份 bad 翻 0) -> 门放行. 门防误杀 (fp) 不防漏判 (fn).

    注意: 本测试不是实验 M4 FAIL 的复现 — 实验场景 P 的污染 (t3 前 5 bad 翻 0)
    在旧账产生 fp_cand=6, 门实际 REJECTED (见 test_selfevolve_replay 场景 P).
    本测试是 M4 局限的独立演示: 无 fp 增长的 fn 方向污染穿过门 (审计 HIGH 1 修正
    错误归因). 防污染真正防线 = 入口过滤 + 上游检查站 (阶段 1+), 门是第二道.
    """
    assert se.cmd_init(_init_args(tmp_path)) == 0
    batch = [_clone(s, f"clB{i:02d}", 0, gatelearn_samples)
             for i, s in enumerate([s for s in gatelearn_samples if s["label"] == 1][6:26])]
    assert _update(tmp_path, batch) == 0
    archive, weights, audit = se.load_state(tmp_path)
    assert archive["exam"]["n_entries"] == 140            # 污染批入档 (门放行, 如实)
    acc = [e for e in audit["events"] if e["kind"] == "accept"]
    assert len(acc) == 1
    assert acc[0]["n_new"] == 20
    assert acc[0]["gate"]["fp_cand"] == 0 and acc[0]["gate"]["fp_cur"] == 0


# ---------------------------------------------------------------------------
# 审计加固覆盖 (2026-08-20): 重复键/超大 int/类型/不变量/非有限 AUC/冒烟
# ---------------------------------------------------------------------------

def test_load_json_dup_keys(tmp_path):
    """重复 JSON 键 -> ValueError (first-wins 解析可绕过验签, 审计 A6)."""
    f = tmp_path / "dup.json"
    f.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="重复键"):
        se.load_json(f)


def test_load_json_broken(tmp_path):
    f = tmp_path / "broken.json"
    f.write_text("{not json")
    with pytest.raises(ValueError, match="JSON 损坏"):
        se.load_json(f)


def test_load_state_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="先 init"):
        se.load_state(tmp_path / "no_such_state")


def test_load_state_invariant_mismatch(tmp_path):
    """weights.meta 与 archive.exam 计数矛盾 -> 报错 (崩溃窗口检测, 审计 A2)."""
    assert se.cmd_init(_init_args(tmp_path)) == 0
    wf = tmp_path / "weights.json"
    d = json.loads(wf.read_text())
    d["meta"]["n_seen"] = d["meta"]["n_seen"] + 1
    sj.save_weights(wf, w=d["w"], med=d["med"], mu=d["mu"], sd=d["sd"], meta=d["meta"])
    with pytest.raises(ValueError, match="state 不一致"):
        se.load_state(tmp_path)


def test_batch_top_not_dict(tmp_path):
    assert se.cmd_init(_init_args(tmp_path)) == 0
    bf = tmp_path / "batch.json"
    bf.write_text("[1, 2]")
    with pytest.raises(ValueError, match="顶层必须是 dict"):
        se.cmd_update(argparse.Namespace(batch=str(bf), state=str(tmp_path)))


def test_samples_not_list(tmp_path):
    assert se.cmd_init(_init_args(tmp_path)) == 0
    bf = tmp_path / "batch.json"
    bf.write_text(json.dumps({"samples": "notalist"}))
    with pytest.raises(ValueError, match="samples 必须是列表"):
        se.cmd_update(argparse.Namespace(batch=str(bf), state=str(tmp_path)))


def test_entry_huge_int_rejected():
    """feat 超大 int (float 不可表示) -> 入口拒绝 (审计 A4)."""
    e = _entry()
    e["feat"]["mean"] = 10 ** 400
    acc, rej = se.filter_batch([e], [])
    assert acc == []
    assert rej[0]["reason"] == "feat.mean 数值不可表示 (超大整数)"


def test_entry_n_err_upper_bound():
    """n_err > 116 表数 -> 拒绝 (审计 A7)."""
    e = _entry()
    e["n_err"] = 117
    e["evidence"]["n_viol"] = 117
    acc, rej = se.filter_batch([e], [])
    assert rej[0]["reason"] == "n_err 超表数"


def test_entry_n_l4_err_upper_bound():
    e = _entry(label=0)
    e["n_err"] = 0
    e["n_l4_err"] = 10
    e["evidence"]["n_viol"] = 0
    e["feat"]["frac_l4_nan"] = 10 / 9
    acc, rej = se.filter_batch([e], [])
    assert rej[0]["reason"] == "n_l4_err 超 L4 输入数"


def test_gate_check_nonfinite_auc_passes():
    """旧账 AUC 非有限 (缺类别) -> 通过 (run_selfevolve.py:151-152)."""
    rng = np.random.default_rng(13)
    Xs = rng.normal(size=(10, 2))
    y = np.zeros(10, int)          # 全 good -> auc NaN
    w = sj.logreg_fit(Xs, y)
    ok, a_c, a_u, fp_c, fp_u = se.gate_check(Xs, y, np.ones(10, bool), w, w)
    assert ok and np.isnan(a_c) and np.isnan(a_u)


def test_status_smoke(tmp_path, capsys):
    assert se.cmd_init(_init_args(tmp_path)) == 0
    assert se.cmd_status(argparse.Namespace(state=str(tmp_path))) == 0
    assert "n_seen=120" in capsys.readouterr().out


def test_main_smoke_replay():
    """main() 分发冒烟: replay 子命令 (直读默认 data, 不落盘)."""
    assert se.main(["replay"]) == 0


def test_replay_after_update(tmp_path, gatelearn_samples):
    """审计 A3: update 改变 state 后 replay 仍可复现 (直读 data, 不依赖 state)."""
    assert se.cmd_init(_init_args(tmp_path)) == 0
    batch = [_clone(s, f"clC{i:02d}", 0, gatelearn_samples)
             for i, s in enumerate([s for s in gatelearn_samples if s["label"] == 0][:8])]
    assert _update(tmp_path, batch) == 0
    assert se.cmd_replay(argparse.Namespace(out="", data=str(DATA))) == 0
