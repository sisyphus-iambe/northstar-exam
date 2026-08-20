"""画像层生产版 (spsl/profile.py) 测试 — 冷启动一致性 / update 重签 / 防污染 / 信任分 / 全局.

形态出处: experiments/profile_2026-08-20 (原型 PASS 7/7, 预注册 README_profile.md),
生产版为同模块扩展: 信任分 (阶段 2) 与全局经验 (阶段 3) 公式见 profile.py 常量.
"""
import json
import shutil
from pathlib import Path

import pytest
from spsl import profile as pp

_NSV2 = Path(__file__).resolve().parents[1]
SRC_DATA = _NSV2 / "experiments" / "gate_learn_2026-08-19" / "gatelearn_data.json"
PROTO = _NSV2 / "experiments" / "profile_2026-08-20" / "profile_data.json"


@pytest.fixture()
def state_dir(tmp_path):
    """临时 state 目录 (每测独立, 防串扰)."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture()
def doc(state_dir):
    """init 后的画像 doc (真实 120 份冷启动)."""
    args = _ns_args("init", state=state_dir)
    assert pp.cmd_init(args) == 0
    return pp.load_state(state_dir)


def _ns_args(cmd, **kw):
    """构造命名空间参数 (绕过 argparse 的快速路径)."""
    import argparse
    ns = argparse.Namespace(state=kw.get("state"), force=kw.get("force", False))
    if cmd == "init":
        ns.data = kw.get("data", str(SRC_DATA))
        ns.state = kw["state"]
        ns.force = kw.get("force", False)
    if cmd == "update":
        ns.batch = kw["batch"]
        ns.state = kw["state"]
    if cmd == "read":
        ns.epoch = kw.get("epoch", 0)
        ns.top = kw.get("top", 3)
        ns.state = kw["state"]
    return ns


# ---------------------------------------------------------------------------
# 冷启动: 与原型 profile_data.json 逐位一致 (移植正确性)
# ---------------------------------------------------------------------------

def test_cold_start_matches_prototype(state_dir):
    """生产 init 的 entries 与原型逐位一致 (含证据链顺序/计数)."""
    args = _ns_args("init", state=state_dir)
    assert pp.cmd_init(args) == 0
    got = pp.load_state(state_dir)
    proto = pp.load_json(PROTO)
    assert got["entries"] == proto["entries"]
    assert got["n_samples"] == 120
    assert got["epoch"] == 0
    assert len(got["entries"]) == 16


def test_init_double_run_deterministic(tmp_path):
    """双跑 init payload_md5 逐字节一致 (确定性)."""
    md5s = []
    for i in range(2):
        d = tmp_path / f"s{i}"
        d.mkdir()
        args = _ns_args("init", state=d)
        assert pp.cmd_init(args) == 0
        md5s.append(pp.load_state(d)["payload_md5"])
    assert md5s[0] == md5s[1]


def test_init_refuses_existing_state(state_dir):
    """已存在 state 且无 --force -> 拒绝."""
    assert pp.cmd_init(_ns_args("init", state=state_dir)) == 0
    assert pp.cmd_init(_ns_args("init", state=state_dir)) == 1
    # --force 重建 OK
    assert pp.cmd_init(_ns_args("init", state=state_dir, force=True)) == 0


# ---------------------------------------------------------------------------
# update: 重签验签 / n_samples 不变式 / 计数更新
# ---------------------------------------------------------------------------

def test_update_applies_and_resigns(state_dir, doc):
    """update 后 load_state 验签过 + verify OK + 计数/n_samples 同步."""
    batch = _batch(state_dir, [{"subject": "vg", "label": 1, "n_err": 2,
                                "n_l4_err": 0, "name": "vg:test_bad_1"}])
    assert pp.cmd_update(_ns_args("update", batch=str(batch), state=state_dir)) == 0
    d = pp.load_state(state_dir)          # 验签不抛 = 重签正确
    ok, problems = pp.verify_profile(d)
    assert ok, problems
    assert d["n_samples"] == 121
    assert d["epoch"] == 1
    vg = next(e for e in d["entries"] if e["subject"] == "vg"
              and e["viol_class"] == "exec_fail")
    assert vg["count"] == 16               # 原型 15 (见 profile_data.json) + 1
    assert any(ev["name"] == "vg:test_bad_1" for ev in vg["evidence"])


def test_update_adds_new_entry(state_dir, doc):
    """新 (subject, viol_class) 组合 -> 新条目, 计数总和仍 == n_samples."""
    batch = _batch(state_dir, [{"subject": "cand", "label": 0, "n_err": 0,
                                "n_l4_err": 1, "name": "cand:test_honest_1"}])
    assert pp.cmd_update(_ns_args("update", batch=str(batch), state=state_dir)) == 0
    d = pp.load_state(state_dir)
    assert d["n_samples"] == 121
    ok, problems = pp.verify_profile(d)
    assert ok, problems
    assert any(e["subject"] == "cand" and e["viol_class"] == "l4_honest"
               and e["count"] == 1 for e in d["entries"])


def test_update_all_rejected_fails(state_dir, doc):
    """全拒批 -> update 返回 1, state 不变."""
    batch = _batch(state_dir, [{"subject": "evil", "label": 1, "n_err": 1,
                                "n_l4_err": 0, "name": "evil:x"}])
    assert pp.cmd_update(_ns_args("update", batch=str(batch), state=state_dir)) == 1
    d = pp.load_state(state_dir)
    assert d["n_samples"] == 120          # 不变
    assert d["epoch"] == 0


# ---------------------------------------------------------------------------
# filter_update: 入口防污染拒绝路径 (J4 生产版)
# ---------------------------------------------------------------------------

def test_filter_update_reject_reasons():
    entries = [{"subject": "vg", "viol_class": "exec_fail", "count": 1,
                "evidence": [{"name": "vg:old"}]}]
    cases = [
        ({"subject": "vg", "label": 1, "n_err": 1, "n_l4_err": 0, "name": "vg:old"},
         "duplicate_name"),
        ({"subject": "evil", "label": 1, "n_err": 1, "n_l4_err": 0, "name": "vg:new1"},
         "非白名单 subject"),
        ({"subject": "vg", "label": True, "n_err": 0, "n_l4_err": 0, "name": "vg:new2"},
         "label 非 int{0,1}"),          # bool 是 int 子类, 必须显式拒
        ({"subject": "vg", "label": 0, "n_err": 3, "n_l4_err": 0, "name": "vg:new3"},
         "label=0 但 n_err>0"),
        ({"subject": "vg", "label": 1, "n_err": 1, "n_l4_err": 0}, "name 缺失"),
    ]
    for sample, reason in cases:
        acc, rej = pp.filter_update(entries, [sample])
        assert acc == []
        assert rej[0]["reason"] == reason


def test_filter_update_accepts_batch_internal_dup():
    """批内重复 name -> 后一条拒 (accepted 已计入去重)."""
    entries = []
    s = {"subject": "vg", "label": 1, "n_err": 1, "n_l4_err": 0, "name": "vg:dup"}
    acc, rej = pp.filter_update(entries, [s, s])
    assert len(acc) == 1
    assert rej[0]["reason"] == "duplicate_name"


# ---------------------------------------------------------------------------
# 阶段 2: 信任分公式 + TRUST_MIN_N 边界
# ---------------------------------------------------------------------------

def _mini_doc(*entries):
    """构造最小画像 doc (epoch=0, n_samples=计数和)."""
    return {"tool": "spsl", "layer": "profile", "epoch": 0,
            "n_samples": sum(e["count"] for e in entries), "entries": list(entries)}


def _entry(subject, cls, count, epoch=0):
    return {"subject": subject, "task_family": "chi2", "viol_class": cls,
            "count": count, "epoch": epoch,
            "evidence": [{"name": f"{subject}:e{i}", "label": 1 if cls in
                          ("exec_fail", "pure_dev") else 0, "n_err": 0, "n_l4_err": 0}
                         for i in range(count)]}


def test_trust_formula():
    """trust = 1 - 衰减后 bad/总; 证据不足 -> 0.5 + not_enough."""
    doc = _mini_doc(
        _entry("a", "pure_dev", 3),      # 3 bad
        _entry("a", "clean", 3),         # 3 good  -> trust = 0.5
        _entry("b", "exec_fail", 3),     # 全 bad  -> trust = 0.0
        _entry("c", "clean", 2),         # 总 2 < TRUST_MIN_N -> not_enough
    )
    t = pp.trust_scores(doc, 0)
    assert t["a"]["trust"] == 0.5 and t["a"]["not_enough"] is False
    assert t["b"]["trust"] == 0.0
    assert t["c"]["trust"] == 0.5 and t["c"]["not_enough"] is True


def test_trust_min_n_boundary():
    """total == TRUST_MIN_N 恰好不算不足 (3 -> 正常算, 2 -> 不足)."""
    doc = _mini_doc(
        _entry("a", "exec_fail", 1), _entry("a", "clean", 2),   # total 3
        _entry("b", "exec_fail", 2),                            # total 2
    )
    t = pp.trust_scores(doc, 0)
    assert t["a"]["not_enough"] is False
    assert t["b"]["not_enough"] is True


def test_trust_decay_applies():
    """衰减影响 bad/good 权重: 旧证据半衰 -> trust 回升."""
    doc = _mini_doc(
        _entry("a", "exec_fail", 6, epoch=0),   # 旧违规
        _entry("a", "clean", 6, epoch=0),
    )
    t0 = pp.trust_scores(doc, 0)               # 6/12 -> 0.5
    t100 = pp.trust_scores(doc, 100)           # 两边同衰减 -> 仍 0.5
    assert t0["a"]["trust"] == 0.5
    assert t100["a"]["trust"] == 0.5
    assert t100["a"]["bad"] == pytest.approx(t0["a"]["bad"] * 0.5 ** 10, abs=1e-6)


# ---------------------------------------------------------------------------
# 阶段 3: 全局经验
# ---------------------------------------------------------------------------

def test_global_insights_deterministic(doc):
    g1 = pp.global_insights(doc, 7)
    g2 = pp.global_insights(doc, 7)
    assert g1 == g2
    # 冷启动 120 份无 clean 样本 (原型 profile_data.json 实证 3 类)
    assert set(g1["agg"].keys()) == {"exec_fail", "pure_dev", "l4_honest"}
    # 反哺候选: 只含 bad 类, 按全局权重降序
    assert g1["exam_focus_candidates"] == sorted(
        ("exec_fail", "pure_dev"), key=lambda k: -g1["agg"].get(k, 0.0))
    # 建议文本含格式化后的权重数字 (验证 {c} 已替换, 无字面量占位符)
    for s in g1["suggestions"]:
        assert "{c}" not in s


def test_global_insights_epoch_shift(doc):
    """epoch 推进 -> 全局权重单调衰减 (同一 doc, 更晚 epoch 权重更小)."""
    g0 = pp.global_insights(doc, 0)
    g50 = pp.global_insights(doc, 50)
    for cls in ("exec_fail", "pure_dev"):
        assert g50["agg"][cls] < g0["agg"][cls]


# ---------------------------------------------------------------------------
# verify: 篡改检测 (J4 生产版)
# ---------------------------------------------------------------------------

def test_verify_tamper_detected(doc):
    """篡改计数 (重签骗过验签) -> verify_profile 报证据矛盾."""
    import copy
    d = copy.deepcopy(doc)
    d["entries"][0]["count"] += 99
    core = {k: v for k, v in d.items() if k not in ("payload_md5", "self_md5")}
    core["payload_md5"] = pp.digest({k: v for k, v in core.items()})
    d2 = dict(core)
    d2["self_md5"] = pp.digest(d2)
    ok, problems = pp.verify_profile(d2)
    assert not ok
    assert any("必须完整" in p or "截断不符" in p or "计数总和" in p for p in problems)


def test_verify_rejects_contradictory_evidence(doc):
    """证据与计数矛盾 (bad 类砍证据) -> 拒绝."""
    import copy
    d = copy.deepcopy(doc)
    bad = next(e for e in d["entries"] if e["viol_class"] == "pure_dev")
    bad["evidence"] = bad["evidence"][:1]
    core = {k: v for k, v in d.items() if k not in ("payload_md5", "self_md5")}
    core["payload_md5"] = pp.digest({k: v for k, v in core.items()})
    d2 = dict(core)
    d2["self_md5"] = pp.digest(d2)
    ok, problems = pp.verify_profile(d2)
    assert not ok
    assert any("必须完整" in p for p in problems)


def test_verify_ok_on_pristine(doc):
    ok, problems = pp.verify_profile(doc)
    assert ok, problems


def test_load_state_rejects_tampered_signature(state_dir):
    """直接篡改磁盘文件 (不重签) -> load_state 验签失败."""
    assert pp.cmd_init(_ns_args("init", state=state_dir)) == 0
    p = state_dir / "profile.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["n_samples"] = 999
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="验签失败"):
        pp.load_state(state_dir)


# ---------------------------------------------------------------------------
# 读端: 红线 (J6 生产版)
# ---------------------------------------------------------------------------

def test_read_advice_respects_redline(doc):
    banned = ("判定", "信任", "ACCEPT", "REJECT", "考卷", "分数")
    for ln in pp.read_profile(doc, 0):
        for b in banned:
            assert b not in ln, f"红线违例: {ln!r} 含 {b!r}"


def test_read_advice_formatted(doc):
    """advice 文本的计数是数字 (非 {c} 占位)."""
    for ln in pp.read_profile(doc, 0):
        assert "{c}" not in ln


# ---------------------------------------------------------------------------
# 并发锁冒烟 (state_lock)
# ---------------------------------------------------------------------------

def test_state_lock_smoke(state_dir):
    with pp.state_lock(state_dir):
        assert (state_dir / ".lock").exists()
    # 锁释放后可再次进入
    with pp.state_lock(state_dir):
        pass


def _batch(state_dir, samples):
    """写 batch JSON 到 state 目录 (更新测试专用)."""
    p = state_dir / "batch.json"
    p.write_text(json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
    return p
