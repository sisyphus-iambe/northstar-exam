"""SPSL 画像层 — 北极星 v3 记忆三层之"画像"生产版.

形态实证: experiments/profile_2026-08-20 (PASS 7/7, 预注册 README_profile.md):
  120 份 REJECT 冷启动 -> 16 画像条目; 确定性生命周期 写/读/忘/防污染.
阶段 2 (信任分) / 阶段 3 (全局经验) 为同模块扩展, 判据/公式见本文件常量与
函数 docstring (预注册口径: 信任分=衰减后违规率推导, 全局=跨主体聚合).

确定性: 无墙钟 (epoch=事件序号替代, 生产映射墙钟); 无随机量; 落盘带
payload_md5/self_md5 双验签; 原子写 (tmp+os.replace) + 并发锁 (fcntl.flock);
零 LLM 判卷; 不接 run.py 四层判定流 (独立能力模块, 零行为变化).

CLI:
  python3 -m spsl.profile init    [--data][--state][--force]   冷启动 (120 份 REJECT)
  python3 -m spsl.profile update --batch <batch.json> [--state] 新 REJECT 到达 -> 计数更新
  python3 -m spsl.profile read    [--epoch <n>][--top <k>][--state]  针对性约束文本
  python3 -m spsl.profile trust   [--epoch <n>][--state]        信任分 (阶段 2)
  python3 -m spsl.profile global  [--epoch <n>][--state]        全局经验 (阶段 3)
  python3 -m spsl.profile verify  [--state]                     验签+白名单+一致性
  python3 -m spsl.profile status  [--state]                     状态
"""
import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEFAULT_STATE = BASE / "state" / "profile"
DEFAULT_DATA = BASE / "experiments" / "gate_learn_2026-08-19" / "gatelearn_data.json"

# 判据常量 (README_profile.md §五/§六, 跑前定死)
WHITELIST_SUBJECTS = ("g3_p1", "g3", "vg", "prog", "cand", "cert", "scon", "p1ch")
TASK_FAMILY = "chi2"
HALF_EPOCHS = 10          # 衰减半衰期 (事件数)
F_HALF = 0.5              # 半衰因子
EVIDENCE_MAX = 20         # good 类证据链上限 (bad 类永不截断, 判据 J3)
N_COLDSTART = 120         # 冷启动样本数
TRUST_MIN_N = 3           # 信任分: 计数 < 3 的条目视为证据不足 (评分保守)

# 阶段 2 信任分公式 (预注册): trust = 1 - (衰减后 bad 权重 / 衰减后总权重)
# 证据不足 (衰减后总数 < TRUST_MIN_N) -> 中性分 0.5 (如实标注 not_enough).
# 阶段 3 全局: 跨主体聚合计数 + 反哺候选 (约束建议/考卷侧重候选, 不自动改考卷).


# ---------------------------------------------------------------------------
# 通用: 落盘/验签/锁 (与 selfevolve 同模式, 2026-08-20 审计加固)
# ---------------------------------------------------------------------------

def _no_dup_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"JSON 重复键: {k!r}")
        d[k] = v
    return d


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_no_dup_keys)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: JSON 损坏: {e}")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def digest(obj):
    import hashlib
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def verify_doc(doc):
    """复算验签: payload_md5=digest(无 md5 核心), self_md5=digest(核心+payload_md5)."""
    core = {k: v for k, v in doc.items() if k not in ("payload_md5", "self_md5")}
    p1 = digest(core)
    signed = dict(core)
    signed["payload_md5"] = p1
    s1 = digest(signed)
    if p1 != doc.get("payload_md5") or s1 != doc.get("self_md5"):
        raise ValueError("state 验签失败 (可能被篡改或损坏)")


@contextmanager
def state_lock(state_dir):
    """state 目录排他锁 (并发 update 串行化)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    f = open(state_dir / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _state_paths(state_dir):
    d = Path(state_dir)
    return d, d / "profile.json"


def load_state(state_dir):
    d, p = _state_paths(state_dir)
    if not p.exists():
        raise ValueError(f"state 未初始化: {p} (先 init)")
    doc = load_json(p)
    verify_doc(doc)
    return doc


# ---------------------------------------------------------------------------
# 违规类型分类 (README_profile.md §四, 从 constraint 失败模式学得)
# ---------------------------------------------------------------------------

def viol_class(label, n_err, n_l4_err):
    if label == 1:
        return "exec_fail" if n_err > 0 else "pure_dev"
    return "l4_honest" if n_l4_err > 0 else "clean"


# ---------------------------------------------------------------------------
# 写: 冷启动 (与原型 run_profile.py cold_start 同逻辑, entries 逐位一致)
# ---------------------------------------------------------------------------

def cold_start(samples):
    entries = {}
    for s in samples:
        subj = s.get("pool")
        cls = viol_class(s.get("label"), s.get("n_err", 0), s.get("n_l4_err", 0))
        key = (subj, TASK_FAMILY, cls)
        e = entries.setdefault(key, {
            "subject": subj, "task_family": TASK_FAMILY, "viol_class": cls,
            "count": 0, "epoch": 0, "evidence": [],
        })
        e["count"] += 1
        e["evidence"].append({
            "name": s.get("name"), "label": s.get("label"),
            "n_err": s.get("n_err", 0), "n_l4_err": s.get("n_l4_err", 0),
        })
    for e in entries.values():
        if e["viol_class"] in ("l4_honest", "clean") and len(e["evidence"]) > EVIDENCE_MAX:
            e["evidence"] = e["evidence"][-EVIDENCE_MAX:]
        e["evidence"].sort(key=lambda x: x["name"])
    return [entries[k] for k in sorted(entries)]


def cmd_init(args):
    with state_lock(args.state):
        d, p = _state_paths(args.state)
        if p.exists() and not args.force:
            print(f"state 已存在 {p} (用 --force 重建)", file=sys.stderr)
            return 1
        doc = load_json(args.data)
        samples = [s for s in doc["exam"]["samples"] if s.get("load_ok")]
        if len(samples) != N_COLDSTART:
            print(f"load_ok 样本数 {len(samples)} != {N_COLDSTART}, 拒绝", file=sys.stderr)
            return 1
        for s in samples:
            if s.get("pool") not in WHITELIST_SUBJECTS:
                print(f"非白名单 subject: {s.get('pool')}", file=sys.stderr)
                return 1
        entries = cold_start(samples)
        core = {
            "tool": "spsl", "layer": "profile",
            "src": str(args.data),
            "n_samples": len(samples),
            "epoch": 0,
            "entries": entries,
        }
        core["payload_md5"] = digest({k: v for k, v in core.items() if k != "payload_md5"})
        core["self_md5"] = digest(core)
        save_json(p, core)
        print(f"画像初始化: {len(entries)} 条目, payload_md5={core['payload_md5']}")
        return 0


# ---------------------------------------------------------------------------
# 忘: 衰减 (README §六, 读时计算不落盘)
# ---------------------------------------------------------------------------

def decay_w(count, epoch_written, epoch_now):
    age = max(0, epoch_now - epoch_written)
    return round(count * F_HALF ** (age / HALF_EPOCHS), 6)


# ---------------------------------------------------------------------------
# 读: 针对性约束文本 (README §六/J6, 确定性模板, 只动非核心层)
# ---------------------------------------------------------------------------

ADVICE = {
    "exec_fail": ("该主体在 chi2 任务族有 {c} 次执行失败型违规 (异常/超时/无返回)。"
                  "建议: 生成时确保实现稳健 — 显式捕获异常、有限循环、避免超时。"),
    "pure_dev": ("该主体在 chi2 任务族有 {c} 次纯数值偏离型违规 (算错不报错)。"
                 "建议: 生成后逐表自查对拍 — 确认每个统计量与参照一致。"),
    "l4_honest": ("该主体在 chi2 任务族有 {c} 次诚实失败 (L4 畸形输入明确说不)。"
                  "合格行为, 无需干预。"),
}


def read_profile(doc, epoch_now, top_n=3):
    lines = []
    bad = [e for e in doc["entries"] if e["viol_class"] in ADVICE]
    bad.sort(key=lambda e: (-decay_w(e["count"], e["epoch"], epoch_now),
                            e["subject"], e["viol_class"]))
    for e in bad[:top_n]:
        c = decay_w(e["count"], e["epoch"], epoch_now)
        lines.append(f"[{e['subject']}/{e['task_family']}/{e['viol_class']}] "
                     + ADVICE[e["viol_class"]].format(c=c))
    return lines


def cmd_read(args):
    with state_lock(args.state):
        doc = load_state(args.state)
        for ln in read_profile(doc, args.epoch, args.top):
            print(ln)
        return 0


# ---------------------------------------------------------------------------
# 阶段 2: 信任分 (预注册公式) + update 防污染
# ---------------------------------------------------------------------------

def trust_scores(doc, epoch_now):
    """每主体在任务族的信任分: 1 - 衰减后违规率. 证据不足 -> 0.5 (not_enough).

    预注册口径 (阶段 2): 只算 bad 类 (exec_fail/pure_dev) 与 good 类
    (l4_honest/clean) 的衰减后权重; l4_honest 是合格行为, 计 good.
    """
    per_subject = {}
    for e in doc["entries"]:
        w = decay_w(e["count"], e["epoch"], epoch_now)
        per_subject.setdefault(e["subject"], {"bad": 0.0, "good": 0.0})
        if e["viol_class"] in ("exec_fail", "pure_dev"):
            per_subject[e["subject"]]["bad"] += w
        else:
            per_subject[e["subject"]]["good"] += w
    out = {}
    for subj, v in sorted(per_subject.items()):
        total = v["bad"] + v["good"]
        if total < TRUST_MIN_N:
            out[subj] = {"trust": 0.5, "not_enough": True, "bad": v["bad"], "good": v["good"]}
        else:
            out[subj] = {"trust": round(1.0 - v["bad"] / total, 4),
                         "not_enough": False, "bad": v["bad"], "good": v["good"]}
    return out


def cmd_trust(args):
    with state_lock(args.state):
        doc = load_state(args.state)
        for subj, v in trust_scores(doc, args.epoch).items():
            tag = " (证据不足)" if v["not_enough"] else ""
            print(f"{subj}: trust={v['trust']}{tag} bad={v['bad']} good={v['good']}")
        return 0


def filter_update(entries, new_samples):
    """update 入口防污染: 白名单 subject + 计数/证据一致性 + 批内去重.

    new_samples: [{subject, label, n_err, n_l4_err, name}], name 不得与
    已有证据重复. 返回 (accepted, rejected). 与 selfevolve filter_batch 同模式.
    """
    accepted, rejected = [], []
    seen = {ev["name"] for e in entries for ev in e["evidence"]}
    for s in new_samples:
        if not isinstance(s, dict):
            rejected.append({"name": None, "reason": "样本非 dict"})
            continue
        name = s.get("name")
        subj = s.get("subject")
        if not isinstance(name, str) or not name:
            rejected.append({"name": name, "reason": "name 缺失"})
            continue
        if name in seen:
            rejected.append({"name": name, "reason": "duplicate_name"})
            continue
        if subj not in WHITELIST_SUBJECTS:
            rejected.append({"name": name, "reason": "非白名单 subject"})
            continue
        label = s.get("label")
        n_err = s.get("n_err", 0)
        n_l4 = s.get("n_l4_err", 0)
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            rejected.append({"name": name, "reason": "label 非 int{0,1}"})
            continue
        if label == 0 and n_err != 0:
            rejected.append({"name": name, "reason": "label=0 但 n_err>0"})
            continue
        accepted.append(s)
        seen.add(name)
    return accepted, rejected


def cmd_update(args):
    with state_lock(args.state):
        doc = load_state(args.state)
        batch = load_json(args.batch)
        if not isinstance(batch, dict) or not isinstance(batch.get("samples"), list):
            print("batch.samples 必须是列表", file=sys.stderr)
            return 1
        accepted, rejected = filter_update(doc["entries"], batch["samples"])
        if not accepted:
            print(f"update: 全部拒绝 ({len(rejected)} 条, 首因: "
                  f"{rejected[0]['reason'] if rejected else '空批'})", file=sys.stderr)
            return 1
        by_key = {}
        for s in accepted:
            cls = viol_class(s["label"], s["n_err"], s["n_l4_err"])
            key = (s["subject"], TASK_FAMILY, cls)
            by_key.setdefault(key, []).append(s)
        for key, ss in by_key.items():
            subj, fam, cls = key
            e = next((e for e in doc["entries"]
                      if (e["subject"], e["viol_class"]) == (subj, cls)), None)
            if e is None:
                e = {"subject": subj, "task_family": fam, "viol_class": cls,
                     "count": 0, "epoch": 0, "evidence": []}
                doc["entries"].append(e)
            e["count"] += len(ss)
            e["evidence"].extend({"name": s["name"], "label": s["label"],
                                  "n_err": s["n_err"], "n_l4_err": s["n_l4_err"]}
                                 for s in ss)
            e["evidence"].sort(key=lambda x: x["name"])
            if e["viol_class"] in ("l4_honest", "clean") and len(e["evidence"]) > EVIDENCE_MAX:
                e["evidence"] = e["evidence"][-EVIDENCE_MAX:]
        doc["epoch"] += 1
        doc["n_samples"] += len(accepted)   # 计数总和 == n_samples 不变式 (verify 校验)
        # 重签: 先剥离旧双签 (旧 self_md5 若残留在 doc 里会污染 digest)
        core = {k: v for k, v in doc.items() if k not in ("payload_md5", "self_md5")}
        payload_md5 = digest(core)
        signed = dict(core)
        signed["payload_md5"] = payload_md5
        self_md5 = digest(signed)
        doc = dict(signed)
        doc["self_md5"] = self_md5
        _, p = _state_paths(args.state)
        save_json(p, doc)
        print(f"update: 接受 {len(accepted)} / 拒绝 {len(rejected)}, "
              f"epoch -> {doc['epoch']}")
        for r in rejected:
            print(f"  REJECT {r.get('name')}: {r['reason']}")
        return 0


# ---------------------------------------------------------------------------
# 阶段 3: 全局经验 (跨主体聚合, 反哺候选 — 不自动改考卷)
# ---------------------------------------------------------------------------

def global_insights(doc, epoch_now):
    """跨主体聚合: {viol_class: 衰减后总权重} + 反哺候选.

    反哺候选 (确定性输出, 供人工/后续阶段决策):
      - 约束建议: 全局 Top 违规类型的建议文本 (同 ADVICE 模板)
      - 考卷侧重候选: 最常见 bad 类违规类型 TopN (结构化), 不自动改考卷
        (改考卷需新验证, 超出本模块作用域)
    """
    agg = {}
    for e in doc["entries"]:
        w = decay_w(e["count"], e["epoch"], epoch_now)
        agg[e["viol_class"]] = round(agg.get(e["viol_class"], 0.0) + w, 6)
    bad_top = sorted((k for k in ("exec_fail", "pure_dev")), key=lambda k: -agg.get(k, 0.0))
    suggestions = []
    for cls in bad_top:
        if agg.get(cls, 0.0) > 0:
            suggestions.append(f"[global/{TASK_FAMILY}/{cls}] 全局衰减后权重 {agg[cls]} — "
                               + ADVICE[cls].format(c=agg[cls]))
    return {"agg": agg, "exam_focus_candidates": bad_top, "suggestions": suggestions}


def cmd_global(args):
    with state_lock(args.state):
        doc = load_state(args.state)
        g = global_insights(doc, args.epoch)
        print(f"全局聚合: {json.dumps(g['agg'], ensure_ascii=False)}")
        print(f"考卷侧重候选: {g['exam_focus_candidates']} "
              f"(不自动改考卷, 需新验证)")
        for s in g["suggestions"]:
            print(s)
        return 0


# ---------------------------------------------------------------------------
# verify / status
# ---------------------------------------------------------------------------

def verify_profile(doc):
    """验签 + 白名单 + 一致性 (防污染, 判据 J4 生产版)."""
    problems = []
    if doc.get("tool") != "spsl" or doc.get("layer") != "profile":
        problems.append("tool/layer 不符")
    if not isinstance(doc.get("epoch"), int) or doc["epoch"] < 0:
        problems.append("epoch 非法")
    seen = set()
    n_bad = n_good = 0
    for e in doc["entries"]:
        if e["subject"] not in WHITELIST_SUBJECTS:
            problems.append(f"非白名单 subject: {e['subject']}")
        if e["task_family"] != TASK_FAMILY:
            problems.append(f"任务族异常: {e['task_family']}")
        if e["viol_class"] not in ("exec_fail", "pure_dev", "l4_honest", "clean"):
            problems.append(f"违规类型非法: {e['viol_class']}")
        if not isinstance(e["count"], int) or e["count"] <= 0:
            problems.append(f"计数非法: {e['subject']}/{e['viol_class']}")
        n_ev = len(e["evidence"])
        if e["viol_class"] in ("exec_fail", "pure_dev"):
            if n_ev != e["count"]:
                problems.append(f"bad 类证据必须完整: {e['subject']}/{e['viol_class']}")
        elif n_ev != min(e["count"], EVIDENCE_MAX):
            problems.append(f"good 类证据截断不符: {e['subject']}/{e['viol_class']}")
        for ev in e["evidence"]:
            k = (ev["name"], e["viol_class"])
            if k in seen:
                problems.append(f"证据重复: {ev['name']}")
            seen.add(k)
        if e["viol_class"] in ("exec_fail", "pure_dev"):
            n_bad += e["count"]
        else:
            n_good += e["count"]
    if n_bad + n_good != doc.get("n_samples"):
        problems.append(f"计数总和 {n_bad + n_good} != n_samples {doc.get('n_samples')}")
    return (not problems), problems


def cmd_verify(args):
    with state_lock(args.state):
        doc = load_state(args.state)
        ok, problems = verify_profile(doc)
        if ok:
            print("verify: OK (验签+白名单+一致性全过)")
            return 0
        for p in problems:
            print(f"FAIL: {p}")
        return 1


def cmd_status(args):
    with state_lock(args.state):
        doc = load_state(args.state)
        n_bad = sum(1 for e in doc["entries"]
                    if e["viol_class"] in ("exec_fail", "pure_dev"))
        print(f"画像状态: {len(doc['entries'])} 条目 ({n_bad} 违规类), "
              f"样本 {doc['n_samples']}, epoch {doc['epoch']}, "
              f"payload_md5={doc['payload_md5'][:12]}...")
        return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--force", action="store_true")
    for name in ("update",):
        u = sub.add_parser(name)
        u.add_argument("--batch", required=True)
        u.add_argument("--state", default=str(DEFAULT_STATE))
    for name in ("read", "trust", "global"):
        r = sub.add_parser(name)
        r.add_argument("--epoch", type=int, default=0)
        r.add_argument("--state", default=str(DEFAULT_STATE))
        if name == "read":
            r.add_argument("--top", type=int, default=3)
    for name in ("verify", "status"):
        s = sub.add_parser(name)
        s.add_argument("--state", default=str(DEFAULT_STATE))
    args = ap.parse_args()
    return {"init": cmd_init, "update": cmd_update, "read": cmd_read,
            "trust": cmd_trust, "global": cmd_global,
            "verify": cmd_verify, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
