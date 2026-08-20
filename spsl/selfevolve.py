"""SPSL 自进化管线 — 判定器持续累积训练 + 外部裁决门 + 入口防污染 (北极星 v3).

形态实证: experiments/selfevolve_2026-08-20 (PARTIAL, 主判据 M1 PASS):
  带门 G=0.9727 vs 冻结 S=0.7893, 零误杀零遗忘 (门拒有害批 t=5 净赚 +0.091).
三层更新形态 (北极星v3.md 自进化节, 正式设计):
  ① 增量学习: 持续累积训练 = 每次从零重训全部已见 (run_selfevolve.py:134-136 fit_seen)
  ② 入口防污染: 污染必须进不了训练集 — M4 教训: 门控污染后永久锁死
     (selfevolve_verdict.json key_findings 3), 故 filter_batch 为第一道防线
  ③ 门控防退化: 外部裁决门, 旧账上 AUC>=旧-0.05 且 fp 不增
     (run_selfevolve.py:141-154 gate_check 照搬)

提交语义 (M4 教训的工程翻译): 门拒绝的批不进档案 (审计留痕), 门是真正提交点.

并发/原子性 (审计加固 2026-08-20): update/init/status 全程持 state 目录锁
(fcntl.flock); 所有落盘为 tmp + os.replace 原子写; load_state 校验跨文件不变量
(weights.meta 计数 == archive.exam 计数), 崩溃窗口/并发交错 -> 明确报错而非静默.

CLI:
  python3 -m spsl.selfevolve init    [--data][--result][--state][--force]  冷启动
  python3 -m spsl.selfevolve update --batch <batch.json> [--state]        新批提交
  python3 -m spsl.selfevolve status [--state]                              状态
  python3 -m spsl.selfevolve replay [--out <json>] [--data]                复现实验

状态目录 state/selfevolve/: archive.json (累积样本档案) / weights.json
(shape_judger 权重) / audit.json (事件日志, 无时间戳保 md5 确定性).

确定性: 无随机量; 全部落盘带 payload_md5/self_md5 (复算验签); 双跑逐字节一致.
零 LLM 判卷; 不接 spsl/run.py 四层判定流 (独立能力模块, 零行为变化).
"""
import argparse
import fcntl
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from spsl import shape_judger as sj

BASE = Path(__file__).resolve().parents[1]
DEFAULT_STATE = BASE / "state" / "selfevolve"
DEFAULT_DATA = (BASE / "experiments" / "gate_learn_2026-08-19" / "gatelearn_data.json")
DEFAULT_RESULT = (BASE / "experiments" / "gate_learn_2026-08-19" / "gatelearn_result.json")

GATE_AUC_TOL = 0.05      # 门: 旧批 AUC 允许退步上限 (run_selfevolve.py:25)
N_POLLUTE = 5            # replay 场景 P: 污染样本数 (run_selfevolve.py:32)
FROZEN_MATCH_TOL = 1e-6  # init: 冷启动权重与冻结权重最大允许差异

# 判据常量 (run_selfevolve.py:26-31 同名同值命名化, 转写时曾内联, 数值逐字验证)
M1_MARGIN = 0.05    # M1: 自进化 >= 静态 + 0.05
M2_MARGIN = 0.05    # M2: 遗忘允许上限
M3_FP_SLACK = 3     # M3: 累计 fp 宽容
M3_FP_RATE = 0.10   # M3: 每批到达误杀率上限
M4_MARGIN = 0.05    # M4: 防污染门 AUC 允许下限
M4_FP_SLACK = 1     # M4: 防污染 fp 宽容

# 入口防污染 (照搬 run_gatelearn.py:171-191 八池 + 对拍证据链约定)
WHITELIST_POOLS = {"g3_p1", "g3", "vg", "prog", "cand", "cert", "scon", "p1ch"}
EVIDENCE_REF = "scipy chi2_contingency"
EVIDENCE_TOL = 1e-6
_PVEC_MD5_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 通用: 落盘/验签 (md5 规范同实验: payload_md5=digest(无 md5), self_md5=digest(含 payload_md5))
# ---------------------------------------------------------------------------

def _no_dup_keys(pairs):
    """object_pairs_hook: 重复 JSON 键 -> 报错.

    Python json 默认 first-wins, 攻击者可构造 {"label": 1, "label": 0} 之类
    绕过验签 (审计发现 2026-08-20).
    """
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
    """原子写 (tmp + os.replace): 崩溃窗口不出现半截文件."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def sign_payload(payload):
    """签名 (幂等: 先剥已有 md5, 已签过的 dict 再签同值)."""
    payload = {k: v for k, v in payload.items() if k not in ("payload_md5", "self_md5")}
    payload["payload_md5"] = sj.digest(payload)
    payload["self_md5"] = sj.digest(payload)
    return payload


def verify_payload(path, layer):
    """读 JSON + 双 md5 复算验签 (失败即报), 返回 dict."""
    doc = load_json(path)
    if layer and doc.get("layer") != layer:
        raise ValueError(f"{path}: layer={doc.get('layer')!r} != {layer!r}")
    core = {k: v for k, v in doc.items() if k not in ("payload_md5", "self_md5")}
    if sj.digest(core) != doc.get("payload_md5"):
        raise ValueError(f"{path}: payload_md5 复算不一致 (文件被改动?)")
    signed = dict(core)
    signed["payload_md5"] = doc["payload_md5"]
    if sj.digest(signed) != doc.get("self_md5"):
        raise ValueError(f"{path}: self_md5 复算不一致 (文件被改动?)")
    return doc


@contextmanager
def state_lock(state_dir):
    """同 state 目录互斥 (fcntl.flock 独占): update/init/status 并发 -> 串行,
    防静默丢更新 (审计 HIGH: 并发 update 5/5 复现丢更新)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_f = open(state_dir / ".lock", "a+")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        lock_f.close()


def load_state(state_dir):
    state_dir = Path(state_dir)
    if not (state_dir / "archive.json").exists():
        raise ValueError(f"state 不存在 {state_dir} (先 init)")
    archive = verify_payload(state_dir / "archive.json", "selfevolve-archive")
    weights = sj.load_weights(state_dir / "weights.json")
    audit = verify_payload(state_dir / "audit.json", "selfevolve-audit")
    # 跨文件不变量 (审计 HIGH: 崩溃窗口/并发交错后三文件各自验签全过但互相矛盾)
    m = weights["meta"]
    if m["n_seen"] != archive["exam"]["n_entries"]:
        raise ValueError(
            f"state 不一致: weights.meta.n_seen={m['n_seen']} != "
            f"archive.exam.n_entries={archive['exam']['n_entries']}")
    if m["n_bad"] != archive["exam"]["n_bad"] or m["n_good"] != archive["exam"]["n_good"]:
        raise ValueError(
            f"state 不一致: weights.meta (bad={m['n_bad']}, good={m['n_good']}) != "
            f"archive.exam (bad={archive['exam']['n_bad']}, good={archive['exam']['n_good']})")
    return archive, weights, audit


def save_state(state_dir, archive, audit):
    state_dir = Path(state_dir)
    save_json(state_dir / "archive.json", sign_payload(archive))
    save_json(state_dir / "audit.json", sign_payload(audit))


def audit_append(audit, event):
    """追加审计事件 (seq 递增, 无时间戳保 md5 确定性)."""
    seq = max((e["seq"] for e in audit["events"]), default=0) + 1
    event = dict(event, seq=seq)
    audit["events"].append(event)
    return event


# ---------------------------------------------------------------------------
# 入口防污染 (北极星v3.md ②; M4 教训: 污染必须进不了训练集)
# ---------------------------------------------------------------------------

def _entry_check(s, seen):
    """单条入口检查: 全过返回 None, 否则返回拒绝 reason."""
    name = s.get("name")
    if not isinstance(name, str) or ":" not in name:
        return "name 结构不符"
    pool, stem = name.split(":", 1)
    if pool not in WHITELIST_POOLS:
        return "白名单外池"
    if s.get("pool") != pool:
        return "name 与 pool 不一致"
    if s.get("stem") != stem:
        return "name 与 stem 不一致"
    if name in seen:
        return "duplicate_name"
    if s.get("load_ok") is not True:
        return "load_ok 非真"
    if s.get("label_ok") is not True:
        return "label_ok 非真"
    label = s.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
        return "label 非 int{0,1}"
    for k in ("n_err", "n_l4_err"):
        v = s.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return f"{k} 非法"
    if s["n_err"] > sj.N_TABLES:
        return "n_err 超表数"          # 116 张表, 每表一次判定, 最多 116 违规
    if s["n_l4_err"] > sj.N_L4:
        return "n_l4_err 超 L4 输入数"  # 9 类畸形输入
    if label == 0 and s["n_err"] != 0:
        return "label=0 但 n_err>0"
    feat = s.get("feat")
    if not isinstance(feat, dict):
        return "feat 缺失"
    for k in sj.FEAT_FIELDS:
        v = feat.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"feat.{k} 非数值"
        try:
            float(v)
        except OverflowError:
            return f"feat.{k} 数值不可表示 (超大整数)"
    if not (abs(feat["frac_l4_nan"] - s["n_l4_err"] / sj.N_L4) < 1e-12):
        return "frac_l4_nan 与 n_l4_err 不一致"
    ev = s.get("evidence")
    if not isinstance(ev, dict):
        return "evidence 缺失"
    if ev.get("ref") != EVIDENCE_REF:
        return "evidence.ref 非对拍来源"
    tol = ev.get("tol")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol != EVIDENCE_TOL:
        return "evidence.tol 非 1e-6"
    n_viol = ev.get("n_viol")
    if isinstance(n_viol, bool) or not isinstance(n_viol, int) or n_viol < 0:
        return "evidence.n_viol 非法"
    if (label == 1) != (n_viol > 0):
        return "n_viol 与 label 矛盾"
    pm = ev.get("pvec_md5")
    if not isinstance(pm, str) or not _PVEC_MD5_RE.fullmatch(pm):
        return "pvec_md5 非 64 位 hex"
    return None


def filter_batch(samples, known_names):
    """入口防污染: 每条全过才收. 返回 (accepted, rejected:[{name, reason}])."""
    if not isinstance(samples, list):
        raise ValueError("batch.samples 必须是列表")
    accepted, rejected = [], []
    seen = set(known_names)
    for s in samples:
        if not isinstance(s, dict):
            rejected.append({"name": None, "reason": "样本非 dict"})
            continue
        name = s.get("name")
        reason = _entry_check(s, seen)
        if reason is None:
            accepted.append(s)
            seen.add(name)
        else:
            rejected.append({"name": name, "reason": reason})
    return accepted, rejected


# ---------------------------------------------------------------------------
# 增量重训 + 外部裁决门 (run_selfevolve.py:136-154 照搬)
# ---------------------------------------------------------------------------

def feat_matrix(rows):
    return np.array([[s["feat"][f] for f in sj.FEAT_FIELDS] for s in rows], dtype=float)


def train_candidate(archive_samples, new_samples, weights):
    """持续累积训练 (fit_seen 口径): 全量重训已见. 统计量冻结, 永不重算."""
    rows = archive_samples + new_samples
    X = feat_matrix(rows)
    y = np.array([s["label"] for s in rows], dtype=int)
    Xf = sj.median_fill(X, np.array(weights["med"]))
    Xs = sj.standardize(Xf, np.array(weights["mu"]), np.array(weights["sd"]))
    return sj.logreg_fit(Xs, y)


def gate_check(Xs, y, old_mask, cand_w, cur_w):
    """外部裁决门 (run_selfevolve.py:141-154 照搬): 旧批上
    AUC_cand >= AUC_cur - GATE_AUC_TOL 且 fp_cand <= fp_cur. 空 mask/缺类 -> 通过."""
    if old_mask.sum() == 0:
        return True, float("nan"), float("nan"), 0, 0
    y_old = y[old_mask]
    p_c = sj.predict_proba(Xs[old_mask], cand_w)
    p_u = sj.predict_proba(Xs[old_mask], cur_w)
    a_c, a_u = sj.auc(y_old, p_c), sj.auc(y_old, p_u)
    fp_c, fp_u = sj.conf(y_old, p_c)["fp"], sj.conf(y_old, p_u)["fp"]
    if not (np.isfinite(a_c) and np.isfinite(a_u)):
        return True, a_c, a_u, fp_c, fp_u
    ok = (a_c >= a_u - GATE_AUC_TOL) and (fp_c <= fp_u)
    return ok, float(a_c), float(a_u), fp_c, fp_u


# ---------------------------------------------------------------------------
# CLI 子命令
# ---------------------------------------------------------------------------

def cmd_init(args):
    """冷启动: 验签源文件 -> 120 份入档案 -> 训练池统计量+权重 (与冻结权重对拍)."""
    state_dir = Path(args.state)
    with state_lock(state_dir):
        if (state_dir / "weights.json").exists() and not args.force:
            print(f"state 已存在 {state_dir}/weights.json (用 --force 重建)", file=sys.stderr)
            return 1
        data = verify_payload(Path(args.data), layer="gate-learn")
        result = verify_payload(Path(args.result), layer="gate-learn")
        samples = [s for s in data["exam"]["samples"] if s.get("load_ok")]
        archive_samples = []
        for i, s in enumerate(samples, 1):
            a = dict(s)
            a["seq"] = i
            a["evidence"] = {"origin": "coldstart", "ref": EVIDENCE_REF,
                             "tol": EVIDENCE_TOL, "n_viol": None, "pvec_md5": None}
            archive_samples.append(a)
        w, med, mu, sd = sj.cold_start_fit(archive_samples)
        frozen = np.asarray(result["result"]["weight"], dtype=float)
        maxdiff = float(np.abs(np.asarray(w, dtype=float) - frozen).max())
        if maxdiff > FROZEN_MATCH_TOL:
            raise SystemExit(f"冷启动权重与冻结权重差异过大 maxdiff={maxdiff} (> {FROZEN_MATCH_TOL}, 实现偏离实验?)")
        n_bad = sum(1 for s in archive_samples if s["label"] == 1)
        n_good = len(archive_samples) - n_bad
        meta = {"n_seen": len(archive_samples), "trained_on": "g3_p1",
                "n_bad": n_bad, "n_good": n_good,
                "std_from": "g3_p1 冷启动训练池(14份, 冻结不重算)"}
        weights = sj.save_weights(state_dir / "weights.json", w=w, med=med, mu=mu, sd=sd, meta=meta)
        archive = {
            "tool": "spsl", "layer": "selfevolve-archive",
            "exam": {"feat_fields": sj.FEAT_FIELDS, "tol_label": sj.TOL,
                     "n_entries": len(archive_samples), "n_bad": n_bad, "n_good": n_good,
                     "preprocess": "median_fill+std, 统计量冻结于冷启动训练池"},
            "samples": archive_samples,
        }
        audit = {"tool": "spsl", "layer": "selfevolve-audit", "events": [
            {"seq": 1, "kind": "init", "n_samples": len(archive_samples),
             "frozen_weight_maxdiff": maxdiff, "weights_md5": weights["self_md5"]}]}
        save_state(state_dir, archive, audit)
        print(f"冷启动完成: {len(archive_samples)} 份 (bad {n_bad}/good {n_good})")
        print(f"  冷启动权重 vs 冻结权重 maxdiff={maxdiff:.3e}")
        print(f"  落盘 {state_dir}/ (archive/weights/audit)")
        return 0


def cmd_update(args):
    """新批: 入口过滤 -> 增量重训候选 -> 门裁决 -> 提交 (追加档案+换权重) / 拒绝.

    全程持 state 锁 (并发 update 串行); 崩溃窗口由原子写 + load_state 不变量兜底.
    """
    state_dir = Path(args.state)
    with state_lock(state_dir):
        archive, weights, audit = load_state(state_dir)
        batch = load_json(args.batch)
        if not isinstance(batch, dict):
            raise ValueError(f"批 JSON 顶层必须是 dict: {args.batch}")
        known = {s["name"] for s in archive["samples"]}
        accepted, rejected = filter_batch(batch.get("samples", []), known)
        for r in rejected:
            audit_append(audit, {"kind": "entry_reject", "name": r["name"],
                                 "reason": r["reason"]})
        if not accepted:
            save_state(state_dir, archive, audit)
            print(f"无通过入口的新样本 (拒绝 {len(rejected)} 条), 权重不变")
            return 0
        cand_w = train_candidate(archive["samples"], accepted, weights)
        rows = archive["samples"] + accepted
        X = feat_matrix(rows)
        y = np.array([s["label"] for s in rows], dtype=int)
        Xf = sj.median_fill(X, np.array(weights["med"]))
        Xs = sj.standardize(Xf, np.array(weights["mu"]), np.array(weights["sd"]))
        old_mask = np.array([True] * len(archive["samples"]) + [False] * len(accepted))
        ok, a_c, a_u, fp_c, fp_u = gate_check(Xs, y, old_mask,
                                              cand_w, np.array(weights["w"], dtype=float))
        if ok:
            next_seq = max(s["seq"] for s in archive["samples"]) + 1
            for j, s in enumerate(accepted):
                s["seq"] = next_seq + j
            archive["samples"].extend(accepted)
            n_bad = archive["exam"]["n_bad"] + sum(1 for s in accepted if s["label"] == 1)
            archive["exam"]["n_entries"] = len(archive["samples"])
            archive["exam"]["n_bad"] = n_bad
            archive["exam"]["n_good"] = len(archive["samples"]) - n_bad
            meta = dict(weights["meta"])
            meta["n_seen"] = len(archive["samples"])
            meta["n_bad"] = n_bad
            meta["n_good"] = archive["exam"]["n_good"]
            new_weights = sj.save_weights(state_dir / "weights.json", w=cand_w,
                                          med=weights["med"], mu=weights["mu"],
                                          sd=weights["sd"], meta=meta)
            audit_append(audit, {"kind": "accept", "n_new": len(accepted),
                                 "gate": {"auc_cand": a_c, "auc_cur": a_u,
                                          "fp_cand": fp_c, "fp_cur": fp_u},
                                 "weights_before_md5": weights["self_md5"],
                                 "weights_after_md5": new_weights["self_md5"]})
            save_state(state_dir, archive, audit)
            print(f"接受 {len(accepted)} 份新样本, 权重已更新: "
                  f"旧账 AUC {a_u:.4f} -> {a_c:.4f}, fp {fp_u} -> {fp_c}")
        else:
            audit_append(audit, {"kind": "reject", "n_pending": len(accepted),
                                 "gate": {"auc_cand": a_c, "auc_cur": a_u,
                                          "fp_cand": fp_c, "fp_cur": fp_u}})
            save_state(state_dir, archive, audit)
            print(f"门拒绝 {len(accepted)} 份新样本 (旧账 AUC {a_u:.4f} -> {a_c:.4f}, "
                  f"fp {fp_u} -> {fp_c}), 权重不变, 批未入档案")
        return 0


def cmd_status(args):
    state_dir = Path(args.state)
    with state_lock(state_dir):
        archive, weights, audit = load_state(state_dir)
        m = weights["meta"]
        kinds = {}
        for e in audit["events"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        print(f"状态: {state_dir}")
        print(f"  权重: n_seen={m['n_seen']} (bad {m['n_bad']}/good {m['n_good']}), "
              f"阈值={weights['params']['threshold']}, std_from={weights['std_from']}")
        print(f"  audit: {kinds} (共 {len(audit['events'])} 事件)")
        print(f"  archive.json  payload_md5={archive['payload_md5'][:16]}...")
        print(f"  weights.json  self_md5   ={weights['self_md5'][:16]}...")
        print(f"  audit.json    payload_md5={audit['payload_md5'][:16]}...")
        return 0


def cmd_replay(args):
    """复现自进化实验 (README_selfevolve.md §三~§五): 直读 gatelearn_data.json
    重建 5 批重放三臂+场景 P (不依赖 state — update 后仍可复现, 审计 A3).

    输出与 experiments/selfevolve_2026-08-20/selfevolve_data.json 逐字一致:
    实验环境 (homebrew python + numpy 2.4.4) 下 payload_md5 == c05b91b6...7555d1e;
    跨 numpy 微版本/BLAS 有 ULP 级浮点差异 (实测 ~1.7e-11, 测试审计 2026-08-20
    homebrew 2.4.4 vs Framework 2.4.3 冻结权重 maxdiff), 判据数字不受影响.
    """
    data = verify_payload(Path(args.data), layer="gate-learn")
    samples = [s for s in data["exam"]["samples"] if s.get("load_ok")]
    rows = [{"name": s["name"], "pool": s["pool"], "label": s["label"],
             "feat": s["feat"]} for s in samples]
    vg_names = sorted([r["name"] for r in rows if r["pool"] == "vg"])
    vg_first = set(vg_names[:35])          # 预注册 §三: vg 按 name 排序前 35 = t3

    def batch_of(r):
        p = r["pool"]
        if p == "g3_p1":
            return 1
        if p == "g3":
            return 2
        if p == "vg":
            return 3 if r["name"] in vg_first else 4
        return 5

    b = np.array([batch_of(r) for r in rows])
    X = feat_matrix(rows)
    y = np.array([r["label"] for r in rows], dtype=int)
    names = np.array([r["name"] for r in rows])
    cnt = [int((b == k).sum()) for k in range(1, 6)]
    if cnt != [14, 19, 35, 36, 16]:
        raise ValueError(f"批次计数不符: {cnt} (期望 [14,19,35,36,16], "
                         f"gatelearn_data.json 与实验口径不一致)")
    m1 = b == 1
    med = np.nanmedian(X[m1], axis=0)
    Xf = sj.median_fill(X, med)
    mu, sd = Xf[m1].mean(axis=0), Xf[m1].std(axis=0)
    Xs = sj.standardize(Xf, mu, sd)
    t3 = np.where((b == 3) & (y == 1))[0]
    t3 = sorted(t3, key=lambda i: names[i])
    pollute_idx = [int(i) for i in t3[:N_POLLUTE]]
    y_p = y.copy()
    y_p[pollute_idx] = 0

    def run_arm(yl, mode):
        """S | I | G (run_selfevolve.py:157-180 照搬)."""
        w_cur = None
        rejections = 0
        gates = []
        traj = {}
        for t in range(1, 6):
            if mode == "S":
                w_cur = sj.logreg_fit(Xs[b == 1], yl[b == 1])
            else:
                m = b <= t
                cand = sj.logreg_fit(Xs[m], yl[m])
                if mode == "I" or t == 1:
                    w_cur = cand
                else:
                    old = (b >= 1) & (b < t)
                    # 复用 gate_check (审计 B5: 消除门双实现, 数值逐字等价)
                    ok, a_c, a_u, fp_c, fp_u = gate_check(Xs, yl, old, cand, w_cur)
                    gates.append({"t": t, "ok": bool(ok), "auc_cand": a_c,
                                  "auc_cur": a_u, "fp_cand": fp_c, "fp_cur": fp_u})
                    if ok:
                        w_cur = cand
                    else:
                        rejections += 1
            traj[t] = w_cur
        return traj, rejections, gates

    def eval_traj(traj, yl, final_mask):
        per_t = {}
        for t, w in traj.items():
            newm = b == t
            p = sj.predict_proba(Xs[newm], w)
            fg = {str(k): float(sj.auc(yl[b == k], sj.predict_proba(Xs[b == k], w)))
                  for k in range(1, t + 1)}
            per_t[t] = {"new_auc": float(sj.auc(yl[newm], p)),
                        "new_conf": sj.conf(yl[newm], p), "forget_auc": fg}
        p_f = sj.predict_proba(Xs[final_mask], traj[5])
        return per_t, {"final_auc": float(sj.auc(yl[final_mask], p_f)),
                       "final_conf": sj.conf(yl[final_mask], p_f)}

    final_mask = b >= 2
    if int(final_mask.sum()) != 106:
        raise ValueError(f"final_mask 行数 {int(final_mask.sum())} != 106 "
                         f"(期望 B2..B5, gatelearn_data.json 与实验口径不一致)")
    traj_S, _, _ = run_arm(y, "S")
    traj_I, _, _ = run_arm(y, "I")
    traj_G, rej_G, gates_G = run_arm(y, "G")
    per_S, fin_S = eval_traj(traj_S, y, final_mask)
    per_I, fin_I = eval_traj(traj_I, y, final_mask)
    per_G, fin_G = eval_traj(traj_G, y, final_mask)
    traj_Ip, _, _ = run_arm(y_p, "I")
    traj_Gp, rej_Gp, gates_Gp = run_arm(y_p, "G")
    per_Ip, fin_Ip = eval_traj(traj_Ip, y_p, final_mask)
    per_Gp, fin_Gp = eval_traj(traj_Gp, y_p, final_mask)

    def b1_auc_at(per, t):
        return per[t]["forget_auc"]["1"]

    m1_i = fin_I["final_auc"] >= fin_S["final_auc"] + M1_MARGIN
    m1_g = fin_G["final_auc"] >= fin_S["final_auc"] + M1_MARGIN
    m1 = m1_i and m1_g
    m2_i = b1_auc_at(per_I, 5) >= b1_auc_at(per_I, 1) - M2_MARGIN
    m2_g = b1_auc_at(per_G, 5) >= b1_auc_at(per_G, 1) - M2_MARGIN
    m2 = m2_i and m2_g

    def fp_rate_ok(per):
        for t in range(1, 6):
            c = per[t]["new_conf"]
            ng = c["fp"] + c["tn"]
            if ng > 0 and c["fp"] / ng > M3_FP_RATE:
                return False
        return True

    m3_i = (fin_I["final_conf"]["fp"] <= fin_S["final_conf"]["fp"] + M3_FP_SLACK
            and fp_rate_ok(per_I))
    m3_g = (fin_G["final_conf"]["fp"] <= fin_S["final_conf"]["fp"] + M3_FP_SLACK
            and fp_rate_ok(per_G))
    m3 = m3_i and m3_g
    m4_a = fin_Gp["final_auc"] >= fin_Ip["final_auc"] - M4_MARGIN
    m4_b = fin_Gp["final_conf"]["fp"] <= fin_Ip["final_conf"]["fp"] + M4_FP_SLACK
    m4 = m4_a and m4_b
    verdict = "PASS" if (m1 and m2 and m3) else ("PARTIAL" if (m1 or m2 or m3) else "FAIL")

    payload = {
        "tool": "spsl", "layer": "selfevolve",
        "exam": {"src": "gatelearn_data.json", "n": len(b), "n_feat": len(sj.FEAT_FIELDS),
                 "batches": {str(k): {"n": int((b == k).sum()),
                                      "bad": int((y[b == k] == 1).sum()),
                                      "good": int((y[b == k] == 0).sum())} for k in range(1, 6)},
                 "final_mask": "B2..B5(106)", "preprocess": "median_fill+std 仅用 B1 统计量",
                 "pollute": {"n": N_POLLUTE, "names": [str(names[i]) for i in pollute_idx]}},
        "result": {
            "scene_A": {
                "S": {"per_t": per_S, "final": fin_S},
                "I": {"per_t": per_I, "final": fin_I},
                "G": {"per_t": per_G, "final": fin_G, "rejections": rej_G, "gates": gates_G},
            },
            "scene_P": {
                "I": {"per_t": per_Ip, "final": fin_Ip},
                "G": {"per_t": per_Gp, "final": fin_Gp, "rejections": rej_Gp, "gates": gates_Gp},
            },
            "weights": {
                "A_S_t5": [float(x) for x in traj_S[5]],
                "A_I_t5": [float(x) for x in traj_I[5]],
                "A_G_t5": [float(x) for x in traj_G[5]],
                "P_I_t5": [float(x) for x in traj_Ip[5]],
                "P_G_t5": [float(x) for x in traj_Gp[5]],
            },
        },
        "verdict": {
            "M1_selfevolve_ge_static_plus_005": {"pass": bool(m1),
                "I_auc": float(fin_I["final_auc"]), "G_auc": float(fin_G["final_auc"]),
                "S_auc": float(fin_S["final_auc"])},
            "M2_no_forget_b1": {"pass": bool(m2),
                "I_b1_t5": float(b1_auc_at(per_I, 5)), "I_b1_t1": float(b1_auc_at(per_I, 1)),
                "G_b1_t5": float(b1_auc_at(per_G, 5)), "G_b1_t1": float(b1_auc_at(per_G, 1))},
            "M3_no_fp_growth": {"pass": bool(m3),
                "I_fp": fin_I["final_conf"]["fp"], "G_fp": fin_G["final_conf"]["fp"],
                "S_fp": fin_S["final_conf"]["fp"]},
            "M4_pollution_gate": {"pass": bool(m4),
                "Gp_auc": float(fin_Gp["final_auc"]), "Ip_auc": float(fin_Ip["final_auc"]),
                "Gp_fp": fin_Gp["final_conf"]["fp"], "Ip_fp": fin_Ip["final_conf"]["fp"]},
            "main": "PASS" if (m1 and m2 and m3) else "FAIL",
            "aux": "PASS" if m4 else "FAIL",
            "verdict": verdict,
        },
    }
    md5 = sj.digest(payload)
    payload["payload_md5"] = md5
    payload["self_md5"] = sj.digest(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"落盘 {out}")
    print(f"复现: M1={bool(m1)} M2={bool(m2)} M3={bool(m3)} M4={bool(m4)} -> {verdict}")
    print(f"payload_md5={md5}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spsl.selfevolve",
        description="SPSL 自进化管线: 判定器持续累积训练 + 外部裁决门 + 入口防污染 (北极星 v3)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="冷启动 (120 份 gatelearn 档案 + 冻结权重)")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--result", default=str(DEFAULT_RESULT))
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--force", action="store_true", help="已存在时重建")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("update", help="新批提交 (入口过滤->重训->门裁决)")
    p.add_argument("--batch", required=True, help="批 JSON: {samples:[...]}")
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("status", help="状态")
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("replay", help="复现自进化实验 (逐字一致, 直读 gatelearn_data.json)")
    p.add_argument("--out", default="", help="落盘路径 (空 = 不落盘)")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
