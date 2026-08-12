"""自动考卷构造器 (③): 离线规则执行 + 评估 (TP/FP/TN/FN + accuracy).

规格: SPEC_插件扩展_2026-08-07.md §3.

纯函数来源: experiments/bootstrap_monotone/run_bootstrap_monotone.py (只抽函数,
同语义实现, 不 import 该实验脚本 — 它有 FAERS/scripts 硬编码绝对路径和 LLM 调用):
  metrics_from_pred / cond_holds / rule_predicates / parse_rule_json / make_split
特征公式 (a/prr/chi2) 与 judge1_common.py 四格表公式同款 (逐位一致口径,
来源注释见 features_from_cells).

输入:
  --rule <rule.json>   R1 同 schema: {"logic": "AND"|"OR", "conditions":
                        [{"stat": "a"|"prr"|"chi2", "year": 2024|2025,
                          "op": ">="|"<=", "value": 数值}]}, 最多 2 条件
  数据 CSV (2x2 四格表格式, 每对每年一行):
      pair,year,a,b,c,d,label
      a = 该药该事件计数, b = 该药其他事件, c = 其他药该事件, d = 其余;
      label = 金标准 (每对恒定, 两行同值; 1 = 正类).

流程 (确定性, 全部固定种子):
  1) 读 CSV -> 每对特征 {a, prr, chi2} x {2024, 2025} (judge1 公式, numpy 同序)
  2) make_split(pairs, seed) (默认 seed=100, 同实验 seed 100 口径):
     400+ 对 -> 采样 400 / 240 训练 / 160 测试 (held-out); 不足 400 对 -> 全量评估并注明
  3) 规则在测试集上评估 (TP/FP/TN/FN + accuracy), 训练集作参考
  4) 输出 JSON (md5 双字段) + HTML 简版

语义说明 (如实): 本子命令 = 9a 的"规则执行+评估"离线化 (R1 规则 held-out
acc 0.994 = 0.99375 可复现, 与判据1 信号定义对拍, 与 calibrator 无关);
"LLM 自动生成规则"部分不接入 (有 API 成本/非确定), 规则 JSON 由用户提供 —
构造器 = 确定性执行器. 未来扩展 (标注, 不做): 自动构造"验证器考卷"型规则
(当前只支持信号检测规则型).
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 纯函数 (同语义复制自 run_bootstrap_monotone.py, 出处标注逐函数)
# ---------------------------------------------------------------------------


def metrics_from_pred(y_true: list, y_pred: list) -> dict:
    """(bootstrap_monotone 同款) 混淆矩阵 + 指标."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    n = tp + fp + fn + tn
    return {
        "accuracy": round((tp + tn) / n, 6) if n else 0.0,
        "precision": round(tp / (tp + fp), 6) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 6) if (tp + fn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def cond_holds(cond: dict, feat: dict) -> bool:
    """(bootstrap_monotone 同款) 单条件判定."""
    v = feat[cond["stat"]][cond["year"]]
    return v >= cond["value"] if cond["op"] == ">=" else v <= cond["value"]


def rule_predicates(rule: dict, feats: list) -> list:
    """(bootstrap_monotone 同款) 把解析后的规则翻译为可执行 predicate, 逐对预测 1/0."""
    out = []
    for feat in feats:
        rs = [cond_holds(c, feat) for c in rule["conditions"]]
        out.append(1 if (all(rs) if rule["logic"] == "AND" else any(rs)) else 0)
    return out


def parse_rule_json(text: str):
    """(bootstrap_monotone 同款) 解析规则 JSON. 非法返回 None."""
    import math

    t = text.strip()
    d = None
    try:
        d = json.loads(t)
    except Exception:
        t2 = t
        if t2.startswith("```"):
            t2 = t2.split("```", 2)[1].lstrip("json \n\r")
        try:
            d = json.loads(t2)
        except Exception:
            return None
    if not isinstance(d, dict):
        return None
    logic = d.get("logic")
    conds = d.get("conditions")
    if logic not in ("AND", "OR"):
        return None
    if not isinstance(conds, list) or not 1 <= len(conds) <= 2:
        return None
    out_conds = []
    for c in conds:
        if not isinstance(c, dict):
            return None
        if c.get("stat") not in ("a", "prr", "chi2"):
            return None
        if c.get("year") not in (2024, 2025):
            return None
        if c.get("op") not in (">=", "<="):
            return None
        v = c.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
        out_conds.append({"stat": c["stat"], "year": c["year"], "op": c["op"], "value": float(v)})
    return {"logic": logic, "conditions": out_conds}


def make_split(pairs: list, seed: int, n_pool: int = 400, n_train: int = 240) -> tuple:
    """(bootstrap_monotone 同款划分规则, 输入改为显式对列表):
    rng = random.Random(seed); pool = rng.sample(sorted(pairs), n_pool); rng.shuffle(pool);
    return (train_pairs, test_pairs, pool_pairs). 对列表 < n_pool 时返回全量作为 test."""
    if len(pairs) < n_pool:
        return [], pairs, pairs
    rng = random.Random(seed)
    pool = rng.sample(sorted(pairs), n_pool)
    rng.shuffle(pool)
    return pool[:n_train], pool[n_train:], pool


# ---------------------------------------------------------------------------
# 特征 (judge1_common.py 同款四格表公式, 出处注释见文件头)
# ---------------------------------------------------------------------------

YEARS = (2024, 2025)


def features_from_cells(cells: dict) -> dict:
    """每对特征: cells = {year: (a, b, c, d)} -> {a, prr, chi2} x {year}.
    公式 (judge1_common.py compute_signals 同款, numpy float64 同序运算, 逐位一致):
      PRR = (a/(a+b)) / (c/(c+d))
      chi2 = n*(a*d - b*c)^2 / ((a+b)*(c+d)*(a+c)*(b+d)), n = a+b+c+d
    退化分母 (c+d=0 等) 行为与 np.errstate(ignore) 相同 (nan/inf 保留)."""
    out = {"a": {}, "prr": {}, "chi2": {}}
    for year in YEARS:
        if year not in cells:
            out["a"][year] = out["prr"][year] = out["chi2"][year] = float("nan")
            continue
        a, b, c, d = (float(x) for x in cells[year])
        # np.float64 标量 + errstate(ignore): 退化表 (a+b=0 等) 行为与 features_for_pairs
        # 的 numpy 数组运算逐位一致 (nan/inf 保留不抛)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_, b_, c_, d_ = np.float64(a), np.float64(b), np.float64(c), np.float64(d)
            n = a_ + b_ + c_ + d_
            prr = (a_ / (a_ + b_)) / (c_ / (c_ + d_))
            chi2 = n * (a_ * d_ - b_ * c_) ** 2 \
                / ((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_))
        out["a"][year], out["prr"][year], out["chi2"][year] = a, float(prr), float(chi2)
    return out


# ---------------------------------------------------------------------------
# CSV 读取 (2x2 四格表格式: pair,year,a,b,c,d,label)
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> dict:
    """读 CSV -> {"pairs": [(drug, event), ...] 按首次出现序, "rows": {pair: {year: (a,b,c,d)}},
    "labels": {pair: int}, "skipped": n}. pair 列格式 = 'DRUG|EVENT' (drug/event 不含 '|',
    FAERS 归一化名可含逗号 — 用 csv 模块解析, 防逗号破坏列)."""
    import csv

    rows = {}
    labels = {}
    order = []
    skipped = 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [c.strip() for c in header]
        if cols != ["pair", "year", "a", "b", "c", "d", "label"]:
            raise ValueError(
                f"CSV 表头须为 pair,year,a,b,c,d,label (当前: {cols!r})")
        for ln, parts in enumerate(reader, start=2):
            if not parts or not any(p.strip() for p in parts):
                continue
            if len(parts) != 7:
                skipped += 1
                continue
            pair_s, year_s, a_s, b_s, c_s, d_s, lab_s = [p.strip() for p in parts]
            try:
                year = int(year_s)
                a, b, c, d = float(a_s), float(b_s), float(c_s), float(d_s)
                lab = int(lab_s)
            except ValueError:
                skipped += 1
                continue
            if year not in YEARS:
                skipped += 1
                continue
            drug, _, event = pair_s.partition("|")
            pair = (drug.strip(), event.strip())
            if not drug or not event:
                skipped += 1
                continue
            if pair not in rows:
                rows[pair] = {}
                order.append(pair)
            rows[pair][year] = (a, b, c, d)
            prev = labels.get(pair)
            if prev is None:
                labels[pair] = lab
            elif prev != lab:
                raise ValueError(
                    f"CSV 第 {ln} 行: 对 {pair_s!r} 的 label 与先前不一致 "
                    f"({prev} vs {lab})")
    if not rows:
        raise ValueError("CSV 无有效行")
    return {"pairs": order, "rows": rows, "labels": labels, "skipped": skipped}


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def run_construct(rule: dict, csv_path: Path, seed: int) -> dict:
    """读取 CSV + 划分 + 规则评估. 返回结果 dict (供 report.py 渲染)."""
    import time

    t0 = time.monotonic()
    data = read_csv(csv_path)
    pairs = data["pairs"]
    feats = [features_from_cells(data["rows"][p]) for p in pairs]
    y = [data["labels"][p] for p in pairs]
    train_pairs, test_pairs, pool = make_split(pairs, seed)
    pos = sum(y)
    info = {
        "n_pairs": len(pairs),
        "n_rows_ok": sum(len(data["rows"][p]) for p in pairs),
        "skipped_rows": data["skipped"],
        "positive_rate": round(pos / len(pairs), 6) if pairs else 0.0,
        "split": ("采样400/240训练/160测试 (held-out, 同实验 seed %d 口径)" % seed
                  if len(pairs) >= 400
                  else f"对列表 {len(pairs)} < 400, 无划分, 全量作测试"),
        "n_train": len(train_pairs), "n_test": len(test_pairs),
        "test_positive_rate": round(sum(1 for p in test_pairs if data["labels"][p]) / len(test_pairs), 6)
        if test_pairs else None,
    }

    test_metrics = {}
    train_metrics = None
    if test_pairs:
        idx = {p: i for i, p in enumerate(pairs)}
        y_test = [data["labels"][p] for p in test_pairs]
        pred = rule_predicates(rule, [feats[idx[p]] for p in test_pairs])
        test_metrics = metrics_from_pred(y_test, pred)
    if train_pairs:
        idx = {p: i for i, p in enumerate(pairs)}
        y_tr = [data["labels"][p] for p in train_pairs]
        pred_tr = rule_predicates(rule, [feats[idx[p]] for p in train_pairs])
        train_metrics = metrics_from_pred(y_tr, pred_tr)

    from verifytool import VERSION

    elapsed = time.monotonic() - t0
    return {
        "tool": "verifytool",
        "version": VERSION,
        "spec": "SPEC_插件扩展_2026-08-07.md",
        "subcommand": "construct",
        "rule": rule,
        "data": {"csv": str(csv_path.resolve()), "info": info},
        "metrics": {"test": test_metrics, "train": train_metrics},
        "elapsed_seconds": round(elapsed, 3),
        "honesty_notes": [
            "构造器 = 9a 的'规则执行+评估'离线化 (确定性执行器), 规则 JSON 由用户提供; "
            "LLM 自动生成规则不接入 (API 成本/非确定).",
            "R1 规则 (a2025>=3 AND prr2025>=2) held-out 基准: acc 0.99375 "
            "(实验 9a, experiments/bootstrap_monotone/results.json seed 100/200/300 一致), "
            "容差 ±0.01 内即通过复现.",
            "判据 = 与判据1 信号定义对拍 (a/prr/chi2 同款四格表公式), 与 calibrator 无关.",
            "未来扩展 (未做): 自动构造'验证器考卷'型规则 (当前只支持信号检测规则型).",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_construct(argv) -> int:
    """python3 -m verifytool construct --rule <rule.json> <数据csv> [--seed N] [--out PATH]."""
    from verifytool.report import render_construct_html, save_json

    parser = argparse.ArgumentParser(
        prog="python3 -m verifytool construct",
        description="构造器 (③): 离线规则执行 + 评估 (TP/FP/TN/FN + accuracy). "
                    "规则 JSON (R1 同 schema) + 数据 CSV (2x2 四格表: pair,year,a,b,c,d,label).")
    parser.add_argument("--rule", required=True, help="规则 JSON 路径 (R1 同 schema)")
    parser.add_argument("csv", help="数据 CSV 路径 (pair,year,a,b,c,d,label)")
    parser.add_argument("--seed", type=int, default=100,
                        help="确定性划分 seed (默认 100, 同实验 9a seed 100 口径)")
    parser.add_argument("--out", default=None, help="HTML 输出路径 (默认 cwd/construct_report_<csv名>.html)")
    args = parser.parse_args(argv)

    rule_path = Path(args.rule)
    if not rule_path.exists():
        print(f"[verifytool construct] 规则文件不存在: {rule_path}", file=sys.stderr)
        return 2
    try:
        rule = parse_rule_json(rule_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"[verifytool construct] 读取规则失败: {exc}", file=sys.stderr)
        return 2
    if rule is None:
        print("[verifytool construct] 规则 JSON 非法 (schema: "
              '{"logic":"AND|OR","conditions":[{"stat":"a|prr|chi2","year":2024|2025,'
              '"op":">=|<=","value":数值}], 1-2 条件}', file=sys.stderr)
        return 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[verifytool construct] 数据 CSV 不存在: {csv_path}", file=sys.stderr)
        return 2

    out_html = (Path(args.out) if args.out
                else Path.cwd() / f"construct_report_{csv_path.stem}.html")
    out_html = out_html.resolve()
    out_json = out_html.with_suffix(".json")

    try:
        result = run_construct(rule, csv_path, args.seed)
    except ValueError as exc:
        print(f"[verifytool construct] 数据读取失败: {exc}", file=sys.stderr)
        return 2

    payload = dict(result)
    payload["command"] = "construct --rule %s %s" % (rule_path, csv_path)
    md5s = save_json(payload, out_json)
    meta = {
        "rule_path": str(rule_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "seed": args.seed,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_path": str(out_html),
        "payload_md5": md5s["payload_md5"],
        "self_md5": md5s["self_md5"],
    }
    out_html.write_text(render_construct_html(result, meta), encoding="utf-8")

    tm = result["metrics"]["test"]
    print("[verifytool construct] 规则: %s" % json.dumps(rule, ensure_ascii=False))
    info = result["data"]["info"]
    print(f"  数据: {info['n_pairs']} 对 ({info['n_rows_ok']} 行, 跳过 {info['skipped_rows']}), "
          f"正类率 {info['positive_rate']}, {info['split']}")
    print(f"  held-out 测试 ({info['n_test']} 对): accuracy {tm['accuracy']} "
          f"(TP={tm['tp']} FP={tm['fp']} FN={tm['fn']} TN={tm['tn']})")
    if result["metrics"]["train"]:
        tr = result["metrics"]["train"]
        print(f"  训练参考 ({info['n_train']} 对): accuracy {tr['accuracy']} "
              f"(TP={tr['tp']} FP={tr['fp']} FN={tr['fn']} TN={tr['tn']})")
    print(f"[verifytool construct] HTML -> {out_html}")
    print(f"[verifytool construct] JSON -> {out_json}")
    print(f"[verifytool construct] payload_md5={md5s['payload_md5']} "
          f"self_md5={md5s['self_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(cmd_construct(sys.argv[1:]))
