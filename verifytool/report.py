"""报告渲染 — HTML (样式复用实验 05 报告) + JSON 落盘 (md5 双字段).

样式与配色来源: ~/Desktop/北极星/北极星质检仪真实演示_2026-08-07.html
(实验 05 报告: 背景 #f6f8fa, 边框 #d8dee4, ok=#dafbe1/#116329,
bad=#ffebe9/#cf222e, mixed=#fff8c5/#7d4e00, kpi 卡片, note 框)。

md5 双字段算法来源: experiments/bootstrap_monotone/run_bootstrap_monotone.py
save_results() 同款:
  payload_md5 = md5(json.dumps(payload, ensure_ascii=False, sort_keys=True))
  self_md5    = md5(json.dumps(去 self_md5, ensure_ascii=False, indent=2, 无 sort_keys))

纯静态 HTML, 无外部 URL, 可独立打开 (规格 §3)。
"""
import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# JSON 落盘 (bootstrap_monotone 同款算法)
# ---------------------------------------------------------------------------


def json_safe(o):
    """np 类型转 Python 原生 (出处: run_bootstrap_monotone.py json_safe())."""
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(x) for x in o]
    return o


def save_json(payload: dict, out_path: Path) -> dict:
    """写 JSON + 附加 md5 双字段. 返回 (payload_md5, self_md5)."""
    payload = json_safe(payload)
    payload["payload_md5"] = hashlib.md5(
        json.dumps({k: v for k, v in payload.items()
                    if k not in ("payload_md5", "self_md5")},
                   ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    without_self = {k: v for k, v in payload.items() if k != "self_md5"}
    payload["self_md5"] = hashlib.md5(
        json.dumps(without_self, ensure_ascii=False, indent=2).encode("utf-8")).hexdigest()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    return {"payload_md5": payload["payload_md5"], "self_md5": payload["self_md5"]}


# ---------------------------------------------------------------------------
# HTML 渲染 (样式: 实验 05 报告)
# ---------------------------------------------------------------------------

_CSS = """
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f6f8fa; color: #1f2328; }
  .wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  .sub { color: #57606a; font-size: 14px; margin-bottom: 24px; }
  h2 { font-size: 19px; margin: 36px 0 12px; border-bottom: 1px solid #d8dee4;
        padding-bottom: 6px; }
  table { border-collapse: collapse; width: 100%; background: #fff;
          border: 1px solid #d8dee4; font-size: 14px; }
  th, td { border: 1px solid #d8dee4; padding: 8px 10px; text-align: center; }
  th { background: #f0f3f6; font-weight: 600; }
  td.pname { text-align: left; font-family: ui-monospace, Menlo, monospace;
             font-size: 13px; }
  td.lnum { text-align: left; font-weight: 600; }
  td.ok { background: #dafbe1; color: #116329; font-weight: 600; }
  td.bad { background: #ffebe9; color: #cf222e; font-weight: 600; }
  td.mixed { background: #fff8c5; color: #7d4e00; font-weight: 600; }
  td.dim { color: #57606a; font-size: 13px; text-align: left; }
  .cnt { display: block; font-size: 11px; color: #57606a; font-weight: 400; }
  .kpi { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
  .kpi .box { flex: 1; min-width: 180px; background: #fff; border: 1px solid #d8dee4;
              border-radius: 8px; padding: 14px 18px; }
  .kpi .num { font-size: 28px; font-weight: 700; }
  .kpi .lbl { color: #57606a; font-size: 13px; margin-top: 2px; }
  .note { background: #fff; border: 1px solid #d8dee4; border-radius: 8px;
          padding: 12px 16px; font-size: 13px; color: #57606a; margin: 12px 0; }
  ul { margin: 8px 0; padding-left: 22px; line-height: 1.7; }
  code { background: #f0f3f6; padding: 1px 5px; border-radius: 4px;
         font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .mono { font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
"""

_VERDICT_CLASS = {
    "PASS": "ok", "ACCEPT": "ok",
    "REJECT": "bad", "FAIL": "bad",
    "MIXED": "mixed", "ABORT": "mixed",
}

_LAYER_LABEL = {
    "L1": "L1 · H0 均匀性",
    "L2": "L2 · 参照对拍",
    "L3": "L3 · 边界泛化",
    "L4": "L4 · 输入缺失",
}


def _cell(verdict: str, extra: str = "") -> str:
    cls = _VERDICT_CLASS.get(verdict, "mixed")
    small = f'<span class="cnt">{extra}</span>' if extra else ""
    return f'<td class="{cls}">{verdict}{small}</td>'


def _fmt(v, nd=3):
    """科学计数格式化 (float), 稳健处理 inf/nan."""
    try:
        if v != v:                      # NaN
            return "NaN"
        if v == float("inf"):
            return "inf"
        return f"{v:.{nd}e}"
    except (TypeError, ValueError):
        return str(v)


def _F(F, alpha):
    """按 α 取值, 兼容内存 dict (float 键) 与 JSON 回读 dict (字符串键)."""
    try:
        return F[alpha]
    except (KeyError, TypeError):
        return F[str(alpha)]


def _l1_brief(d):
    return (f'连续区 KS D={d["cont_ks_D"]:.4f} (p={d["cont_ks_p"]:.3f}) '
            f'mean={d["cont_mean"]:.4f} F̂(0.05)={_F(d["cont_F"], 0.05):.4f} '
            f'<span class="mono">|</span> '
            f'离散区 mean={d["disc_mean"]:.4f} F̂(0.10)={_F(d["disc_F"], 0.10):.4f} '
            f'(阈值 α+0.02=0.12)')


def _pair_brief(d):
    if d.get("ref_abort"):
        return f"参照自检 {_fmt(d['worst_ref_dev'])} > 1e-9 -> ABORT"
    return (f"max_dev={_fmt(d['max_dev'])} n_viol={d['n_viol']}/{d['n_tables']} "
            f"(容差 1e-6, 参照自检 {_fmt(d['worst_ref_dev'], 1)})")


def _l4_brief(d):
    if "note" in d:
        return d["note"]
    return f"{d['n_fail']}/{d['n_inputs']} 类畸形输入返回有限值 (幻觉填补)"


def _layer_key_rows(four) -> list:
    """[(层, 判定, 拒绝计数, 关键数字, 崩溃标注)]"""
    diag_run = next((r for r in four["per_run"] if "crash" not in r), four["first"])
    crashed = [r["seed"] for r in four["per_run"] if "crash" in r]
    crash_note = ("; 候选在 seed %s 的合法考卷表上抛异常"
                  % ", ".join(map(str, crashed))) if crashed else ""
    briefs = {"L1": _l1_brief, "L2": _pair_brief, "L3": _pair_brief, "L4": _l4_brief}
    rows = []
    for layer in ("L1", "L2", "L3", "L4"):
        lv = four["layers"][layer]
        d = diag_run.get(f"{layer}_diag")
        brief = ("候选崩溃, 无诊断" if d is None else briefs[layer](d)) + crash_note
        rows.append((layer, lv["verdict"], f'{lv["rej"]}/{lv["runs"]}', brief))
    tv = four["total_verdict"]
    rows.append(("总判定", tv, f'{four["reject_runs"]}/{four["n_runs"]}',
                 "任意一层 REJECT 即总 REJECT; <=10% 拒 = ACCEPT"))
    return rows


def render_html(result: dict, meta: dict) -> str:
    four = result["four_layers"]
    cap = result["capability_map"]
    blind = result["blind_spots"]

    # ---- ① 头部 ----
    head = f"""
  <h1>验证器质检报告 — {meta["validator_name"]}</h1>
  <div class="sub">被考验证器: <code>{meta["validator_path"]}</code> ·
  接口 <code>chi2_pvalue(observed) -> float</code> (r×c 计数表, 边缘全正) ·
  {meta["n_runs"]} runs (seeds {", ".join(str(s) for s in four["seeds"])}) ·
  verifytool {result["version"]} · 运行时刻 {meta["run_at"]} ·
  耗时 {result["elapsed_seconds"]} s{(' <b style="color:#cf222e">(超出 2 分钟护栏)</b>' if result.get("time_warn") else '')}</div>
"""

    # ---- ② 四层判定表 ----
    rows_html = "".join(
        f"<tr><td class='lnum'>{lab}</td>"
        f"{_cell(verdict, cnt)}"
        f"<td class='dim'>{brief}</td></tr>"
        for lab, verdict, cnt, brief in _layer_key_rows(four)
    )
    section2 = f"""
  <h2>① 四层考卷判定 (L1-L4; 判定 = {meta["n_runs"]} runs 汇总, 小字 = 拒绝计数)</h2>
  <table>
    <tr><th>层</th><th>判定</th><th>拒绝计数</th><th>关键数字 (首次 run 诊断)</th></tr>
    {rows_html}
  </table>
  <div class="note">判定协议 (同实验 05 results.json comparison_rule, 按 run 数归一):
  拒绝计数占比 &lt;= 10% = PASS, &gt;= 90% = REJECT, 其余 MIXED。
  L2/L3/L4 为固定考卷确定性判定, 多 run 间结果恒定。
  L1 只抓"让 p 值分布脱离 H0 均匀"的错误; L2/L3 抓公式类误差 (1e-6 对拍);
  L4 抓"畸形输入下幻觉填补" (返回有限 p 值 = 静默修补, 须诚实失败)。</div>
"""

    # ---- ③ 能力地图 ----
    def cap_cell(c):
        if c["verdict"] == "ABORT":
            return _cell("ABORT", f"参照自检 {_fmt(c['worst_ref_dev'], 1)}")
        extra = "抛异常" if c["max_dev"] == float("inf") else f"max_dev={_fmt(c['max_dev'])}"
        return _cell(c["verdict"], f"{extra} ({c['n_viol']}/{c['n_tables']} 违规)")

    cap_cells = []
    for r in cap:
        l4_extra = "%d/%d" % (r["L4"]["n_fail"], r["L4"]["n_inputs"])
        cap_cells.append(
            "<tr><td class='pname'>%s</td><td>%s</td>%s%s%s"
            "<td class='dim'>考卷形状域: %s (出处: calibrator/template_exam.py)</td></tr>"
            % (r["template"], r["domain"], cap_cell(r["L2"]), cap_cell(r["L3"]),
               _cell(r["L4"]["verdict"], l4_extra), r["domain"]))
    cap_rows = "".join(cap_cells)
    section3 = f"""
  <h2>② 能力地图 — 9 模板考卷 × L2-L4 (固定考卷, 毫秒级)</h2>
  <table>
    <tr><th>模板</th><th>形状域</th><th>L2 参照对拍</th><th>L3 边界泛化</th>
        <th>L4 输入缺失</th><th>考卷来源</th></tr>
    {cap_rows}
  </table>
  <div class="note">对拍参照 = <code>calibrator/reference.py</code> 的 ref_hand (手写 gammq)
  + ref_scipy (chi2_contingency), 两参照先自检一致到 1e-9 才采信; 容差 1e-6 (与 l2.L2_TOL 同值)。
  每个模板的固定考卷表 (手选 + 种子化随机, 全部边缘 &gt; 0) 来自 calibrator/template_exam.py。
  本图只考"卡方实现正确性" (候选 p 值与卡方参照一致到 1e-6), 不考方法选择
  (用卡方还是费雪是方法层, 不在本图判定内)。L4 列与模板无关 (同一 9 类畸形输入), 全行相同。</div>
"""

    # ---- ④ 盲点清单 ----
    hallu_items = blind["l4_hallucination"]
    if hallu_items:
        hallu_html = "".join(
            f"<li><b>{it['name']}</b> ({it['shape']}) — 返回有限 p 值 "
            f"<code>{it['returned']}</code>: 畸形输入被静默修补后给出看似合理的数字 (幻觉填补)</li>"
            for it in hallu_items)
        hallu_count = f"共 {len(hallu_items)} 类"
    else:
        hallu_html = "<li>无 — 9/9 类畸形输入均诚实失败 (返回非有限值或抛异常)</li>"
        hallu_count = "0 类"

    pair_html = ""
    for layer in ("L2", "L3"):
        p = blind["pair_dev"][layer]
        if p.get("ref_abort"):
            pair_html += (f"<li><b>{layer}</b>: 参照自检 {_fmt(p['worst_ref_dev'], 1)} "
                          f"&gt; 1e-9, 校准层拒绝采信参照 (层自身问题, 非候选问题)</li>")
        elif p.get("n_viol", 0) == 0:
            pair_html += f"<li><b>{layer}</b>: 无违规表 (0/{'40' if layer == 'L2' else '76'} 超容差)</li>"
        else:
            tops = "".join(
                f"<tr><td class='pname'>{list(t['shape'])}</td><td>{t['n']}</td>"
                f"<td>{_fmt(t['dev'])}</td>"
                f"<td class='dim'>候选 {_fmt(t['cand_p'])} vs 参照 {_fmt(t['ref_p'])}</td></tr>"
                for t in p["top"])
            pair_html += (
                f"<li><b>{layer}</b>: {p['n_viol']} 张表超 1e-6 容差, 偏差最大的 "
                f"{len(p['top'])} 张 (形状 / n / 偏差 / p 值):"
                f"<table><tr><th>形状</th><th>n</th><th>偏差</th><th>p 值对照</th></tr>"
                f"{tops}</table></li>")

    cov = blind["coverage"]
    covered_str = ", ".join(
        f"{s[0]}x{s[1]}" for s in cov["covered_shapes"])
    uncov_html = "".join(f"<li>{u}</li>" for u in cov["uncovered"])
    section4 = f"""
  <h2>③ 盲点清单 (自动生成, 零 LLM)</h2>
  <ul>
    <li><b>L4 幻觉填补 ({hallu_count})</b>: {hallu_html}</li>
    <li><b>L2/L3 对拍偏差</b>: {pair_html}</li>
    <li><b>覆盖边界 (工具边界, 如实声明)</b>: 固定考卷覆盖形状 = {covered_str}
      (行 &lt;= {cov['max_rows']}, 列 &lt;= {cov['max_cols']}, n &lt;= 20000)。
      未覆盖类别:<ul>{uncov_html}</ul></li>
  </ul>
"""

    # ---- ⑤ 背景与诚实标注 ----
    section5 = f"""
  <h2>④ 背景与诚实标注</h2>
  <h3>工具校准层自己的校准记录</h3>
  <ul>
    <li>合成世界自证 (2026-08-06, calibrator/run_experiment.py, 5 程序 × 100 seeds):
      灵敏度 <b>1.000</b> (400/400), 误杀率 <b>0.020</b> (2/100)。出处: calib_exp/RESULTS.md。</li>
    <li>真实世界演示 (2026-08-07, 实验 05 qualcheck_realworld, 8 个真实 scipy 程序 × 100 runs):
      灵敏度 <b>1.000</b> (6/6 错程序, 600/600 runs), 误杀率 <b>0.020</b> (手写正确实现 2/100)。
      出处: experiments/qualcheck_realworld/results.json。</li>
    <li>误杀机理 (两处一致): 全来自 L1 离散区 α=0.10 逐点检查 — 卡方逼近在 n=30 下
      α=0.10 轻微反保守 (F̂(0.10) 实测 0.1205/0.1255 > 阈值 0.12), 非实现错误, 在设计预算内。</li>
  </ul>
  <h3>工具边界 (规格 §7)</h3>
  <ul>
    <li>只考实现正确性, 不考方法选择 (用卡方还是费雪 / Yates 校正取舍属方法层)。</li>
    <li>频率学派 p 值框架; 贝叶斯验证器需预测区间覆盖率考官 (未实现)。</li>
    <li>零行/零列退化表协议排除 (scipy 抛 ValueError, 属边界声明而非实现差异)。</li>
    <li>9 模板 L1 慢速考卷 (barnard 类) 不在单次运行内 — 那是模板库自校准档案, 本报告
      只含单验证器的 L1 (通用 3x4/2x3 域) + 能力地图固定考卷。</li>
    <li>单验证器报告 = 一次采样快照, 非统计保证 (与合成自证口径一致;
      L1 抽样区误杀率 ~2%, 多 run 判定可降低随机波动影响)。</li>
    <li>本报告唯一非确定性数据 = 运行耗时 (elapsed_seconds) 及派生的两个 md5;
      判定/诊断/计数等全部字段重跑逐字节一致。</li>
  </ul>
"""

    # ---- footer: md5 指纹 ----
    section6 = f"""
  <div class="note"><b>JSON 验证指纹</b>: payload_md5 = <code class="mono">{meta["payload_md5"]}</code>
  &nbsp;·&nbsp; self_md5 = <code class="mono">{meta["self_md5"]}</code>
  (bootstrap_monotone 同款算法, 主终端可独立复算)</div>
  <div class="sub" style="margin-top:24px">复现: python3 -m verifytool {meta["validator_path"]}
  --runs {meta["n_runs"]} --out {meta["out_path"]} ·
  纯程序零 LLM 零网络 ¥0 · 确定性可复现 (calibrator/ programs/ 零改动)</div>
"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>验证器质检报告 — {meta["validator_name"]} (verifytool {result["version"]})</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
{head}
{section2}
{section3}
{section4}
{section5}
{section6}
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 插件扩展报告模板 (规格 SPEC_插件扩展_2026-08-07.md; 只新增, 不动 render_html)
# ---------------------------------------------------------------------------


def render_exam_html(result: dict, meta: dict, note: str = "") -> str:
    """exam 子命令: 与主命令同格式 (render_html), 额外注入口径差异说明 (模板库自校准档案)."""
    html = render_html(result, meta)
    if note:
        note_div = (f'<div class="note"><b>口径提示</b>: {note}</div>\n'
                    '<div class="wrap">')
        html = html.replace('<div class="wrap">', note_div, 1)
    return html


def render_construct_html(result: dict, meta: dict) -> str:
    """construct 子命令: 规则执行 + 评估 (TP/FP/TN/FN + accuracy) 简版."""
    rule = result["rule"]
    conds = " AND ".join(
        f"{c['stat']}{c['year']} {c['op']} {c['value']:g}"
        for c in rule["conditions"])
    rule_txt = f"{rule['logic']}: {conds}"
    info = result["data"]["info"]
    tm, tr = result["metrics"]["test"], result["metrics"]["train"]

    def m_rows(tag, m, n):
        return (f"<tr><td class='lnum'>{tag}</td><td>{m['tp']}</td><td>{m['fp']}</td>"
                f"<td>{m['fn']}</td><td>{m['tn']}</td>"
                f"<td class='ok'>{m['accuracy']}</td>"
                f"<td>{m['precision']}</td><td>{m['recall']}</td>"
                f"<td>{n}</td></tr>")

    notes = "".join(f"<li>{n}</li>" for n in result["honesty_notes"])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>构造器报告 — {meta["csv_path"]} (verifytool {result["version"]})</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>自动考卷构造器报告 (③ · 离线规则执行)</h1>
  <div class="sub">规则: <code>{rule_txt}</code> · 数据: <code>{meta["csv_path"]}</code> ·
  划分 seed {meta["seed"]} · verifytool {result["version"]} · 运行时刻 {meta["run_at"]} ·
  耗时 {result["elapsed_seconds"]} s</div>
  <h2>① 数据与划分</h2>
  <table>
    <tr><th>对数</th><th>行数 (含)</th><th>跳过行</th><th>正类率</th>
        <th>训练</th><th>测试</th><th>划分规则</th></tr>
    <tr><td>{info["n_pairs"]}</td><td>{info["n_rows_ok"]}</td><td>{info["skipped_rows"]}</td>
        <td>{info["positive_rate"]}</td><td>{info["n_train"]}</td><td>{info["n_test"]}</td>
        <td class="dim">{info["split"]}</td></tr>
  </table>
  <h2>② 规则评估 (TP/FP/TN/FN + accuracy; 判定 = 规则预测 vs label)</h2>
  <table>
    <tr><th>划分</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th>
        <th>accuracy</th><th>precision</th><th>recall</th><th>n</th></tr>
    {m_rows("held-out 测试", tm, info["n_test"])}
    {m_rows("训练参考", tr, info["n_train"]) if tr else ""}
  </table>
  <div class="note">判据 = 与判据1 信号定义对拍 (a/prr/chi2 同款四格表公式), 与 calibrator 无关。
  规则 JSON 由用户提供, 本工具是确定性执行器 (零 LLM, 零网络)。</div>
  <h2>③ 诚实标注</h2>
  <ul>{notes}</ul>
  <div class="note"><b>JSON 验证指纹</b>: payload_md5 = <code class="mono">{meta["payload_md5"]}</code>
  &nbsp;·&nbsp; self_md5 = <code class="mono">{meta["self_md5"]}</code></div>
  <div class="sub" style="margin-top:24px">复现: python3 -m verifytool construct --rule {meta["rule_path"]}
  {meta["csv_path"]} --seed {meta["seed"]} · 纯程序零 LLM 零网络 ¥0</div>
</div>
</body>
</html>
"""


def render_prune_html(result: dict, meta: dict) -> str:
    """prune 子命令: 三臂真检出/fp 对比 + 调度轨迹简版."""
    agg = result["aggregate"]
    arms = ("MAP_CRIT", "ONESHOT", "FULL")

    if "true_det_mean" in agg["MAP_CRIT"] and "recall" in agg["MAP_CRIT"]:
        # 合成世界: 真/假检出 + recall/fp_rate + 形态分解
        rows = "".join(
            f"<tr><td class='lnum'>{a}</td>"
            f"<td>{agg[a]['true_det_mean']}</td><td>{agg[a]['false_det_mean']}</td>"
            f"<td>{agg[a]['recall']:.4f}</td><td>{agg[a]['fp_rate']:.4f}</td>"
            f"<td>{agg[a]['verifiable_true_det_mean']}</td>"
            f"<td>{agg[a]['blind_true_det_mean']}</td>"
            f"<td>{agg[a]['n_tests_total']}</td></tr>" for a in arms)
        head = ("<tr><th>臂</th><th>真检出/seed</th><th>假阳性/seed</th><th>recall</th>"
                "<th>fp_rate</th><th>可验证真检出/seed</th><th>盲点真检出/seed</th>"
                "<th>检验数</th></tr>")
    elif "true_det_mean" in agg["MAP_CRIT"]:
        # cands + truth: 真/假检出均值 (无 recall/fp_rate/形态分解)
        rows = "".join(
            f"<tr><td class='lnum'>{a}</td><td>{agg[a]['true_det_mean']}</td>"
            f"<td>{agg[a]['false_det_mean']}</td>"
            f"<td>{agg[a]['n_tests_total']}</td></tr>" for a in arms)
        head = ("<tr><th>臂</th><th>真检出/seed</th><th>假阳性/seed</th>"
                "<th>检验数</th></tr>")
    else:
        # cands 无 truth: 只有 n_detected
        rows = "".join(
            f"<tr><td class='lnum'>{a}</td><td>{agg[a]['n_detected_mean']}</td>"
            f"<td class='dim'>无 truth (无真/假检出)</td>"
            f"<td>{agg[a]['n_tests_total']}</td></tr>" for a in arms)
        head = ("<tr><th>臂</th><th>n_detected/seed (BH q&lt;0.05)</th>"
                "<th>说明</th><th>检验数</th></tr>")

    # 调度轨迹
    trj = result["trajectory"]
    if "MAP_CRIT" in trj and "per_round" in trj["MAP_CRIT"]:
        pr = trj["MAP_CRIT"]["per_round"]
        pr_rows = "".join(
            f"<tr><td>r{r['round']}</td><td>{r['n_pool']}</td>"
            f"<td>{r['n_true_pool']}/{r['n_false_pool']}</td>"
            f"<td>{r['false_ratio']:.3f}</td><td>{r['n_topup']}</td></tr>"
            for r in pr) or "<tr><td colspan='5' class='dim'>无加投轮</td></tr>"
        traj_html = f"""
  <h2>③ 调度轨迹 (合成世界, {result["seeds"][0]}..{result["seeds"][-1]} 聚合)</h2>
  <table>
    <tr><th>MAP_CRIT 轮</th><th>池大小</th><th>真/假入池</th><th>假占比</th><th>加投数</th></tr>
    {pr_rows}
  </table>
  <div class="note">ONESHOT 加投 {trj["ONESHOT"]["n_topup_done"]} 次 (剩余预算不用, 设计如此);
  FULL 加投 {trj["FULL"]["n_topup_total"]} 次 (随机全池, 独立 rng 流 seed*7919+预算)。</div>
"""
    else:
        traj_html = f"""
  <h2>③ 调度轨迹</h2>
  <div class="note">{trj.get("note", "")}</div>
"""

    notes = "".join(f"<li>{n}</li>" for n in result["honesty_notes"])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>剪枝调度报告 — {meta["world"]} (verifytool {result["version"]})</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>剪枝调度报告 (④ · 三臂对比)</h1>
  <div class="sub">世界: {meta["world"]} · 预算 {meta["budget"]} 检验/seed ·
  {meta["n_seeds"]} seeds (seeds {result["seeds"][0]}..{result["seeds"][-1]}) ·
  verifytool {result["version"]} · 运行时刻 {meta["run_at"]} ·
  耗时 {result["elapsed_seconds"]} s</div>
  <h2>① 三臂对比 (检出 = BH α=0.05 校正后 q&lt;0.05, m=候选数)</h2>
  <table>{head}{rows}</table>
  <h2>② 判据口径 (与 v7/4h 一致)</h2>
  <div class="note">FULL = 全候选首检 + 剩余预算均匀随机加投; MAP_CRIT = 140 可验证首检 +
  递归临界救援 (每轮池 = 未检出候选按组合 p 升序, 全部加投, 立即更新组合 p); ONESHOT = 140 首检 +
  单轮加投 (每候选最多 1 次, 剩余预算不用)。组合 = Fisher (combine_k, 自由度 2k);
  盲点候选 MAP_CRIT/ONESHOT 永不加投 (final_p=1.0, BH 下永不检出)。</div>
  {traj_html}
  <h2>④ 诚实标注 (边界与出处)</h2>
  <ul>{notes}</ul>
  <div class="note"><b>JSON 验证指纹</b>: payload_md5 = <code class="mono">{meta["payload_md5"]}</code>
  &nbsp;·&nbsp; self_md5 = <code class="mono">{meta["self_md5"]}</code></div>
  <div class="sub" style="margin-top:24px">复现: python3 -m verifytool prune --world {result["mode"]}
  --budget {meta["budget"]} --n-seeds {meta["n_seeds"]} · 纯程序零 LLM 零网络 ¥0 ·
  确定性可复现 (唯一非确定 = elapsed_seconds 及其派生 md5)</div>
</div>
</body>
</html>
"""
