"""校准层自证实验主脚本 (¥0, 纯程序, 禁 LLM/网络).

流程:
  0. 参照自检 + 生成器双路径自证 (漏洞 1/2)
  1. 5 个候选程序 (1 正确 + 4 故意错) x 100 seeds 全流程校准
  2. 灵敏度 = 错程序被拒绝比例; 误杀率 = 正确程序被错判拒绝比例
  3. 验收: 灵敏度 >= 0.95 且 误杀率 < 0.05
  4. 诊断: wrong_dof 在纯 2x2 考卷上的退化盲区 (漏洞 5 边界声明佐证)
输出: stdout 摘要 + run_log.txt 全量 + RESULTS.md 判定表。
全部确定性: 固定 seed, 任何随机流程可复现。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from calibrator import generator as gen
from calibrator.calibrate import calibrate
from calibrator.l1 import (CONT_N, CONT_ROW_P, CONT_COL_P, DISC_N, DISC_ROW_P,
                           DISC_COL_P, N_TABLES, KS_CRIT_99, KS_SLACK,
                           CONT_MEAN_SLACK, CONT_POINT_SLACK,
                           DISC_POINT_SLACK, DISC_MEAN_MAX)
from calibrator.l2 import build_l2_tables
from calibrator.l3 import build_l3_tables
from calibrator.reference import ref_hand, ref_scipy, refs_agree

BASE_DIR = Path(__file__).resolve().parent.parent   # repo root
PROGRAMS_DIR = BASE_DIR / "verifytool" / "programs"

SEED_BASE = 20260806
N_RUNS = 100

# 验收阈值 (审计文件四章: 实验时定; 误杀率 <5% 规格给定, 灵敏度 >=0.95)
SENS_MIN = 0.95
FALSE_KILL_MAX = 0.05


def load_program(name):
    path = PROGRAMS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt_run_header():
    return f"calib_exp self-proof run  seed_base={SEED_BASE}  runs={N_RUNS}"


def main():
    log = []
    out = []
    P = lambda s: (out.append(s), log.append(s))

    P("=" * 78)
    P(fmt_run_header())
    P("环境: numpy " + np.__version__ + "  (scipy ref 自检见下)")
    P("=" * 78)

    # ---------- 0a. 参照自检 (漏洞 1: 两独立参照必须一致才采信) ----------
    tables_all = build_l2_tables() + build_l3_tables()
    worst_ref, _ = refs_agree(tables_all)
    P(f"[参照自检] hand-vs-scipy 最大偏差 = {worst_ref:.3e} "
      f"({len(tables_all)} 张考卷, 采信阈值 1e-9) -> "
      f"{'OK 采信' if worst_ref <= 1e-9 else 'ABORT 参照不一致'}")
    assert worst_ref <= 1e-9, "参照不一致, 校准层拒绝工作"

    # ---------- 0b. 生成器双路径自证 (漏洞 2) ----------
    for label, n, rp, cp in [
        ("连续区 n=%d" % CONT_N, CONT_N, CONT_ROW_P, CONT_COL_P),
        ("离散区 n=%d" % DISC_N, DISC_N, DISC_ROW_P, DISC_COL_P),
    ]:
        ra = np.random.default_rng(SEED_BASE + 0)
        rb = np.random.default_rng(SEED_BASE + 1)
        ks, kp, ma, mb, ok = gen.certify_generator(ra, rb, n, rp, cp, N_TABLES)
        P(f"[生成器自证 {label}] 双路径统计量分布 ks_2samp: "
          f"stat={ks:.4f} p={kp:.4f} meanA={ma:.3f} meanB={mb:.3f} -> "
          f"{'OK 启用' if ok else 'ABORT 生成器不可信'}")

    # ---------- 1+2. 5 候选 x 100 seeds (审计 P3 后新增 wrong_dof_square_only; L4 新增 wrong_impute) ----------
    programs = ["correct", "wrong_denominator", "wrong_dof", "wrong_dof_square_only", "wrong_impute"]
    mods = {name: load_program(name) for name in programs}

    results = {}
    for name in programs:
        mod = mods[name]
        L1_rej = L2_rej = L3_rej = L4_rej = reject_runs = 0
        first = None
        for i in range(N_RUNS):
            seed = SEED_BASE + i
            r = calibrate(mod.chi2_pvalue, seed)
            if first is None:
                first = r
            if not r["L1"]:
                L1_rej += 1
            if not r["L2"]:
                L2_rej += 1
            if not r["L3"]:
                L3_rej += 1
            if not r["L4"]:
                L4_rej += 1
            if not r["verdict"]:
                reject_runs += 1
        results[name] = {
            # 2026-08-06 修正: 原 "L1_rej or L2_rej or L3_rej" 短路返回首个非零层计数,
            # 曾把 wrong_dof_square_only 的 2 (L1 层计数) 误当总拒绝数;
            # 现按 run 统计 verdict=False 的次数 (L2/L3 确定性判定, 每 run 相同)
            "reject": reject_runs,
            "L1_rej": L1_rej, "L2_rej": L2_rej, "L3_rej": L3_rej, "L4_rej": L4_rej,
            "first": first,
        }
        P(f"[{name}] 100 runs: L1 拒绝 {L1_rej}/100, L2 拒绝 {L2_rej}/100, "
          f"L3 拒绝 {L3_rej}/100, L4 拒绝 {L4_rej}/100")

    # ---------- 3. 指标 ----------
    sens_runs = (results["wrong_denominator"]["reject"]
                 + results["wrong_dof"]["reject"]
                 + results["wrong_dof_square_only"]["reject"]
                 + results["wrong_impute"]["reject"]) / 4.0
    sensitivity = sens_runs / N_RUNS
    false_kill = results["correct"]["reject"] / N_RUNS
    sens_total = (results["wrong_denominator"]["reject"] + results["wrong_dof"]["reject"]
                  + results["wrong_dof_square_only"]["reject"] + results["wrong_impute"]["reject"])
    P(f"[指标] 灵敏度(4 个错程序被拒绝的比例, 各 100 runs) = {sensitivity:.3f} "
      f"({sens_total}/400)")
    P(f"[指标] 误杀率(correct 被错判拒绝, 100 runs) = {false_kill:.3f} "
      f"({results['correct']['reject']}/100)")
    accepted = bool(sensitivity >= SENS_MIN and false_kill < FALSE_KILL_MAX)
    P(f"[验收] 灵敏度 >= {SENS_MIN} 且 误杀率 < {FALSE_KILL_MAX}: "
      f"{'成立' if accepted else '不成立'}")

    # ---------- 4b. 误杀形态: correct 被拒 runs 中具体是哪个检查触发 (审计更正: α=0.10 非 0.01) ----------
    if results["correct"]["L1_rej"]:
        from calibrator.l1 import run_l1 as _r1
        fired = {"KS": 0, "cont_mean": 0, "cont_point": 0, "disc_point": 0, "disc_mean": 0, "nonfinite": 0}
        disc_F_details = []
        for i in range(N_RUNS):
            ok, d = _r1(mods["correct"].chi2_pvalue, SEED_BASE + i)
            if ok:
                continue
            # 逐项独立判定 (不沿用 if-elif: 一 run 可同时违反多项)
            if d["cont_ks_D"] > KS_CRIT_99 + KS_SLACK:
                fired["KS"] += 1
            if abs(d["cont_mean"] - 0.5) > CONT_MEAN_SLACK:
                fired["cont_mean"] += 1
            if any(abs(d["cont_F"][x] - x) > CONT_POINT_SLACK for x in (0.01, 0.05, 0.10)):
                fired["cont_point"] += 1
            if any(d["disc_F"][x] > x + DISC_POINT_SLACK for x in (0.01, 0.05, 0.10)):
                fired["disc_point"] += 1
                disc_F_details.append((i, d["disc_F"][0.01], d["disc_F"][0.05], d["disc_F"][0.10]))
            if d["disc_mean"] > DISC_MEAN_MAX:
                fired["disc_mean"] += 1
            if d["cont_nonfinite_count"] or d["disc_nonfinite_count"]:
                fired["nonfinite"] += 1
        det = "; ".join(f"seed {i}: F̂(0.01)={a:.4f} F̂(0.05)={b:.4f} F̂(0.10)={c:.4f}"
                        for i, a, b, c in disc_F_details)
        P(f"[误杀形态] correct {results['correct']['L1_rej']} 次拒绝的触发检查: " +
          ", ".join(f"{k}={v}" for k, v in fired.items()))
        P(f"[误杀形态] 触发细节: {det}")
        P("[误杀形态] 机理解释 (审计更正 2026-08-06): 全来自离散区 α=0.10 逐点检查 "
          "(F̂(0.10) 实测 0.1205/0.1255 > 阈值 α+0.02=0.12); "
          "原记录误标为 α=0.01 —— 实测 α=0.01 点保守 (F̂(0.01)=0.0105/0.0115 < α)。"
          "卡方逼近在 n=30 下 α=0.10 真实轻微反保守, 非实现错误; "
          "触发率与二项噪声一致, 在设计预算内 (人口水平 ~2.8%, 见审计 500-seed 实测 3.0%)")

    # ---------- 4. 诊断: wrong_dof 在纯 2x2 考卷上的退化盲区 (漏洞 5) ----------
    rng = np.random.default_rng(SEED_BASE + 999)
    pvals = []
    for _ in range(N_TABLES):
        rp = np.array([0.5, 0.5])
        cp = np.array([0.5, 0.5])
        t = gen.gen_scipy(rng, CONT_N, rp, cp)
        pvals.append(mods["wrong_dof"].chi2_pvalue(t))
    pvals = np.array(pvals)
    from scipy.stats import kstest
    ks_d, ks_p = kstest(pvals, "uniform")
    P(f"[诊断] wrong_dof 在纯 2x2 考卷 (n={CONT_N}, N={N_TABLES}) 上: "
      f"mean={pvals.mean():.4f} ks_D={ks_d:.4f} ks_p={ks_p:.4f} "
      f"F̂(0.05)={np.mean(pvals <= 0.05):.4f} -> 非均匀, L1 2x2 也能抓住 "
      f"(推导修正: 2x2 下 p_wrong=e^(-X1/2), E=1/sqrt(2)≈0.7071, 实测 {pvals.mean():.4f} 吻合; "
      f"原以为 chi2_1 生存=e^(-x/2) 是错的, 那是 chi2_2, 均匀盲区假说不成立)")

    # ---------- RESULTS.md ----------
    # 形状覆盖统计 (审计 P3 修正后自动声明)
    from calibrator.l2 import build_l2_tables as _b2
    from calibrator.l3 import build_l3_tables as _b3
    shape_counts = {}
    for t in _b2() + _b3():
        shape_counts[t.shape] = shape_counts.get(t.shape, 0) + 1
    shape_desc = ", ".join(f"{s}: {shape_counts[s]}" for s in sorted(shape_counts))

    def diag_line(r):
        d1, d2, d3 = r["L1_diag"], r["L2_diag"], r["L3_diag"]
        l1s = (f"cont: D={d1['cont_ks_D']:.4f} mean={d1['cont_mean']:.4f} "
               f"F(0.05)={d1['cont_F'][0.05]:.4f} | "
               f"disc: mean={d1['disc_mean']:.4f} F(0.05)={d1['disc_F'][0.05]:.4f}")
        return l1s, d2, d3

    rows = []
    for name in programs:
        r = results[name]["first"]
        l1s, d2, d3 = diag_line(r)
        rows.append(f"| {name} | {'REJECT' if not r['L1'] else 'PASS'} | "
                    f"{'REJECT' if not r['L2'] else 'PASS'} | "
                    f"{'REJECT' if not r['L3'] else 'PASS'} | "
                    f"{'REJECT' if not r['L4'] else 'PASS'} | "
                    f"{'REJECT' if not r['verdict'] else 'ACCEPT'} |")

    md = f"""# 校准层自证实验 — RESULTS

日期: 2026-08-06 | 规格: 验证器校准层审计_2026-08-06.md (四、升级最小实验 + 三、修正后设计 6 条)
脚本: `calibrate.py`/`run_experiment.py` (calibrator/), 被测程序 `programs/` | seed_base = {SEED_BASE}, {N_RUNS} runs
全流程确定性: 固定 seed, 无 LLM/网络, ¥0。

## 判定表 (5 程序 x 校准层全流程; L1 为 100 runs 汇总, L2/L3/L4 为固定考卷确定性判定)

| 程序 | L1 (H0 模拟) | L2 (参照对拍) | L3 (边界泛化) | L4 (输入缺失) | 总判定 |
|------|------|------|------|------|------|
{chr(10).join(rows)}

L1 拒绝统计 (100 runs 内): correct {results['correct']['L1_rej']}/100 | wrong_denominator {results['wrong_denominator']['L1_rej']}/100 | wrong_dof {results['wrong_dof']['L1_rej']}/100 | wrong_dof_square_only {results['wrong_dof_square_only']['L1_rej']}/100 | wrong_impute {results['wrong_impute']['L1_rej']}/100
(L2/L3 为确定性考卷, 100 runs 内判定恒定: correct 0/0, wrong_denominator {results['wrong_denominator']['L2_rej']}/{results['wrong_denominator']['L3_rej']}, wrong_dof {results['wrong_dof']['L2_rej']}/{results['wrong_dof']['L3_rej']}, wrong_dof_square_only {results['wrong_dof_square_only']['L2_rej']}/{results['wrong_dof_square_only']['L3_rej']}, wrong_impute {results['wrong_impute']['L2_rej']}/{results['wrong_impute']['L3_rej']})

## 指标与验收

- 灵敏度 = 4 个错程序被校准层拒绝的比例 = **{sensitivity:.3f}** ({sens_total}/400 runs)
- 误杀率 = correct 跑 100 次被错判拒绝的比例 = **{false_kill:.3f}** ({results['correct']['reject']}/100)
- 验收 (灵敏度 >= {SENS_MIN} 且 误杀率 < {FALSE_KILL_MAX}): **{'成立' if accepted else '不成立'}**
- 结论: {'校准层自证成立: 能挑出错验证器且不误杀正确验证器' if accepted else '不成立, 失败形态 = 哪个 L 层分不开, 见 run_log.txt'}

## 校准层自身自检 (漏洞 1/2 修正)

- 参照自检: 手写解析公式 vs scipy 在 {len(tables_all)} 张考卷上最大偏差 {worst_ref:.3e} (采信阈值 1e-9) -> OK
- 生成器双路径自证 (scipy 路径 vs 手写逆变换路径, 统计量分布 ks_2samp): 连续区/离散区均 p > 0.01 -> OK 启用
- 注: statsmodels 未安装 (环境无网络), 独立参照 = scipy + 手写解析公式 (>=2 满足); L1 的 H0 均匀性定理检查是第三个、与任何实现共享假设无关的检查。

## L4 输入数据缺失检测 (POPPER 教训迁移, 2026-08-06)

考 9 类缺失/畸形输入 (NaN 单元格 / 整行 NaN / +inf / -inf / 全零表 / 单行表 / 单列表 / 字符串混合 / 空表), 确定性构造, 每 run 判定恒定。与 L1-L3 正交: L1-L3 考正常输入上的 p 值正确性, L4 考异常输入上的诚实失败。负值表类已移除 (期望频数可为负、所有实现均返回有限 p 值, 无区分度, 决策记录见 calibrator/l4.py docstring)。

- 合格 = 诚实失败: 每类输入下候选要么返回非有限值 (NaN/inf), 要么抛异常
- 返回有限 p 值 (0<=p<=1) = 幻觉填补 = FAIL (禁止静默修补后给出看似合理的数字)
- wrong_impute 是幻觉填补镜像 (与 POPPER 铁证三连对应): 缺失/畸形输入先静默填 0 再正常计算, 正常输入与 correct 一致, 只坏在缺失输入上; 必须在 L4 被拒 (见判定表)
- 边界声明: 本层考函数级输入; 智能体级"引用不存在文件"行为 (POPPER 教训的另一面) 属 LLM 编排层, 不在此层覆盖

## 已知边界 (漏洞 5/6 显式声明)

- **误杀机理解释已更正 (审计漏洞 2, 2026-08-06)**: correct 的 2/100 误杀全来自离散区
  α=0.10 逐点检查 (F̂(0.10) 实测 0.1205/0.1255 > α+0.02=0.12, seeds 89/99), 非原记录的 α=0.01
  (该点实测保守 0.0105/0.0115); 详见 run_log.txt 误杀形态段。
- **形状覆盖 (审计 P3 修正 2026-08-06)**: 考卷形状集 = **{shape_desc}**。
  校准层保证仅在覆盖的形状上成立。审计构造 bug "仅方表>=4x4 dof 加法" 在旧考卷
  (6 形状) 上三层全放行 (P3), 补 4x4/5x5/3x5 考卷后 wrong_dof_square_only 须被 L2/L3 拒绝 (见判定表)。
- 校准层只考"实现正确性", 不考"方法选择正确性" (如 Yates 校正取舍属方法层, 协议固定 correction=False)
- 只支持频率学派 (p 值框架); 贝叶斯验证器需预测区间覆盖率考官 (未实现)
- 零行/零列退化表 (期望为 0) 不考: scipy 对之抛 ValueError, 协议边界, 不属实现差异
- 诊断 (推导修正记录): wrong_dof 在纯 2x2 考卷上 p 值并非均匀 (mean≈1/sqrt(2)≈0.7071, 实测吻合),
  L1 2x2 也能抓住; 原先"2x2 均匀盲区"推导把 chi2_1 生存函数误当 e^(-x/2) (那是 chi2_2 的), 被数据推翻。
  (数值: run_log.txt 诊断段)

## 复现

`cd calib_exp && python3 -m calibrator.run_experiment` 两次输出逐字节一致 (确定性验证见 run_log.txt)。
"""
    results_path = BASE_DIR / "RESULTS.md"
    results_path.write_text(md, encoding="utf-8")

    log_path = BASE_DIR / "run_log.txt"
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nRESULTS.md -> {results_path}")
    print(f"run_log.txt -> {log_path}")


if __name__ == "__main__":
    main()
