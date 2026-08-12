# verifytool — 验证器质检 CLI (MVP + 插件式扩展)

把"校准层四层考卷"装成命令行工具: 喂一个统计验证器 (一个 .py 文件), 吐一份
校准报告 (HTML + JSON)。只考"实现正确性", 不考方法选择, 不联网, 零 pip, ¥0。

插件式扩展 (产品②模板库 / ③自动考卷构造器 / ④剪枝调度, 2026-08-07 规格):
一个程序 + 四个子命令, 无子命令时行为与 MVP 完全一致。

规格: `SPEC_MVP工具形态_2026-08-07.md` (①主命令, 权威) +
`SPEC_插件扩展_2026-08-07.md` (②③④, 权威)。

## 用法

```bash
cd calib_exp   # 校准层包所在根目录
python3 -m verifytool <验证器.py> [--runs N] [--out PATH]
```

- `<验证器.py>`: 必须暴露 `chi2_pvalue(observed) -> float`
  (任意 r×c 计数表, 边缘全正 → p 值)。
- `--runs N`: L1 抽样 run 数, 默认 3 (seeds = 20260807/20260808/20260909)。
  层判定按 run 数归一 (同实验 05 的 100-runs 协议: 拒 <=10% = PASS,
  >=90% = REJECT, 其余 MIXED)。
- `--out PATH`: HTML 输出路径, 默认 `cwd/verifytool_report_<验证器名>.html`;
  JSON = 同名 `.json`。

输出两份:
- **HTML 报告**: ① 四层判定表 (L1 H0 均匀性 / L2 参照对拍 / L3 边界泛化 /
  L4 输入缺失诚实失败) + ② 能力地图 (9 模板考卷 × L2-L4) + ③ 盲点清单
  (L4 幻觉填补 / L2-L3 对拍偏差 top5 / 覆盖边界) + ④ 背景与诚实标注。
- **JSON**: 全量结构化结果, 含 md5 双字段 (bootstrap_monotone 同款算法),
  主终端可独立复算验真。

## 插件扩展: ②③④ 子命令

```bash
python3 -m verifytool templates [list|info <名>]                 # ② 模板库
python3 -m verifytool exam <模板名> [--runs N] [--out PATH]      # ② 考任意模板
python3 -m verifytool construct --rule <rule.json> <数据csv> [--seed N] [--out PATH]  # ③
python3 -m verifytool prune [--world synthetic|--cands <p.json>] [--budget B] [--n-seeds N] [--out PATH]  # ④
```

**② templates / exam**: 显式注册表 22 条目 (programs/ 9 考卷模板 + 5 错误对照 +
programs_real/ 8 演示样), 统一契约 `chi2_pvalue(observed) -> float`。
`exam` = 加载模板走主管道 (四层+能力地图+盲点), 输出 `exam_<模板名>.html/.json`。
口径提示 (写进每份输出): exam 考的是"单模板实现正确性"; 模板库自校准档案
(calibrator/template_exam.py, 9 域 L1-L4 慢速考卷 100 seeds, 9 模板全 FAIL —
F̂(0.10) 越 0.12 阈值的小样本波动) 是另一套口径, 不可互相替代。

**③ construct (自动考卷构造器, 离线版)**: 规则 JSON (R1 同 schema:
`{"logic":"AND|OR","conditions":[{"stat":"a|prr|chi2","year":2024|2025,
"op":">="|"<=","value":数值}]}`) + 数据 CSV (2x2 四格表:
`pair,year,a,b,c,d,label`, pair 格式 `药|事件`, a/b/c/d = 该药该事件/该药其他/
其他药该事件/其余) → 特征 (judge1 同款公式) → 确定性划分 (seed 默认 100,
同实验 9a 口径; <400 对则全量评估) → TP/FP/TN/FN + accuracy。
复现基准: R1 规则 (a2025>=3 AND prr2025>=2) 在演示数据
(`demo_data/r1_faers_seed100.csv`, FAERS S24 全池 12699 对) 上 held-out
accuracy = 0.99375 (与实验 9a 逐位一致)。不含 LLM 生成规则 (API 成本/非确定) —
规则由用户提供, 构造器是确定性执行器。

**④ prune (剪枝调度)**: 三臂 FULL (全池首检+均匀随机加投) / MAP_CRIT (可验证
首检+递归临界救援) / ONESHOT (首检+单轮加投), Fisher 组合 (combine_k) + BH
判定 (α=0.05, m=候选数)。合成世界 (默认, v7c 同款: 200 候选 = 70 超弱 + 70
中弱 + 30 盲点 M7 + 30 盲点 M6, seeds = 20260806+i) 或 `--cands <p.json>`
(`{"p_first":[...], "p_slots":[[...],...], "truth":[...可选]}`)。
纯函数逐行复制自 design_prune_v7c.py (v6/v7 锚点红线, 逐位一致)。输出三臂
真检出/fp 对比 + 调度轨迹 (JSON+HTML)。演示: `--budget 200` 预算稀缺条件,
MAP_CRIT 真检出 ≥ FULL (50 seeds: 27.16 vs 20.36, +6.80, 零 fp 代价)。

## 演示样例 (实验 05 真实程序)

## 三个演示样例 (实验 05 真实程序)

```bash
cd calib_exp
python3 -m verifytool experiments/qualcheck_realworld/programs_real/scipy_direct.py --runs 3
python3 -m verifytool experiments/qualcheck_realworld/programs_real/wrong_fillna.py --runs 3
python3 -m verifytool experiments/qualcheck_realworld/programs_real/wrong_dof.py --runs 3
```

预期 (与实验 05 results.json 判定表对照, 详见状态文件 §17):

| 验证器 | L1 | L2 | L3 | L4 | 总判定 |
|--------|----|----|----|----|--------|
| scipy_direct (教科书直连) | PASS | PASS | PASS | REJECT | REJECT |
| wrong_fillna (静默填补) | PASS | PASS | PASS | REJECT | REJECT |
| wrong_dof (自由度公式错) | REJECT | REJECT | REJECT | REJECT | REJECT |

scipy_direct 的 L4 REJECT 是"真抓"不是误杀: scipy 1.18 的 chi2_contingency
对 1x5/5x1 单行/单列退化表 (dof=0) 静默返回有限 p=1.0, 按校准层"畸形输入须
诚实失败"的标准判为幻觉填补 (实验 05 头号发现)。

## 运行时构成 (全部毫秒-秒级, 单验证器 < 2 分钟)

1. **动态加载** (`loader.py`): importlib 加载用户 .py, 查 `chi2_pvalue`;
   缺失/不可调用 → 友好报错并列出找到的函数 (契约: `chi2_pvalue(observed) -> float`)。
2. **四层考卷** (`run_verify.py`): 调 `calibrator/calibrate.py` 的 `calibrate(func, seed)`
   — L1 随机 4000 表 × runs 数 + L2-L4 固定考卷。
3. **能力地图**: 9 模板 (chi2_rxc / ratio / fisher / barnard / trend / slope /
   replicate / stratified / strat_bidir) 的固定考卷表 × L2/L3 对拍 —
   参照 = `calibrator/reference.py` 的 ref_hand + ref_scipy (先自检 1e-9),
   容差 1e-6。只考卡方实现正确性, 不考方法选择。
4. **盲点清单**: 零 LLM, 全自动 — L4 哪几类畸形输入被静默填补 /
   L2/L3 偏差最大的 5 张表 (形状 / n / 偏差) / 考卷覆盖不到的形状如实列出。

## 诚实标注 (MVP 边界 + 插件扩展边界)

- MVP = 工具形态第一版, 覆盖"验证器质检"主链路。9 模板的 L1 慢速考卷
  (barnard 类) 不在单次运行内 — 那是模板库自校准档案 (`calibrator/template_exam.py`),
  本工具单次运行只有通用 L1 (3x4/2x3 域) + 能力地图固定考卷, 如实说明。
- 只考实现正确性, 不评判方法选择 (用卡方还是费雪); 频率学派 p 值框架;
  零行/零列退化表协议排除。
- 单验证器报告 = 一次采样快照, 非统计保证 (与合成自证口径一致:
  L1 抽样区对正确实现有 ~2% 随机误杀率, 多 run 判定降低波动)。
- 工具校准层自己的校准记录 (写进每份报告): 合成自证灵敏度 1.000/误杀 0.020
  (2026-08-06) + 实验 05 真实演示灵敏度 1.000/误杀 0.020 (2026-08-07)。
- **插件扩展 = 组装既有已验证组件, 不产生新实验结果**; 所有数字引用原实验出处
  (9a: bootstrap_monotone/results.json; v7c: design_prune/results_v7c.json;
  4h: finance_weaksig/results.json)。
- ③构造器不含 LLM 生成 (API 成本/非确定, 规则 JSON 由用户提供)。
- ④不含真实数据场景 (finance_weaksig 数据依赖未接, 只接合成世界 + 用户候选 p);
  4h 结论"部分成立"照录 (方向同向, 零 fp 未复制: FULL 71真/4fp vs MAP_CRIT
  73真/5fp, 单实例无显著性检验)。
- ⑤平台端到端集成仍未做 (研究资产, 不在本工具内)。

## 文件清单

```
verifytool/
├── __init__.py      # 包标识 + 版本
├── __main__.py      # CLI 入口 (子命令分发; 无子命令 → 原主命令, 行为不变)
├── loader.py        # 动态加载 + 契约检查 (友好报错)
├── run_verify.py    # 四层考卷 + 能力地图 + 盲点清单编排 (只 import calibrator/)
├── report.py        # HTML 渲染 + JSON 落盘 (md5 双字段); 插件扩展报告模板
├── templates.py     # ②模板注册表 (22 条目) + list/info/exam 编排
├── constructor.py   # ③规则执行 + 评估 (离线, 不 import 实验脚本)
├── prune.py         # ④三臂调度 + Fisher 合并 + BH (v7c 纯函数逐位一致)
├── demo_data/       # ③演示数据: r1_rule.json + r1_faers_seed100.csv + 生成脚本
├── README.md        # 本文件
├── SPEC_MVP工具形态_2026-08-07.md   # 规格 ① (权威)
└── SPEC_插件扩展_2026-08-07.md      # 规格 ②③④ (权威)
```

依赖: 只 import `calibrator/` 现有模块 (calibrate / l1-l4 / reference / generator /
template_exam)。`template_exam/` 实验数据只读引用, 不作为运行时依赖。
不改 calibrator/ programs/ 任何现有文件。

## 确定性

全部固定种子 (L1: 20260807+i; L2/L3: 考卷内建固定种子), 判定/诊断/计数
重跑逐字节一致。唯一非确定性数据 = `elapsed_seconds` (运行耗时) 及由它派生的
两个 md5 字符串 (同实验 05 口径)。
