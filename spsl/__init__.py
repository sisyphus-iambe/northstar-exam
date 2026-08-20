"""SPSL — 规格驱动的统计实现质检.

规格 JSON (~15 行, 字段 inputs/outputs/statistic/null_dist/function
+reference: 参照来源 + 自检协议)
-> 编译四层考卷 JSON (envelope: L1 H0 模拟校准 + L2 双参照对拍 +
L3 边界泛化 + L4 畸形输入诚实失败) -> 对任意候选验证器执行 (run).

能力出处: 规格编译 (SPSL).
schema/compile_l1/run_l1 (v0.1.0).  compile_l2/l3/l4/envelope/golden/run
(v0.2.0, 完整四层).  shape_judger/selfevolve/profile (v0.3.0, 自进化判定器
+ 记忆画像: 增量学习 + 入口防污染 + 外部裁决门 + 三层时标记忆).
"""

VERSION = "0.3.0"
