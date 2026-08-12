"""verifytool — 验证器质检 CLI (MVP).

输入一个 .py 验证器 (暴露 chi2_pvalue(observed) -> float),
输出四层考卷校准报告: HTML (四层判定 + 能力地图 + 盲点清单) + JSON (md5 双字段).
规格: SPEC_MVP工具形态_2026-08-07.md. 纯组装 calibrator/ 现有模块, 零新发明.
"""

VERSION = "0.1.0"
