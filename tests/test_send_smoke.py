"""test_send_smoke — spsl.run 端到端冒烟: 正确考生 ACCEPT, 错误考生 REJECT.

考卷: exams/exam_pearson_full.json (仓库内唯一完整四层 pearson 考卷;
任务书所指 exam_pearson_demo.json 在仓库中不存在, 以 full 版替代).
考生: tests/candidates/correct_chi2.py (scipy + 诚实失败预检) 与
tests/candidates/wrong_chi2.py (恒返 0.5).

断言绑定行为 (run.py:339-362 main / _run_statistical):
  - 正确考生: 总判定 ACCEPT, 四层全 PASS
  - 错误考生: 总判定 REJECT
  - 合法候选加载失败 (如 smoke check 拒 NaN) -> 退出码 2
"""
import json
from pathlib import Path

import pytest

from spsl.run import main

BASE = Path(__file__).resolve().parents[1]
EXAM = BASE / "exams" / "exam_pearson_full.json"
CORRECT = BASE / "tests" / "candidates" / "correct_chi2.py"
WRONG = BASE / "tests" / "candidates" / "wrong_chi2.py"
NAN = BASE / "candidates" / "nan_always.py"


def _run(validator, out_path):
    rc = main([str(EXAM), str(validator), "--out", str(out_path)])
    assert rc == 0, f"spsl.run 退出码 {rc}"
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_correct_candidate_accepted(tmp_path):
    """scipy + 诚实失败预检的正确考生 -> ACCEPT, L1-L4 全 PASS."""
    out = tmp_path / "verdict_correct.json"
    verdict = _run(CORRECT, out)
    assert verdict["layer"] == "FULL"
    assert verdict["four_layers"]["total_verdict"] == "ACCEPT"
    assert verdict["four_layers"]["reject_runs"] == 0
    for layer in ("L1", "L2", "L3", "L4"):
        assert verdict["four_layers"]["layers"][layer]["verdict"] == "PASS", layer


def test_wrong_candidate_rejected(tmp_path):
    """恒返 0.5 的错误考生 -> REJECT."""
    out = tmp_path / "verdict_wrong.json"
    verdict = _run(WRONG, out)
    assert verdict["four_layers"]["total_verdict"] == "REJECT"
    assert verdict["four_layers"]["reject_runs"] == verdict["four_layers"]["n_runs"]
    for layer in ("L1", "L2", "L3", "L4"):
        assert verdict["four_layers"]["layers"][layer]["verdict"] == "REJECT", layer


def test_nan_candidate_fails_load():
    """恒返 NaN 的候选在加载冒烟被拒 -> 退出码 2 (run.py:267-276)."""
    rc = main([str(EXAM), str(NAN), "--out", str(Path("/tmp/unused_nan.json"))])
    assert rc == 2
