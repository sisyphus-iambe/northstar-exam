"""test_determinism — 同命令连跑两次, 判定 JSON 剥离非确定字段后逐字节一致.

非确定字段 (run.py:23-24 声明): elapsed_seconds 及派生的 payload_md5/
self_md5; command 为 sys.argv 快照. 剥离这 4 个字段后, 规范化序列化
(ensure_ascii=False, sort_keys=True) 逐字节一致; four_layers 结构深层一致.
"""
import json
from pathlib import Path

from spsl.run import main

BASE = Path(__file__).resolve().parents[1]
EXAM = BASE / "exams" / "exam_pearson_full.json"
CORRECT = BASE / "tests" / "candidates" / "correct_chi2.py"

# 每次运行必然变化的派生字段 (elapsed_seconds 及其 md5 派生, command 快照)
_NONDET = ("elapsed_seconds", "command", "payload_md5", "self_md5")


def _run_once(out_path):
    rc = main([str(EXAM), str(CORRECT), "--out", str(out_path)])
    assert rc == 0
    return json.loads(out_path.read_text(encoding="utf-8"))


def _strip_and_normalize(payload):
    for key in _NONDET:
        payload.pop(key, None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def test_two_runs_byte_identical_after_stripping(tmp_path):
    run1 = _run_once(tmp_path / "det1.json")
    run2 = _run_once(tmp_path / "det2.json")
    assert _strip_and_normalize(run1) == _strip_and_normalize(run2)


def test_four_layers_deep_equal_across_runs(tmp_path):
    """four_layers 逐 run 深层一致 (L1 抽样种子固定 -> 确定性)."""
    run1 = _run_once(tmp_path / "det3.json")
    run2 = _run_once(tmp_path / "det4.json")
    a = json.dumps(run1["four_layers"], sort_keys=True,
                   allow_nan=True, separators=(",", ":"))
    b = json.dumps(run2["four_layers"], sort_keys=True,
                   allow_nan=True, separators=(",", ":"))
    assert a == b


def test_elapsed_seconds_is_the_only_difference_source(tmp_path):
    """剥离后字段集外仅 elapsed_seconds 可变 (两个原始 payload 键集相同)."""
    run1 = _run_once(tmp_path / "det5.json")
    run2 = _run_once(tmp_path / "det6.json")
    assert set(run1) == set(run2)
    assert "elapsed_seconds" in run1
