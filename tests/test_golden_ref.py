"""test_golden_ref — 金标准参照注册表 (golden.py): 取用可调用, 已知答案核对.

已知答案: 恒等 2x2 表 [[1,1],[1,1]] 卡方 p 值解析值 = 1.0 (考试规格
reference.self_check 同款); 双参照 (手写 vs scipy) 偏差 < 1e-6
(任务书判据; 考试层自检容差为 1e-9).
"""
import pytest

from spsl.golden import REGISTRY, get_ref

KNOWN_IDENTITY = {"type": "contingency_table", "table": [[1, 1], [1, 1]]}
STANDARD_TABLE = {"type": "contingency_table", "table": [[10, 20], [30, 40]]}


def test_get_ref_returns_callable():
    """登记名取回可调用实现 (pearson 双参照)."""
    for name in ("ref_hand_chi2", "ref_scipy_chi2"):
        impl = get_ref(name)
        assert callable(impl)


def test_unknown_ref_raises():
    """未登记名 -> ValueError."""
    with pytest.raises(ValueError):
        get_ref("no_such_ref_impl")


def test_known_answer_identity_2x2():
    """已知答案: 恒等表 p 值 = 1.0 (参照自检 1e-9 同款用例)."""
    hand = float(get_ref("ref_hand_chi2")(KNOWN_IDENTITY))
    scipy = float(get_ref("ref_scipy_chi2")(KNOWN_IDENTITY))
    assert abs(hand - 1.0) < 1e-6
    assert abs(scipy - 1.0) < 1e-6


def test_hand_scipy_agree_on_standard_table():
    """标准表上双参照偏差 < 1e-6 (独立代码路径互证)."""
    hand = float(get_ref("ref_hand_chi2")(STANDARD_TABLE))
    scipy = float(get_ref("ref_scipy_chi2")(STANDARD_TABLE))
    assert abs(hand - scipy) < 1e-6


def test_registry_has_four_entries():
    """登记表含 pearson 双参照 + wilcoxon 双参照."""
    assert set(REGISTRY) == {
        "ref_hand_chi2", "ref_scipy_chi2",
        "ref_hand_ranksum", "ref_scipy_ranksums",
    }
