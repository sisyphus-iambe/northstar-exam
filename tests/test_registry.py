"""test_registry — 考官注册表 (registry.py): statistical 族可查, 分发可调用."""
from spsl.registry import (
    DEFAULT_CONSTRAINT_TYPE,
    EXAMINER_REGISTRY,
    input_types_for,
)


def test_registry_contains_statistical_family():
    """statistical 族登记在册, 且为缺省 constraint_type."""
    assert "statistical" in EXAMINER_REGISTRY
    assert DEFAULT_CONSTRAINT_TYPE == "statistical"


def test_registry_contains_platform_families():
    """阶段1 平台化登记族存在 (demo_data / state_estimator)."""
    assert "demo_data" in EXAMINER_REGISTRY
    assert "state_estimator" in EXAMINER_REGISTRY


def test_statistical_dispatch_returns_callables():
    """查表分发 (compile_fn, run_fn) 两个均可调用."""
    compile_fn, run_fn = EXAMINER_REGISTRY[DEFAULT_CONSTRAINT_TYPE]
    assert callable(compile_fn)
    assert callable(run_fn)


def test_non_statistical_dispatch_returns_callables():
    """非 statistical 族分发同样返回可调用对."""
    for ct in ("demo_data", "state_estimator"):
        compile_fn, run_fn = EXAMINER_REGISTRY[ct]
        assert callable(compile_fn), ct
        assert callable(run_fn), ct


def test_input_types_view():
    """statistical 族输入类型登记视图保持原两型."""
    assert input_types_for("statistical") == ("contingency_table", "two_samples")
