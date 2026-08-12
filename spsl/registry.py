"""SPSL 考官注册表 — constraint_type -> (compile_fn, run_fn) (注册表平台化).

出处: 注册表平台化方案 (考官注册表);
注册模式: 键 -> 实现查表 (编译/执行成对).

约定:
  - statistical 族 = L1-L4 四层协议 (envelope/compile_l1-l4/golden), 不删不迁;
    compile = spsl.envelope.build_full_exam, run 由 spsl.run 原路径承接 (零行为变化).
  - 非 statistical 族: 规格即考卷 (compile = 恒等), run = 各族执行器, 惰性加载
    (本模块模块级只依赖标准库, 避免 schema <-> envelope 循环引用).
"""
import importlib

DEFAULT_CONSTRAINT_TYPE = "statistical"

# constraint_type -> (执行器模块, 该族 INPUT_TYPES 声明; statistical 族原两型)
_EXAMINERS = {
    "demo_data": ("spsl.examiners.demo_data", ()),
    "state_estimator": ("spsl.examiners.state_estimator", ()),
}


def _load(ct: str):
    mod_name, _ = _EXAMINERS[ct]
    return importlib.import_module(mod_name)


def _compile_statistical(spec: dict) -> dict:
    """compile_fn (statistical): 规格 -> 完整四层考卷 (spsl.envelope 原逻辑)."""
    from spsl.envelope import build_full_exam

    return build_full_exam(spec)


def _run_statistical(*_args, **_kwargs):
    raise AssertionError(
        "the run for the statistical family is handled by spsl.run's original path, not registered here")


def _compile_identity(spec: dict) -> dict:
    """compile_fn (非 statistical 族): 规格即考卷, 原样返回."""
    return spec


def _run_demo_data(spec: dict, validator_path, out_path):
    return _load("demo_data").run_spec(spec, validator_path, out_path)


def _run_state_estimator(spec: dict, validator_path, out_path):
    return _load("state_estimator").run_spec(spec, validator_path, out_path)


EXAMINER_REGISTRY = {
    DEFAULT_CONSTRAINT_TYPE: (_compile_statistical, _run_statistical),
    "demo_data": (_compile_identity, _run_demo_data),
    "state_estimator": (_compile_identity, _run_state_estimator),
}


def input_types_for(ct: str) -> tuple:
    """INPUT_TYPES 登记视图: statistical 族保持原两型, 其余族按登记声明."""
    if ct == DEFAULT_CONSTRAINT_TYPE:
        return ("contingency_table", "two_samples")
    return _EXAMINERS[ct][1]
