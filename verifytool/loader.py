"""动态加载用户验证器 + 契约检查 (规格 §2.1; 北极星 v2 D1 按规格解绑).

输入: 一个 .py 文件 + 目标函数名 func_name (默认 "chi2_pvalue" = v1 旧契约,
向后兼容; 新函数族名由 spsl 编译考卷 JSON 的 spec.function 传入).
加载失败/契约不符 -> 友好报错退出 (列出找到的函数清单, 提示契约), 不裸崩.
"""
import importlib.util
from pathlib import Path


def load_validator(path: str | Path, func_name: str = "chi2_pvalue") -> tuple:
    """加载验证器. 返回 (func_name 对应的可调用对象, 模块名).

    func_name: 按考卷规格加载的目标函数名 (spsl 考卷 JSON 的 spec.function,
    北极星 v2 规格解绑); 默认 "chi2_pvalue" = v1 旧契约, 向后兼容.
    失败时 raise RuntimeError, 消息面向用户 (含诊断信息).
    """
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"file not found: {p}")
    if p.suffix != ".py":
        raise RuntimeError(f"input must be a .py file (got: {p.suffix or 'no suffix'})")

    mod_name = p.stem
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to parse file: {p} (spec_from_file_location returned None)")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise RuntimeError(
            f"module load failed (executing {p} top-level code raised): {type(exc).__name__}: {exc}") from exc

    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        found = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
        raise RuntimeError(
            f"module {mod_name} does not expose a callable {func_name}."
            f"contract: {func_name}(...) -> float."
            f"callables found: {found if found else '(none)'}"
        )

    if func_name == "chi2_pvalue":
        _smoke_check(fn, mod_name)  # v1 旧契约冒烟 (2x2 表); 新函数族冒烟由
                                    # spsl 侧按考卷规格执行 (run_l1 全表即冒烟)
    return fn, mod_name


def _smoke_check(fn, mod_name: str) -> None:
    """契约冒烟: 对一张合法 2x2 表调一次, 必须是可转 float 的有限值 (0<=p<=1).

    早失败带友好消息; 真正的行为判定交给四层考卷.
    """
    import numpy as np

    table = np.array([[10, 20], [30, 40]], dtype=float)
    try:
        p = float(fn(table))
    except Exception as exc:
        raise RuntimeError(
            f"{mod_name}.chi2_pvalue raised on legal input [[10,20],[30,40]]: "
            f"{type(exc).__name__}: {exc}") from exc
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"{mod_name}.chi2_pvalue returned {p!r}, outside [0,1] (contract: p must be in [0,1])")
