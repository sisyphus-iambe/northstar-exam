"""SPSL 规格 schema — 校验 + 指纹.

规格 JSON 字段 (出处: 规格设计):
  name        str   检验族唯一名 (必填)
  family      str   检验族分类 (必填, 如 pearson_chi2 / wilcoxon)
  description str   可选, 人类可读描述
  inputs      dict  {type, shape, value} — 输入结构 (必填)
  outputs     dict  {type, range} — 输出契约 (必填; 输出: p 值, range=[0,1])
  statistic   dict  {name, ...} — 统计量描述 (必填)
  null_dist   dict  {type, cont, disc} — H0 生成器配置 (必填, 驱动 L1 抽样)
  function    str   候选模块中待加载的函数名 (loader 解绑依据, 必填)
  reference   dict  可选 (完整四层考卷必需): 参照来源 + 自检协议
                     {refs: [>=2 个 golden.py 登记名], agree_tol: 双参照一致容差
                     (1e-9), self_check: [已知答案用例 {name, input, expected}],
                     self_check_tol: 已知答案核对容差}
  l1          dict  可选, L1 参数覆盖 (缺省值 = v1 L1 参数)
  constraint_type str  可选, 考官族 (注册表平台化): 缺省 "statistical" (L1-L4 协议);
                       demo_data / state_estimator 为注册表新族, 见 spsl/registry.py
"""
import hashlib
import json

from spsl.registry import (  # noqa: E402  (registry 模块级零 spsl 依赖, 无环)
    DEFAULT_CONSTRAINT_TYPE,
    EXAMINER_REGISTRY,
    input_types_for,
)

REQUIRED_TOP = ("name", "family", "inputs", "outputs", "statistic",
                "null_dist", "function")
SUPPORTED_GENERATORS = ("multinomial_independence", "iid_two_samples")
# INPUT_TYPES = registry 登记视图 (注册表平台化): statistical 族 = 原两型
INPUT_TYPES = input_types_for(DEFAULT_CONSTRAINT_TYPE)
DISTRIBUTIONS = ("uniform",)

# L1 缺省参数 (v1 同款)
DEFAULT_L1 = {
    "n_tables": 2000,
    "points": [0.01, 0.05, 0.10],
    "ks_slack": 0.02,            # l1.py:30
    "cont_point_slack": 0.02,    # l1.py:31
    "cont_mean_slack": 0.03,     # l1.py:32
    "disc_point_slack": 0.02,    # l1.py:33
    "disc_mean_max": 0.54,       # l1.py:34
    "runs": 3,                   # v1 默认 runs
    "seed_base": 20260807,       # v1 默认 seed_base
}


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"spsl spec validation failed [{path}]: {msg}")


def _check(cond: bool, path: str, msg: str) -> None:
    if not cond:
        raise _err(path, msg)


def _require_dict(x, path: str) -> None:
    _check(isinstance(x, dict), path, "must be a JSON object")


def _require_str(x, path: str) -> None:
    _check(isinstance(x, str) and bool(x), path, "must be a non-empty string")


def _validate_zone(z, gtype: str, path: str) -> None:
    """null_dist.cont/disc 结构校验 (按生成器类型)."""
    if gtype == "multinomial_independence":
        _check(isinstance(z.get("n"), int) and z["n"] > 0,
               f"{path}.n", "must be a positive integer")
        for key in ("row_p", "col_p"):
            ps = z.get(key)
            ok = (isinstance(ps, list) and len(ps) >= 2
                  and all(isinstance(x, (int, float)) and x > 0 for x in ps))
            _check(ok, f"{path}.{key}",
                   "must be a non-empty list of positive probabilities (length >= 2; all-positive cell probabilities -> all-positive expected counts, protocol boundary)")
            _check(abs(sum(ps) - 1.0) <= 1e-6, f"{path}.{key}",
                   f"probabilities must sum to 1 (got {sum(ps)})")
    elif gtype == "iid_two_samples":
        for key in ("n1", "n2"):
            _check(isinstance(z.get(key), int) and z[key] > 0,
                   f"{path}.{key}", "must be a positive integer")
        _check(z.get("distribution") in DISTRIBUTIONS,
               f"{path}.distribution", f"supported distributions ∈ {DISTRIBUTIONS}")
        lo, hi = z.get("lo"), z.get("hi")
        _check(isinstance(lo, (int, float)) and isinstance(hi, (int, float))
               and lo < hi, f"{path}.lo/hi", "must be numeric with lo < hi")


def _validate_l1(l1) -> None:
    _require_dict(l1, "l1")
    if "n_tables" in l1:
        _check(isinstance(l1["n_tables"], int) and l1["n_tables"] >= 100,
               "l1.n_tables", ">= 100")
    if "points" in l1:
        ps = l1["points"]
        _check(isinstance(ps, list) and len(ps) >= 1
               and all(isinstance(x, (int, float)) and 0 < x < 1 for x in ps),
               "l1.points", "list of alpha values, all in (0,1)")
    for key in ("ks_slack", "cont_point_slack", "cont_mean_slack",
                "disc_point_slack"):
        if key in l1:
            _check(isinstance(l1[key], (int, float)) and l1[key] >= 0,
                   f"l1.{key}", ">= 0")
    if "disc_mean_max" in l1:
        _check(isinstance(l1["disc_mean_max"], (int, float))
               and 0.5 < l1["disc_mean_max"] <= 1,
               "l1.disc_mean_max", "∈ (0.5, 1]")
    if "runs" in l1:
        _check(isinstance(l1["runs"], int) and l1["runs"] >= 1, "l1.runs", ">= 1")
    if "seed_base" in l1:
        _check(isinstance(l1["seed_base"], int) and l1["seed_base"] >= 0,
               "l1.seed_base", "non-negative integer")


def _validate_reference(ref) -> None:
    """reference 结构校验 (双参照 + 自检协议)."""
    _require_dict(ref, "reference")
    refs = ref.get("refs")
    _check(isinstance(refs, list) and len(refs) >= 2
           and all(isinstance(x, str) and bool(x) for x in refs),
           "reference.refs", "must register >= 2 reference implementation names (dual references; v1 baseline)")
    _check(len(set(refs)) == len(refs), "reference.refs", "reference names must be unique")
    for key, path in (("agree_tol", "reference.agree_tol"),
                      ("self_check_tol", "reference.self_check_tol")):
        v = ref.get(key)
        _check(isinstance(v, (int, float)) and 0 < v <= 1e-3, path,
               "must be a positive tolerance in (0, 1e-3] (default 1e-9)")
    sc = ref.get("self_check")
    _check(isinstance(sc, list) and len(sc) >= 1, "reference.self_check",
           "must have >= 1 known-answer case (self-check protocol: verify reference output on known-answer inputs)")
    for i, case in enumerate(sc):
        _require_dict(case, f"reference.self_check[{i}]")
        _require_str(case.get("name"), f"reference.self_check[{i}].name")
        _require_dict(case.get("input"), f"reference.self_check[{i}].input")
        itype = case["input"].get("type")
        _check(itype in INPUT_TYPES, f"reference.self_check[{i}].input.type",
               f"known-answer input type must be in {INPUT_TYPES}")
        _check(isinstance(case.get("expected"), (int, float))
               and 0 <= case["expected"] <= 1,
               f"reference.self_check[{i}].expected", "must be in [0,1]")


def validate_spec(raw: dict) -> dict:
    """校验规格 JSON, 通过则返回原样 dict; 失败 raise ValueError (带字段路径).

    注册表平台化: constraint_type 缺省
    "statistical" (向后兼容 —— 现有 spec/exam 无该字段照常通过, spec_md5 不变,
    已编译考卷不重编译不挂); 非 statistical 族跳过 statistical 专属强校验
    (inputs/outputs/statistic/null_dist/function 对其不适用, 由各族执行器校验).
    """
    _require_dict(raw, "spec")
    ct = raw.get("constraint_type", DEFAULT_CONSTRAINT_TYPE)
    _require_str(ct, "constraint_type")
    _check(ct in EXAMINER_REGISTRY, "constraint_type",
           f"unregistered constraint_type ∈ {sorted(EXAMINER_REGISTRY)}")
    _require_str(raw.get("name"), "name")
    _require_str(raw.get("family"), "family")

    if ct != DEFAULT_CONSTRAINT_TYPE:
        # 非 statistical 族: 规格即考卷, statistical 专属字段
        # (inputs/outputs/statistic/null_dist/function) 不强制, 族专属字段由
        # 执行器校验 (demo_data: root/seed/阈值/注入数; state_estimator:
        # functions/seed/tol)
        return raw

    for f in REQUIRED_TOP:
        _check(f in raw, f, "missing required field (spec requires: "
                            "inputs/outputs/statistic/null_dist/function)")
    _require_str(raw["function"], "function")

    inp = raw["inputs"]
    _require_dict(inp, "inputs")
    _check(inp.get("type") in INPUT_TYPES, "inputs.type",
           f"supported input types ∈ {INPUT_TYPES}")
    _require_str(inp.get("shape"), "inputs.shape")
    _require_str(inp.get("value"), "inputs.value")

    out = raw["outputs"]
    _require_dict(out, "outputs")
    _check(out.get("type") == "pvalue", "outputs.type", "output must be a p-value")
    rng = out.get("range")
    _check(isinstance(rng, list) and len(rng) == 2
           and all(isinstance(x, (int, float)) for x in rng),
           "outputs.range", "must be two numbers [lo, hi]")
    _check(rng[0] <= rng[1], "outputs.range", "lo must be <= hi")

    stat = raw["statistic"]
    _require_dict(stat, "statistic")
    _require_str(stat.get("name"), "statistic.name")

    nd = raw["null_dist"]
    _require_dict(nd, "null_dist")
    _check(nd.get("type") in SUPPORTED_GENERATORS, "null_dist.type",
           f"supported generators ∈ {SUPPORTED_GENERATORS}")
    for zone in ("cont", "disc"):
        _require_dict(nd.get(zone), f"null_dist.{zone}")
        _validate_zone(nd[zone], nd["type"], f"null_dist.{zone}")

    if "l1" in raw:
        _validate_l1(raw["l1"])
    if "reference" in raw:
        _validate_reference(raw["reference"])
    return raw


def merged_l1(spec: dict) -> dict:
    """规格 l1 覆盖 + v1 缺省值合并."""
    l1 = dict(DEFAULT_L1)
    l1.update(spec.get("l1", {}))
    return l1


def spec_md5(spec: dict) -> str:
    """输入规格 md5: 规范化序列化 (ensure_ascii=False, sort_keys=True)."""
    return hashlib.md5(
        json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
