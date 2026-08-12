"""test_schema — validate_spec 合法通过/非法拒绝 + spec_md5 确定性.

断言绑定行为: 合法规格原样返回, 非法规格 raise ValueError (带字段路径),
同规格 md5 两次一致 (schema.py:145-220).
"""
import copy
import json

import pytest

from spsl.schema import spec_md5, validate_spec

BASE = __import__("pathlib").Path(__file__).resolve().parents[1]
SPEC = json.loads((BASE / "specs" / "spec_pearson.json").read_text(encoding="utf-8"))


def test_valid_spec_passes_unchanged():
    """合法规格 (spec_pearson.json) 校验通过, 返回原样 dict."""
    result = validate_spec(SPEC)
    assert result is SPEC


def test_missing_required_field_rejected():
    """缺必填字段 (REQUIRED_TOP 之一) -> ValueError."""
    for field in ("name", "family", "inputs", "outputs", "statistic",
                  "null_dist", "function"):
        bad = copy.deepcopy(SPEC)
        del bad[field]
        with pytest.raises(ValueError):
            validate_spec(bad)


def test_wrong_type_rejected():
    """类型错 (name 非字符串) -> ValueError."""
    bad = copy.deepcopy(SPEC)
    bad["name"] = 42
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_bad_inputs_type_rejected():
    """inputs.type 不在登记输入类型 -> ValueError."""
    bad = copy.deepcopy(SPEC)
    bad["inputs"]["type"] = "no_such_input_type"
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_bad_outputs_type_rejected():
    """outputs.type 非 pvalue -> ValueError."""
    bad = copy.deepcopy(SPEC)
    bad["outputs"]["type"] = "effect_size"
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_bad_null_dist_generator_rejected():
    """null_dist.type 非支持生成器 -> ValueError."""
    bad = copy.deepcopy(SPEC)
    bad["null_dist"]["type"] = "mystery_sampler"
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_bad_row_p_sum_rejected():
    """null_dist.cont.row_p 概率和 != 1 -> ValueError."""
    bad = copy.deepcopy(SPEC)
    bad["null_dist"]["cont"]["row_p"] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_bad_reference_refs_rejected():
    """reference.refs 少于 2 个 -> ValueError (双参照要求)."""
    bad = copy.deepcopy(SPEC)
    bad["reference"]["refs"] = ["ref_hand_chi2"]
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_bad_l1_n_tables_rejected():
    """l1.n_tables < 100 -> ValueError."""
    bad = copy.deepcopy(SPEC)
    bad["l1"]["n_tables"] = 10
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_spec_md5_deterministic():
    """同规格两次 md5 相同; 改动规格 md5 改变."""
    assert spec_md5(SPEC) == spec_md5(SPEC)
    changed = copy.deepcopy(SPEC)
    changed["description"] = "改了一个字段"
    assert spec_md5(changed) != spec_md5(SPEC)


def test_spec_md5_matches_compiled_exam():
    """规格 md5 与已编译考卷内嵌 spec_md5 一致 (验签链完整性)."""
    exam = json.loads((BASE / "exams" / "exam_pearson_full.json")
                      .read_text(encoding="utf-8"))
    assert spec_md5(SPEC) == exam["spec_md5"]
