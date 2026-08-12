"""恒等式知识库 v1 (北极星 v3 级 1, D1) — 无参照验证的第三类锚.

恒等式 = 同一实现的多个入口必须满足的数学自洽性 (零模拟零参照零 LLM):
  - 与 L2 参照对拍正交: 不依赖任何参照实现, 抓"方向语义"类静默错误
    (外部背书: scipy/scipy#19872 + PR #20765 官方回归测试断言同一恒等式,
    该 bug 类连 scipy 都带过上线; scanpy#698 两样本秩和 ties 未校正)
  - 判据 (round9_A §③ 入库硬条款, 伪公理审查防"数学上像公理其实不是"):
    1. swap_inputs 必须 true (输入交换配对, 非同参配对)
    2. (CDF,SF) 方向配对缺一不可 — 伪公理 #3 (CDF,CDF) 配对实测 98/100 违规,
       E1v1 同参配对 495/500 违规 -> 恒等式不能靠拍脑袋写
    3. 每条必须有实测档案: >=3 真实参照 0 违规 @1e-6 (参照零误杀)
    4. 离散修正三要素: ①输入交换配对 ②ties/cont 分区采样 ③异常/非有限计违规

用法 (判定层, D2 run_inv.py 调用):
    check_invariant(name, fn_a, fn_b, rng, n, R, condition, tol)
    -> {invariant, condition, viol, R, worst_dev, nonfinite, crashed}
采样档位由规格 invariants[].condition 引用 (SAMPLERS 注册表);
fn_a/fn_b = 规格 pair 解析后的候选入口函数 (transpose 自配对: fn_b 忽略).
"""
import json

import numpy as np

__all__ = ["INVARIANTS", "SAMPLERS", "check_invariant", "list_invariants"]

TOL_DEFAULT = 1e-6


# ---------------------------------------------------------------------------
# 采样档位 (生成器注册表, 规格 condition 引用; E1v2 gen_ties/gen_cont 同款)
# ---------------------------------------------------------------------------

def _gen_ties_fractional_w(rng, n):
    """重度 ties: 整数网格 0-4 -> 分数秩 (E1v2 gen_ties)."""
    return (rng.integers(0, 5, size=n).astype(float),
            rng.integers(0, 5, size=n).astype(float))


def _gen_cont_stdnormal(rng, n):
    """连续: 标准正态 (E1v2 gen_cont)."""
    return rng.standard_normal(n), rng.standard_normal(n)


def _gen_ties_mild(rng, n):
    """轻度 ties: 整数网格 0-9 (标定扩展档, D3)."""
    return (rng.integers(0, 10, size=n).astype(float),
            rng.integers(0, 10, size=n).astype(float))


def _gen_chi2_table(rng, n):
    """chi2 计数表 4x4: 单元计数 1-7 (转置对称用, 无空表)."""
    return (rng.integers(1, 8, size=(4, 4)).astype(float),)


def _non_zero_d(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return d[d != 0.0]


def _w_obs_paired(x, y):
    """观测正秩和 (zero 剔除 + averaged ranks, 可非整数). d = x - y.
    (exp5/run_d5.py:52-56 同款, 条件化采样谓词用)"""
    v = _non_zero_d(x, y)
    if len(v) == 0:
        return 0.0
    order = np.argsort(np.abs(v), kind="stable")
    n = len(v)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and np.abs(v[order[j + 1]]) == np.abs(v[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return float(ranks[v > 0].sum())


def _accept_w_noninteger(x, y):
    """条件化谓词: w 非整数 (半整数) — 原子形状 (#19872 修复前截断 bug)
    的检测面 (D5: 无条件化 240/500 vs 条件化 210/210, 预注册判据: 条件化 >=90%)."""
    return _w_obs_paired(x, y) != int(_w_obs_paired(x, y))


def _w_obs_ranksum(x, y):
    """两样本观测 x 秩和 (averaged ranks, ties 时非整数).
    (exp5 cand_atomic_exam.py:38-54 _w_obs 同款算法)"""
    v = np.concatenate([np.asarray(x, dtype=float),
                        np.asarray(y, dtype=float)])
    n1 = len(x)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    i = 0
    n = len(v)
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return float(ranks[:n1].sum())


def _accept_w_noninteger_ranksum(x, y):
    """两样本条件化谓词: x 秩和 w 非整数 (半整数) — 与配对族同一理由
    (半整数 w 上原子 bug 检测面最大, D3 双口径 47.4% -> 100%)."""
    w = _w_obs_ranksum(x, y)
    return w != int(w)


SAMPLERS = {
    "ties_fractional_w": {"sampler": _gen_ties_fractional_w, "accept": None},
    "cont_stdnormal": {"sampler": _gen_cont_stdnormal, "accept": None},
    "ties_mild": {"sampler": _gen_ties_mild, "accept": None},
    "chi2_table_44": {"sampler": _gen_chi2_table, "accept": None},
    # 条件化档 (预注册判据 ①): 只采样 w 半整数的输入对 (拒绝采样)
    "ties_fractional_w_cond": {"sampler": _gen_ties_fractional_w,
                               "accept": _accept_w_noninteger},
    # 两样本条件化档 (D4): x 秩和 w 半整数的输入对 (拒绝采样, 生成器与配对档共用)
    "ties_fractional_w_cond_ranksum": {"sampler": _gen_ties_fractional_w,
                                       "accept": _accept_w_noninteger_ranksum},
}


# ---------------------------------------------------------------------------
# 恒等式知识库 v1 (3 条)
# ---------------------------------------------------------------------------

INVARIANTS = {
    "direction_complement": {
        "families": ["paired_rank", "ranksum", "two_sample_location"],
        "swap_inputs": True,
        "math": ("p_less(x,y) = p_greater(y,x): 单侧备择在输入交换下互为补事件"
                 "重标记, 正确实现 (含 ties) 必须精确成立; "
                 "违反 = 方向语义 bug, 统计功效归零而代码看似正常 (D1 实测 "
                 "违背率 14.8% 的头号 bug 类)"),
        "pair_semantics": ("pair = [left_side, right_side]; "
                           "dev = |pair[0](x,y) - pair[1](y,x)| "
                           "(right 侧必须输入交换)"),
        "n_inputs_default": 500,
        "archive": [
            # 参照零误杀实测 (E1v2 已实测, 出处 e1v2_wilcoxon_out.json)
            {"ref": "scipy wilcoxon cont n=20", "viol": 0, "R": 500,
             "worst": 0.0, "source": "expA3/e1v2_wilcoxon_out.json"},
            {"ref": "scipy wilcoxon cont n=50", "viol": 0, "R": 500,
             "worst": 0.0, "source": "expA3/e1v2_wilcoxon_out.json"},
            {"ref": "scipy wilcoxon ties n=20", "viol": 0, "R": 500,
             "worst": 0.0, "source": "expA3/e1v2_wilcoxon_out.json"},
            {"ref": "scipy wilcoxon ties n=50", "viol": 0, "R": 500,
             "worst": 0.0, "source": "expA3/e1v2_wilcoxon_out.json"},
            {"ref": "scipy ttest_ind cont n=20/50 (闭式)", "viol": 0, "R": 500,
             "worst": 0.0, "source": "expA3/e1v2_wilcoxon_out.json"},
        ],
        "external": ["scipy/scipy#19872 + PR #20765: 官方回归测试断言 "
                     "p_less(x,y)==p_greater(y,x) (paired 符号秩修复)"],
    },
    "transpose_symmetry": {
        "families": ["pearson_chi2"],
        "swap_inputs": True,
        "math": ("p(table) = p(table.T): 卡方统计量对行/列转置不变 "
                 "(单元格期望频数 E_ij = r_i c_j / N 对转置对称); "
                 "真实 LLM 代码 23 份 0 违 (e2_llm_out.json)"),
        "pair_semantics": ("pair = [f, f] (转置是单输入自配对); "
                           "dev = |pair[0](t) - pair[0](t.T)|"),
        "n_inputs_default": 500,
        "archive": [
            # 真实参照零误杀 (D1 实测, run_d1_verify.py 落盘 out/d1_axioms_archive.json)
            {"ref": "scipy chi2_contingency", "viol": 0, "R": 500,
             "worst": 6.661e-16, "source": "out/d1_axioms_archive.json"},
            {"ref": "手写卡方 (无 Yate's)", "viol": 0, "R": 500,
             "worst": 6.106e-16, "source": "out/d1_axioms_archive.json"},
            {"ref": "E2 真实 LLM 代码 23 份", "viol": 0, "R": None,
             "worst": None, "source": "expA3/e2_llm_out.json"},
        ],
        "external": [],
    },
    "swap_symmetry": {
        "families": ["ranksum"],
        "swap_inputs": True,
        "math": ("p(x,y) = p(y,x): 两样本交换对称. 补充锚 (低信息量: "
                 "ranksums 实测 0/500 无信息量, 双侧统计量对称), 不进主判据"),
        "pair_semantics": ("pair = [f, f] (同函数自配对); "
                           "dev = |pair[0](x,y) - pair[0](y,x)|"),
        "n_inputs_default": 500,
        "archive": [
            {"ref": "scipy ranksums ties n=20 (低信息量对照)", "viol": 0,
             "R": 500, "worst": 0.0, "source": "expA3/e1v2_wilcoxon_out.json"},
        ],
        "external": [],
    },
}


# ---------------------------------------------------------------------------
# 判定 (D2 run_inv.py 调用; 异常/非有限计违规 = 离散修正三要素 ③)
# ---------------------------------------------------------------------------

def _dev_direction_complement(fn_a, fn_b, x, y):
    """方向互补: 右入口必须输入交换 (硬条款 1)."""
    return abs(float(fn_a(x, y)) - float(fn_b(y, x)))


def _dev_transpose_symmetry(fn_a, fn_b, t):
    """转置对称: 单输入自配对 (t vs t.T)."""
    return abs(float(fn_a(t)) - float(fn_a(np.asarray(t).T)))


def _dev_swap_symmetry(fn_a, fn_b, x, y):
    """交换对称: 同函数输入交换."""
    return abs(float(fn_a(x, y)) - float(fn_a(y, x)))


_DEVS = {
    "direction_complement": _dev_direction_complement,
    "transpose_symmetry": _dev_transpose_symmetry,
    "swap_symmetry": _dev_swap_symmetry,
}

_TWO_SAMPLE = ("direction_complement", "swap_symmetry")


def check_invariant(name, fn_a, fn_b, rng, n, R, condition, tol=TOL_DEFAULT):
    """对两个入口函数执行一条恒等式考卷 (fn_b 对 transpose 忽略).

    condition: 采样档名 (SAMPLERS 注册表, 规格 invariants[].condition).
    返回 {invariant, condition, viol, R, worst_dev, nonfinite, crashed}.
    """
    entry = INVARIANTS.get(name)
    if entry is None:
        raise KeyError(f"未知恒等式 {name!r}; 注册表: {list(INVARIANTS)}")
    if condition not in SAMPLERS:
        raise KeyError(f"未知采样档 {condition!r}; 档表: {list(SAMPLERS)}")
    dev_fn = _DEVS[name]
    sdef = SAMPLERS[condition]
    sampler, accept = sdef["sampler"], sdef["accept"]
    viol = nf = cr = 0
    worst = 0.0
    for _ in range(R):
        for _rej in range(1000):
            args = sampler(rng, n)
            if accept is None or accept(*args):
                break
        else:
            raise RuntimeError(f"拒绝采样超限 (condition={condition}) "
                               f"— 采样面可能为空")
        try:
            if name in _TWO_SAMPLE:
                x, y = args
                dev = dev_fn(fn_a, fn_b, x, y)
            else:
                (t,) = args
                dev = dev_fn(fn_a, fn_b, t)
        except Exception:
            cr += 1
            viol += 1
            continue
        if not np.isfinite(dev):
            nf += 1
            viol += 1
            continue
        if dev > tol:
            viol += 1
        worst = max(worst, dev)
    return {"invariant": name, "condition": condition, "viol": viol, "R": R,
            "worst_dev": worst, "nonfinite": nf, "crashed": cr}


def list_invariants():
    return list(INVARIANTS)


def archive_to_json(path):
    json.dump({k: {"archive": v["archive"], "families": v["families"],
                   "math": v["math"]}
               for k, v in INVARIANTS.items()},
              open(path, "w"), ensure_ascii=False, indent=2)
