"""候选链: 完整案例 (自检锚) — 自 run_defect_catalog.py chain_listwise 逐行提取.

契约: NAME + run_chain(rng) -> float p (与前置实验链契约一致).
预期校准 (前置实测 F(0.05)=0.0535, max_dev=0.0035 PASS).
"""
import numpy as np
from scipy.stats import ttest_ind

NAME = "chain_listwise"
N1 = N2 = 50
MISS = 0.30


def run_chain(rng):
    x = rng.normal(0, 1, N1)
    y = rng.normal(0, 1, N2)
    mask = rng.random(N2) < MISS
    y_obs = y[~mask]
    return float(ttest_ind(x, y_obs, equal_var=True).pvalue)
