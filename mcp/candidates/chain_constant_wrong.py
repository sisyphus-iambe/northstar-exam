"""故意错误候选 (smoke/批量判定用): 恒返回 p=0.5.

与链考卷校准契约 (零假设下 p 均匀) 矛盾: 恒 0.5 时
F(0.01)=F(0.05)=F(0.10)=0, worst dev = 0.10 > slack 0.02 -> REJECT.
本文件是被考对象, 不含任何判定逻辑; 判定权威 = stage2 run_chain.py.
"""
NAME = "chain_constant_wrong"


def run_chain(rng):
    return 0.5
