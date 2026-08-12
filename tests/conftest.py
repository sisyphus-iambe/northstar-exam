"""pytest 共享配置 — 仅保证 import spsl 可用 (不动北极星任何现有代码).

northstar_v2/spsl 依赖 cwd 在仓库根; 以 `python -m pytest tests/` 方式运行
时 cwd 即根目录, 此处为其他调用方式 (IDE runner / 裸 pytest) 兜底.
"""
import sys
from pathlib import Path

_NSV2 = Path(__file__).resolve().parents[1]
if str(_NSV2) not in sys.path:
    sys.path.insert(0, str(_NSV2))
