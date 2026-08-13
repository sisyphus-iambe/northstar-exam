#!/usr/bin/env python3
"""北极星 v3 阶段 4 — 级 4 形态: MCP 网关 (stdio, 调用层包装, 零判定逻辑).

3 工具:
  submit_candidate(candidate_name, language, source_path|source_text, family)
      -> 登记候选 (注册表 + 源码落盘, 相对路径)
  run_exam(spec_name, candidate_name=None)
      -> 按规格 family 分派到既有考官, subprocess 调用, 返回 verdict JSON
  get_verdict(run_id 或 spec_name+candidate_name)
      -> 返回最近判定结果 JSON (从 out/verdicts/ 读)

设计 (round9_A §三④ + 阶段 4 任务书):
  - 判定数字 100% 来自既有考官 (stage2 run_chain.py / stage3 conclusion_anchor.py),
    本文件不含任何统计/判定代码, 只传参 + 返回结果.
  - 确定性: run_id = "<spec_name>__<candidate_name>" (结论锚无候选时为 spec_name),
    同参重跑覆盖同一文件; 考官输出 JSON 无 elapsed/command/时间戳, 同路径重跑
    逐字节一致 (验收判据①).
  - 零硬编码数据路径: 规格目录/考官根目录由启动参数指定, 规格内数据集/阈值
    全由规格 JSON 声明 (级 2/3 既有机制), 唯一固定键 = 注册表文件名.
  - 候选源码以相对工作区根路径落盘 (mcp/candidates/<name>/), 不落绝对路径.
  - 被判定候选在子进程运行且 stdout 捕获, 防止污染 MCP stdio 协议.
  - 支持族 = {chain, conclusion_anchor}; family=inv 显式拒绝: stage1 run_inv.py
    输出含 elapsed_seconds/command (设计如此, 非确定性), 不进入 stage4 的
    确定性 MCP/CI 面 (理由见 README §能力边界).

启动: python3 mcp/northstar_mcp.py --specs-dir specs
  (可选 --examiners-root, 默认工作区父目录, 考官 = <root>/northstar_v3_stage2|3/)
依赖: /tmp/g2venv, pip install "mcp==1.29.0"
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

MCP_NAME = "northstar-v3"
WORKSPACE = Path(__file__).resolve().parents[1]
REGISTRY_REL = "mcp/candidates_registry.json"
CANDIDATES_REL = "mcp/candidates"
VERDICTS_REL = "out/verdicts"
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
LANGS = {"py": ".py", "node": ".js"}
FAMILIES = ("chain", "conclusion_anchor")
EXAMINERS = {
    "chain": ("northstar_v3_stage2", "run_chain.py"),
    "conclusion_anchor": ("northstar_v3_stage3", "conclusion_anchor.py"),
}


def _log(msg: str) -> None:
    print(f"[northstar_mcp] {msg}", file=sys.stderr)


class NorthstarGate:
    """MCP 网关状态: 注册表 + 落盘目录 (纯调用层, 无判定逻辑)."""

    def __init__(self, specs_dir: Path, examiners_root: Path):
        self.specs_dir = specs_dir.resolve()
        self.examiners_root = examiners_root.resolve()
        self.registry_path = WORKSPACE / REGISTRY_REL
        self.candidates_dir = WORKSPACE / CANDIDATES_REL
        self.verdicts_dir = WORKSPACE / VERDICTS_REL
        self.verdicts_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    # ---- 注册表 ----

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            self.registry = json.loads(
                self.registry_path.read_text(encoding="utf-8"))
        else:
            self.registry = {"schema": 1, "candidates": {}}

    def _save_registry(self) -> None:
        self.registry_path.write_text(
            json.dumps(self.registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    def register(self, name: str, language: str, source: str, family: str) -> dict:
        if not NAME_RE.match(name):
            raise ValueError(f"invalid candidate_name: {name!r} "
                             f"(must match {NAME_RE.pattern})")
        if language not in LANGS:
            raise ValueError(f"invalid language: {language!r} (choose from {sorted(LANGS)})")
        if family not in FAMILIES:
            raise ValueError(f"invalid family: {family!r} (supported {FAMILIES}; "
                             f"see README section on capability boundaries for why inv is excluded)")
        cand_dir = self.candidates_dir / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        rel = f"mcp/candidates/{name}/{name}{LANGS[language]}"
        (cand_dir / f"{name}{LANGS[language]}").write_text(source, encoding="utf-8")
        entry = {"name": name, "language": language, "family": family,
                 "source_rel": rel}
        self.registry.setdefault("candidates", {})[name] = entry
        self._save_registry()
        return entry

    def resolve_candidate(self, name: str) -> Path:
        entry = self.registry["candidates"].get(name)
        if entry is None:
            raise ValueError(f"candidate not registered: {name!r} (call submit_candidate first)")
        resolved = (WORKSPACE / entry["source_rel"]).resolve()
        if not resolved.is_relative_to(WORKSPACE):
            raise ValueError(f"candidate source escapes the workspace: {resolved}")
        return resolved

    # ---- 规格 ----

    def load_spec(self, spec_name: str) -> dict:
        p = Path(spec_name)
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        spec_path = (self.specs_dir / p).resolve()
        if not spec_path.is_relative_to(self.specs_dir):
            raise ValueError(f"spec path escapes the specs dir: {spec_path}")
        if not spec_path.is_file():
            raise ValueError(f"spec not found: {spec_path}")
        return json.loads(spec_path.read_text(encoding="utf-8"))

    # ---- 分派 (判定权威 = 既有考官, 本方法只拼命令行 + 读回结果) ----

    def run_spec_candidate(self, spec_name: str,
                           candidate_name: str | None) -> dict:
        spec = self.load_spec(spec_name)
        family = spec.get("family")
        if family not in EXAMINERS:
            raise ValueError(
                f"规格 {spec_name} 的 family={family!r} 不在支持集 {FAMILIES}; "
                f"inv 排除理由见 README §能力边界")

        out_path = (self.verdicts_dir / f"{spec_name}__{candidate_name}.json"
                    if candidate_name else self.verdicts_dir / f"{spec_name}.json")
        out_path = out_path.resolve()
        if not out_path.is_relative_to(self.verdicts_dir):
            raise ValueError(f"output path escapes the verdicts dir: {out_path}")
        stage, script = EXAMINERS[family]
        cmd = [sys.executable, str(self.examiners_root / stage / script),
               str((self.specs_dir / (spec_name if spec_name.endswith(".json")
                                      else spec_name + ".json")).resolve())]
        if family == "chain":
            cmd += [str(self.resolve_candidate(candidate_name))]
        cmd += ["--out", str(out_path)]
        proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-8:]
            raise RuntimeError(f"examiner failed rc={proc.returncode}: " + " | ".join(tail))
        verdict = json.loads(out_path.read_text(encoding="utf-8"))
        return {"run_id": out_path.stem, "spec_name": spec_name,
                "candidate_name": candidate_name, "family": family,
                "verdict_file": f"{VERDICTS_REL}/{out_path.name}",
                "verdict": verdict}

    def read_verdict(self, run_id: str | None, spec_name: str | None,
                     candidate_name: str | None) -> dict:
        if run_id:
            p = self.verdicts_dir / f"{run_id}.json"
        elif spec_name:
            rid = spec_name if not candidate_name else f"{spec_name}__{candidate_name}"
            p = self.verdicts_dir / f"{rid}.json"
        else:
            raise ValueError("get_verdict requires run_id or (spec_name + candidate_name)")
        p = p.resolve()
        if not p.is_relative_to(self.verdicts_dir):
            raise ValueError(f"verdict path escapes the verdicts dir: {p}")
        if not p.is_file():
            raise ValueError(f"no verdict result: {p} (run_exam first)")
        return {"run_id": p.stem, "verdict_file": f"{VERDICTS_REL}/{p.name}",
                "verdict": json.loads(p.read_text(encoding="utf-8"))}


GATE: NorthstarGate = None  # type: ignore[assignment]


def _gate() -> NorthstarGate:
    if GATE is None:  # pragma: no cover
        raise RuntimeError("server not initialized")
    return GATE


mcp = FastMCP(MCP_NAME)


@mcp.tool()
def submit_candidate(candidate_name: str, language: str,
                     source_path: str | None = None,
                     source_text: str | None = None,
                     family: str = "chain") -> str:
    """登记候选: 源码落盘 + 注册表 (source_path 与 source_text 二选一)."""
    try:
        if (source_path is None) == (source_text is None):
            raise ValueError("exactly one of source_path / source_text required")
        if source_path is not None:
            src = Path(source_path)
            if not src.is_absolute():
                src = (WORKSPACE / src).resolve()
            src = src.resolve()
            if not str(src).startswith(str(WORKSPACE)):
                raise ValueError(f"source_path must be inside the workspace: {src}")
            source = src.read_text(encoding="utf-8")
        else:
            source = source_text
        entry = _gate().register(candidate_name, language, source, family)
        return json.dumps({"ok": True, "candidate": entry}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 — 工具边界, 显式失败
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def run_exam(spec_name: str, candidate_name: str | None = None) -> str:
    """送考: 按规格 family 分派到既有考官 (chain -> run_chain.py, "
    "conclusion_anchor -> conclusion_anchor.py), 返回 verdict JSON."""
    try:
        res = _gate().run_spec_candidate(spec_name, candidate_name)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def get_verdict(run_id: str | None = None, spec_name: str | None = None,
                candidate_name: str | None = None) -> str:
    """查最近判定: 传 run_id, 或 (spec_name + candidate_name)."""
    try:
        return json.dumps(_gate().read_verdict(run_id, spec_name, candidate_name),
                          ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 mcp/northstar_mcp.py",
                                 description="Northstar v3 MCP gateway (stdio)")
    ap.add_argument("--specs-dir", default=str(WORKSPACE / "specs"),
                    help="specs directory (default <workspace>/specs)")
    ap.add_argument("--examiners-root", default=str(WORKSPACE.parent),
                    help="examiners root (default workspace parent; examiners = stage2/3 subdirectories)")
    args = ap.parse_args(argv)

    global GATE
    GATE = NorthstarGate(Path(args.specs_dir), Path(args.examiners_root))
    _log(f"specs_dir={GATE.specs_dir} examiners_root={GATE.examiners_root} "
         f"registry={GATE.registry_path}")
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
