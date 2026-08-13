#!/usr/bin/env python3
"""北极星公开仓库 MCP 网关 (stdio, 调用层包装, 零判定逻辑).

3 工具:
  submit_candidate(candidate_name, language, source_path|source_text, family)
      -> 登记候选 (注册表 + 源码落盘, 相对路径)
  run_exam(spec_name, candidate_name=None)
      -> 按规格 family 分派到 spsl.run (判定权威), subprocess 调用, 返回 verdict JSON
  get_verdict(run_id 或 spec_name+candidate_name)
      -> 返回最近判定结果 JSON (从 out/verdicts/ 读)

分派 (判定数字 100% 来自 spsl.run 的 main; 本文件只拼命令 + 读回结果):
  - statistical 族 (规格 family = pearson_chi2 | wilcoxon): 规格 -> 预编译四层
    考卷 (exams/, 映射见 STAT_EXAMS; 送考前校验考卷内嵌规格与当前规格逐字节
    一致, 不一致即报错). 命令 = python3 -m spsl.run <考卷.json> <候选.py> --out ...
  - 规格即考卷族 (family = demo_data | state_estimator | conclusion_anchor):
    规格 JSON 直接交 spsl.run (state_estimator 需候选模块; demo_data /
    conclusion_anchor 无考生).
  - 恒等式族 (spec_name = exam_wsr_inv | exam_ranksum_inv, 考卷在 exams/):
    考卷 JSON + 候选模块交 spsl.run, 其 main 依 layer=INV 委托 spsl.run_inv
    判定 (PASS/REJECT). 恒等式候选按考卷 family 登记: wsr 考卷 family=wilcoxon,
    ranksum 考卷 family=ranksum.

确定性 / 运行语义:
  - run_id = "<spec_name>__<candidate_name>" (无候选时 = spec_name), 同参重跑
    覆盖同一文件.
  - spsl.run 判定/诊断字段确定性, 重跑一致; verdict payload 含
    elapsed_seconds / command / payload_md5 / self_md5 字段 (管线设计如此,
    本文件不做任何改写).
  - 候选源码以相对工作区根路径落盘 (mcp/candidates/<name>/), 不落绝对路径;
    仓库自带候选以 registry 的 source_rel 指向真实文件 (candidates/ 等).
  - 被判定候选在子进程运行且 stdout/stderr 捕获, 防止污染 MCP stdio 协议.
  - 零硬编码数据路径: specs 目录由 --specs-dir 指定; 考卷映射表为仓库内
    预编译产物名 (编译路径在仓库内可用: spsl.envelope.build_full_exam 与
    exams/ 逐字节一致, 见 STAT_EXAMS 校验).

启动: python3 mcp/northstar_mcp.py --specs-dir specs
依赖: 环境需 mcp 包 (pip install "mcp==1.29.0") + numpy/scipy (spsl.run 子进程);
      spsl.run 以 `python3 -m spsl.run` 在仓库根 (WORKSPACE) 执行.
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

# statistical 族: 规格名 -> 预编译四层考卷 (exams/, spsl.envelope 产物).
# 送考前校验: 考卷内嵌规格 == 当前规格文件 (逐字节), 防规格修改后静默用旧卷.
STAT_EXAMS = {
    "pearson_chi2": "exams/exam_pearson_full.json",
    "pearson_chi2_demo": "exams/exam_pearson_demo.json",
    "wilcoxon_rank_sum": "exams/exam_wilcoxon_full.json",
}
STAT_FAMILIES = ("pearson_chi2", "wilcoxon")
# 规格即考卷族: 规格 JSON 直接交 spsl.run (注册表 EXAMINER_REGISTRY 承接).
SPEC_AS_EXAM_FAMILIES = ("demo_data", "state_estimator", "conclusion_anchor")
# 恒等式 (INV) 考卷: 在 exams/ 而非 specs/, 以考卷文件名引用.
INV_EXAMS = {
    "exam_wsr_inv": "exams/exam_wsr_inv.json",
    "exam_ranksum_inv": "exams/exam_ranksum_inv.json",
}
FAMILIES = STAT_FAMILIES + SPEC_AS_EXAM_FAMILIES + ("ranksum",)


def _log(msg: str) -> None:
    print(f"[northstar_mcp] {msg}", file=sys.stderr)


class NorthstarGate:
    """MCP 网关状态: 注册表 + 落盘目录 (纯调用层, 无判定逻辑)."""

    def __init__(self, specs_dir: Path):
        self.specs_dir = specs_dir.resolve()
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
                             f"恒等式考卷候选按其考卷 family 登记: wilcoxon/ranksum)")
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
        if not resolved.is_file():
            raise ValueError(f"candidate source file not found: {resolved}")
        return resolved

    # ---- 规格 ----

    def _spec_path(self, spec_name: str) -> Path:
        p = Path(spec_name)
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        spec_path = (self.specs_dir / p).resolve()
        if not spec_path.is_relative_to(self.specs_dir):
            raise ValueError(f"spec path escapes the specs dir: {spec_path}")
        return spec_path

    def load_spec(self, spec_name: str) -> dict:
        spec_path = self._spec_path(spec_name)
        if not spec_path.is_file():
            raise ValueError(f"spec not found: {spec_path}")
        return json.loads(spec_path.read_text(encoding="utf-8"))

    def _resolve_workspace_path(self, rel: str, kind: str) -> Path:
        """WORKSPACE 相对路径 -> 绝对路径 (防逃逸 + 存在性检查)."""
        p = (WORKSPACE / rel).resolve()
        if not p.is_relative_to(WORKSPACE):
            raise ValueError(f"{kind} path escapes the workspace: {p}")
        if not p.is_file():
            raise ValueError(f"{kind} file not found: {p}")
        return p

    # ---- 分派 (判定权威 = spsl.run 子进程, 本方法只拼命令行 + 读回结果) ----

    def run_spec_candidate(self, spec_name: str,
                           candidate_name: str | None) -> dict:
        out_path = (self.verdicts_dir / f"{spec_name}__{candidate_name}.json"
                    if candidate_name else self.verdicts_dir / f"{spec_name}.json")
        out_path = out_path.resolve()
        if not out_path.is_relative_to(self.verdicts_dir):
            raise ValueError(f"output path escapes the verdicts dir: {out_path}")

        inv_key = spec_name[:-5] if spec_name.endswith(".json") else spec_name
        if inv_key in INV_EXAMS:
            exam_path = self._resolve_workspace_path(INV_EXAMS[inv_key], "inv exam")
            family = json.loads(exam_path.read_text(encoding="utf-8"))["family"]
            return self._run_examiner(spec_name, family, exam_path,
                                      candidate_name, out_path)

        spec = self.load_spec(spec_name)
        family = spec.get("family")
        if family in STAT_FAMILIES:
            exam_rel = STAT_EXAMS.get(spec.get("name"))
            if exam_rel is None:
                raise ValueError(
                    f"no compiled exam for spec name {spec['name']!r}; "
                    f"compiled exams available for: {', '.join(sorted(STAT_EXAMS))}")
            exam_path = self._resolve_workspace_path(exam_rel, "exam")
            embedded = json.loads(exam_path.read_text(encoding="utf-8"))["spec"]
            if embedded != spec:
                raise ValueError(
                    f"spec {spec_name} is out of sync with its compiled exam "
                    f"{exam_rel} (embedded spec differs); recompile the exam first")
            return self._run_examiner(spec_name, family, exam_path,
                                      candidate_name, out_path)

        if family in SPEC_AS_EXAM_FAMILIES:
            return self._run_examiner(spec_name, family, self._spec_path(spec_name),
                                      candidate_name, out_path)

        raise ValueError(
            f"规格 {spec_name} 的 family={family!r} 不在支持集 {FAMILIES}")

    def _run_examiner(self, spec_name: str, family: str, target: Path,
                      candidate_name: str | None, out_path: Path) -> dict:
        cmd = [sys.executable, "-m", "spsl.run", str(target)]
        if candidate_name is not None:
            cmd.append(str(self.resolve_candidate(candidate_name)))
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
                     family: str | None = None) -> str:
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
    """送考: 按规格 family 分派到 spsl.run (statistical 族 -> 预编译四层考卷,
    规格即考卷族 -> 规格 JSON, 恒等式 -> exams/*_inv.json), 返回 verdict JSON."""
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
                                 description="Northstar MCP gateway (stdio)")
    ap.add_argument("--specs-dir", default=str(WORKSPACE / "specs"),
                    help="specs directory (default <workspace>/specs)")
    args = ap.parse_args(argv)

    global GATE
    GATE = NorthstarGate(Path(args.specs_dir))
    _log(f"specs_dir={GATE.specs_dir} registry={GATE.registry_path}")
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
