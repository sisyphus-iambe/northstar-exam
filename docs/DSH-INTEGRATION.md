# Northstar as a DeepSeek Harness (DSH) plugin

This guide wires the Northstar exam hall into [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) as an MCP server via the official `@deepseek-ai/dsh-mcp-client` bridge. The agent then sees three tools — `mcp__northstar__submit_candidate`, `mcp__northstar__run_exam`, `mcp__northstar__get_verdict` — and can send AI-generated code or conclusions through the four-layer exam pipeline, with the verdict determined by `spsl.run` (the gateway contains zero judgement logic).

## Prerequisites

- DeepSeek Harness installed (`npx @deepseek-ai/dsh web` on Node 22.19+/24; see the [DSH README](https://github.com/deepseek-ai/deepseek-harness) for install options)
- This repository cloned somewhere stable, e.g. `~/northstar-exam`
- Python 3 with the exam environment: `numpy`, `scipy`, and the MCP SDK pinned per the gateway (`pip install "mcp==1.29.0"`). A dedicated venv is recommended so the `python3` executable below is unambiguous:

```sh
python3 -m venv ~/northstar-exam/.venv
~/northstar-exam/.venv/bin/pip install numpy scipy "mcp==1.29.0"
```

## Wire it in

Add a row to your DSH profile's `cordis.patch.yml` (or to the home-level `cordis.yml`), following the shape of the official [`mcp-reference-memory.cordis.yml`](https://github.com/deepseek-ai/deepseek-harness/tree/master/examples/mcp-memory) example:

```yaml
- insert:
    - id: northstar-mcp
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: northstar
        transport: stdio
        command: /absolute/path/to/northstar-exam/.venv/bin/python
        args:
          - mcp/northstar_mcp.py
          - --specs-dir
          - specs
        cwd: /absolute/path/to/northstar-exam
```

Replace both `/absolute/path/to/northstar-exam` occurrences with the actual checkout path. Key semantics (matching DSH's MCP client contract):

| Field | Meaning |
|---|---|
| `serverName` | Namespace prefix; the agent sees tools as `mcp__northstar__<tool>` |
| `transport: stdio` | DSH spawns the command as a child process and manages its lifecycle (launch/stop with the plugin) — no server to keep running |
| `command` | Use the venv's `python` binary so the MCP/numpy/scipy packages resolve |
| `cwd` | Must be the repository root: `northstar_mcp.py` resolves the workspace from its own path (`parents[1]`), and `spsl.run` runs as a subprocess with `cwd=WORKSPACE` |

If your DSH installation has no `cordis.patch.yml` yet, create one or use the profile's own patch layer — see [Profiles and bundles](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md#profiles-and-bundles) for layer ordering.

## Verify

With DSH running (`dsh web` or `dsh --profile headless`), ask the agent to list its tools, or check the assembled config tree:

```sh
dsh --profile web --dump-config   # the northstar-mcp row should appear
```

The agent should be able to name the three tools and their schemas.

## Usage example

```text
Agent: "The candidate model just wrote a Pearson chi-square implementation at
       mcp/candidates/pearson_ai.py. Run it through the exam hall and report the verdict."

Tools:
  mcp__northstar__submit_candidate(candidate_name="pearson_ai", language="py",
                                   source_path="mcp/candidates/pearson_ai.py", family="pearson_chi2")
  mcp__northstar__run_exam(spec_name="pearson_chi2", candidate_name="pearson_ai")
  mcp__northstar__get_verdict(run_id="pearson_chi2__pearson_ai")
```

Fast smoke test without a candidate: `run_exam(spec_name="demo_data")` — the spec-as-exam family runs the demo-data spec directly with no candidate.

## Tool reference

| Tool | Purpose | Notes |
|---|---|---|
| `submit_candidate(name, language, source_path\|source_text, family)` | Register a candidate (source saved to `mcp/candidates/<name>/`, registry updated) | `language` ∈ {`py`, `node`}; `family` ∈ {`pearson_chi2`, `wilcoxon`, `ranksum`, `demo_data`, `state_estimator`, `conclusion_anchor`}; paths are workspace-relative and escape-checked |
| `run_exam(spec_name, candidate_name?)` | Dispatch to the authoritative `spsl.run` subprocess and return the verdict JSON | Statistical families use the pre-compiled four-layer exams (`exams/`); spec-as-exam families run the spec directly; invariant exams via `exam_wsr_inv` / `exam_ranksum_inv`; the compiled exam's embedded spec is byte-compared against `specs/` before every run |
| `get_verdict(run_id? \| spec_name + candidate_name?)` | Read back the latest verdict JSON | Written to `out/verdicts/<spec>__<candidate>.json`; deterministic on rerun |

## Notes

- **Judgement authority stays local.** The gateway only composes commands and reads back verdicts; all decision numbers come from `spsl.run` running as a subprocess inside this repository. DSH never sees the exams or the judgement logic.
- **Determinism.** Verdicts are deterministic under the pinned Python/numpy/scipy environment; rerunning the same `run_exam` overwrites the same verdict file.
- **Candidate sources.** Paths must resolve inside the workspace (`is_relative_to` check) and are saved relative to the repo root — the gateway refuses anything that escapes.
- **Environment hygiene.** DSH's stdio bridge strips credential-named ambient variables before spawning the child; if the pipeline ever needs a secret (it does not today), pass it via the row's `config.env`, never in the YAML literal.
- **Deep integration (optional).** The MCP row gives the agent *self-service* access to the exam hall. A native Cordis plugin could instead listen to `tools/post-execute` / `agent/turn-stopping` and make verification a *mandatory gate* (blocking FAIL verdicts, logging every exam into the session log). The MCP row is the zero-cost first step; the native plugin is a follow-up.

## Filing issues

Bug reports and integration feedback: [GitHub Discussions](https://github.com/deepseek-ai/deepseek-harness/discussions) for DSH-side issues, this repo's issues for Northstar-side behavior. If you ship a Northstar integration, tag the repo with the `dsh-plugin` topic for discoverability.
