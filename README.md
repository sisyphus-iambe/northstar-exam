# Northstar — a trust checkpoint for AI-generated artifacts

Submit an AI-generated implementation, get a four-layer verdict (L1 calibration / L2 cross-reference / L3 coverage / L4 degenerate inputs). Correct → ACCEPT, wrong → REJECT. No LLM self-evaluation anywhere — the verdict is computed by deterministic programmatic checks against independent references.

Northstar treats a statistical implementation like a candidate sitting an exam: you submit a `.py` implementation, the platform runs a four-layer exam and returns a per-layer verdict (L1 distribution / L2 cross-reference / L3 coverage / L4 degenerate inputs) together with a report card. It is especially sensitive to code that "looks right but computes wrong" — wrong degrees of freedom, wrong p-value formulas, and silent p-value filling on degenerate inputs (on the classes where the reference itself fails honestly) are exactly the bug classes it catches.

## Quick start

```bash
# 1. Dependencies: Python 3.14.6 + NumPy + SciPy
pip install numpy scipy

# 2. Write your implementation (contract: chi2_pvalue(observed) -> float, see example below)
#    File: my_chi2.py

# 3. Submit for examination
python3 -m spsl.run exams/exam_pearson_demo.json my_chi2.py --out verdict.json
```

Example output (correct implementation — textbook formula, example below):

```
[spsl] exam pearson_chi2_demo — my_chi2.chi2_pvalue, 3 runs (seeds 20260807..20260809), took 0.3 s
  L1: PASS   (rejected 0/3)
  L2: PASS   (rejected 0/3)
  L3: PASS   (rejected 0/3)
  L4: PASS   (rejected 0/3)
  Verdict: ACCEPT (rejected 0/3)
[spsl] JSON -> verdict.json
```

Example output (wrong implementation — always returns 0.5):

```
  L1: REJECT (rejected 3/3)
  L2: REJECT (rejected 3/3)
  L3: REJECT (rejected 3/3)
  L4: REJECT (rejected 3/3)
  Verdict: REJECT (rejected 3/3)
```

`my_chi2.py` — a minimal correct implementation (Pearson chi-square test of independence, textbook formula):

```python
import numpy as np
from scipy.special import gammaincc

def chi2_pvalue(observed):
    obs = np.asarray(observed, dtype=float)
    row_tot, col_tot = obs.sum(axis=1), obs.sum(axis=0)
    n = float(obs.sum())
    expected = np.outer(row_tot, col_tot) / n
    chi2 = float(np.sum((obs - expected) ** 2 / expected))
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return float(gammaincc(dof / 2.0, chi2 / 2.0))
```

## What it catches

The four layers exist because these bug classes are real, common, and silent. All numbers below are measured in our internal acceptance evaluations.

- **Silent p-value filling on degenerate tables (L4)** — on the 7/9 malformed-input classes where the reference itself fails honestly (NaN/Inf/zero table/string/empty table — it raises or returns non-finite), a candidate that silently returns a finite p-value is flagged as hallucinated filling and rejected. The call looks like a pass but tests nothing. (The 2 remaining classes — single-row / single-column tables — are mathematically definable and the reference itself returns a finite p=1.0; there the candidate is not judged, only a `ref_dev` diagnostic is recorded.)
- **Broken direction complementarity** — we had an LLM hand-write 60 statistical implementations (the prompt forced the normal approximation plus ties correction). Of the 54 that could be submitted for examination, 8 (14.8%) violated the direction-complementary identity — p_less(x, y) vs p_greater(y, x) — and spot checks confirmed every one was a real bug. Direction-flipped variants were REJECTED 492/500 times; correct implementations drew 0/500 false positives.
- **"Computes right but lies" is the norm, not the edge case** — in a question-bank evolution evaluation (22 human templates → 100 mutated questions × 3 generations), 17 fresh LLM-written chi-square implementations were graded by per-table gold truth: 9 were genuinely wrong, and of the 8 that computed correctly, 7 (87.5%) silently returned finite p-values on malformed inputs instead of failing honestly. Only 1 of 17 passed all four layers (5.9% pass rate). Northstar is a strict examiner of LLM output.
- **Dirty samples in robot demonstration data** — on 402 real Unitree G1 demonstration trajectories we injected three classes of dirty samples (NaN / out-of-range values / jumps): 60/60 detected, 0/30 false positives.
- **Type-confused data ships in the wild** — in our evaluation of a popular open-source robot data tooling stack (v0.18.22), type-confusion testing showed it silently passed 0/20 type-confused samples — outright-corrupted data was treated as normal data — and crashed on 20/20. These bug classes ship in the wild.

## Platform structure

```
spec (spec JSON) → compile (exam JSON) → submit (candidate .py) → four-layer report
```

- **Spec** — describes the contract of a test family: input structure, output p-value, H0 generator, reference source.
- **Compile** — compiles a spec into a complete four-layer exam (frozen L1 parameters + L2/L3 exam tables + L4 malformed inputs), fingerprinted with spec_md5 / content_md5 to prevent tampering.
- **Submit** — `spsl.run` loads the candidate implementation and runs the four-layer verdict.
- **Four-layer report**:
  - **L1 distribution calibration** — H0 simulation sampling: p-values must be uniform on the continuous region and conservative on the discrete region; NaN scores nothing.
  - **L2 cross-reference** — cross-validation against two independent reference implementations (known-answer check + reference agreement); the candidate must match the references to 1e-6, otherwise FAIL.
  - **L3 coverage** — boundary generalization: 5 shapes × 76 inputs (tiny samples / large samples / strongly skewed / zero cells / mixed).
  - **L4 degenerate inputs** — 9 classes of malformed inputs (NaN/Inf/zero table/single row/single column/string/empty table): on classes where the reference itself fails honestly (raises or returns non-finite — 7 of the 9), the candidate must fail honestly too; a silently filled finite p-value there is hallucinated filling = FAIL. On the 2 classes where the reference itself returns a finite value (single-row/single-column), the candidate is not judged (ref_dev diagnostic only).
- **Extensible examiner registry** — `spsl/registry.py` dispatches by `constraint_type`; examiner families register their own (compile_fn, run_fn) and INPUT_TYPES. In this repo today:
  - `statistical` — the four-layer exam above (pearson_chi2, wilcoxon families).
  - `conclusion_anchor` — anchors a capability conclusion on real data: the candidate estimates coverage of a stated conclusion, and the examiner verifies it against the underlying dataset (generator-based: AR(1) / normal, or self-provided parquet data).
  - `demo_data` — dirty-sample detection on robot demonstration data (specs/spec_demo_data.json; the dataset itself is not bundled — point `root` at your own data).
  - `state_estimator` — state-estimation accuracy against a reference implementation on simulated data.
  - `invariant` (v3 stage 1) — identity-based, reference-free judgement: checks that a candidate's functions satisfy mathematical identities (e.g. direction complementarity p_less(x,y) vs p_greater(y,x)) with zero references, catching flip-style bugs at 500-sample resolution. Exams live in `experiments/v3_stage1/exams/` (also mirrored in `exams/`; the main `spsl.run` pipeline dispatches them on `layer=INV`).

## Everything is open

This repository ships the **full exam-setting kit**: the four-layer engine, the spec schema, the examiner registry, the compiled exams for both statistical families (`exams/`, four-layer and L1-only), the invariant exams and their knowledge base, the question-bank evolution pipeline (`experiments/exam_evolution/`), the self-proof calibration experiment (`calibrator/`), and the local determinism checks (`ci/batch_determinism.py`). No closed exams — compile, inspect, verify, and re-derive everything yourself.

Every exam JSON is tamper-evident: `content_md5` is recomputed on load against the normalized content, and `spec_md5` against the embedded spec. You can recompile any exam from its spec with `python3 -m spsl.compile_l1/l2/l3/l4` (invariant exams: `python3 -m spsl.compile_inv`) and diff the fingerprints.

## Tooling

- **Validator QC** — `python3 -m verifytool my_chi2.py` runs the full four-layer pipeline with per-run rejection counts and produces an HTML report card plus a JSON verdict with payload_md5/self_md5 double fingerprints. `python3 -m verifytool templates list` shows the 22-template library (9 exam templates + 5 error controls + 8 real-world demos).
- **Invariant examiner** — `python3 -m spsl.run_inv experiments/v3_stage1/exams/exam_ranksum_inv.json candidates/ranksum_correct.py` runs the reference-free identity exam (candidates: `ranksum_correct.py` → PASS, `ranksum_flip.py` → REJECT).
- **MCP gateway** — `mcp/northstar_mcp.py` exposes the exam pipeline as an MCP stdio server: submit candidates, run exams (statistical four-layer via spec name, conclusion_anchor / demo_data / state_estimator via spec, invariant via `exam_wsr_inv` / `exam_ranksum_inv`), fetch verdicts. Requires the `mcp` package (`pip install "mcp==1.29.0"`) plus numpy/scipy. Dispatch delegates to `spsl.run` subprocesses — the gateway contains no judgement logic. For DeepSeek Harness: see [DSH-INTEGRATION.md](docs/DSH-INTEGRATION.md).
- **Self-proof** — `calibrator/run_experiment.py` re-derives the calibration-layer acceptance numbers (reference agreement, generator dual-path, L1 uniformity, sensitivity ≥ 0.95, false-kill ≤ 5%) from first principles.

## Capabilities

- **Statistics four-layer exams** — chi-square (pearson_chi2) and rank-sum (wilcoxon) families, fully open: `exams/exam_pearson_full.json`, `exams/exam_wilcoxon_full.json` (+ L1-only variants, + `exam_pearson_demo.json` for a fast smoke test).
- **Invariant exams** — ranksum and signed-rank identity exams (v3 stage 1, `experiments/v3_stage1/`).
- **Conclusion anchors** — coverage-anchored verdicts on real data (generator-backed, no external data required).
- **Embodied-data QC** — dirty-sample detection examiner (data self-provided, see `specs/spec_demo_data.json`).

## License

This repository is licensed under **BUSL 1.1** (Business Source License 1.1): the source code is visible, and non-commercial use (personal learning / teaching / evaluation / research) is free; commercial use requires a license. After the Change Date (2030-08-11) it converts to Apache License 2.0. See [LICENSE](./LICENSE).

**For commercial use, please contact the maintainer** (open an issue or use any contact channel on the repository page) — the exam-hall model (licensed use of the exam-setting kit) is also available.

## Environment

- Python 3.14.6 + NumPy 2.4.4 + SciPy 1.18.0 (all exams in this repo were compiled and validated in this environment).
- All verdict fields are deterministic (except elapsed_seconds and the derived payload_md5 / self_md5): re-running in the same environment gives byte-identical output; for other environments, rely on this repo's compiled artifacts or recompile yourself (local determinism checks: `ci/batch_determinism.py` — 5 exam pairs byte-identical on re-run; `tests/`, 27 tests).
- The `demo_data` examiner requires a parquet dataset (AppleToPlate-style trajectories) — set `root` in `specs/spec_demo_data.json` to your own data; the other examiners need no external data.
