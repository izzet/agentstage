# Paper Evaluations

Pytest-based evaluation suite for the AgentStage paper. Each test file encodes
one hypothesis from `AGENTSTAGE.md` §2. Each `assert` verifies a quantitative
claim from §3 (C1-C10) or an evaluation from §11.5 (E1-E11) against captured
data. If all tests pass, the paper's claims have live evidence.

This suite is independent of `tests/`, which contains fast unit tests over
`src/agentstage/`. Running `uv run pytest` covers `tests/` only; the paper
evals run with an explicit path:

```bash
# Trace-only evaluations against the 88-probe PoC corpus
uv run pytest paper_evals/ --trace-root poc/runs

# Single hypothesis (e.g. the headline H3 predictability claim)
uv run pytest paper_evals/ -m h3 --trace-root poc/runs

# Once end-to-end staging campaigns exist (E5+)
uv run pytest paper_evals/ \
    --trace-root poc/runs \
    --staging-root results/staging/<date> \
    --rule-library-version v1
```

## CLI options

| Option | Default | Meaning |
|---|---|---|
| `--outputs-root` | `outputs` | Root for campaign run outputs (trace + end-to-end share this root) |
| `--legacy-trace-root` | `poc/runs` | Legacy 88-probe PoC corpus for E2 re-score |
| `--io-report-root` | _none_ | Historical io_report.json files for empirical GT (e.g. $SCIIOBENCH_ROOT/outputs) |
| `--min-seeds` | `3` | Min seeds per (task, model, prompt) cell to include |
| `--rule-library-version` | _none_ | Expected frozen rule version (H6/H7 assert match) |
| `--report-dir` | `paper_evals/.results` | Where `report.json` is written |

## Hypotheses → claims map

| File | Hypothesis | Serves | Where claim originates |
|---|---|---|---|
| `test_h1_slack.py` | Reasoning slack windows are large and reliable | C1, C6 | §3, §6.1 |
| `test_h2_intent.py` | Streaming thinking reveals file-access intent | C2, C5 | §3, §6.5 |
| `test_h3_predictability.py` | Tiered predictor achieves ≥0.85 byte recall, ≤1.5× overfetch | C2, C3 (headline) | §3, §6.2 |
| `test_h4_tiering.py` | Tiering survives the worst-case workload (6042 files) | C2/C3 case | §6.3 |
| `test_h5_planning_prompts.py` | Planning prompts give 2–10× slack multiplier | C6 | §6, §7.1 |
| `test_h6_frozen_rules_crosscorpus.py` | Frozen rules transfer to SAB + SWE-bench Lite | L2 genericity (E9, E10) | §11.6 |
| `test_h7_leave_one_out.py` | Frozen rules transfer to held-out AgentIOBench workload | L1 genericity (E3) | §11.6 |
| `test_h8_staging_effectiveness.py` | End-to-end staging reduces per-tool first-read P95 | C8, E5 | §11.5 |
| `test_h9_bandwidth_sensitivity.py` | Speedup is monotone in cold-tier bandwidth | E6 | §11.5 |
| `test_h10_proxy_overhead.py` | Proxy overhead ≤1% p99; no-thinking case is baseline-identical | E4, E7 | §11.5 |

## Report output

After every run, `paper_evals/.results/report.json` is written by the
`pytest_sessionfinish` hook. Structure:

```json
{
  "timestamp": "2026-05-19T...",
  "exitstatus": 0,
  "summary": {"passed": N, "failed": 0, "skipped": K, "error": 0},
  "tests": [{"nodeid": "...", "outcome": "passed"}, ...],
  "data": {
    "table_slack_distribution": ...,
    "figure_tier1_recall_per_config": ...,
    "figure_goes_collapse": ...,
    "figure_bandwidth_curve": ...
  }
}
```

Tests record into `data` via the `report` fixture:

```python
def test_something(outputs_root, report):
    report.record("table_slack_distribution", {...})
    report.append("figure_per_seed_points", {...})
```

Plotting scripts consume `data.*` keys; the rest is suite metadata.

## Data shapes consumed

Trace-only run directory layout (one per seed, under `--trace-root`):

```
<timestamp>_<task>_<model>_<turn>_<budget>_<prompt-variant>_s<seed>/
├── stream.jsonl          # raw SSE events, t_ms relative to urlopen start
├── summary.json          # block-level timing (first-thinking, first-tool, ...)
├── prediction.json       # per-rule activations + tier outputs
└── byte_metrics.json     # per-tier byte recall + overfetch
```

End-to-end staging run directory (one per seed, under `--staging-root`):

```
<timestamp>_<task>_<model>_<config>/
├── staging_report.json   # per-tool-call first-read latencies, prestage hits
├── io_report.json        # DFTracer summary
└── stream.jsonl          # full SSE for replay/audit
```

The `Campaign` / `RunResult` analogs that index these will live in
`src/agentstage/workloads/` once written, and the conftest fixtures will
delegate to them. Today the conftest exposes only the raw path fixtures and
the `report` collector; each test file will load its own slice until the
workloads module materializes.
