# Paper Evaluations

Pytest-based evaluation suite for the AgentStage paper. Each test file encodes
one hypothesis (H1-H10), and each `assert` verifies a quantitative claim
against captured run data. If all tests pass, the paper's claims have live
evidence.

This suite is independent of `tests/`, which contains fast unit tests over
`src/agentstage/`. Running `uv run pytest` covers `tests/` only; the paper
evals run with an explicit path:

```bash
# Run against the full campaign root (includes PoC at outputs/poc/ if present)
uv run pytest paper_evals/ --outputs-root outputs/

# Single hypothesis (e.g. the headline H3 predictability claim)
uv run pytest paper_evals/ -m h3 --outputs-root outputs/

# With historical empirical-GT root for E2 re-score
uv run pytest paper_evals/ \
    --outputs-root outputs/ \
    --io-report-root $SCIIOBENCH_ROOT/outputs \
    --rule-library-version v1
```

## CLI options

| Option | Default | Meaning |
|---|---|---|
| `--outputs-root` | `outputs` | Root for all campaign run outputs (PoC corpus at outputs/poc/ if present, plus new trace + end-to-end runs) |
| `--io-report-root` | _none_ | Historical io_report.json files for empirical GT (e.g. $SCIIOBENCH_ROOT/outputs) |
| `--min-seeds` | `3` | Min seeds per (task, model, prompt) cell to include |
| `--rule-library-version` | _none_ | Expected frozen rule version (H7 asserts match) |
| `--report-dir` | `paper_evals/.results` | Where `report.json` is written |

## Hypotheses → claims map

Every paper claim must have a hypothesis test here. The test STRUCTURE has to exist even if the data hasn't caught up yet — failing/skipping tests are fine, missing tests are not. When a paper section commits to a new quantitative claim, add (or extend) the corresponding `test_h<N>_*.py` first, then write the prose.

| File | Hypothesis | Serves | Where claim originates |
|---|---|---|---|
| `test_h1_slack.py` | Reasoning slack windows are large and reliable; cross-provider consistency | C1, C6 | §3, §6.1 |
| `test_h2_intent.py` | Thinking reveals file-access intent; HOT high-precision low-recall; semantic rules carry the load | C2, C5 | §3, §6.5 |
| `test_h3_predictability.py` | Tiered detector ≥0.85 byte recall, ≤1.5× overfetch; cross-provider consistency (HEADLINE) | C2, C3 | §3, §6.2 |
| `test_h4_tiering.py` | aiob_107 GOES collapse 6078× → ≤1.5×; tier-3 recall stays high on high-fanout workloads | C2/C3 case | §6.3 |
| `test_h7_leave_one_out.py` | No single rule load-bearing; min-recall-after-drop ≥ 0.80 | L1 genericity (E3) | §11.6 |
| `test_h8_staging_effectiveness.py` | E2E speedup aiob_110 (s3 ≥1.3×, local ≥1.0×); decompression-staging ≥ plain; per-benchmark median ≥1.1×, max ≥1.5× on DSBench + MLE-bench (the result trio alongside AIOB); speedup attributable to staging | C8, E5 | §11.5 |
| `test_h9_bandwidth_sensitivity.py` | Speedup monotone in cold-tier BW; figure_bytes_moveable_per_backend spans ≥50× for Fig 1b | E6 | §11.5 |
| `test_h10_proxy_overhead.py` | Proxy p99 overhead ≤1%; auto-rule p95 ≤1ms; no-thinking pathway baseline-identical; passthrough byte-identical when detector disabled | E4, E7 | §11.5 |
| `test_h11_first_tool_prior.py` | First tool is filesystem probe (`list_dir`) on ≥90% of runs (96.1% per E-033); per-model ≥60% | C2 (subset selection) | §1 P3, §4 predictor |
| `test_h12_pathful_prompt_fails.py` | Pathful prompts do not improve HOT or tier-1 recall paired; justifies auto-rules | C1 (path extraction) | §1 P4, §4 auto-rules |

## Adding a new claim

1. Find the hypothesis the claim belongs to. If it doesn't fit any, add `test_h<N+1>_*.py` and register the marker in `pytest.ini`.
2. Pick the existing test in that file structurally closest to the new claim. Copy its skeleton (fixture imports, `pytest.skip` if-data-missing pattern, `report.record` / `report.append` calls).
3. The assertion threshold must match what the paper text says — verbatim. If the paper claims "≥85%", the assertion is `>= 0.85`. No fuzzing.
4. Run `uv run pytest paper_evals/ --outputs-root outputs/`. Skipping is fine; failing is information ("we haven't run enough yet" vs "the claim is false").
5. Update this README's hypotheses table.

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
├── detection.json       # per-rule activations + tier outputs
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
