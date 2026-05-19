# AgentStage Campaign Plan

The experimental campaign that produces the data backing the eScience '26
paper. This is the source of truth for which probes / end-to-end runs we
intend to execute, against which models, on which workloads, with what
seed budget, and at what cost.

Locked decisions (2026-05-19):
- **Three-model trio:** Claude Haiku 4.5, Gemini 2.5 Flash, one self-hosted
  OSS reasoning model (Qwen3-Thinking or DeepSeek-R1-Distill — pick during
  Day-2 vLLM setup based on Ares GPU memory headroom).
- **Spending ceiling: $150** (hard). Forces discipline; matches the
  cheap-model framing.
- **Framing decided on the fly** when writing §1 of the paper. The PoC
  Sonnet/Pro corpus stays available as cross-cost-tier validation.

## Why this trio

| Model | Provider family | Role | Per-probe cost |
|---|---|---|---:|
| Claude Haiku 4.5 | Anthropic | Primary headline model; signed-thinking-block multi-turn validation; cross-Anthropic verification against PoC Sonnet | $0.04 |
| Gemini 2.5 Flash | Google | Cross-vendor + cross-cost-tier (vs PoC Pro) verification | $0.015 |
| OSS reasoning model (self-hosted) | Open weights | Third provider family; zero per-token cost on Ares GPUs | $0 |

Three families preserves C1's cross-vendor claim. Haiku and Flash are both
already validated in the 88-probe PoC corpus (so we know the rules fire and
slack windows exist on these cost tiers). The OSS slot replaces the
OpenRouter-DeepSeek path from the PoC — same cross-vendor signal at zero
marginal cost, at the price of a one-day vLLM setup on Day 2.

## What we already have vs. what we need to run

The 88-probe PoC corpus at `poc/runs/` is **not discarded.** Three uses:

1. **Re-score against the frozen rule library** (E2). The PoC byte metrics
   were computed with PoC-time rules; once we lock `RULE_LIBRARY_VERSION`
   on Day 1, we re-run the predictor over existing `stream.jsonl` files.
   No new LLM calls. This is the empirical-ground-truth re-score described
   in §11.8 Day 1.
2. **PoC Sonnet + Gemini-Pro data stays as cross-cost-tier validation.**
   Reviewers see "cheap-model trio is the headline; PoC shows the same
   architecture works on the expensive tier too."
3. **The leave-one-out (E3) and auto-rule (E11) work runs against the
   existing trace corpus**, no new LLM calls.

What we **do** need to run (Campaigns A + B below).

## Campaign A — Trace-only (Days 1-4)

Single-turn probes to characterize slack and predictor accuracy on the new
model trio. Re-uses the workspace_prior + planning_prompt machinery from the
PoC script (will live in `src/agentstage/cli/probe.py` after the port).

### Matrix

| Workload | Haiku 4.5 | Gemini Flash | OSS | Total |
|---|---|---|---|---:|
| aiob_104 IGSR | 5 × {none, PP, strict-PP} × T1 = 15 | 5 × {none, PP} × T1 = 10 | 3 × PP × T1 | 28 |
| aiob_107 GOES | 15 | 10 | 3 | 28 |
| aiob_110 NWB | 15 | 10 | 3 | 28 |
| code_repo | 15 | 10 | 3 | 28 |
| aiob_101 ERA5 (edge case) | 3 × PP × T1 | 3 × PP × T1 | — | 6 |
| **Subtotal Campaign A** | **63** | **43** | **12** | **118 new probes** |
| E8 thinking-budget sweep (aiob_110 only) | 3 × PP × T1 × {1k, 2k, 4k, 8k, 16k} budgets = 15 | — | — | 15 |
| **Total trace probes** | **78** | **43** | **12** | **~133 probes** |

Plus turn-2 opportunistic probes (Anthropic signed-thinking-block passthrough
exercise, per §6.8): 3 tasks × Haiku × T2 × PP × 3 seeds = 9 probes.

**Cost: 78 × $0.04 + 43 × $0.015 + 12 × $0 + 9 × $0.04 = ~$4.20**

### Per-cell acceptance criteria (Campaign A)

A cell is **done** when:
- All planned seeds produced a non-empty `stream.jsonl`
- Aggregated `byte_metrics.json` shows tier-1 byte recall computed against
  the cell's ground truth (static + empirical where available)
- The (task, model, prompt, turn) tuple has a row in `paper_evals/.results/report.json`'s
  `table_tier1_byte_recall_per_config` after running `pytest paper_evals/ -m h3`

Failed seeds (HTTP errors, empty thinking blocks on Anthropic turn-2,
provider rate limits) are **re-run up to 2 times**, then either filled with
a stand-in seed or recorded as a known gap in the cell.

## Campaign B — End-to-end (Days 5-10)

Multi-turn agent runs through the full proxy + predictor + stager + path-
rewriting shim. Each run is a complete task execution; cost is dominated by
turn count × per-turn token cost.

### Matrix

| Eval | Workload | Models | Configs | Seeds | Runs | $/run | Subtotal |
|---|---|---|---|---:|---:|---:|---:|
| E5 staging effectiveness | aiob_107, aiob_110 | Haiku, OSS | {with-stager, baseline} | 5 | 40 | $0.40 | $16 |
| E6 BW sensitivity | aiob_107 | Haiku | 1 measured BW × {with, baseline} | 5 | 10 | $0.40 | $4 |
| E7 graceful degradation | aiob_110 | Haiku, Flash | {proxy-on, proxy-off} | 3 | 12 | $0.30 | $3.60 |
| E9 SAB end-to-end (mandatory L2) | 3 SAB tasks | Haiku, Flash | {with, baseline} | 5 | 60 | $0.65 | $39 |
| E10 SWE-bench Lite (mandatory L2) | 2 instances | Haiku | {with, baseline} | 3 | 12 | $1.00 | $12 |
| **End-to-end subtotal** | | | | | **134 runs** | | **~$75** |

### Cost projection summary

| Bucket | Cost |
|---|---:|
| Campaign A (trace-only, new probes) | $4.20 |
| Campaign B (end-to-end) | $75 |
| 1.5× re-run / debugging buffer | $40 |
| **Projected total** | **~$120** |

Headroom under the $150 ceiling: **~$30** for unanticipated re-runs,
expanded SAB task set (3 → 4), or an OSS-model retry if the Day-2 vLLM
choice doesn't behave.

### Per-cell acceptance criteria (Campaign B)

A run is **done** when:
- The agent executed at least one tool call (i.e., not blocked on prompt)
- `staging_report.json` contains per-tool-call first-read latencies for both
  baseline and with-stager configs
- For SAB / SWE-bench: the benchmark's verdict (`task.solved` or equivalent)
  is recorded. AgentStage is **not** required to improve solution accuracy
  — it must not degrade it.

## Ground-truth provenance per workload

| Workload | Static GT source | Empirical GT source |
|---|---|---|
| aiob_104 IGSR | `src/agentstage/workloads/aiob.py::AIOB_104_GROUND_TRUTH` (ported from PoC line ~485) | `/home/iyildirim/projects/sciiobench/outputs/aiob_104_.../io_report.json` (gpt-4.1 run) |
| aiob_107 GOES | port from PoC | sciiobench gpt-4.1 io_report |
| aiob_110 NWB | port from PoC | sciiobench gpt-4.1 io_report |
| aiob_101 ERA5 | port from PoC (kept as honest edge case) | none — structural-ambiguity workload |
| code_repo | `src/agentstage/workloads/aiob.py::CODE_REPO_GROUND_TRUTH` | none — static GT only |
| SAB tasks #1-3 | extract from SAB task spec (`external/benchmarks/scienceagentbench/`) | end-to-end run produces `io_report.json` |
| SWE-bench instances | extract from instance spec | end-to-end run |

The empirical-ground-truth join (E2 re-score) reads `io_report.json`
files in sciiobench and intersects with `prediction.json` per-tier outputs.
Implementation lives in `src/agentstage/metrics/empirical_gt.py` (to be
written on Day 1).

## OSS model setup (Day 2)

To-be-decided model: Qwen3-Thinking 14B or DeepSeek-R1-Distill-Qwen-14B.
Decision criteria (in order of importance):

1. Fits in available GPU memory on the Ares allocation we have access to
2. Has clean SSE thinking semantics (`<think>` tags or equivalent) that the
   proxy can parse without per-vendor special-casing
3. Produces non-trivial thinking content (≥ 1 s of slack) on at least
   aiob_110 and code_repo

Setup steps:
1. Identify Ares GPU node with ≥ 32 GB VRAM headroom
2. Install vLLM (uv-managed in a separate venv or via the project's `[dependency-groups].vllm` optional group)
3. Serve the chosen model with `--enable-reasoning` (vLLM 0.7+) and SSE
   streaming on `localhost:8000`
4. Add `OSS_MODEL_BASE_URL=http://localhost:8000` to `.env`
5. Verify with a one-shot probe on aiob_110

If neither candidate produces useful thinking signal in a half-day, drop
the OSS slot and run the campaign with Haiku + Flash only. Updates
[[project-escience26-deadline]]'s plan accordingly.

## Rerun policy

- **No silent data accumulation.** Each probe / run records the git commit
  hash of `agentstage` and the `RULE_LIBRARY_VERSION` it used. Mixed-version
  data in the same cell is a **bug**, not an option.
- **Frozen rules cannot be quietly updated.** Bumping
  `RULE_LIBRARY_VERSION` is a deliberate, commit-message-documented event.
  Re-runs of completed cells against a new version are tagged in their
  output directory and excluded from H6's frozen-rules-cross-corpus
  assertion unless the version pin is updated.
- **Per-cell rerun cap = 2 retries.** After that, document the failure in
  this file (a row in the "Known gaps" section below) rather than retrying
  indefinitely.

## Known gaps (populated as the campaign progresses)

_None yet — campaign begins after Day 1 rule freeze._
