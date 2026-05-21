# AgentStage Campaign Plan

The experimental campaign that produces the data backing the eScience '26
paper. Source of truth for which probes / end-to-end runs we intend to
execute, against which models, on which workloads, with what seed budget,
under what protocol, and at what cost.

## Locked decisions (2026-05-19)

- **Three-model trio:** Claude Haiku 4.5, Gemini 2.5 Flash, one self-hosted
  OSS reasoning model (Qwen3-Thinking or DeepSeek-R1-Distill — pick during
  Day-2 vLLM setup based on Ares GPU memory headroom).
- **Spending ceiling: $150 (hard).** Projected ~$50 with the protocol
  decisions below; the rest is re-run buffer.
- **Client-library-primary architecture.** `src/agentstage/client/`
  provides drop-in `AnthropicClient`, `OpenAIClient`, `GeminiClient` that
  wrap the underlying SDKs, intercept streaming events, run the detector,
  dispatch prefetch to the stager, and optionally return `DataHint`
  objects to the caller. The HTTP proxy becomes a thin wrapper around the
  client lib for non-Python harnesses (e.g. SWE-bench in Docker).
- **Outputs live under `outputs/`** (gitignored). PoC corpus moved to
  `outputs/poc/` as one sub-campaign; the `poc/` directory holds only
  the historical `probe_reasoning_slack.py` script for reference.
  Matches AgentIOBench convention.
- **15-turn hard cap on end-to-end runs**, with soft termination at "first
  significant write" (≥ 1 KB to the task output dir, after ≥ 3 tool calls).
- **Cold cache mandatory for end-to-end runs.** Reuse
  `agentiobench.utils.cache.evict_dataset` (POSIX_FADV_DONTNEED, no root)
  and log `measure_temperature` into each run's `verdict.json`.
- **agentiobench is a submodule** at `external/benchmarks/agentiobench/`
  pinned at `29a2070`. `src/agentstage/workloads/aiob.py` is a thin
  adapter over its task config YAMLs. AgentStage integration goes on
  the `feat/agentstage-integration` branch of the agentiobench repo
  (see "AIOB integration branch" below).
- **External benchmarks: SAB + KramaBench** (replaces the earlier
  SWE-bench plan). SAB at ICLR 2025; KramaBench is preprint (MIT DB
  Lab, `@misc{lai2025KramaBench}`) — flagged in the paper with a
  footnote. KramaBench picked over SWE-bench Lite because its I/O
  profile (multi-domain raw-data pipelines, 1.7 GB across 1764 files)
  is far closer to AgentStage's scientific-HPC use case than
  SWE-bench's small-Python-file repos.
- **Framing decided on the fly** when writing §1.

## Why this trio

| Model | Provider family | Role | $/probe | $/multi-turn run (15 turns) |
|---|---|---|---:|---:|
| Claude Haiku 4.5 | Anthropic | Primary headline; signed-thinking-block multi-turn; Anthropic-family cross-check vs PoC Sonnet | $0.04 | $0.30-0.50 |
| Gemini 2.5 Flash | Google | Cross-vendor + cross-cost-tier (vs PoC Pro) | $0.015 | $0.10-0.20 |
| OSS reasoning (self-hosted vLLM) | Open weights | Third provider family; zero per-token cost on Ares GPUs | $0 | $0 |

Three families preserves C1's cross-vendor claim. All three already
validated in the PoC corpus (Haiku and Pro in outputs/poc/; Flash via Pro's
SSE structure shared across the Gemini family; DeepSeek-R1 via OpenRouter
sample). The OSS slot replaces OpenRouter-DeepSeek at zero marginal cost.

## What we already have vs. what we need to run

The PoC corpus, now at `outputs/poc/`, is **not discarded.** Three uses:

1. **Re-score against the frozen rule library** (E2): re-run the detector
   over the existing 88 `stream.jsonl` files using the frozen rules. No
   LLM calls; cheap. PoC traces are just another set of output dirs under
   `--outputs-root outputs/`; the Campaign indexer treats them like any
   other sub-campaign.
2. **PoC Sonnet + Gemini-Pro data stays as cross-cost-tier validation.**
3. **Leave-one-out (E3) and auto-rule (E11) work runs against existing
   trace data**, no new LLM calls.

What we **do** need new runs for:

- Fresh probes on Haiku / Flash / OSS for the Campaign-A trace matrix
- All end-to-end runs (Campaign B) — the PoC was trace-only
- Cross-corpus runs on SAB and SWE-bench Lite (E9, E10)

## Output directory layout

All new runs land under `outputs/<run_id>/`. Convention:

```
outputs/
├── <task>_<model>_<config>_s<seed>/
│   ├── stream.jsonl           # raw SSE events (trace + end-to-end)
│   ├── summary.json           # block-level timing
│   ├── detection.json        # per-rule activations + tier outputs
│   ├── byte_metrics.json      # per-tier byte recall + overfetch
│   ├── cost.json              # input/thinking/output tokens + USD spend
│   ├── verdict.json           # end-to-end only: trajectory, cache temperature, validator outcome
│   ├── io_report.json         # end-to-end only: DFTracer-produced I/O summary
│   └── staging_report.json    # end-to-end only: per-tool-call first-read latencies
```

`<config>` encodes the cell: e.g. `trace_t1_b8k_pp` for a trace-only turn-1
probe at 8k thinking budget with planning prompt; `e2e_with_stager_50mbps`
for an end-to-end run with stager at 50 MB/s cold-tier bandwidth.

## Campaign A — Trace-only single-turn probes (Days 1-4)

Single-turn LLM calls to characterize slack + detector accuracy. Runs via
`agentstage.client.AnthropicClient` etc. in `--trace-only` mode (no stager,
no real workspace I/O). Cold cache not required (no agent file reads).

### Matrix

| Workload | Haiku 4.5 | Gemini Flash | OSS | Total |
|---|---|---|---|---:|
| aiob_104 IGSR | 5 × {none, PP, strict-PP} × T1 = 15 | 5 × {none, PP} × T1 = 10 | 3 × PP × T1 | 28 |
| aiob_107 GOES | 15 | 10 | 3 | 28 |
| aiob_110 NWB | 15 | 10 | 3 | 28 |
| code_repo | 15 | 10 | 3 | 28 |
| aiob_101 ERA5 (edge case) | 3 × PP × T1 | 3 × PP × T1 | — | 6 |
| **Subtotal main** | **63** | **43** | **12** | **118** |
| E8 budget sweep (aiob_110, Haiku) | 3 × PP × T1 × {1k, 2k, 4k, 8k, 16k} | — | — | 15 |
| Turn-2 opportunistic (Haiku signed-thinking-block) | 3 tasks × 3 seeds | — | — | 9 |
| **Campaign A total** | **87** | **43** | **12** | **142 probes** |

**Cost: 87 × $0.04 + 43 × $0.015 + 12 × $0 = ~$4.20**

## Campaign B — End-to-end multi-turn runs (Days 5-10)

Full agent runs through `AgentStageClient` + stager + path-rewriting shim.
Each run executes the workload's task; detector and stager run live.

### Per-run protocol

1. **Pre-run cold-cache eviction:** call
   `agentiobench.utils.cache.evict_dataset(task.dataset_root)`
2. **Temperature snapshot:** `measure_temperature(task.dataset_root)` →
   log result into `verdict.json["pre_run_temperature"]`
3. **Run agent** with `max_turns=15` and soft-stop callback
4. **Soft-stop trigger:** when agent writes ≥ 1 KB to the task output dir,
   AND `turn_idx >= 3`, inject a "stage 1 complete; stop here" tool
   result so the agent terminates gracefully on its next turn
5. **Post-run temperature snapshot** → `verdict.json["post_run_temperature"]`
6. **DFTracer → io_report.json** via dfanalyzer

### Matrix

| Eval | Workload | Models | Configs | Seeds | Runs | $/run | Subtotal |
|---|---|---|---|---:|---:|---:|---:|
| E5 staging effectiveness | aiob_104, aiob_107, aiob_110 | Haiku, OSS | {with-stager, baseline} | 5 | 60 | $0.30 | $18 |
| E6 BW sensitivity | aiob_107 | Haiku | 1 measured BW × {with, baseline} | 5 | 10 | $0.30 | $3 |
| E7 graceful degradation | aiob_110 | Haiku, Flash | {hint-on, hint-off} | 3 | 12 | $0.20 | $2.40 |
| E9 SAB end-to-end | 3 SAB tasks | Haiku, Flash | {with, baseline} | 5 | 60 | $0.35 | $21 |
| E10 KramaBench end-to-end | 3 tasks (Astronomy + Biomedical + Wildfire) | Haiku, Flash | {with, baseline} | 5 | 60 | $0.30 | $18 |
| **Campaign B total** | | | | | **202 runs** | | **~$63** |

### Cost projection summary

| Bucket | Cost |
|---|---:|
| Campaign A (trace-only, new probes) | $4 |
| Campaign B (end-to-end, 15-turn cap) | $63 |
| 1.5× re-run / debugging buffer | $32 |
| **Projected total** | **~$99** |

Headroom under the $150 ceiling: **~$50** for OSS-model thinking-content
surprises, expanded SAB or KramaBench task sets, Sonnet sanity probes
against the frozen rules, or a fourth-model expansion if a reviewer asks.

## Per-cell acceptance criteria

A trace-only cell (Campaign A) is **done** when:
- All planned seeds produced non-empty `stream.jsonl`
- `byte_metrics.json` exists with tier-1 + tier-3 byte recall vs static GT
- The cell appears in `paper_evals/.results/report.json`'s
  `table_tier1_byte_recall_per_config` after `pytest paper_evals/ -m h3`

An end-to-end cell (Campaign B) is **done** when:
- All planned seeds completed (terminated by soft-stop or hit turn cap)
- `verdict.json` records `pre_run_temperature.resident_frac ≤ 0.05`
  (cold-cache eviction worked)
- `io_report.json` produced and contains `file_name_view` entries
- `staging_report.json` records per-tool-call first-read latencies for
  both the `with-stager` and `baseline` configs
- For SAB / SWE-bench: benchmark's verdict (`task.solved` or equivalent)
  is recorded. AgentStage is **not** required to improve solution
  accuracy — it must not degrade it (paired sign test, α = 0.05)

### Sample-thinness gate

Drop any end-to-end run from analysis where `len(trajectory) < 3`. Too
thin a sample to score the working-set detector against.

## Cold cache + temperature logging

```python
from agentiobench.utils.cache import evict_dataset, measure_temperature

# Before each end-to-end run:
evict_dataset(task.dataset_root)
verdict["pre_run_temperature"] = measure_temperature(task.dataset_root)

# After:
verdict["post_run_temperature"] = measure_temperature(task.dataset_root)
```

`measure_temperature` returns `{resident_frac, eviction_works,
page_cache_active}`. Acceptance gate: `pre_run_temperature.resident_frac
≤ 0.05` on the eviction-supported filesystems (XFS, ext4 — Ares default).

If `eviction_works == False` (filesystem doesn't honor
POSIX_FADV_DONTNEED, e.g. some NFS configs), document the gap in the
"Known gaps" section and either skip the cell or use a different mount.

## Turn limit + soft termination

```python
# In src/agentstage/runner.py
MAX_TURNS = 15
SOFT_STOP_MIN_TURNS = 3

for turn in range(MAX_TURNS):
    response = client.next_turn(messages)
    # ... handle tool calls ...
    if turn >= SOFT_STOP_MIN_TURNS and benchmark_runner.wrote_significant_output(task):
        messages.append(_stop_message("Stage 1 complete. Reading-phase
                          measurement done; stopping here for the campaign."))
        break
```

`wrote_significant_output(task)` is **per-benchmark-configurable** —
each benchmark's runner defines what counts as "first significant write
to the task output target":

| Benchmark | Predicate |
|---|---|
| AIOB | `Path(task.dataset_root / "output" / task.output_fname).stat().st_size >= 1024` (uses `TaskConfig.output_fname` from the task YAML) |
| SAB | a notebook cell or analysis output file ≥ 1 KB in SAB's output dir |
| KramaBench | a pipeline-script write or final-answer artifact ≥ 1 KB in `{domain}_results/` |

The default 1 KB threshold is configurable per-benchmark via the runner's
constructor; sub-1 KB defaults exist for benchmarks where outputs are
expected to be small (e.g. one-line answers).

## Ground-truth provenance

| Workload | Static GT source | Empirical GT source |
|---|---|---|
| aiob_104 IGSR | `external/benchmarks/agentiobench/agentiobench/config/task/aiob_104_*.yaml` | E2 re-score against `$SCIIOBENCH_ROOT/outputs/aiob_104_.../io_report.json` (historical gpt-4.1 runs); E5+ runs produce their own io_report.json |
| aiob_107 GOES | same | same |
| aiob_110 NWB | same | same |
| aiob_101 ERA5 | same (kept as honest edge case) | structural-ambiguity workload; static GT only |
| code_repo | static enumeration in `src/agentstage/workloads/code_repo.py` (ported from PoC) | E5+ runs produce their own io_report.json |
| SAB tasks | extract from SAB task spec (`external/benchmarks/scienceagentbench/`) | E9 runs produce their own io_report.json |
| KramaBench tasks | extract from `external/benchmarks/kramabench/data/<domain>/{domain}.json` (per-task `reference pipeline` + sub-task list) | E10 runs produce their own io_report.json; KramaBench's own gold pipelines define the eventual working set |

The empirical-GT join (E2 re-score) is implemented in
`src/agentstage/metrics/empirical_gt.py`: reads `io_report.json`, extracts
files with `posix_count_sum > 0` and `posix_read_size_sum > 0` from
`file_name_view[*]`, intersects with the detector's tiered detected set
from `detection.json`.

## OSS model setup (Day 2) — via NCSA Delta, not Ares

Model: **Qwen/Qwen3.6-27B with thinking enabled**. Reuses the model
AgentIOBench already validated on Delta (HF cache, vLLM venv, Lmod stack
all pre-built); only the chat-template kwarg `enable_thinking` flips from
false to true, and `--reasoning-parser qwen3` becomes load-bearing
(separates thinking from visible content in the SSE stream).

Topology (mirrors AIOB's two-cluster setup):

```
  Ares (this repo, this machine)                  Delta AI (NCSA, bekn alloc)
  ──────────────────────────────                  ────────────────────────────
  agentstage.client.OpenAIClient                  gh<NNN> compute node
      │                                                  │
      ▼  SSH tunnel: localhost:8002 → gh<NNN>:8002       │
   POST /v1/chat/completions ───────────────────────►  vLLM serving
                                                        Qwen/Qwen3.6-27B
                                                        --reasoning-parser qwen3
                                                        enable_thinking=true
```

Workflow:

```bash
# ── On Delta AI (gh-login01.delta.ncsa.illinois.edu) ──────────────────
# 1. 2-GPU allocation, 2h timebox (T15-equivalent)
salloc -A bekn-dtai-gh -p ghx4-interactive \
       --gres=gpu:2 --cpus-per-task=32 --mem=300G \
       --time=02:00:00

# 2. Sync agentstage's launcher scripts to Delta (rsync from Ares)
rsync -av <ares>:agentstage/scripts/delta/ ~/dtai/agentstage/
cd ~/dtai/agentstage

# 3. Launch — sources bekn policy from ~/.bashrc, prints "BOTH SERVERS READY"
./run_vllm_qwen3_thinking.sh
# Note the gh<NNN> hostname from the banner

# ── On Ares ───────────────────────────────────────────────────────────
# 4. Tunnel
./scripts/delta/tunnel.sh gh<NNN>

# 5. Verify reasoning_content streams (in another shell)
./scripts/delta/verify_vllm_thinking.sh

# 6. Set in .env:
#    OSS_MODEL_BASE_URL=http://localhost:8002/v1
#    OSS_MODEL_NAME=Qwen/Qwen3.6-27B
```

**2-hour timebox** (down from the originally-budgeted 4 hours — the hard
parts are pre-solved by AIOB). If thinking content fails to stream after
2 hours, drop the OSS slot and run with Haiku + Flash only (loses
third-provider-family C1 framing in the **new** campaign, but the PoC
DeepSeek-R1 traces re-scored against frozen rules on Day 1 still carry
the third-family signal across the combined dataset — see project memory
`project_campaign_plan.md`).

Important client-side implication: vLLM's `--reasoning-parser` extends the
OpenAI-compatible API with a `delta.reasoning_content` field separate from
`delta.content` in streaming chunks. `src/agentstage/client/openai.py` (and
`client/http.py` for the urllib path) must read both fields and feed
`reasoning_content` to the detector as the thinking stream. Documented in
T19 and T22.

## Client library architecture

The drop-in `AgentStageClient` is the primary integration path:

```python
from agentstage import AnthropicClient   # was: from anthropic import Anthropic

client = AnthropicClient(
    api_key=...,
    workspace=workspace_prior_for_task,
    stager=local_stager_or_none,            # None → trace-only mode
    rule_library_version="v1",              # asserts matching frozen rules
)

# 100% compatible with the underlying SDK call:
response = client.messages.create(model="claude-haiku-4-5", ..., stream=True)
for chunk in response:
    # caller sees stream identical to direct SDK; AgentStage tees
    # internally to the detector + stager
    ...

# Optional: pull hints back
for hint in client.last_data_hints():
    # hint = DataHint(detected_files, tier, fired_at_ms, rule_id, byte_estimate)
    log.info(f"Prestaged tier-{hint.tier}: {len(hint.detected_files)} files")
```

Three client implementations (Day 3-4):
- `client/anthropic.py` — wraps `anthropic.Anthropic`
- `client/openai.py` — wraps `openai.OpenAI`
- `client/gemini.py` — wraps `google.generativeai` (or `google-genai`)
- `client/http.py` — raw urllib for matching PoC + AgentIOBench style;
  bypasses the SDK for benchmark environments where it isn't installed

The proxy (`src/agentstage/proxy/server.py`) becomes a thin HTTP wrapper
around the client library. Same detector + stager; just exposed over
HTTP for non-Python harnesses (e.g. SWE-bench in Docker, future external
research agents). **Lower priority than the client lib.**

## AIOB integration branch

AgentIOBench is our project. Integration goes on a branch of the
upstream repo (`git@github.com:grc-iit/agentiobench.git`) named
**`feat/agentstage-integration`**. Two minimal upstream additions:

1. **Stable public API re-export** in `agentiobench/__init__.py`:
   re-export `TaskConfig`, `evict_dataset`, `measure_temperature`,
   `dftracer_context` so AgentStage imports go through
   `from agentiobench import ...` without reaching into private modules.

2. **Optional pre/post-turn callbacks** in `agentiobench/runner.py::run_task`:
   ```python
   def run_task(..., pre_turn_hook=None, post_turn_hook=None):
       for turn in range(max_turns):
           if pre_turn_hook: pre_turn_hook(turn, messages)
           response = client.complete(messages)
           if post_turn_hook: post_turn_hook(turn, response, trajectory)
           ...
   ```
   AgentStage's runner registers these to drive the client + stager +
   cache logging without forking AIOB's agent loop.

Our submodule pin (`external/benchmarks/agentiobench/`) tracks
`feat/agentstage-integration` until those hooks merge to AIOB's main
when the AIOB-companion paper ships.

## External benchmark integration

| Benchmark | Mechanism | Effort |
|---|---|---|
| AIOB | Own runner (`src/agentstage/runners/aiob_runner.py`) importing primitives from the branch above | Day 1-4 alongside the rest of the package |
| SAB | Runtime monkey-patch of `openai.OpenAI` before SAB's harness imports it | Day 8 (T41) |
| KramaBench | Same monkey-patch pattern as SAB; KramaBench's `systems/baseline_example.py` uses standard SDK clients | Day 9 (replaces T46-T47) |

**Footnote in paper:** "We evaluate on two externally-released
benchmarks: ScienceAgentBench (Chen et al., ICLR 2025) and KramaBench
(Lai et al., 2025; preprint at MIT DB Lab). KramaBench's preprint
status reflects its recency as a benchmark designed for end-to-end
data-science agents; we treat it as a credible cross-corpus probe but
acknowledge it has not yet completed peer review."

## Rerun policy

- **Each run records:** `agentstage_commit` (git SHA), `rule_library_version`,
  `agentiobench_commit`, model name, prompt mode, seed, turn cap, cold-cache
  temperature snapshots
- **Mixed-version data in the same cell is a bug.** When
  `RULE_LIBRARY_VERSION` is bumped, all cells in `paper_evals/` runs use
  the new version; older runs get re-tagged as "version <X>" and excluded
  from H6's frozen-rules-cross-corpus assertion unless explicitly opted in
- **Per-cell retry cap: 2.** After that, record in "Known gaps" below
- **Soft-stop is not a failure.** A run that hits soft-stop is *complete*

## Known gaps (populated as the campaign progresses)

_None yet — campaign begins after Day 1 rule freeze._

---

## Campaign C — I/O-leakage-bias ablations (Days 5-10, parallel with B)

Added 2026-05-20 in response to the I/O-leakage audit
([`IO_LEAKAGE_AUDIT.md`](IO_LEAKAGE_AUDIT.md)). Smoke versions of these
ran on a single workload × single model and are recorded as E-011
through E-017. This section sizes the **paper-grade** ablation suite.

### Goals (each measured per cell)

For every (model, workload, regime) triple, we record four numbers:

1. **Activation count by source** — how many rules fire, attributed to
   `thinking` / `text` / `tool_result` (detector-extension contribution)
2. **Precision / recall / Jaccard** — false-positive rate on prefetched
   files (E-016 methodology)
3. **Oracle vs. realistic wall-time** — what the architecture *could*
   save vs. what this run *did* save (E-017 methodology)
4. **Slack-window utilization** — did the detector's first-turn hint
   land inside the agent's thinking-to-first-tool slack? (yes/no)

### Per-cell protocol

1. Cold-cache eviction + temperature snapshot (same as Campaign B)
2. Run `path_b_multiturn.py --prompt-mode <hinted|sparse>` with up to
   15 turns (paper-grade) or 8 turns (smoke). For each run:
   - Save full multi-turn corpus under
     `outputs/multi_turn/<model>_<workload>_<regime>_<seed>_<ts>/`
3. **Post-run analyses** (offline, no extra API cost):
   - `path_b_falsepos.py` → `falsepos.json` (E-016 metrics)
   - `path_b_walltime.py` → `walltime_replay.json` (E-017 metrics)
   - `path_b_replay.py`   → `replay_variants.json` (Variant A→D lift)

### Matrix (paper-grade)

| Workload | Models | Regimes | Seeds | Runs |
|---|---|---|---:|---:|
| aiob_107 (or aiob_107_s3) | Haiku, Gemini Flash, OSS | hinted, sparse | 3 | 18 |
| aiob_110 | Haiku, Gemini Flash, OSS | hinted, sparse | 3 | 18 |
| aiob_104 | Haiku, Gemini Flash, OSS | hinted, sparse | 3 | 18 |
| KramaBench (natural-sparse, Regime C) | Haiku, Flash | (single regime) | 3 | 6 |
| **Total** | | | | **60 runs** |

**Cost projection (Campaign C):**
60 runs × $0.30/run ≈ **$18** (similar to Campaign B's E5 cell — same
multi-turn cost profile).

**Total campaign with Campaign C added: ~$117** under the $150 ceiling.
Drops the "1.5× re-run buffer" margin from $32 to $14, which is still
safe given each cell only burns ~$0.30.

### Smoke-vs-full comparison

| Dimension | Smoke (E-011 — E-017) | Full Campaign C |
|---|---|---|
| Models | Haiku only | Haiku + Gemini Flash + OSS |
| Workloads | aiob_107_s3 only | aiob_107 + aiob_110 + aiob_104 + KramaBench |
| Regimes | hinted, sparse, sparse_live (1 each) | hinted × 3 seeds, sparse × 3 seeds |
| Turn cap | 8 | 15 |
| Ablation runs | 3 (E-016, E-017, E-012 ablation triplet) | 60 |
| Cost | ~$2 (one session) | ~$18 |
| Coverage | "exists" proof | per-model, per-workload, statistical replicates |

### What the smoke ablations already establish (and what they don't)

**Established by smoke** (and need only confirmation, not novelty):
- The architecture works end-to-end (E-010, E-015 confirm shim + stager + multi-turn agent all compose)
- The multi-turn detector extensions (`tool_result`-aware + `text`-aware) lift activation count by +100% to +300% over thinking-only
- Hinted-vs-sparse regime separation produces a measurable behavior delta (E-016: 100% vs 0% Jaccard; E-017: 3886× vs 1.0× realized wall-time)

**NOT established by smoke** (needs full Campaign C):
- Per-model variance — only Haiku tested so far; Gemini Flash and OSS may show different rule-activation patterns
- Per-workload generality — only aiob_107 tested; aiob_110 (NWB) and aiob_104 (genomics) have different file naming conventions and may stress different rules
- Statistical confidence — single-seed observations; 3-seed minimum needed for per-cell mean ± std
- KramaBench naturalistic regime (Regime C) — Campaign B's E10 cell already plans this, but we haven't run it; comparing C to A+B is novel
- Whether sparse-mode 0% precision is a STABLE finding or a single-seed unlucky draw

### Schedule fit

Campaign C runs on top of Campaign B's existing infrastructure:
- Day 5-6 (T29-T35): builds the Path B multi-turn runner. **Already done** as part of the E-011 smoke work.
- Day 7-8 (T36-T42): runs Campaign B's E5 cell (3 workloads × 2 models × 5 seeds × 2 configs = 60 runs). Adding Campaign C means *also* running the same workloads × models × **hinted/sparse** axis. Doubles the count but same machinery.
- Day 9-10 (T43-T49): adds the 3 external benchmarks (KramaBench, SAB) end-to-end. KramaBench is "naturally sparse" per the audit; running it satisfies both Campaign B's E10 cell AND Campaign C's Regime C cell with zero extra work.

### Acceptance criteria for Campaign C

A Campaign C cell is **done** when:
- All planned seeds produced a captured corpus under `outputs/multi_turn/`
- `falsepos.json` written per corpus (precision, recall, Jaccard)
- `walltime_replay.json` written per corpus (oracle, realistic speedup)
- `replay_variants.json` written per corpus (detector-extension lift)
- The per-model, per-workload, per-regime aggregate is written to
  `outputs/campaign_c/<model>_<workload>_<regime>_aggregate.json`

A Campaign C cell **passes** the H3 hypothesis test (working-set
predictability) at:
- **Regime A (hinted)**: byte recall ≥ 0.85, byte overfetch ≤ 1.5× (paper claim C2 baseline)
- **Regime B (sparse)**: Jaccard ≥ 0.50, realistic wall-time speedup ≥ 2× over baseline (lower bar — acknowledges static rule library limitation)
- **Regime C (KramaBench)**: byte recall ≥ 0.50 (per AGENTSTAGE.md §11.10 Level 2)

### Why split the campaign rather than rolling C into B

Campaign B was designed around the existing single-regime hypothesis
(H3). Campaign C is a *threats-to-validity* ablation — it answers
"would your numbers survive a stripped prompt?" rather than "does
your system work?" Bundling them risks reviewers confusing "we ran 60
end-to-end multi-turn runs" with "the existing C2 claim is now
multi-regime." Keeping them separate clarifies that C is the *audit*
arm and B is the *main results* arm.

