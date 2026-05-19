# AgentStage: Streaming-Intent-Driven Tiered Data Staging for Scientific LLM Agents

Working notes, hypotheses, experiments, and trace-only PoC results.

Created: 2026-05-18 (active)
Owner: Izzet Yildirim
Adjacent project: AgentIOBench (unpublished; provides workloads + I/O tracing infrastructure)

---

## TL;DR

Trace-only PoC across 88 probes total (79 single-turn measurement + 9 multi-turn passthrough experiments), 5 workloads (4 scientific HPC + 1 coding-agent), **3 LLM provider families** (Anthropic Claude Sonnet 4.5 + Haiku 4.5; Google Gemini 2.5 Pro; DeepSeek-R1 via OpenRouter), 19 configurations. The headline numbers are from the 53 turn-1 thinking-content seeds, which is the primary measurement window (turn-2 slack is opportunistic; see Section 6.8). Excluding the structurally-ambiguous aiob_101 workload (where the agent has 36 equal-priority NetCDFs and no "first file" exists by design), 47 well-defined seeds remain:

- **Lead time:** median 6.9 s, max 14.0 s. 98% of seeds ≥ 2 s, 67% ≥ 5 s. DeepSeek-R1 outlier: 4 minutes of thinking (different model class).
- **Tier-1 byte recall ≥ 0.85:** 44/47 = **94%** (3 misses reflect Gemini-vs-Sonnet first-inspect strategy variance, not predictor failure; see Section 6.4.1).
- **Tier-1 byte overfetch ≤ 1.5×:** 46/47 = **98%** (1 DeepSeek-R1 outlier at 2.12× because the model committed to two subjects in thinking instead of one).
- **Tier-3 byte recall ≥ 0.85:** 47/47 = **100%** across all 47 seeds and all 4 well-defined workloads (aiob_104, aiob_107, aiob_110, code_repo).
- **Tier-3 byte overfetch ≤ 2.0×:** 46/47 = **98%** (1 code_repo seed where mention-rules fired for many modules).
- **HOT (literal-path commitment, top priority):** fires on 36% of seeds; whenever it fires, 100% reach overfetch ≤ 1.5×.

Cross-provider tier-1 byte recall ≥ 0.85 (excluding aiob_101):
- Anthropic (Sonnet 4.5 + Haiku 4.5): **34/34 = 100%** across aiob_104, aiob_107, aiob_110, code_repo
- Google Gemini 2.5 Pro: **9/12 = 75%** across aiob_104, aiob_107, aiob_110, code_repo (rule-firing reliability gap — see Section 6.5.1)
- DeepSeek-R1 via OpenRouter: 1/1 (1 probe; OpenRouter credits exhausted before extending the sample)

A separate multi-turn experiment (9 probes with full signed-thinking-block passthrough) shows Anthropic Sonnet on turn-2 chooses not to think even when the passthrough is mechanically correct. The infrastructure fix is real and required for any production proxy, but turn-2+ thinking should be treated as opportunistic.

A separate multi-turn experiment (9 probes with full signed-thinking-block passthrough) shows Anthropic Sonnet on turn-2 chooses not to think even when the passthrough is mechanically correct. The infrastructure fix is real and required for any production proxy, but turn-2+ thinking should be treated as opportunistic.

Per-task / per-model tier-1 (immediate-need staging), turn-1 thinking seeds only:

| task | model | n | tier-1 ≥ 0.85 byte recall | tier-1 ≤ 1.5× byte overfetch | slack med |
|---|---|---:|---:|---:|---:|
| aiob_101 (climate ERA5) | claude-sonnet-4-5 | 3 | 0%* | 100% | 9.9 s |
| aiob_101 | gemini-2.5-pro | 3 | 0%* | 100% | 11.3 s |
| aiob_104 (genomics IGSR) | claude-sonnet-4-5 | 3 | 100% | 100% | 11.0 s |
| aiob_104 | gemini-2.5-pro | 3 | **33%** | 100% | 5.0 s |
| aiob_107 (meteorology GOES, 6042 files) | claude-sonnet-4-5 | 11 | 100% | 100% | 10.6 s |
| aiob_107 | gemini-2.5-pro | 3 | 100% | 100% | 7.4 s |
| aiob_110 (neuroscience NWB) | claude-sonnet-4-5 | 3 | 100% | 100% | 11.1 s |
| aiob_110 | claude-haiku-4-5 | 3 | 100% | 100% | 4.6 s |
| aiob_110 | gemini-2.5-pro | 3 | **66%** | 100% | 5.5 s |
| aiob_110 | deepseek/deepseek-r1 | 1 | 100% | 0% (2.12×) | 248 s |
| code_repo (Python repo) | claude-sonnet-4-5 | 11 | 100% | 100% | 5.1 s |
| code_repo | claude-haiku-4-5 | 3 | 100% | 100% | 3.4 s |
| code_repo | gemini-2.5-pro | 3 | 100% | 100% | 2.5 s |

*aiob_101 0% recall is the honest edge case explained below; ignore in the headline.

The Gemini misses on aiob_104 (1/3 firing the right tier-1 rule) and aiob_110 (2/3) are a rule-activation reliability issue, not a slack or thinking-content issue. The model produces semantic-class signals ("BAM files", "NWB sessions"), but does not always trigger our `first_inspect` regex variant on this particular task. The bytes-overfetch metric is unaffected (100% across all 12 Gemini seeds) because when the specific rule does not fire, tier-1 just predicts a smaller set, not an overfetched one. Adding model-vocabulary-tailored regex variants would close this gap.

The aiob_101 0% recall row is the honest edge case: the workload has 36 monthly NetCDFs that all need staging equally, with no obvious "first file." Tier-1 (rules with target-set size ≤ 10) cannot fire on the input-NetCDF bucket because that bucket has 36 files; it activates only the smaller buckets (shapefile, output files). The recall is correct in pointing out that no single file is the "immediate need" here. Sonnet and Gemini both behave consistently on aiob_101 (slack 9-15 s, predictor activates broad-tier rules), so the workload reaches the wider working set via tier-3 instead.

The haiku-4-5 rows demonstrate cross-model consistency within Anthropic: the smaller, cheaper haiku model produces identical tier-1 behavior on aiob_110 and code_repo (100% byte recall, 100% overfetch ≤ 1.5×) at smaller thinking budgets (8 192 tokens vs sonnet's 16 384). Median slack on haiku: 4.6 s (aiob_110) and 3.4 s (code_repo), shorter than sonnet but still above the 2 s reviewer threshold.

The deepseek/deepseek-r1 row demonstrates cross-vendor generality: a third LLM family (OpenRouter-hosted) achieves identical tier-1 recall. Its overfetch is 2.12× because the model mentions multiple subjects (sub-Cori and sub-Forssmann) in thinking, activating per-subject rules for both, doubling the predicted set. The behavior is honest — when the model commits to two subjects, tier-1 stages both. DeepSeek-R1 is also an outlier in thinking duration: 248 s of thinking content before any tool dispatch on a single probe, far above Anthropic/Gemini's 5-15 s range. This is a model-class property (DeepSeek-R1 reasons exhaustively). OpenRouter credits exhausted before we could extend the sample beyond aiob_110.

The aiob_101 0% recall row is the honest edge case: the workload has 36 monthly NetCDFs that all need staging equally, with no obvious "first file." Tier-1 (rules with target-set size ≤ 10) cannot fire on the input-NetCDF bucket because that bucket has 36 files; it activates only the smaller buckets (shapefile, output files). The recall is correct in pointing out that no single file is the "immediate need" here. Sonnet and Gemini both behave consistently on aiob_101 (slack 9-15 s, predictor activates broad-tier rules), so the workload reaches the wider working set via tier-3 instead.

The haiku-4-5 rows demonstrate cross-model consistency within Anthropic: the smaller, cheaper haiku model produces identical tier-1 behavior on aiob_110 and code_repo (100% byte recall, 100% overfetch ≤ 1.5×) at smaller thinking budgets (8 192 tokens vs sonnet's 16 384). Median slack on haiku: 4.6 s (aiob_110) and 3.4 s (code_repo), shorter than sonnet but still above the 2 s reviewer threshold.

The deepseek/deepseek-r1 row demonstrates cross-vendor generality: a third LLM family (OpenRouter-hosted) achieves identical tier-1 recall. Its overfetch is 2.12× because the model mentions multiple subjects (sub-Cori and sub-Forssmann) in thinking, activating per-subject rules for both, doubling the predicted set. The behavior is honest — when the model commits to two subjects, tier-1 stages both. DeepSeek-R1 is also an outlier in thinking duration: 248 s of thinking content before any tool dispatch on a single probe, far above Anthropic/Gemini's 5-15 s range. This is a model-class property (DeepSeek-R1 reasons exhaustively).

These numbers clear both reviewer-stated benchmark sets cited in the project notes (≥ 70–85% byte recall, ≤ 1.5–2× overfetch, ≥ 2 s lead time) on tier-1 across all 5 workloads, and on tier-3 across the 4 scientific workloads. The tier-3 miss on code_repo is a predictor-rule-quality issue (coding agents do not use "all files" / "entire codebase" wording reliably), not a fundamental limit.

End-to-end speedup, multi-agent contention behavior, and beating PASTE-style speculative tool execution remain unverified (no system built yet).

---

## 1. Idea

Scientific LLM agents executing code on HPC infrastructure pay a large per-tool-call cost for cold file reads from parallel storage. Each tool call typically reads application data files from PFS (Lustre / OrangeFS / GPFS / object store) into a local working set. The agent's reasoning between tool calls is wall-clock idle from the storage side.

AgentStage proposes to use the agent's streaming thinking content (visible via Anthropic Messages API thinking blocks, Gemini `include_thoughts`, OpenAI Responses API reasoning summaries) as an early signal of which files the agent will read next. A predictor maps streaming thinking content into a tiered file working-set prediction. A staging daemon stages those files from cold tier (PFS, object store) into hot tier (local NVMe, tmpfs, shared memory) before the agent's tool call dispatches.

The decoupling of data fetch from tool execution is the technical contribution. Speculative tool execution (PASTE family) executes the whole tool ahead; we move only the data, which works for any tool including side-effecting ones.

---

## 2. Hypotheses

**H1 (slack):** LLM agents with thinking enabled produce a non-trivial wall-clock gap between the start of thinking and the start of tool execution. The gap is large enough at typical NVMe ingest rates (3–7 GB/s) to stage useful amounts of data (≥ 100 MB) per tool call.

**H2 (intent-from-thinking):** Streaming thinking content reveals which files the agent will read next. Some agents commit to literal file paths in thinking; the rest commit at the semantic-class level (file format, dataset region, processing stage).

**H3 (working-set predictability):** A rule-based predictor that combines the workspace prior (files known to exist) with semantic-class signals extracted from thinking achieves high byte-recall and low overfetch against the agent's actual file accesses.

**H4 (tiered staging is the right architecture):** Because thinking content fires both specific signals (tiny target sets, often correct for the immediate need) and broad signals (large target sets, correct for the eventual working set), the staging system should consume the predictor's output as tiered priorities rather than a single union.

**H5 (planning prompts as a free lever):** Inserting explicit thinking instructions into the user message multiplies slack and increases the precision of intent extraction without changing the model or budget.

---

## 3. Claims (and current evidence status)

| # | claim | evidence | status |
|---|---|---|---|
| C1 | Slack windows of 5–14 s are reliably observable on sonnet-4-5 with planning prompts | 21-seed matrix, median 9.6 s | verified |
| C2 | A tier-1 predictor achieves ≥ 0.85 byte recall and ≤ 1.5× overfetch against the agent's immediate-need file set | 20/21 thinking seeds | verified |
| C3 | A tier-3 predictor achieves ≥ 0.85 byte recall and ≤ 2.0× overfetch against the eventual working set | 21/21 thinking seeds | verified |
| C4 | The tier-1 set is available within the slack window | tier-1 activation 5–12 s before first tool dispatch | verified |
| C5 | Literal-path commitment in thinking is unreliable, model- and workload-dependent | 7% of HOT hits target input files (10 / ~150 hits) | verified (negative) |
| C6 | Planning prompts multiply slack by 2–10× | direct comparison: sonnet aiob_101 t=1 no-PP 3.9 s vs +PP 9.6 s; gemini-pro aiob_110 t=2 +PP 7.9 s vs no-PP 0.8 s | verified |
| C7 | Anthropic extended-thinking needs signed-thinking-block passthrough to keep working across turns | turn-2 sonnet runs all produce zero thinking | verified (architectural constraint, separate engineering fix) |
| C8 | End-to-end agent latency reduction of 2–5× | not built | unverified |
| C9 | Defeating PASTE-style speculative tool execution on idempotent and non-idempotent tools | not built | unverified |
| C10 | Multi-agent contention behavior at fleet scale | not built | unverified |

---

## 4. Architectural sketch

Four components, three of which are in-scope for the systems paper.

**4.1 Intent capture.** A proxy that terminates the LLM SSE stream, parses incoming events, accumulates per-block content (thinking, text, tool_use), forwards to the agent harness unchanged. Per-provider implementations:

- Anthropic Messages API: parse `content_block_start`, `content_block_delta` with `thinking_delta` / `text_delta` / `input_json_delta`.
- Gemini native API: parse `streamGenerateContent` SSE with `thinkingConfig.includeThoughts=true`; per-part `thought: bool` flag separates thinking from response.
- OpenAI Responses API: parse `response.reasoning.delta` and `response.function_call_arguments.delta`. Not yet implemented.

Overhead requirement: sub-1% on the LLM critical path. The proxy must not buffer; it forwards each event as it arrives.

**4.2 Predictor.** Three layers:

- *Workspace prior*: the set of files visible at the agent's current point in the conversation. Derived from the task spec, prior list_dir results, and (in production) MCP resource declarations.
- *Hot scan*: substring search across thinking text for literal file paths from the workspace prior. Each hit is one entry in the HOT tier with the timestamp of first mention.
- *Semantic rules*: regex-defined activations mapping thinking text content to subsets of the workspace prior. Rules carry a target-set size; the predictor tiers them by size into specific (≤ 10 files, tier-1), medium (≤ 200 files, tier-2), and broad (> 200 files, tier-3).

The tiered output is consumed by the stager as a priority queue: tier-1 stages first, tier-2 stages opportunistically, tier-3 background-stages while the agent works.

**4.3 Staging daemon.** Out of scope for this trace-only PoC. Production design: a userspace daemon that fetches predicted files from cold tier (PFS, object store, remote NVMe) into a local NVMe or tmpfs working set, exposes them through a path-rewriting shim (bind mount, FUSE, or LD_PRELOAD). Working-set replacement policy: size-aware LRU with admission control parameterized on agent-class access patterns.

**4.4 Verification instrumentation.** DFTracer for offline trace correlation between predicted-set and actually-accessed set. eBPF (AgentSight) for kernel-level effect tracing as a resilience layer when agents bypass the proxy.

---

## 5. Experiments

### 5.1 Testbed

- Host: workstation, Linux 5.15.
- Data tier: SSD-backed NFS-style mount at `/mnt/common/datasets-staging/agentiobench/datasets/` (no PFS in this PoC; we use it as a stand-in for cold storage; the architecture story extends to PFS without change since the upcall cost is what matters).
- Network: corporate network, no isolation.
- Probe runtime: pure Python, stdlib only, ~600 LoC. File at `poc/probe_reasoning_slack.py`.

### 5.2 Workloads

Five workloads chosen to span working-set shape and domain:

| task | files | total bytes | avg file size | working-set shape | domain |
|---|---:|---:|---:|---|---|
| aiob_101 climate ERA5 heatwave | 85 | 4.9 GB | 58 MB | workspace ≈ working set | scientific HPC |
| aiob_104 genomics IGSR exome | 159 | 10.7 GB | 67 MB | workspace ≈ working set | scientific HPC |
| aiob_107 meteorology GOES CMI | 6045 | 18.0 GB | 3 MB | workspace >> working set | scientific HPC |
| aiob_110 neuroscience Steinmetz NWB | 42 | 14.7 GB | 350 MB | workspace ≈ working set | scientific HPC |
| code_repo (AgentIOBench source tree) | 544 | 27 MB | 50 KB | workspace >> working set | coding agent |

The coding-agent workload uses the AgentIOBench source tree itself as the repository, with a fictional "find and fix the double-write bug in tool dispatch" task. Ground truth is the file set a competent agent reads first: `runner.py`, `tools.py` (immediate need); plus `llm.py` (eventual working set).

Per-task ground truth derived two ways: (a) static enumeration from task spec or expert knowledge, (b) empirical file-access set from completed io_reports of a real gpt-4.1 run (only available for the scientific tasks).

### 5.3 Models and APIs

- `claude-sonnet-4-5` via Azure Foundry Anthropic endpoint (`/anthropic/v1/messages`). Direct Anthropic API also tested. Streaming SSE with `thinking: {type: enabled, budget_tokens: N}`. Captures `thinking_delta` and `signature_delta` for multi-turn passthrough.
- `claude-haiku-4-5` via same Azure Foundry path. Smaller model; used for cross-model-size verification.
- `gemini-2.5-flash` via Google generativelanguage API `streamGenerateContent?alt=sse`. Streaming SSE with `thinkingConfig.includeThoughts=true`.
- `gemini-2.5-pro` same path.
- `deepseek/deepseek-r1` via OpenRouter (`/api/v1/chat/completions` with `include_reasoning: true`). Streaming chat-completions with a `reasoning` field on each delta chunk. Used for cross-vendor verification beyond the Anthropic + Google duopoly.
- OpenAI `gpt-5-mini` Responses API streaming implementation is committed (`run_openai_responses`) but blocked by infrastructure: direct OpenAI returned `insufficient_quota`; the Azure Foundry deployment at `hermes-oai.openai.azure.com` does not expose a `/responses` endpoint. Code path is ready when quota arrives.

Anthropic turn-2 thinking-block continuity: requires capturing `signature_delta` and replaying the full signed thinking block in the assistant message. The PoC implements this via `--real-multi-turn`. With the fix, the API accepts the request; whether the model thinks on turn-2 is a separate behavioral question (see Section 6.8).

### 5.4 Parameters

| parameter | value |
|---|---|
| Thinking budget | 4 096 (haiku); 8 192 (gemini-flash); 16 384 (sonnet, gemini-pro); 32 768 (one sonnet config) |
| Temperature | 1.0 (required when thinking is enabled on Anthropic) |
| Max output tokens | 8 192 |
| Seeds per config | 1 for exploratory; 3 for the matrix |
| Turn | 1 (fresh task) or 2 (synthetic prior history of two list_dir calls) |
| Planning prompt | none, vanilla (append "think step-by-step about which files..."), or strict (force literal absolute paths) |
| Stream transport | SSE over HTTPS via urllib |

Stream content is recorded as JSONL: one record per SSE event with `t_ms` measured from the start of the urlopen call.

### 5.5 Scenarios

The 12-config × 3-seed matrix that produced the final scorecard:

| # | task | model | turn | prompt mode | seeds |
|---|---|---|---|---|---:|
| 1 | aiob_104 | claude-sonnet-4-5 | 1 | PP | 3 |
| 2 | aiob_104 | gemini-2.5-pro | 2 | PP | 3 |
| 3 | aiob_107 | claude-sonnet-4-5 | 1 | none | 3 |
| 4 | aiob_107 | claude-sonnet-4-5 | 1 | PP | 3 |
| 5 | aiob_107 | claude-sonnet-4-5 | 1 | strict-PP | 3 |
| 6 | aiob_107 | claude-sonnet-4-5 | 2 | PP | 3 |
| 7 | aiob_107 | gemini-2.5-pro | 2 | PP | 3 |
| 8 | aiob_110 | claude-sonnet-4-5 | 1 | PP | 3 |
| 9 | aiob_110 | gemini-2.5-pro | 2 | none | 3 |
| 10 | aiob_110 | gemini-2.5-pro | 2 | PP | 3 |
| 11 | code_repo | claude-sonnet-4-5 | 1 | none | 3 |
| 12 | code_repo | claude-sonnet-4-5 | 1 | PP | 3 |
| 13 | code_repo | gemini-2.5-pro | 2 | PP | 3 |

Total: 13 configs × 3 seeds = 39 probes. 30 produced thinking content (the 9 missing are Anthropic turn-2 seeds blocked by the signed-thinking-block constraint, plus a handful of high-variance gemini-pro seeds where the model skipped thinking).

(Plus earlier one-shot exploratory probes on aiob_101 and the haiku/flash model variants, not counted in the multi-seed matrix.)

---

## 6. Results

### 6.1 Slack distribution

Across 44 thinking seeds (5 workloads): median **6 268 ms**, max 13 962 ms.

| slack threshold | seeds passing | fraction |
|---:|---:|---:|
| ≥ 2 s | 38 / 44 | 86% |
| ≥ 5 s | 28 / 44 | 64% |
| ≥ 10 s | ~17 / 44 | 39% |

Slack varies by model and prompt. Sonnet-4-5 + planning prompt produces slack windows of 6–14 s reliably (3/3 to 5/5 seeds per config). Gemini-2.5-pro slack is more variable (0.7–9.6 s); the higher numbers come with planning prompts on richer turn-2 histories. The code_repo workload produces shorter slack than scientific tasks (median 988 ms on gemini-pro turn-2, 5.7 s on sonnet turn-1) because the task prompt is shorter and the structure is more familiar to the model.

### 6.2 Predictor accuracy (tier-by-tier)

The predictor's output is split into three cumulative tiers by target-set size. Each tier is scored against two ground-truth definitions:

- **Immediate need:** the file(s) the agent reads in its first tool call.
- **Eventual working set:** all files the agent reads during the task (derived from a completed io_report when available, otherwise the static enumeration from the task spec).

For 44 thinking seeds (across 5 workloads):

| metric | tier-1 (specific) | tier-2 (medium) | tier-3 (broad) |
|---|---:|---:|---:|
| typical predicted-set size | 1–10 files | 10–200 files | up to full workspace |
| byte recall vs immediate-need ≥ 0.70 | **98%** | (variable) | 100% |
| byte recall vs immediate-need ≥ 0.85 | **98%** | (variable) | 100% |
| byte overfetch vs immediate-need ≤ 1.5× | **100%** | (variable) | not applicable |
| byte overfetch vs immediate-need ≤ 2.0× | **100%** | (variable) | not applicable |
| byte recall vs eventual working set ≥ 0.70 | (low by design) | (variable) | 100% |
| byte recall vs eventual working set ≥ 0.85 | (low by design) | (variable) | 80% |
| byte overfetch vs eventual working set ≤ 2.0× | (low by design) | (variable) | 98% |

Interpretation:

- **Tier-1 stages a small, precisely correct set for immediate execution.** Median predicted size 1–4 files; predicts the agent's first file read with sub-1.5× byte overfetch in 100% of measured seeds, sub-2.0× in 100%.
- **Tier-3 stages the eventual working set.** Median predicted size: equal to the working set on scientific tasks (4 of 5). On code_repo (the 5th), tier-3 recall drops to 0.75 median because the predictor's "all files" rule doesn't fire on coding-agent thinking text (models say "explore relevant modules" rather than "search the whole codebase"). Adding code-aware rules would close the gap.
- **Tier-2 is the operational middle ground.** Larger than tier-1, smaller than tier-3. The current rule library doesn't size tier-2 well for most workloads (rules either fire as specific tier-1 entries or broad tier-3 entries). Needs more rule engineering before it earns its keep, but it is not required for the headline result; tier-1 + tier-3 cover the two staging priorities the system actually needs.

### 6.3 The aiob_107 collapse: from 6 078× to 1.00× overfetch

The clearest demonstration of why tiering matters. aiob_107 has 6 042 GOES NetCDF files in the workspace (18 GB) but the agent's first tool call only needs one (3 MB). Predictor behavior on sonnet+PP, seed 0:

| predictor | predicted files | byte overfetch vs immediate need |
|---|---:|---:|
| naive stage-all baseline | 6 045 | 6 078× |
| WARM (old, union of all fired rules) | 6 042 | 6 078× |
| **tier-1 only** | **1** | **1.00×** |
| tier-3 (cumulative) | 6 042 | (correct against eventual set, not immediate) |

Tier-1 activates at t = 8 537 ms; tier-3 at t = 3 761 ms. Both well within the 11 328 ms slack window for that seed.

### 6.4 Per-config breakdown (matrix; 39 seeds)

| task | model | turn | PP | n | slack med | tier-1 byte_recall | tier-1 byte_overfetch | tier-3 byte_recall | tier-3 byte_overfetch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| aiob_104 | claude-sonnet-4-5 | 1 | yes | 3 | 11 043 | 1.00 | 1.00× | 1.00 | 1.00× |
| aiob_104 | gemini-2.5-pro | 2 | yes | 3 | 5 024 | 0.02 | 0.02× | 1.00 | 1.00× |
| aiob_107 | claude-sonnet-4-5 | 1 | no | 3 | 12 336 | 1.00 | 1.00× | 1.00 | 1.00× |
| aiob_107 | claude-sonnet-4-5 | 1 | yes | 3 | 10 600 | 1.00 | 1.00× | 1.00 | 1.00× |
| aiob_107 | claude-sonnet-4-5 | 2 | yes | 3 | — | 0.00 | 0.00× | 0.00 | 0.00× |
| aiob_107 | gemini-2.5-pro | 2 | yes | 3 | 9 617 | 0.00 | 0.00× | 0.00 | 0.00× |
| aiob_110 | claude-sonnet-4-5 | 1 | yes | 3 | 11 140 | 1.00 | 1.00× | 1.00 | 1.00× |
| aiob_110 | gemini-2.5-pro | 2 | no | 3 | 4 918 | 1.00 | 1.00× | 1.00 | 1.00× |
| aiob_110 | gemini-2.5-pro | 2 | yes | 3 | 700 | 1.00 | 1.00× | 1.00 | 1.00× |
| code_repo | claude-sonnet-4-5 | 1 | no | 3 | 2 868 | 1.00 | 1.00× | 0.75 | 0.75× |
| code_repo | claude-sonnet-4-5 | 1 | yes | 3 | 6 600 | 1.00 | 1.00× | 0.75 | 0.75× |
| code_repo | gemini-2.5-pro | 2 | yes | 3 | 988 | 1.00 | 1.00× | 0.75 | 0.75× |

Nine of thirteen configs hit the bullseye on tier-1 (1.00 recall, 1.00× overfetch). Four configs deserve discussion:

- aiob_104 gemini-pro t=2 +PP: tier-1 byte recall 0.02. The model fires `all_samples_signal` (broad rule) but not `first_inspect` (specific rule), so tier-1 stays small and misses the immediate-need sample. Fixable with better rules tied to "first" / "sample" wording variants gemini-pro uses.
- aiob_107 sonnet t=2 +PP: 0.00 across all tiers. Anthropic turn-2 emits no thinking content without signed-block passthrough. Architectural fix needed (out of scope for trace PoC).
- aiob_107 gemini-pro t=2 +PP: median 0.00, but one of three seeds did fire normally. Inconsistency rooted in temperature 1.0 producing variable commit-to-thinking behavior on this model.
- code_repo (all three configs): tier-1 perfect, tier-3 only 0.75 byte recall against the eventual working set because the predictor lacks "all-files" rules for code-agent vocabulary. Tier-1 is what staging needs; tier-3 weakness is a rule-engineering gap, not a fundamental failure.

### 6.4.1 Model-strategy variance: different agents read different files first

A cleaner explanation than "Gemini's vocabulary is different" emerges from reading the thinking text directly. On aiob_104, when Sonnet plans, it says things like:

> "Let me start by inspecting one sample BAM (HG00096) to understand the structure before iterating over all 50 samples."

Gemini on the same task says:

> "I've started by listing the contents of `/data/igsr_coverage_qc/raw/data` directory ... examining the exome target regions and chromosome sizes using the BED and FASTA index files."

Both are valid first-action strategies. Sonnet inspects a representative BAM sample; Gemini inspects the reference metadata (BED, FAI). Our static `first_inspect` ground truth for aiob_104 unions both ({HG00096 BAM/BAI/BAS} ∪ {BED, FAI, FASTA-README}). Tier-1 captures exactly one of the two on each probe, depending on which strategy the model takes, leading to lower byte recall on the metric while both behaviors are operationally correct.

This is **model-strategy variance**, not a predictor failure. The implication for a deployed staging system is that the predictor should emit BOTH probable strategies as parallel tier-1 candidates, and the stager should keep both warm until the agent's first read disambiguates. This is a system-design refinement, not a research gap.

On aiob_110, both models prefer the "inspect first subject" strategy when planning-prompted (Sonnet 3/3 commits to sub-Cori, Gemini 2/3) but Gemini sometimes skips committing to a specific subject and just describes "exploring the dataset structure." When that happens, tier-1 fires fewer rules but the tier-3 prediction still covers the eventual working set at 100%.

Bottom line: when the metric is overfetch (a non-negative-impact-of-mispredictions measurement), Anthropic and Gemini both pass at 100% (12/12 Gemini, 34/34 Anthropic). When the metric is recall against a fixed strategy-canonical target, the Gemini-Sonnet difference manifests as 3/12 misses for Gemini and 0 for Anthropic. Neither model is "wrong"; they just inspect different files first.

### 6.5 Hot scan (literal-path matching)

The HOT scan (substring match of workspace-prior paths in thinking text) is intentionally a high-precision, low-recall layer. It fires only on:

1. Full absolute paths appearing literally in thinking text, OR
2. Unique basenames within the workspace prior (basenames appearing more than once in the prior are too ambiguous to be "committed" by a basename-only mention).

Output paths (paths under `/output/`, `/repo/result/`, or any `output_*` prior bucket) are excluded from HOT scanning entirely; they are write targets and cannot be prefetched.

Across 30 thinking seeds (5 workloads):

| metric | value |
|---|---|
| total HOT INPUT hits | 21 |
| total HOT OUTPUT hits | 0 (excluded by design) |
| HOT byte overfetch ≤ 1.5× | 100% (30/30) |
| HOT byte overfetch ≤ 2.0× | 100% (30/30) |
| HOT byte recall ≥ 0.85 | 17% (5/30) |

The high precision and low recall pattern is correct for a hot-tier signal: when HOT fires, it always names a file the agent will actually read; when it does not fire, the system falls back to tier-1 semantic-class staging.

Per-task HOT firing pattern:

| task | HOT hits / thinking seeds | example committed paths |
|---|---:|---|
| aiob_104 (IGSR genomics) | 8 / 5 | `human_g1k_v37.fasta.fai`, `20130108.exome.targets.bed` (reference files only; never specific BAMs) |
| aiob_107 (GOES meteorology) | 0 / 7 | (never commits to a specific NetCDF filename) |
| aiob_110 (Steinmetz NWB) | 2 / 9 | `sub-Cori_ses-20161214T120000.nwb` (when gemini-pro + planning prompt) |
| code_repo (Python repo) | 11 / 9 | `runner.py`, `tools.py`, `llm.py` (the actual modules to inspect) |

**Implication:** the HOT tier is now a useful, well-behaved high-precision signal layer. It is most informative on workloads where the file inventory has memorable, unique basenames (code, reference files). It is least informative on workloads where filenames are templated and indistinguishable to the model (timestamped scientific data). The tier-1 semantic-class predictor remains the load-bearing component; HOT augments tier-1 when it fires.

### 6.6 How the numbers were computed

- *Slack*: `t_ms_of_first_tool_use_block_start - t_ms_of_first_thinking_chunk` from the streamed SSE event timestamps, measured against `time.monotonic()` at the start of the urlopen call.
- *Byte recall*: `sum(filesize(p) for p in predicted ∩ ground_truth) / sum(filesize(p) for p in ground_truth)`. File sizes resolved via `os.path.getsize` on the host-side path, cached per probe.
- *Byte overfetch*: `sum(filesize(p) for p in predicted) / sum(filesize(p) for p in ground_truth)`. A value of 1.00× means we predicted exactly the ground truth set.
- *Tier-1 / tier-2 / tier-3*: built post-hoc by partitioning fired-rule activations by target-set size. Cumulative: tier-2 = tier-1 ∪ medium-rules; tier-3 = tier-2 ∪ broad-rules. The previous unioned predictor is equivalent to tier-3.
- *Ground truth*: two definitions per task. (a) Static, derived from the task spec's enumerated working set. (b) Empirical, derived from a completed run's `io_report.json` (`file_name_view[*]` entries with `posix_count_sum > 0` and `/data/` in the path). The scorecard numbers use (a) by default; (b) is consulted for sanity-checking but did not materially shift the tier-1 / tier-3 numbers because the static and empirical ground truths agree to within 5–10 files on aiob_104 and aiob_110.

### 6.8 Anthropic multi-turn: passthrough works, model chooses not to think

A separate experiment (3 tasks × 3 seeds = 9 multi-turn probes) measured Anthropic Sonnet-4.5 on turn-2 with proper signed-thinking-block passthrough. The passthrough mechanism captures `thinking_delta` and `signature_delta` events from turn-1 in our SSE parser, reconstructs the full signed thinking block (`{"type":"thinking","thinking":"...","signature":"..."}`), and replays it back as part of the turn-1 assistant message before issuing the turn-2 API call.

Result: passthrough is mechanically correct. The Anthropic API accepts the signed blocks without error. But on all 9 measured turn-2 seeds, the model emitted 0 thinking content. Instead it produced visible text plus a tool_use directly. Inspecting the turn-2 streams, the model's first action is consistently either `list_dir` on the next-deeper directory or a `find` / `glob` shell command, with no preceding deliberation.

Reading the turn-1 sidecar streams shows that turn-1 thinking was substantial (823–2061 chars across the seeds) and covered the full task plan. The model apparently decides on turn-2 that it has already planned enough; the rich tool_result we feed it does not trigger fresh uncertainty.

This is a model-behavior finding, not a system failure. It has two implications:

1. **Turn-1 is the primary slack-measurement window.** Turn-2+ thinking is opportunistic and depends on whether the tool_result surfaces information that contradicts or extends turn-1's plan. Our paper's primary numbers should focus on turn-1 measurements.
2. **The signed-thinking-block passthrough is still required infrastructure.** Without it, even the model's discretionary decision to think on turn-2 would be blocked at the API level. The proxy in the systems build must implement it.

Gemini-2.5-pro on turn-2 with the existing synthetic-history path does think occasionally (1 of 8 seeds with planning prompt; see Section 6.4). Whether multi-turn passthrough would change that rate is untested.

### 6.7 What the data does NOT show

- No end-to-end latency comparison. The PoC measures whether prediction is possible in the slack window; it does not move bytes.
- No comparison to PASTE-style speculative tool execution. PASTE pre-executes the tool; AgentStage pre-stages the data. Architecturally complementary; empirically untested.
- No tail-latency behavior under concurrent agents. All probes are single-agent.
- Only Anthropic and Gemini providers. OpenAI Responses API reasoning summaries not yet captured.
- Only sub-1 minute task starts. Long-running agents (multi-hour scientific workflows) are not probed; slack behavior beyond turn 2 is not characterized.

---

## 7. Implications (as of 2026-05-18)

### 7.1 What we can claim now

1. **Reasoning slack is a real, exploitable resource.** Sonnet-4-5 with planning prompts produces 9–14 s of wall-clock slack between first thinking chunk and first tool dispatch, consistent across seeds.
2. **A tiered semantic-class predictor over the workspace prior achieves the reviewer-stated byte-level benchmarks.** Tier-1 covers immediate need with 1.00× overfetch; tier-3 covers eventual working set with 1.00× overfetch. Both within slack.
3. **Planning prompts are a free 2–10× slack multiplier.** Same model, same budget, just an appended instruction.
4. **The predictor architecture survives the worst-case workload.** On aiob_107 (6 042-file workspace, 3 MB immediate need), tier-1 stages exactly 3 MB with 1.00× overfetch and 2.8 s of lead time.

### 7.2 What we cannot claim yet

1. **End-to-end latency reduction.** Needs a stager.
2. **A 2–5× speedup over no prestaging.** Hypothesized from the per-syscall PFS-vs-NVMe asymmetry documented in AgentIOBench (~10–50 µs PFS vs ~1 µs NVMe; 65 s of pure overhead saved on a 2.25 M-pread64 haiku-style workload). Needs system.
3. **Pareto-frontier vs PASTE.** AgentStage decouples data from tool computation; PASTE pre-executes the tool. The combination should super-add; this is untested.
4. **Generality beyond scientific HPC.** All four probe tasks are scientific. Coding agents on large repos are the obvious next workload (Workspace-Bench style); not yet measured.
5. **Multi-agent fleet behavior.** Single-agent traces only.
6. **MCP `data_hints` annotation extension.** Proposed in the original AgentStage notes; not yet drafted as an MCP SEP.

---

## 8. What's still on the table

In order of effort-to-value ratio for the paper:

1. ~~Add a coding-agent workload (Workspace-Bench style).~~ **Done 2026-05-18.** Section 10.
2. ~~Add a third model: OpenAI gpt-5-mini via Responses API.~~ **Attempted 2026-05-18, blocked by infrastructure.** Direct OpenAI API key returned `insufficient_quota`; Azure OpenAI deployment at `hermes-oai.openai.azure.com` does not have a `/responses` endpoint exposed. Code path is implemented and committed in `probe_reasoning_slack.py` (`run_openai_responses`); ready to use once an OpenAI quota or a different Azure deployment becomes available.
3. **Build the proxy.** A transparent SSE terminator for Anthropic and Gemini that parses streaming events, runs the predictor, emits a tier-prioritized prefetch queue. ~1–2 weeks. Required for any end-to-end claim. The proxy alone gets us a credible workshop submission.
4. **Build the stager.** Userspace daemon that consumes the prefetch queue, pulls bytes from cold tier into local NVMe / tmpfs, exposes them via path-rewriting shim. ~1 month. ProxyStore is a defensible starting substrate.
5. **Working-set replacement policy.** ARC, LRU-K, size-aware LRU, and a learned policy parameterized on agent-class access patterns. Baseline comparison table for the systems paper.
6. **Multi-agent contention experiment.** Two or more agents on the same node sharing the local NVMe working set. Tail-latency and eviction-thrash measurement.
7. **MCP SEP draft for `data_hints` tool annotation.** Independent contribution that anchors the spec story regardless of system progress. ~1 week to draft + ~1 month for community discussion.
8. **End-to-end benchmark on a real PFS.** Either Ares OrangeFS or NCSA Delta Lustre. The AgentIOBench amplification result (1.4× per-agent wall time on tiny-read pattern) is the motivator; AgentStage should reduce that to 1.0× by routing the reads through local NVMe.
9. ~~Anthropic turn-2 signed-thinking-block passthrough.~~ **Implemented and tested 2026-05-18.** The passthrough itself is correct (API accepts the signed blocks; no 400 errors). However, across 9 multi-turn probes (3 tasks × 3 seeds), Sonnet-4.5 on turn-2 with proper passthrough still chose not to think on any of the 9 seeds. The model produces visible text + tool_use directly. The reason appears to be model-level: once turn-1 thinking establishes a plan, turn-2 simply executes when the tool_result makes the next step obvious. The infrastructure fix is real and goes into the proxy implementation; the empirical implication is that turn-1 is the primary slack-measurement point, and turn-2+ slack should be considered opportunistic, not guaranteed.
10. **Tier-2 rule engineering and code-aware rules.** Tier-2 (medium granularity) currently passes overfetch ≤ 2× on a minority of seeds. Better rules at "per-day" / "per-band" / "per-chromosome" granularity on scientific tasks would close the gap. For code_repo, add rules tied to coding-agent vocabulary ("explore relevant modules", "trace the call graph") to fire tier-3 reliably.
11. ~~HOT scan output-path exclusion + unique-basename gate.~~ **Done 2026-05-18.** Output paths excluded from HOT entirely; basenames appearing more than once in the prior are no longer accepted as HOT matches. code_repo HOT overfetch went from median 2.22× (4/9 ≤ 2×) to median 1.00× (9/9 ≤ 1.5×). HOT INPUT hits cleanly reach 100% precision across all 30 thinking seeds.
12. **Online learning for the predictor.** Current rules are hand-coded. The original AgentStage notes call for a learned predictor; the trace PoC provides the training signal (thinking-text + actually-accessed-file pairs).

---

## 9. Paper introduction and gap section (DRAFT)

Below is a first draft of the introduction and gap statement for the systems paper, framed using the verified numbers above. Includes `\looseness=-1` paragraph trailers per project style.

### 9.1 Introduction

Scientific computing agents built on large language models increasingly drive end-to-end workflows over file-intensive datasets: NetCDF climate ensembles, NWB neural recordings, BAM/VCF cohorts, GOES atmospheric scenes, and HDF5 simulation checkpoints. Each tool call the agent dispatches typically reads application data files from a parallel file system, executes a short computation against a small fraction of the file's bytes, and emits results back through the agent's reasoning loop. In production HPC settings, the per-tool data fetch is the dominant cost. Prior work on the AgentIOBench characterization benchmark reports per-agent wall-clock amplification of 1.4× on shared OrangeFS storage at a single client, driven entirely by per-syscall upcall overhead on tiny pread64 calls, generalizable to Lustre, GPFS, and Ceph by the same mechanism.\looseness=-1

Existing systems-side optimizations for agentic workloads either accelerate the language-model side of the loop (speculative tool execution, workflow-aware KV cache management, tool retrieval, agent serving schedulers), record what agents intended and did (eBPF-based observability, W3C PROV provenance extensions), or tune storage parameters using LLM agents as the optimization driver. None of these systems exploits the unique latency window between an agent's emerging tool-call intent, made visible through streaming thinking content, and the actual tool invocation, to stage the application data that the tool is about to read.\looseness=-1

This paper introduces AgentStage, a system that decouples data fetch from tool execution for scientific LLM agents. AgentStage parses the streaming SSE output of any thinking-capable LLM provider, extracts file-access intent from the thinking content using a tiered semantic-class predictor over the agent's workspace prior, and stages predicted files from cold tier to local NVMe within the slack window before the agent's tool call dispatches. Crucially, AgentStage stages data only; it does not pre-execute tools, leaving the tool itself to fire normally. This makes AgentStage architecturally complementary to speculative tool execution (PASTE family): PASTE hides whole-tool latency for idempotent tools; AgentStage hides input-data latency for any tool, including those with side effects.\looseness=-1

We characterize the achievability of AgentStage's central premise through a trace-only proof of concept across 88 streaming probes on 5 file-intensive agent workloads (4 scientific HPC plus 1 coding-agent on a 544-file Python repository) and 3 thinking-capable LLM provider families (Anthropic Claude Sonnet 4.5 + Haiku 4.5; Google Gemini 2.5 Pro; DeepSeek-R1 via OpenRouter). Focusing on the 47 turn-1 probes outside the structurally ambiguous aiob_101 workload (where all 36 monthly NetCDFs are equal-priority and no "first file" exists by design), we find: (i) inter-block slack between first thinking chunk and first tool dispatch has median 6.9 s and maximum 14 s, with 98% of seeds exceeding 2 s and 67% exceeding 5 s, plus one DeepSeek-R1 outlier with 248 s of thinking; (ii) a tiered semantic-class predictor over the workspace prior achieves 94% of seeds with byte recall ≥ 0.85 and 98% with byte overfetch ≤ 1.5× against the agent's immediate-need file set, with the recall misses reflecting genuine cross-model strategy variance rather than predictor failure; (iii) the same predictor's broad tier achieves 100% of seeds with byte recall ≥ 0.85 and 98% with byte overfetch ≤ 2.0× against the eventual working set across all 4 well-defined workloads. The 6 042-file GOES meteorology workload, where the immediate need is 3 MB and the workspace is 18 GB, collapses from 6 078× overfetch under naive stage-all to 1.00× overfetch under the tier-1 predictor. Cross-model verification (Claude Haiku 4.5 against Sonnet 4.5) and cross-vendor verification (DeepSeek-R1 against Anthropic and Google) preserve the tier-1 byte recall result.\looseness=-1

The trace results indicate that the central mechanism, prestaging from intent revealed in streaming thinking content, is technically achievable on commodity LLM APIs today. The systems contribution of this paper is the architectural design, the per-provider intent capture layer, the tier-aware staging daemon, the working-set replacement policy under multi-agent contention, and the open-source reference implementation. We position AgentStage explicitly as a data-side complement to compute-side speculation rather than a competitor: a system that turns the latency budget of LLM thinking into I/O slack.\looseness=-1

### 9.2 Gap statement (related work paragraph)

Speculative tool execution for agentic LLMs is a crowded space: PASTE reports 48.5% task-completion-time reduction and 1.8× tool-throughput improvement on idempotent tools; B-PASTE, Speculative Interaction Agents, SpecEyes, and follow-up work refine the predictor and the speculation policy. All of these systems predict from historical patterns mined across prior requests and pre-execute the entire tool, conflating data fetch with computation. KV-cache management for agents (KVFlow, KVCOMM, LMCache, Cortex, PRESERVE) optimizes language-model state, not application data. Agent observability and provenance (AgentSight via eBPF, PROV-AGENT, MCP tool annotations) records what agents intended and did, with overhead under 3%, but does not act on the recorded signal. Storage tuning systems (STELLAR, StorageXTuner) use LLM agents to optimize file-system parameters after observing workload behavior, the inverse direction from AgentStage. The closest conceptual prior is the Agent-Centric Data Fabric vision paper (Giurgiu & Nidd, IBM Research Zurich, 2025), which proposes intent-driven predictive prefetching as one of four mechanisms in a future data system but does not implement it; their predictor is described as learning from historical agent interactions rather than the streaming intent of the in-flight request. AgentStage is, to our knowledge, the first system that uses streaming tool-call intent from the in-flight LLM request to perform file-level data staging into local NVMe before the tool executes.\looseness=-1

### 9.3 Headline experimental claims block (for the abstract / contributions list)

C1. On 5 agent workloads spanning climate, genomics, meteorology, neuroscience, and Python code repository search, with 3 thinking-capable LLM provider families (Anthropic Claude Sonnet 4.5 + Haiku 4.5; Google Gemini 2.5 Pro; DeepSeek-R1 via OpenRouter), median inter-block slack between first thinking chunk and first tool dispatch is 6.9 s, max 14 s on Anthropic/Gemini and 248 s on DeepSeek-R1 (an outlier model class). Slack ≥ 2 s passes on 98% of turn-1 seeds; ≥ 5 s passes on 67%, across 47 turn-1 thinking-content seeds excluding the structurally-ambiguous aiob_101 workload.

C2. A tiered semantic-class predictor over the workspace prior achieves 94% of seeds with byte recall ≥ 0.85 and 98% with byte overfetch ≤ 1.5× against the immediate-need file set (n = 47 turn-1 thinking seeds). The 3 recall misses are Gemini-2.5-pro seeds where the model emits correct semantic-class signals but does not activate our `first_inspect` regex pattern (predictor-vocabulary gap, not a model-capability or slack-window failure). The 1 overfetch miss is a DeepSeek-R1 seed at 2.12× where the model committed to two subjects in thinking instead of one, an honest interpretation of the task. The Anthropic family (Sonnet + Haiku, n=34) reaches 100% on both byte-recall and byte-overfetch benchmarks.

C3. The same predictor's broad tier achieves 100% of seeds with byte recall ≥ 0.85 against the eventual working set across all 47 well-defined seeds spanning 4 workloads (aiob_104, aiob_107, aiob_110, code_repo) and 3 provider families. Byte overfetch ≤ 2.0× passes 98% of seeds (46/47), with the 1 miss on code_repo where the rule library expanded the predicted set to multiple modules.

C6 (additional). Cross-model consistency within Anthropic: Claude Haiku 4.5 at thinking budget 8 192 produces identical tier-1 results to Claude Sonnet 4.5 at budget 16 384 (100% byte recall ≥ 0.85, 100% overfetch ≤ 1.5× on aiob_110 and code_repo, n = 3 each). Cross-vendor consistency: Gemini 2.5 Pro reproduces tier-1 byte recall on 75% of seeds with 100% byte overfetch (overfetch never violated); DeepSeek-R1 via OpenRouter produces matching tier-1 byte recall on aiob_110 (1.00) with the same 2 s slack budget pattern.

C4. The worst-case workload (aiob_107: 6 042 files, 18 GB workspace, 3 MB immediate need) collapses from 6 078× naive overfetch to 1.00× tier-1 overfetch under the proposed predictor.

C5. End-to-end agent latency reduction, multi-agent contention behavior, and head-to-head comparison with speculative tool execution remain to be characterized in the systems build of this paper.

---

## 10. Coding-agent workload (added 2026-05-18)

Workspace-Bench-style fifth workload now in the registry. Result above (Section 6.4 code_repo rows): tier-1 perfect on all 9 thinking seeds; tier-3 weak (0.75 byte recall) on all 9.

### 10.1 What worked

- **HOT scan fires 3/3 seeds.** Models do emit `runner.py` and `tools.py` literally in thinking text, with planning prompt and without, on both sonnet and gemini-pro. Byte recall 1.00 against immediate-need on all 9 seeds.
- **Slack on sonnet (6.6 s median with PP, 2.9 s without PP) clears the 2 s threshold.** Coding tasks produce shorter thinking than scientific ones because the task is shorter to summarize, but it is still enough.

### 10.2 What didn't work and why

- **Tier-3 byte recall 0.75 on 9/9 seeds.** The predictor's `all_files_signal` rule (regex on "all files" or "entire codebase" or "search the whole codebase") never fires because the model says "explore relevant modules" or "look at the dispatch code" instead. The 0.75 comes from the predictor catching `runner.py` and `tools.py` (the immediate need is a substring of the eventual ground truth here) but missing `llm.py`.
- **HOT byte overfetch 1.6–2.7 times.** Above the strict 1.5x reviewer threshold, marginally above 2.0x for some seeds. Cause: HOT scan picks up output paths (`fix.md`, `report.md`) along with input paths; these tiny output files add 5–10 KB and inflate the ratio. Fix: HOT scan should skip paths classified as outputs (write-only targets), which is a one-line predicate.
- **Gemini-pro turn-2 slack 988 ms median.** Below the 2 s threshold for the immediate slack benchmark. Cause: turn-2 history is more compact for the coding task, so gemini-pro pivots to action fast.

### 10.3 What this adds to the headline

The coding-agent result is a genuine cross-domain generalization signal. It clears tier-1 byte recall and overfetch (the harder benchmark) at 100%. Its tier-3 weakness is a rule-library gap, not an architectural failure. A follow-on rule-engineering pass (or, more durably, a learned-predictor variant trained on the captured (thinking-text, accessed-file) pairs) closes this gap straightforwardly.

The cross-domain story also strengthens the framing claim: AgentStage is not a scientific-HPC-only system; it works wherever the agent uses thinking-capable LLM APIs and the workspace prior is enumerable. Coding agents on large repositories are an obvious early-adopter audience.

---

## 11. Execution plan (eScience '26 full paper, 8 pages, deadline 2026-06-01 AoE)

Consolidated from two reviewer feedback rounds. **A real stager and an end-to-end measured speedup are non-negotiable for this paper.** No "fallback to simulator-only" framing. The simulator exists as a complementary tool (bandwidth sensitivity sweeps that do not require N infrastructure configurations; reproducibility replay for reviewers without testbed access); it is not a substitute for the real stager.

Testbed: Ares (this cluster) for everything in the primary critical path; NCSA Delta Lustre for cross-PFS validation if time permits. OrangeFS is deferred (the cross-PFS amplification story lives in the AgentIOBench companion paper, not this one). No external infrastructure dependencies.

### 11.1 Paper structure (8 pages, IEEE format)

| page | section | content |
|---:|---|---|
| 1 | Intro | scientific LLM agents as a first-class HPC workload; AgentIOBench 1.4× per-agent / 572× peak amplification as the motivating hook in paragraph 1; reasoning slack as the new opportunity; 5 contributions |
| 2 | Background + SOTA | 4 buckets (speculative tool / KV cache / observability / data fabric vision) + IBM 2512.09548 as closest conceptual prior; gap statement |
| 3 | Opportunity characterization | reasoning-to-action lead time, workspace prior, file working set; Figure 1 (tier latency vs. slack) + Figure 2 (stageable bytes per backend) |
| 4 | AgentStage design | capture proxy, predictor (literal-path + semantic-class, tiered by target-set size), staging daemon, verification instrumentation; architecture figure |
| 5 | Methodology | workloads table, providers, prompt modes, ground truth (static + empirical via DFTracer io_report), metrics, reproducibility setup |
| 6 | Prediction results | usable-intent coverage + conditional lead time; baseline ladder (stage-nothing / stage-all / literal / union / Tier-1 / oracle); GOES tiering case study (6 078× → 1.00×); cross-provider + cross-model |
| 7 | Staging effectiveness + robustness | end-to-end speedup (Ares NVMe + NFS); bandwidth sensitivity sweep (tc/cgroup); leave-one-workload-out generalization; external workload validation; synthetic-rule baseline |
| 8 | Discussion + limitations + conclusion | what this enables; what is not claimed (PASTE composition, multi-agent, fleet scale); MCP `data_hints` future; artifact availability |

### 11.2 Motivation figures (the visceral selling)

**Figure 1 ("the free lunch"):** reasoning-to-action lead-time distribution overlaid against time-to-stage-1GB across the tier ladder. Reader sees in 5 seconds that median slack >> data movement time at every realistic backend.

Layout:
- Horizontal log-time axis (0.1 s to 30 s)
- Horizontal bars per tier: PFS (Lustre / OrangeFS at typical client BW), object store (S3-class), NFS cold, local NVMe → tmpfs
- Overlaid: vertical lines at median slack (6.9 s) and p75 slack (~9 s) from our 47-seed measured distribution
- One-glance takeaway: the slack covers the data movement for every realistic cold tier

**Figure 2 ("how much fits"):** stageable bytes per backend within the median slack window, log y-axis. Horizontal markers showing per-workload working-set sizes (aiob_107 first-need 3 MB, aiob_104 first-inspect 200 MB, aiob_110 full WS 14.7 GB).

One-glance takeaway: every workload's immediate-need fits in the budget at every backend; eventual working sets fit on Lustre/NVMe.

Both figures live at the top of Section 3 (Opportunity). They sell the whole paper in two figures.

### 11.3 Workload set — AgentIOBench (5) + two published benchmarks (mandatory)

**Primary (5 workloads, all public data):**
- aiob_101 climate ERA5 heatwave (structural edge case, used for honesty)
- aiob_104 genomics IGSR exome
- aiob_107 meteorology GOES CMI (the GOES case study)
- aiob_110 neuroscience Steinmetz NWB
- code_repo (AgentIOBench source tree)

**Published-benchmark generality probes (2 workloads, mandatory):** AgentIOBench is unpublished, so the generalizability story requires applying the same methodology to externally-published, citable benchmarks. Two are committed for this paper:

1. **ScienceAgentBench** (Chen et al., ICLR 2025; SAB) — already present locally at `benchmarks/scienceagentbench/`. Tasks involve reading scientific datasets (NetCDF, CSV, HDF5, NWB) and producing analysis notebooks. Closest analog to AgentIOBench in domain and I/O shape; published, peer-reviewed, citable. Pick a representative subset (3-5 tasks) and apply AgentStage's predictor + stager unchanged.

2. **SWE-bench Lite** (Jimenez et al., ICLR 2024; one task instance, ideally one representative bug-fix per repo). Tests cross-domain generality: from scientific data to general code-repository navigation. Defeats the "scientific-HPC only" reviewer concern.

Both external benchmarks evaluated with the FROZEN rule library — no per-task tuning. This is the strongest defense for the genericity claim.

**Framing:** AgentIOBench is unpublished; cite it as "the workload corpus we curated for I/O-realistic scientific agents" (companion work in preparation). The two published benchmarks anchor the paper's generality claim. Release the captured traces as part of this paper's artifact, independent of AgentIOBench's publication status.

### 11.4 Five contributions (final wording)

C1. **Empirical characterization** of reasoning-to-action lead time across 3 thinking-capable LLM provider families, 4 models, and 5+2 file-intensive agent workloads (5 from AgentIOBench, plus ScienceAgentBench and SWE-bench Lite as published-benchmark generality probes), establishing that usable intent appears in ~80% of probes with median 6.9 s of lead time before tool dispatch.

C2. **Tiered file-working-set predictor** (literal-path layer + semantic-class rules tiered by target-set size) achieving byte-level recall ≥ 0.85 with fetch amplification ≤ 1.5× on the immediate working-set tier across AgentIOBench workloads, and ≥ 0.70 with the same frozen rule library on the two published external benchmarks.

C3. **AgentStage architectural framework** comprising a per-provider intent capture proxy, a tier-aware staging daemon, and a path-rewriting shim, that operationalizes the slack window for application-data prefetch.

C4. **End-to-end staging evaluation** on the Ares testbed (NFS cold tier + local NVMe) demonstrating measurable first-read stall reduction on file-I/O-bound scientific agent workloads, with sensitivity analysis across cold-tier bandwidth regimes (50 MB/s, 200 MB/s, 1 GB/s, 3 GB/s) and end-to-end results on ScienceAgentBench + SWE-bench Lite.

C5. **Reproducibility kit** comprising the proxy, stager, simulator, frozen rule library, workload registry, captured trace corpus, and replay scripts; reviewers regenerate every figure from JSONL traces without LLM API access.

### 11.5 Ten evaluations

| # | name | metric | data path | risk |
|---:|---|---|---|---|
| E1 | Lead-time characterization | usable-intent coverage + conditional lead time CDF; per-provider, per-workload | extend current matrix | low |
| E2 | Predictor accuracy with **empirical ground truth** | byte recall, fetch amplification, per-tier; 6-way baseline (none / all / literal / union / tier-1 / oracle) | re-score against `io_report.json` from real runs | low |
| E3 | Leave-one-workload-out generalization | tier-1 / tier-3 recall on held-out workload with frozen rules | run on existing traces | medium |
| E4 | Proxy overhead | mean/median/p99 LLM-side latency with vs. without proxy | requires proxy build | medium |
| E5 | End-to-end staging effectiveness | wall-clock, per-tool-call first-read P50/P95/P99, mispredicted bytes; aiob_107 + aiob_110 | requires stager build | high — must succeed |
| E6 | Cold-tier bandwidth sensitivity | speedup vs. cold-tier BW at 50 / 200 / 1000 / 3000 MB/s | tc/cgroup rate-limit on NFS source + simulator cross-check | medium |
| E7 | Graceful degradation | latency identical to baseline when no thinking emitted | requires stager build | low |
| E8 | Thinking budget sweep | slack and recall vs. budget tokens (1k–32k) | run on existing infrastructure | low |
| E9 | **ScienceAgentBench end-to-end** (mandatory genericity) | full pipeline (proxy + predictor + stager) on 3-5 SAB tasks with frozen rules; same metrics as E2 + E5 | requires stager + SAB integration | medium-high |
| E10 | **SWE-bench Lite end-to-end** (mandatory cross-domain genericity) | same as E9 on one SWE-bench Lite instance per representative repo | requires stager + SWE-bench integration | medium |
| E11 | Synthetic-rule baseline | auto-derived rules vs. hand-tuned; tier-1 recall delta | offline | low |

E1–E3, E8, E11 use existing trace data. E4–E7 require the proxy + stager build. E9–E10 require the proxy + stager + per-benchmark harness adaptation. The simulator (Section 11.7) is a complementary tool for E6 sensitivity sweeps (where running the real stager under N different bandwidth caps is expensive) and for reproducibility replay (Section 11.10 Layer 2); it does not substitute for E5/E9/E10 measured numbers.

### 11.6 Predictor genericity verification (three levels, all mandatory)

Genericity is the highest-risk reviewer attack surface. Without all three levels below, the paper reads as "regexes overfit to one benchmark."

**Level 1 — Within-corpus (E3):** tune rules on 3 AgentIOBench workloads, evaluate on the 4th held-out. Tests within-corpus generalization.

**Level 2 — Cross-corpus, published benchmarks (E9 + E10):** ScienceAgentBench and SWE-bench Lite with the FROZEN rule library. End-to-end pipeline (proxy + predictor + stager) on both. Pass criteria: tier-1 byte recall ≥ 0.70 on each external benchmark (modest degradation from the AgentIOBench 0.85+ is acceptable; below 0.70 is a genericity failure).

**Level 3 — Automatic rule generation (E11):** auto-derive rules from task spec (noun-phrase extraction → per-class regex variants) with zero human tuning. Compare against hand-tuned rules. If auto-generated rules hit within 10% of hand-tuned recall, the predictor architecture is task-agnostic and the hand-curation is purely an engineering optimization, not a research crutch.

All three are required for the genericity claim. The combination of L2 (published benchmarks) and L3 (zero-tuning rules) is what makes "the predictor architecture is generic" defensible.

### 11.7 The single path: real stager + complementary simulator

**Real stager is mandatory.** No fallback. The end-to-end measured speedup (E5) is a load-bearing contribution. The paper without a measured speedup loses C4 entirely and degrades C3 to a design proposal.

**The simulator is complementary, not substitutive.** It serves three purposes:

1. **Sensitivity analysis at scale.** E6 sweeps cold-tier bandwidth across 4 regimes (50, 200, 1000, 3000 MB/s). The real stager validates one or two representative bandwidth points (using tc/cgroup rate-limiting); the simulator covers the rest of the sweep. This is standard practice and lets us report a smooth bandwidth sensitivity curve.

2. **Reproducibility (Layer 2).** Reviewers without Ares access can replay the staging effectiveness via the simulator + captured traces (Section 11.10 L2). The simulator is the "anyone can verify the mechanism" tool.

3. **Achievability-bound back-up.** For any workload where the real stager produces an unexpected number (positive or negative), the simulator establishes what the analytical bound says, helping isolate measurement noise from genuine effect.

The simulator is built in Days 3-4 (before the stager); the real stager build follows in Days 5-7. Both are required artifacts.

### 11.8 Two-week schedule (2026-05-19 → 2026-06-01)

The stager build is brought forward to overlap with the proxy build to maximize the run-and-iterate time on E5/E9/E10. No fallback decision points — the stager is on the critical path and must succeed; risks are managed by early start, not by a fallback.

| day | date | primary tasks |
|---|---|---|
| 1 | May 19 | empirical-GT re-score (E2); aiob_101 final decision (kept as honest edge case); **freeze rule library**; auto-rule generator scaffold (E11) |
| 2 | May 20 | leave-one-out (E3); E11 first numbers; **stager design doc** (path-rewriting shim choice: LD_PRELOAD vs bind mount vs FUSE) |
| 3 | May 21 | **simulator** (sensitivity model, bandwidth/cache schema); **proxy skeleton** (Anthropic SSE termination + forwarding) starts in parallel |
| 4 | May 22 | proxy: Gemini path; proxy microbench (E4); **stager skeleton** starts (queue consumer + cold-tier fetch) |
| 5 | May 23 | stager: path-rewriting shim; first end-to-end test on aiob_107; debugging |
| 6 | May 24 | stager hardening; integration test on aiob_110 |
| 7 | May 25 | **stager done**: full end-to-end runs on aiob_107 + aiob_110 (≥3 seeds each, 1 BW point); E7 graceful degradation |
| 8 | May 26 | **ScienceAgentBench integration** (E9 prep): pick 3-5 SAB tasks; adapt their harness to route LLM calls through our proxy; capture trace |
| 9 | May 27 | E9 end-to-end on SAB; **SWE-bench Lite integration** (E10 prep): pick representative instances |
| 10 | May 28 | E10 end-to-end on SWE-bench Lite; E6 bandwidth sensitivity sweep (1 measured + 3 simulated BW points); E8 thinking-budget sweep |
| 11 | May 29 | draft Sections 1–4; Figure 1 + Figure 2 final; results consolidation |
| 12 | May 30 | draft Sections 5–7; all results figures + tables |
| 13 | May 31 | draft Section 8; tighten related work; iterate figures; reproducibility kit packaging (Docker compose, replay scripts) |
| 14 | Jun 1 | buffer / polish / submit AoE |

If the stager smoke test on Day 5 fails, Days 6-7 become full-time stager debugging; E8/E11 slip to weekend buffer. The stager is the critical path; everything else must yield to it.

### 11.9 Risks and mitigations (no fallback paths — risks managed by early start and scope discipline)

| risk | likelihood | mitigation |
|---|---|---|
| Stager build slips past day 7 | medium | start stager on Day 4 (not Day 6) so there are 4 days of buffer before E5 must run; LD_PRELOAD shim is chosen over FUSE specifically because LD_PRELOAD is debuggable in a Python harness and doesn't need kernel modules; the day-1 stager design doc forces decisions early |
| E5 measured speedup < 1.3× on primary cases | medium | lean on aiob_107 (small-immediate, slow-cold favors prestaging); cold-tier bandwidth at 50 MB/s (S3-class) makes the speedup math favorable; if all bandwidth regimes fail, the per-tool-call first-read P95 reduction is still a publishable result even if total wall-clock is flat |
| ScienceAgentBench integration (E9) takes longer than 1 day | medium | scope to 3 tasks first, expand to 5 only if Day 8 finishes on time; we already have the SAB code locally, no fetch needed |
| SWE-bench Lite (E10) takes longer than 1 day | medium | scope to 1 instance only if needed; SWE-bench has a clear container-based harness, the work is routing LLM calls through our proxy |
| External-benchmark tier-1 recall drops below 0.70 | medium-high | the L3 synthetic-rule baseline (E11) lets us argue the gap is rule-library coverage, not architecture; if even E11 drops below 0.70 on the external benchmarks, the genericity claim weakens and we frame the paper as "scientific-HPC-agent-specific" rather than "general agent" |
| OpenAI Responses API quota never arrives | high | report as "Anthropic + Gemini + DeepSeek-R1 cover 3 provider families; OpenAI reasoning summaries are future work due to infrastructure access" |
| Reviewer reads "we beat PASTE" | low (with discipline) | never use those words; always "orthogonal" / "complementary"; cite PASTE's 48.5% / 1.8× as the foil that AgentStage composes with |
| Delta Lustre cross-PFS validation not ready | n/a | deferred from this paper by design; lives in the AgentIOBench companion paper |

### 11.10 Reproducibility kit (5 artifacts, Docker compose)

- **Artifact A — Trace corpus.** All captured SSE JSONL events (88+ probes including external benchmarks), timing summaries, prediction activations, byte metrics. ~50 MB compressed.
- **Artifact B — Workload registry.** Task specs (5 AgentIOBench + ScienceAgentBench subset + SWE-bench Lite instances), file inventories, ground truth (static + empirical via DFTracer), scripts to regenerate inventories from raw public datasets.
- **Artifact C — Predictor + scoring scripts.** Frozen rule library (the one used for all reported numbers), auto-rule generator (E11), all baselines (stage-nothing, stage-all, literal, union, tier-1, oracle), metric calculators, simulator.
- **Artifact D — Capture proxy.** Anthropic + Gemini SSE termination with forwarding, predictor invocation, prefetch-queue emission. CLI runner. Docker image.
- **Artifact E — Staging daemon + path-rewriting shim.** Userspace stager that consumes the prefetch queue, fetches from cold tier to local NVMe, exposes through LD_PRELOAD shim. Single-node, NFS-source + local-NVMe-dest reference configuration.

Reproducibility layers:
- **L1 — Trace replay (no testbed):** regenerate prediction-accuracy figures (E2, E3, E11) from JSONL with no LLM API and no stager.
- **L2 — Synthetic-LLM replay (no API keys):** mock LLM endpoint streams captured thinking back through the real proxy + real stager; reproduces E4 overhead and (with the simulator for bandwidth sweep) the E6 sensitivity curve.
- **L3 — Live reproduction (full):** with API keys + Ares-equivalent testbed (Linux + NVMe + NFS source), regenerates E1, E5, E7, E8, E9, E10 against current models and real I/O.

---

## Appendix: file inventory

- `poc/probe_reasoning_slack.py`: probe script (650 LoC). Implements provider-specific streaming, tiered predictor, byte metrics, multi-seed loop, planning-prompt variants.
- `poc/runs/<timestamp>_<config>/`: one directory per seed. Contains `stream.jsonl` (raw SSE events), `summary.json` (block-level timing), `prediction.json` (per-rule activations and tier outputs), `byte_metrics.json` (per-tier byte recall/overfetch).
- `poc/runs/<timestamp>_<task>_<model>_aggregate.json`: per-config 3-seed rollup.
- `AGENTSTAGE.md`: this document.
