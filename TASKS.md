# Tasks

Persistent task list across sessions. Designed for the project owner +
Claude sessions, not external contributors. Mark `[x]` when done; preserve
history (don't delete completed entries — they're the only record of what
was done by which session).

Authoritative schedule: `AGENTSTAGE.md` §11.8.
Source of truth for hypotheses + claims: `AGENTSTAGE.md` §2-3.
Campaign matrix + protocol: `CAMPAIGN.md`.

Task IDs are `T<NN>`. Format: `- [ ] T01 <subject> — <served-by>`.

## Day 1 — 2026-05-19 (Scaffolding + rule freeze + E2 re-score)

- [x] T01 Scaffold uv package + benchmark submodules — _commit `eb422fd`_
- [x] T02 Scaffold paper_evals H1-H10 stubs — _commit `9f0d6a6`_
- [x] T03 Lay down CAMPAIGN/TASKS/README/.env.example — _commit `748a91d`_
- [x] T04 Add agentiobench submodule + revise architecture (client-lib primary, outputs/, 15-turn cap, cold cache mandatory) — _this commit_
- [x] T05 Port rule library `poc/probe_reasoning_slack.py` → `src/agentstage/predictor/rules.py` with `Rule` + `RuleSet` dataclasses and per-rule `origin` tagging for leave-one-out. 105 rules across 5 workloads (58 aiob_104 / 16 aiob_110 / 13 code_repo / 10 aiob_107 / 8 aiob_101; 16 tagged "general"). HOT scan + engine deferred to T07-T08.
- [x] T06 `RULE_LIBRARY_VERSION="v1"` + `RULE_LIBRARY_HASH` (sha256 over canonical serialization) in `src/agentstage/predictor/rules.py`; `tests/test_rules_freeze.py` pins hash + per-workload counts + origin distribution + leave-one-out filter behavior (9 tests, all green).
- [x] T07 `src/agentstage/workloads/{aiob,code_repo}.py` — `Workload` dataclass + per-task loaders (`load_aiob_101/104/107/110/code_repo`). `TaskConfig` mirrored locally to avoid AIOB's eager-numpy import (switches to `from agentiobench import TaskConfig` when T13b lands). Workspace priors keyed to match the rule library's `target_keys`. `AGENTIOBENCH_DATA_ROOT` env-resolved.
- [x] T08 `src/agentstage/metrics/byte_metrics.py` — `ByteScore` frozen dataclass + `byte_score()` + `file_size()` with `lru_cache` size lookup and prefix_map resolution. GOES collapse reproduces at 6078× overfetch in the smoke test.
- [x] T09 `src/agentstage/metrics/empirical_gt.py` — `load_empirical_reads()` parses io_report.json `file_name_view[*]` filtering on `posix_count_sum>0` + `posix_read_size_sum>0`, drops `/output/` and `/repo/result/` write targets. Verified on a real sciiobench io_report (aiob_101 sonnet-4-5: 39 files, 25.4 GB read).
- [x] T10 H3 headline assertions live: tier_1_first byte recall ≥0.85 on ≥90% (got 93% of 67 seeds), tier_1_first overfetch ≤1.5× on ≥95% (99%), tier_3_full recall ≥0.85 on ≥95% (99%), tier_3_full overfetch ≤2.0× on ≥95% (99%), Anthropic family 100%, Gemini family ≥95% overfetch. PoC corpus re-scored against frozen v1 rules. Required side-effects: ported predictor engine (`src/agentstage/predictor/engine.py` — parse_anthropic_stream + parse_gemini_stream + hot_path_scan + run_predictor), Campaign indexer (`src/agentstage/workloads/campaign.py`), rescore helper (`src/agentstage/metrics/rescore.py`), and `campaign` session fixture in `paper_evals/conftest.py`.
- [x] T11 H1 slack assertions live: median slack ≥5s, ≥80% of seeds clear 2s, ≥50% clear 5s, every provider family with ≥3 seeds has median ≥2s. Reads `slack_ms` derived from summary.json blocks (first thinking_delta → first tool_use).
- [x] T12 `src/agentstage/predictor/auto_rules.py` scaffolded — `AutoRuleGenerator` class with TODO docstring, returns empty RuleSet. Real implementation deferred to T13 / Day 2.

## Day 2 — 2026-05-20 (Leave-one-out + stager design + OSS setup + fetch scripts + AIOB branch)

- [ ] T13 Implement `test_h7_leave_one_out.py` 4-way parametrized assertion using the per-rule origin tagging from T05
- [ ] T13a Create `feat/agentstage-integration` branch on the agentiobench upstream repo (you create + push; this is the branch we'll bump our submodule pin to)
- [ ] T13b Add minimal hooks on that branch: (1) stable public API re-export in `agentiobench/__init__.py` (TaskConfig, evict_dataset, measure_temperature, dftracer_context); (2) optional `pre_turn_hook` and `post_turn_hook` parameters in `agentiobench/runner.py::run_task`. Bump our submodule pin to the new HEAD.
- [x] T14 Write `STAGER_DESIGN.md` — LD_PRELOAD shim, minimal syscall set (`openat`/`openat2`/`creat` + `statx`/`newfstatat`/`stat`/`lstat` + `access`/`faccessat`; reads/mmap follow the fd transparently), atomicity via rename-into-place, 32 GB NVMe hot tier (`/scratch/agentstage`), opportunistic LRU eviction on ENOSPC, in-process thread-pool stager (no daemon), DFTracer load order (dftracer first to log intent, agentstage second to redirect), 20 ms retry-spin on ENOENT, write pass-through
- [ ] T15a Confirm bekn-dtai-gh allocation: hours remaining via `accounts` on Delta + scope is OK for AgentStage (not exclusively AIOB). **Blocks T15.**
- [ ] T15 OSS model serving via NCSA Delta vLLM (**2-hour timebox**, down from 4 — AIOB has pre-built the vLLM stack on Delta):
    - rsync `scripts/delta/` to `~/dtai/agentstage/` on Delta
    - `salloc -A bekn-dtai-gh -p ghx4-interactive --gres=gpu:2 --cpus-per-task=32 --mem=300G --time=02:00:00`
    - `./run_vllm_qwen3_thinking.sh` (serves Qwen/Qwen3.6-27B with `enable_thinking=true` + `--reasoning-parser qwen3`)
    - On Ares: `./scripts/delta/tunnel.sh gh<NNN>`
    - `./scripts/delta/verify_vllm_thinking.sh` (asserts `reasoning_content` chunks stream)
    - Set `OSS_MODEL_BASE_URL=http://localhost:8002/v1`, `OSS_MODEL_NAME=Qwen/Qwen3.6-27B` in `.env`
    - **Fallback if no thinking signal in 2h:** drop OSS slot from new campaign; rely on PoC DeepSeek-R1 traces (re-scored Day 1) for third-family signal.
- [ ] T16 First end-to-end aiob_110 probe through the Delta-tunneled OSS model. **Blocked by T22** (need `src/agentstage/client/http.py` to handle `delta.reasoning_content`).
- [ ] T17 Write `scripts/fetch_datasets.sh` — SAB + KramaBench data pulls into `external/datasets/`

## Day 3 — 2026-05-21 (Client library + simulator)

- [ ] T18 `src/agentstage/client/base.py` — `AgentStageClient` ABC + `DataHint` dataclass (`predicted_files`, `tier`, `fired_at_ms`, `rule_id`, `byte_estimate`, `signature`). Tee-stream semantics: caller sees identical chunks; predictor sees identical chunks; stager sees prefetch dispatches.
- [ ] T19 `src/agentstage/client/anthropic.py` — wraps `anthropic.Anthropic.messages.create(stream=True)`; intercepts `thinking_delta`/`signature_delta`/`text_delta`/`input_json_delta`; runs predictor live; dispatches to stager. **Replaces previously-planned `src/agentstage/proxy/anthropic.py`.** Also handle OpenAI-shape clients: `delta.reasoning_content` (vLLM extension) is the thinking field for Qwen-served-via-vLLM and any future reasoning-parser-enabled OpenAI-compatible endpoint.
- [ ] T20 `src/agentstage/simulator/bandwidth.py` — bandwidth-vs-speedup sensitivity model (E6 backbone)

## Day 4 — 2026-05-22 (More client wrappers + cache integration + Campaign A start)

- [ ] T21 `src/agentstage/client/gemini.py` — wraps `google-genai`; parses `thinkingConfig.includeThoughts=true` per-part `thought: bool`
- [ ] T22 `src/agentstage/client/http.py` — raw urllib path matching the PoC's `run_*` functions; needed for OSS-vLLM endpoint (parses `delta.reasoning_content` separately from `delta.content`) and for benchmark environments where the SDK isn't installed
- [ ] T23 Cache eviction integration: `src/agentstage/runner.py` imports `agentiobench.utils.cache.evict_dataset, measure_temperature`; logs `pre_run_temperature` and `post_run_temperature` into `verdict.json`
- [ ] T24 Soft-stop logic: detect ≥ 1 KB write to task output dir after ≥ 3 tool calls; inject "stage 1 complete" stop message
- [ ] T25 Campaign-A orchestrator: `src/agentstage/cli/campaign.py` reading `CAMPAIGN.md`'s matrix, resume-by-cell-presence, provider-aware concurrency (3-5 inflight per provider)
- [ ] T26 **Campaign A — Haiku passes** on aiob_104, aiob_107, aiob_110, code_repo (60 probes) — see `CAMPAIGN.md` matrix
- [ ] T27 **Campaign A — Gemini Flash passes** on the same 4 workloads (40 probes)
- [ ] T28 Proxy microbench (E4) for client library: `experiments/E4_overhead/` — client-wrapped vs direct-SDK latency CDF; assert p99 delta ≤ 1%

## Day 5 — 2026-05-23 (Stager: shim + first end-to-end + dfanalyzer/dftracer)

- [ ] T29 Add `external/dfanalyzer` and `external/dftracer` submodules (pinned to the same commits sciiobench uses)
- [ ] T30 `src/agentstage/stager/daemon.py` — queue consumer + cold-tier fetcher
- [ ] T31 LD_PRELOAD shim (`src/agentstage/stager/shim/agentstage_shim.c`) + Makefile
- [ ] T32 First end-to-end smoke test on aiob_107 with Haiku, 15-turn cap, soft-stop enabled — measure first-read P50/P95 with vs without stager
- [ ] T33 Debug whatever breaks (highest schedule risk per §11.9)

## Day 6 — 2026-05-24 (Stager hardening + aiob_110 + E7)

- [ ] T34 Stager: size-aware LRU eviction (per `STAGER_DESIGN.md`)
- [ ] T35 End-to-end aiob_110 with Haiku and OSS — record `staging_report.json`
- [ ] T36 E7 graceful-degradation pairs (hint-on vs hint-off via client `data_hints_enabled=False`)

## Day 7 — 2026-05-25 (Stager DONE; full E5 runs)

- [ ] T37 5-seed E5 runs on aiob_107 + aiob_110 (Haiku + OSS, with/without stager — 40 runs)
- [ ] T38 Implement `test_h8_staging_effectiveness.py::test_first_read_p95_speedup`
- [ ] T39 Implement `test_h10_proxy_overhead.py::test_p99_latency_overhead_below_1pct` (now "client overhead", same idea)

## Day 8 — 2026-05-26 (ScienceAgentBench integration)

- [ ] T40 Pick 3 SAB tasks (criteria: NetCDF/CSV/HDF5/NWB I/O; representative, not the easiest)
- [ ] T41 Adapt SAB harness: monkey-patch its LLM client at runtime to use `agentstage.client.{Anthropic,OpenAI}Client` — no edits to the SAB submodule
- [ ] T42 **Sentinel run**: 1 SAB task end-to-end with Haiku, measure actual turn count vs CAMPAIGN.md estimate; re-budget E9 if off by > 2×
- [ ] T43 Capture SAB trace runs (12 probes: 3 tasks × 2 models × 2 seeds)

## Day 9 — 2026-05-27 (SAB end-to-end + KramaBench integration)

- [ ] T44 E9 end-to-end on SAB: 3 tasks × 2 models × 2 configs × 5 seeds = 60 runs
- [ ] T45 Implement `test_h6_frozen_rules_crosscorpus.py::TestFrozenRulesOnScienceAgentBench`
- [ ] T46 KramaBench task selection: one each from Astronomy (1556 files / 486 MB), Biomedical (7 files / 175 MB), Wildfire (23 files / 1 GB). Read `external/benchmarks/kramabench/data/<domain>/{domain}.json` for task picks.
- [ ] T47 Adapt KramaBench harness: monkey-patch its LLM client at runtime via `patch_openai_sdk` (`systems/baseline_example.py` uses the openai SDK). No edits to the KramaBench submodule.

## Day 10 — 2026-05-28 (KramaBench end-to-end + BW sweep + budget sweep)

- [ ] T48 E10 end-to-end on KramaBench: 3 tasks × 2 models × 2 configs × 5 seeds = 60 runs
- [ ] T49 Implement `test_h6_frozen_rules_crosscorpus.py::TestFrozenRulesOnKramaBench` (3 parametrized domains)
- [ ] T50 E6 bandwidth sensitivity: 1 measured BW point with tc/cgroup rate-limit + 3 simulator points
- [ ] T51 Implement `test_h9_bandwidth_sensitivity.py`
- [ ] T52 E8 thinking-budget sweep on aiob_110 (15 probes at varying budgets)

## Day 11 — 2026-05-29 (Paper draft §1-4 + figures)

- [ ] T53 Generate Figure 1 (slack vs data-movement-time) from `paper_evals/.results/report.json`
- [ ] T54 Generate Figure 2 (stageable bytes per backend)
- [ ] T55 Draft §1 Intro · T56 Draft §2 Background + SOTA · T57 Draft §3 Opportunity · T58 Draft §4 AgentStage design

## Day 12 — 2026-05-30 (Paper draft §5-7 + all figures)

- [ ] T59 Generate all results figures from `report.json`
- [ ] T60 Draft §5 Methodology · T61 §6 Prediction results · T62 §7 Staging effectiveness + robustness

## Day 13 — 2026-05-31 (Paper draft §8 + reproducibility kit)

- [ ] T63 Draft §8 Discussion + limitations + conclusion
- [ ] T64 Tighten Related Work
- [ ] T65 Package reproducibility kit (Docker compose, replay scripts) — Artifact A-E in §11.10
- [ ] T66 README.md final pass

## Day 14 — 2026-06-01 (Submit)

- [ ] T67 Polish pass on figures + tables
- [ ] T68 Final read-through
- [ ] T69 **Submit (AoE deadline)**

## Backlog (not on critical path; pick up if time allows)

- [ ] B01 OpenAI Responses API integration (gated on quota / Azure deployment)
- [ ] B02 Tier-2 rule engineering for medium-granularity (per-day, per-band) signals
- [ ] B03 MCP `data_hints` SEP draft (independent contribution; the client lib's DataHint maps directly to this)
- [ ] B04 NCSA Delta Lustre cross-PFS validation
- [ ] B05 Online-learned predictor (uses captured trace pairs as training data)
- [ ] B06 Sonnet sanity-check sub-matrix on aiob_110 + code_repo with frozen rules (~$3)
- [ ] B07 Full HTTP proxy implementation (beyond the thin `proxy/server.py` wrapper) — for harnesses that can't import agentstage at all
- [ ] B08 SWE-bench Lite end-to-end (Docker integration) — reconsidered for a future version of the paper; KramaBench replaces it for the eScience submission because of its better I/O profile

## Conventions

- **Marking done:** `- [x] T01 ...` — keep the task in place, don't delete
- **Discovering a new task mid-day:** add it to the appropriate day with the next available T number
- **Blocking notes:** add a sub-bullet `  - blocked by: T05 (frozen rules)`
- **Per-task commit:** when a task lands in git, append `— _commit `<hash>`_` to the line
- **Backlog graduation:** when a B-task gets pulled into a day, move (don't copy) it under that day with a new T number
