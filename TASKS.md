# Tasks

Persistent task list across sessions. Designed for the project owner +
Claude sessions, not external contributors. Mark `[x]` when done; preserve
history (don't delete completed entries — they're the only record of what
was done by which session).

Authoritative schedule: `AGENTSTAGE.md` §11.8.
Source of truth for hypotheses + claims: `AGENTSTAGE.md` §2-3.
Campaign matrix + cost: `CAMPAIGN.md`.

Task IDs are `T<NN>`. Format: `- [ ] T01 <subject> — <served-by>`.

## Day 1 — 2026-05-19 (Rule freeze + E2 re-score)

- [x] T01 Scaffold uv package + benchmark submodules — _commit `eb422fd`_
- [x] T02 Scaffold paper_evals H1-H10 stubs — _commit `9f0d6a6`_
- [ ] T03 Port rule library from `poc/probe_reasoning_slack.py` → `src/agentstage/predictor/rules.py` (workspace prior loader, HOT scan, semantic-class regex rules, tiering by target-set size) — **unblocks T04-T07**
- [ ] T04 Define `RULE_LIBRARY_VERSION` (semver + sha256 of rule definitions) in `src/agentstage/predictor/__init__.py`
- [ ] T05 Add `tests/test_rules_freeze.py` pinning the rule hash — the frozen-contract unit test
- [ ] T06 Port workload definitions (workspace inventories, static GT) from PoC `WORKSPACE_PRIOR_*` and `GROUND_TRUTH_*` blocks → `src/agentstage/workloads/aiob.py`
- [ ] T07 Write `src/agentstage/metrics/byte_metrics.py` (byte recall, overfetch) consuming `prediction.json` + GT — needed by H1-H4 tests
- [ ] T08 Write `src/agentstage/metrics/empirical_gt.py` to load `io_report.json` from sciiobench and produce per-cell empirical ground truth
- [ ] T09 Implement `test_h3_predictability.py` headline assertions (drop the `pytest.skip` in `TestTier1ImmediateNeed`) — re-score existing 88 PoC probes against frozen rules
- [ ] T10 Implement `test_h1_slack.py` median + per-provider asserts — direct read from `summary.json`
- [ ] T11 Scaffold auto-rule generator at `src/agentstage/predictor/auto_rules.py` (E11; empty class with TODO docstring is enough for Day 1)

## Day 2 — 2026-05-20 (Leave-one-out + stager design + OSS setup)

- [ ] T12 Tag each rule in `rules.py` with its origin workload (for leave-one-out filtering)
- [ ] T13 Implement `test_h7_leave_one_out.py` 4-way parametrized assertion
- [ ] T14 Write `STAGER_DESIGN.md` — LD_PRELOAD shim choice, syscall set to intercept, atomicity model, cache eviction policy
- [ ] T15 OSS model selection on Ares: identify GPU node with ≥32 GB headroom, install vLLM, serve Qwen3-Thinking-14B or DeepSeek-R1-Distill-Qwen-14B with SSE thinking enabled, verify on aiob_110 — see `CAMPAIGN.md` "OSS model setup"
- [ ] T16 Capture aiob_110 single-probe trace on the OSS model; commit `stream.jsonl` to `poc/runs/` (gitignored) and verify thinking content is non-empty
- [ ] T17 Write `scripts/fetch_datasets.sh` (SAB + SWE-bench data pulls into `external/datasets/`)

## Day 3 — 2026-05-21 (Simulator + proxy skeleton)

- [ ] T18 Write `src/agentstage/simulator/bandwidth.py` — bandwidth-vs-speedup sensitivity model (E6 backbone)
- [ ] T19 Scaffold `src/agentstage/proxy/anthropic.py` — SSE termination + forwarding, `content_block_start` / `thinking_delta` / `text_delta` / `input_json_delta` parsing
- [ ] T20 Scaffold `src/agentstage/cli/proxy.py` — `agentstage-proxy --provider anthropic --listen 127.0.0.1:9999 --upstream ...`

## Day 4 — 2026-05-22 (Gemini proxy + proxy microbench + stager skeleton)

- [ ] T21 `src/agentstage/proxy/gemini.py` — `streamGenerateContent` SSE with `thinkingConfig.includeThoughts=true`, `thought: bool` per-part
- [ ] T22 Proxy microbench (E4): `experiments/E4_proxy_overhead/` — proxy-on vs proxy-off latency CDF; assert p99 delta ≤ 1%
- [ ] T23 Scaffold `src/agentstage/stager/daemon.py` — queue consumer + cold-tier fetcher (no shim yet)
- [ ] T24 Run **Campaign A** Haiku passes on aiob_104, aiob_107, aiob_110, code_repo (60 probes) — see `CAMPAIGN.md` matrix
- [ ] T25 Run **Campaign A** Gemini Flash passes on the same 4 workloads (40 probes)

## Day 5 — 2026-05-23 (Stager: path-rewriting shim + first end-to-end)

- [ ] T26 LD_PRELOAD shim source (`src/agentstage/stager/shim/agentstage_shim.c`) + Makefile
- [ ] T27 First end-to-end smoke test on aiob_107 with Haiku — measure first-read P50/P95 with vs without stager
- [ ] T28 Debug whatever breaks in T27 (this is the day with the most schedule risk per §11.9)

## Day 6 — 2026-05-24 (Stager hardening + aiob_110 integration test)

- [ ] T29 Stager: cache eviction policy (size-aware LRU per `STAGER_DESIGN.md`)
- [ ] T30 End-to-end aiob_110 with both Haiku and OSS — record `staging_report.json`
- [ ] T31 Run E7 graceful-degradation pairs (no-thinking case with proxy-on vs proxy-off)

## Day 7 — 2026-05-25 (Stager DONE; E5 full runs)

- [ ] T32 Stager production: 5-seed E5 runs on aiob_107 + aiob_110, both models, with/without stager (40 runs)
- [ ] T33 Implement `test_h8_staging_effectiveness.py` (drop skip on `test_first_read_p95_speedup`)
- [ ] T34 Implement `test_h10_proxy_overhead.py` (drop skip on `test_p99_latency_overhead_below_1pct`)

## Day 8 — 2026-05-26 (ScienceAgentBench integration)

- [ ] T35 Pick 3 SAB tasks (criteria: NetCDF/CSV/HDF5/NWB I/O; representative of the SAB corpus, not the easiest tasks)
- [ ] T36 Adapt SAB harness to route LLM calls through `agentstage-proxy` (monkey-patch their LLM client at runtime, per `project_package_layout` memory)
- [ ] T37 Capture SAB trace runs (12 probes: 3 tasks × 2 models × 2 seeds for trace-only first pass)

## Day 9 — 2026-05-27 (SAB end-to-end + SWE-bench integration)

- [ ] T38 E9 end-to-end on SAB: 3 tasks × 2 models × 2 configs × 5 seeds = 60 runs
- [ ] T39 Implement `test_h6_frozen_rules_crosscorpus.py::TestFrozenRulesOnScienceAgentBench` (drop skip)
- [ ] T40 SWE-bench Lite instance selection (2 representative instances)
- [ ] T41 Route SWE-bench harness through proxy (containerized — needs `--network=host` or proxy-as-sidecar)

## Day 10 — 2026-05-28 (SWE-bench end-to-end + BW sweep + budget sweep)

- [ ] T42 E10 end-to-end on SWE-bench Lite: 2 instances × 1 model × 2 configs × 3 seeds = 12 runs
- [ ] T43 Implement `test_h6_frozen_rules_crosscorpus.py::TestFrozenRulesOnSWEbenchLite`
- [ ] T44 E6 bandwidth sensitivity: 1 measured BW point with tc/cgroup rate limit + 3 simulator points
- [ ] T45 Implement `test_h9_bandwidth_sensitivity.py`
- [ ] T46 E8 thinking-budget sweep on aiob_110 (15 probes at varying budgets)

## Day 11 — 2026-05-29 (Paper draft §1-4 + figures)

- [ ] T47 Generate Figure 1 (slack vs data-movement-time) from `paper_evals/.results/report.json`
- [ ] T48 Generate Figure 2 (stageable bytes per backend) — partly analytical, partly from staging traces
- [ ] T49 Draft §1 Intro
- [ ] T50 Draft §2 Background + SOTA
- [ ] T51 Draft §3 Opportunity characterization
- [ ] T52 Draft §4 AgentStage design

## Day 12 — 2026-05-30 (Paper draft §5-7 + all figures)

- [ ] T53 Generate all results figures from `report.json`
- [ ] T54 Draft §5 Methodology
- [ ] T55 Draft §6 Prediction results
- [ ] T56 Draft §7 Staging effectiveness + robustness

## Day 13 — 2026-05-31 (Paper draft §8 + reproducibility kit)

- [ ] T57 Draft §8 Discussion + limitations + conclusion
- [ ] T58 Tighten Related Work
- [ ] T59 Package reproducibility kit (Docker compose, replay scripts) — Artifact A-E in §11.10
- [ ] T60 README.md final pass

## Day 14 — 2026-06-01 (Submit)

- [ ] T61 Polish pass on figures + tables
- [ ] T62 Final read-through
- [ ] T63 **Submit (AoE deadline)**

## Backlog (not on critical path; pick up if time allows)

- [ ] B01 OpenAI Responses API integration (gated on quota / Azure deployment)
- [ ] B02 Tier-2 rule engineering for medium-granularity (per-day, per-band) signals
- [ ] B03 MCP `data_hints` SEP draft (independent contribution)
- [ ] B04 NCSA Delta Lustre cross-PFS validation
- [ ] B05 Online-learned predictor (uses captured trace pairs as training data)
- [ ] B06 Sonnet sanity-check sub-matrix on aiob_110 + code_repo with frozen rules (anchors PoC numbers in fresh data; ~$3)

## Conventions

- **Marking done:** `- [x] T01 ...` — keep the task in place, don't delete
- **Discovering a new task mid-day:** add it to the appropriate day with the next available T number, even if out of numeric order
- **Blocking notes:** add a sub-bullet under the blocked task: `  - blocked by: T03 (frozen rules)`
- **Per-task commit:** when a task lands in git, append `— _commit `<hash>`_` to the line
- **Backlog graduation:** when a B-task gets pulled into a day, move (don't copy) it under that day with a new T number
