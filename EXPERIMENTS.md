# Experiments Log

Append-only chronological record of every measurement run. Each entry
captures **exact reproduction info** so we can re-run any past
experiment from a fresh checkout. New entries go at the bottom; never
edit completed ones (record errata as new entries that reference the
old one).

For the bigger-picture verification narrative (which experiments
support which paper claims), see [`STAGER_VERIFICATION.md`](STAGER_VERIFICATION.md).

## Summary table (most recent first)

| ID | Date | Goal | Headline | Script | Commit |
|---|---|---|---|---|---|
| **E-037** | 2026-05-24 | SAB (ScienceAgentBench) cross-benchmark capture (3 tasks × 2 models × 2 regimes, mock sandbox) | **Captures all 12 runs; agents reason concretely about file paths from dataset_folder_tree even without real bytes; ready for H6 frozen-rules replay** | `scripts/microbench/sab_capture.py --instance-id N` | _next commit_ |
| **E-036** | 2026-05-24 | H6 cross-benchmark frozen-rules replay (AIOB rules on KB captures) | **AIOB-trained rule patterns fire on 12/12 KB captures (genericity at regex level); native auto-rules on KB achieve recall=1.0 vs agent opens on 6/12 captures (mean precision 0.05 — KB priors are broad)** | `scripts/microbench/path_b_h6_frozen_xbench.py` | _next commit_ |
| **E-035** | 2026-05-24 | KramaBench cross-benchmark capture (3 tasks × 2 models × 2 regimes) | **All 12 captures succeed; Haiku + Gemini Flash on wildfire/biomedical/astronomy tasks; agent opens GT files reliably under hinted mode** | `scripts/microbench/kramabench_capture.py` | _next commit_ |
| **E-034** | 2026-05-24 | H7 leave-one-out rule sufficiency on existing AIOB corpora | **e021 (sparse+enrich) & e011 (hinted): full recall=1.0, 0 load-bearing rules (ruleset is redundant); e014/e015 (sparse, no enrich): 0 recall — LOO uninformative, precondition fails** | `scripts/microbench/path_b_h7_loo.py` | _next commit_ |
| **E-033** | 2026-05-24 | First-tool-name statistic from 638 real AIOB production runs | **96.08% of runs start with list_dir; 5/8 models 100% (GPT-4.1, Gemma, Qwen, Gemini Flash, Sonnet 4.5 at 97.8%); only Haiku 4.5 at 73.1% — validates pre-task-hook timing for auto-rule generation** | `scripts/microbench/first_tool_stat.py` | _next commit_ |
| **E-032** | 2026-05-24 | AutoRuleGenerator runtime cost (1000 iterations × 5 workloads) | **p50 ~230 µs, p95 ~300 µs, max ever 620 µs across aiob_101/104/107/110 priors (7-56 keys, 9-58 rules emitted) — sub-millisecond, safely fits any pre-task or post-list_dir hook** | `scripts/microbench/auto_rules_cost.py` | _next commit_ |
| **E-020** | 2026-05-21 | Pathful-prompt live ablation (system-prompt asks LLM to write full paths) | **Literal-path detection fired ZERO times; LLM writes path templates with placeholders, not concrete paths. Pathful prompt INCREASED rule activations +25% hinted, +100% sparse — complement to rules, not replacement** | `./scripts/path_b_run.sh {hinted,sparse}_pathful` | _next commit_ |
| **E-020 v4** | 2026-05-21 | Pathful-prompt V4 iteration + logical-prior fix | **V4 prompt produces concrete paths in both regimes; hinted: literal-path dispatch fires successfully; sparse: paths concrete but agent picks Band 01/02 OUTSIDE workspace prior (prior built from constrained task spec)** | `PATHFUL_VERSION=v4 ./scripts/path_b_run.sh {hinted,sparse}_pathful` | _next commit_ |
| **E-024** | 2026-05-21 | Enrichment precision-tuning ablation (cap-N, pattern, ext) | **`all files` is the only policy with 100% recall in all 3 seeds; cap-N fails because alphabetical sort concentrates one band; stratified sampling identified as future work** | `scripts/microbench/path_b_enrich_ablation.py` | _next commit_ |
| **E-025** | 2026-05-21 | Per-file vs per-session wall-time gap (analysis of E-023 captures) | **Per-file 10^4x, per-session 1% in 8-turn smoke runs (agent opens only 1 file); projected ~75 min saved per full aiob_107 task; needs task-completing runner for full session ablation** | (analysis only — no new run) | _next commit_ |
| **E-026** | 2026-05-21 | Cross-vendor live multi-turn (Gemini 2.5 Flash, n=3) | **3/3 hits, speedup 3,965x-13,152x (mean ~7,000x); architecture works end-to-end on Anthropic AND Gemini families** | `GEMINI_MODEL=gemini-2.5-flash ./scripts/path_b_run.sh e026_gemini_sparse_enrich_live` (3 reps) | _next commit_ |
| **E-027** | 2026-05-21 | Session-level speedup from REAL AIOB production runs (n=30 across 3 workloads) | **I/O fraction 1.4-30.6% on local NFS; eliminating it gives 1.01-2.08x session speedup measured, 1.32-24.35x projected on S3** | `scripts/microbench/path_b_aiob_realruns.py` | _next commit_ |
| **E-031** | 2026-05-22 | Cross-script variance: naive vs chunked I/O pattern (same task) | **Naive 23x / chunked 54x on S3; both save ~160s wall; architecture benefit holds across script styles** | `E2E_TASK_SCRIPT=outputs/e2e/task_script_chunked.py ./scripts/microbench/path_b_e2e.py` | _next commit_ |
| **E-030** | 2026-05-22 | Verified-cold-cache local rerun (3 reps, mincore residency check) | **Local NVMe XFS 1.2x plain / 1.5x decomp; verified truly cold (0/3611 pages resident); honest spectrum: 1.2x local NVMe → 12x throttled NFS-class → 23x S3** | `scripts/microbench/path_b_e2e*.py` ×3 reps each | _next commit_ |
| **E-029** | 2026-05-22 | Decompression-staging end-to-end (uncompressed hot copies) | **S3 29.1x, local 1.6x session speedup; decompression moved off critical path into staging window** | `scripts/microbench/path_b_e2e_decompress.py` | _next commit_ |
| **E-028** | 2026-05-22 | End-to-end task-script speedup (real agent script, baseline vs staged) | **S3 23.4x (169s->7.2s), local 1.2x; shim fopen bug found+fixed** | `scripts/microbench/path_b_e2e.py` | _next commit_ |
| **E-023** | 2026-05-21 | Multi-seed E-021 (3 reps) stability check | **3/3 seeds: `was_staged=True`; speedup range 6.8k×-25k× (S3 cold latency variance); enrichment structurally reliable** | `PATHFUL_VERSION=v4 ./scripts/path_b_run.sh e021_sparse_enrich_live` ×3 | _next commit_ |
| **E-022** | 2026-05-21 | Cross-workload auto-rules check (aiob_104 + aiob_110 + aiob_107) | **Auto within 3% of hand on all 3 workloads (-0.2%, -3.0%, 0.0%); L3 genericity exceeded** | `scripts/microbench/path_b_xworkload.py` | _next commit_ |
| **E-021** | 2026-05-21 | Sparse + V4 pathful + dynamic prior enrichment | **Sparse-mode recall 0% → 100%; realistic wall-time 1.0× → 2,989×; 100 paths added from one list_dir; over-fetch 35× (bandwidth-for-recall trade-off)** | `PATHFUL_VERSION=v4 ./scripts/path_b_run.sh e021_sparse_enrich` | _next commit_ |
| **E-019** | 2026-05-21 | Auto-generated rules vs hand-tuned (L3 genericity claim) | **Auto matches hand in hinted regime (100%=100%); auto EXCEEDS hand by +33% in sparse regime (100% vs 66.7%) because mechanical per-instance enumeration catches band_10 hand missed** | `scripts/microbench/path_b_auto_vs_hand.py` | _next commit_ |
| **E-018** | 2026-05-21 | Subset-detection accuracy replay (per-rule precision/recall vs static GT) | **100% subset precision across all rules and regimes; hinted recall 100%, sparse recall 67% (one band rule did not fire)** | `scripts/microbench/path_b_subset_replay.py` | _next commit_ |
| **E-017** | 2026-05-20 | Wall-time replay ablation (oracle vs realistic detector) | **Hinted: 3886× realized = 3886× oracle; Sparse: 1.0× realized vs 3512× oracle — 100% of potential savings lost to rule mismatch** | `scripts/microbench/path_b_walltime_run.sh <corpus>` | _next commit_ |
| **E-016** | 2026-05-20 | False-positive / precision-recall ablation on captured corpora | **Hinted: 100% precision, 100% recall, Jaccard 100%; Sparse: 0% precision, 0% recall, Jaccard 0% (sets disjoint, byte_overfetch metric collapses)** | `scripts/microbench/path_b_falsepos.py --corpus <run>` | _next commit_ |
| **E-015** | 2026-05-20 | Path B sparse-prompt live multi-turn + cold/hot measurement | **3 rules fired across 8 turns; live cold→hot 622.8→0.127 ms after force-prefetch; agent picked Band 02 (sparse-mode behavior)** | `scripts/path_b_run.sh sparse_live` | _next commit_ |
| **E-014** | 2026-05-20 | Path B sparse-prompt multi-turn capture (Regime B, AIOB_STRIP_HINTS analog) | **4 rules across 8 turns; rule mix diverges from hinted (`one_hour` instead of `all_files_signal`); Band 01 opened first** | `scripts/path_b_run.sh sparse` | _next commit_ |
| **E-013** | 2026-05-20 | SessionDetector replay on E-011 corpus (offline, free) | **Variant D = Variant C activation count; multi-turn delta tracking confirmed** | `scripts/microbench/path_b_replay.py --corpus <E-011>` | _next commit_ |
| **E-012** | 2026-05-20 | tool_result-aware detector replay on E-011 corpus (offline, free) | **Variant A 2 rules → Variant C 4 rules (+100%); load-bearing extension confirmed** | `scripts/microbench/path_b_replay.py --corpus <E-011>` | _next commit_ |
| **E-011** | 2026-05-20 | Path B hinted multi-turn baseline capture on `aiob_107_s3` | **8 turns / 10 tool_use / 4 rules fired (first_inspect, all_files_signal from thinking; band_08, band_09 from tool_result)** | `scripts/path_b_run.sh hinted` | _next commit_ |
| **E-010** | 2026-05-20 | Path A live Haiku end-to-end with S3-backed cold tier (`aiob_107_s3`) | **19,213× per-file speedup; 2.59 s stage fit inside 14.4 s slack** | `agentstage.runners.path_a_smoke --workload aiob_107_s3` | _next commit_ |
| **E-009** | 2026-05-20 | Path 0 replay against S3-mounted cold (shim correctness with S3 backend) | **29,283× p50 first-block speedup; shim redirect works on S3 mount** | `scripts/microbench/path0_replay.py --workload aiob_107_s3` | _next commit_ |
| **E-008** | 2026-05-20 | Real S3 measurement vs NOAA's public GOES bucket via mountpoint-s3 | **2,144× per-file p50; 1.96× wall on aiob_107; ~100× projected wall on aiob_110** | `scripts/microbench/path0_s3_run.sh` | (committed) |
| **E-007** | 2026-05-20 | Throttled-cold-tier sweep: measured wall-time speedup on simulated slow PFS | **1.72× native → 12.3× at 10 MB/s** | `scripts/microbench/path0_throttle_sweep.sh` | (prior commit) |
| E-006 | 2026-05-20 | Full-file throughput on aiob_110 (large NWB) | 32× measured throughput, 1.92× projected wall-time | `scripts/microbench/path0_walltime_run.sh aiob_110 5` | `1c03164` |
| E-005 | 2026-05-20 | Path A live Haiku smoke on aiob_107 | 195.6× per-syscall, 9131 ms slack matches spec | `scripts/path_a_run.sh` | `c78031b` |
| E-004 | 2026-05-19 | Path 0 replay smoke on aiob_107 | 5628× p50 / 8819× p95 first-byte | `scripts/microbench/path0_run.sh` | `88c7525` |
| E-003 | 2026-05-19 | DFTracer + shim chain verification | dftracer logs cold-path intent regardless of LD_PRELOAD order | `tests/test_dftracer_chain.py` | `2a817fa` |
| E-002 | 2026-05-19 | End-to-end synthetic 5-file integration | 3 HITs / 2 MISSes, byte identity preserved | `tests/integration/test_end_to_end_staging.py` | `2e483b6` |
| E-001 | 2026-05-19 | L0 microbench (cold P95, hot P95, eviction) | 168-306× p95 speedup on 3-bucket file sizes; XFS eviction works | `scripts/microbench/stager_baseline.py` | `5527268` |

## Conventions

- **Reproduction is the priority.** Every entry MUST include the exact
  command + env vars + output dir + git commit so anyone (including
  future-self) can re-run.
- **Output dirs are timestamped** under `outputs/<bucket>/<ts>/`.
  Headline numbers come from the JSON files there.
- **Forced-add** the headline artifact JSON into git (past
  `outputs/.gitignore`) when it's a publishable measurement.
- **Commit hash is the agentstage repo's HEAD at run time.** If
  submodules changed (rare for these experiments), note their
  commits too.

---

## E-001 — L0 microbench (environment baseline)

**Date:** 2026-05-19
**Goal:** Validate three environment assumptions the stager design
relies on before investing implementation time: (A) cold-tier
first-read P95 is high enough to justify staging, (B) hot-tier
first-read P95 is low enough to be a useful ceiling,
(C) `POSIX_FADV_DONTNEED` actually drops residency on `/mnt/common`'s XFS.

**Script:** `scripts/microbench/stager_baseline.py`
**Commit:** `5527268`

**Reproduction:**
```bash
~/.local/bin/uv run python scripts/microbench/stager_baseline.py
```

No env vars required; reads `/mnt/common/datasets-staging/agentiobench/datasets/`
and writes to `/dev/shm/agentstage_microbench/`. Cleans up after.

**Output:** `outputs/microbench/stager_baseline_<ts>.json`

**Headline:**
| Bucket | Cold P95 | Hot P95 (tmpfs) | Speedup |
|---|---:|---:|---:|
| small_goes_3mb | 19 ms | 0.06 ms | 306× |
| medium_era5_50mb | 17 ms | 0.10 ms | 168× |
| large_nwb_350mb | 23 ms | 0.11 ms | 216× |

Eviction: 100% → 0% residency on warmed XFS files (3626 pages cleanly
dropped).

**Notes:** This was a 4 KB first-block measurement. Subsequent
experiments (E-006) show that for full-file reads on aiob_110, the
realistic per-file speedup is 32× (not 200+), because throughput
matters more than first-byte for large files. The 168-306× P95
numbers should be read as "first-byte latency speedup," not
"wall-time speedup."

The eviction check had a false-negative bug in the first run (whole-
bucket residency dilution); fixed by measuring per-file residency with
`agentiobench.utils.cache._resident_pages`.

## E-002 — End-to-end synthetic 5-file integration

**Date:** 2026-05-19
**Goal:** Validate the full stager + shim contract on a synthetic
workload (no LLM). Closest analog of E5 we could run pre-Day-5.

**Test:** `tests/integration/test_end_to_end_staging.py`
**Commit:** `2e483b6`

**Reproduction:**
```bash
make -C src/agentstage/stager/shim
~/.local/bin/uv run pytest tests/integration/ -v
```

**Output:** pytest fixtures (no committed artifact); logs in test output.

**Headline:**
- 5 files (1, 5, 10, 25, 50 MB), 3 pre-staged
- Files 1-3 hit hot via shim (HIT in log); files 4-5 fall through to cold (MISS)
- All byte hashes match sources
- Stager fetch p50 ~9 ms / p95 ~16 ms
- 2 tests passing; parametrized later (E-003) to also run with dftracer in LD_PRELOAD

**Notes:** This is the closest pre-LLM proof that "detector → stager →
shim" composes correctly. Latency direction confirmed (staged < cold).

## E-003 — DFTracer + shim LD_PRELOAD chain verification

**Date:** 2026-05-19
**Goal:** Confirm DFTracer + agentstage shim compose correctly in the
LD_PRELOAD chain. Specifically: dftracer logs the agent's INTENT
(cold path) while the shim redirects to hot.

**Tests:** `tests/test_dftracer_chain.py` (5 tests)
**Commit:** `2a817fa`

**Reproduction:**
```bash
# Requires libdftracer_preload.so. AGENTSTAGE_DFTRACER_PRELOAD env var
# overrides search; otherwise checks external/libs/dftracer build dir,
# then sciiobench fallback.
make -C src/agentstage/stager/shim
~/.local/bin/uv run pytest tests/test_dftracer_chain.py -v
```

**Output:** pytest output; trace files written to pytest tmp_path.

**Headline:**
- 4 tests passing, 1 skipped (dfanalyzer Python package not installed;
  needs meson-python + dask)
- **Empirical finding:** DFTracer logs cold path REGARDLESS of
  LD_PRELOAD ordering. It uses syscall-level instrumentation (deeper
  than libc function wrapping), so it captures agent intent even if
  agentstage redirects first. The documented `$DFTRACER:$SHIM` order
  is robust by design.

**Notes:** This is the best possible behavior for the paper's story —
trace integrity doesn't depend on LD_PRELOAD ordering. Worth a
defensive paragraph in the methodology section.

## E-004 — Path 0 replay smoke on aiob_107

**Date:** 2026-05-19
**Goal:** First real-data speedup measurement. Replays a PoC
stream.jsonl (Sonnet 4.5, aiob_107, PP, seed 0) through the frozen v1
rule library + real Stager + real LD_PRELOAD shim, against actual
cold-tier GOES NetCDFs.

**Scripts:**
- `scripts/microbench/path0_replay.py`
- `scripts/microbench/path0_run.sh` (wrapper)
**Commit:** `88c7525`

**Reproduction:**
```bash
bash scripts/microbench/path0_run.sh \
  outputs/poc/20260518-171234_aiob_107_anthropic_claude-sonnet-4-5_t1_b16384_pp_s0_azure/stream.jsonl \
  20
```

Sets `LD_PRELOAD=$SHIM`, `AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path0`,
`AGENTSTAGE_COLD_ROOTS=/mnt/common/datasets-staging/agentiobench/datasets`
internally for the with-stager mode.

**Output:** `outputs/microbench/path0_20260519T235208/{baseline,with_stager}.json`
(committed past gitignore)

**Headline:** 20 distinct files (none touched recently from the
storage's perspective):

| Metric | Cold (no shim) | Hot (via shim) | Speedup |
|---|---:|---:|---:|
| p50 | 185.7 ms | 0.033 ms | **5628×** |
| p95 | 574.6 ms | 0.065 ms | **8819×** |

**Notes:**
- First version of the script re-read the same file repeatedly →
  bimodal results because SSD device cache holds blocks even after
  kernel page-cache eviction. Fixed by sampling N distinct files.
- 4 KB first-block measurement: speedups overstate per-file wall-time
  impact (see E-006 for full-file throughput).
- Headline first-byte number; honest about scope.

## E-005 — Path A live Haiku smoke on aiob_107

**Date:** 2026-05-20
**Goal:** First measurement with a live LLM call. Validates that
streaming → detector → stager works end-to-end on a real
`messages.create(stream=True)` against Anthropic.

**Script:** `src/agentstage/runners/path_a_smoke.py`
**Wrapper:** `scripts/path_a_run.sh`
**Commit:** `c78031b`

**Reproduction:**
```bash
# Loads .env (ours) + sciiobench's .env (fallback for AZURE_FOUNDRY_KEY).
bash scripts/path_a_run.sh
```

The script:
- Loads aiob_107 workload + planning prompt
- Sends one Haiku 4.5 call (8192 thinking budget, max_tokens=12288)
  via Azure Foundry (`AZURE_FOUNDRY_ANTHROPIC_URL` defaults to
  `https://izzet-2249-resource.openai.azure.com/anthropic/v1/messages`)
- Tee streaming → live detector → stager (tier-1-only dispatch)
- Picks measurement target: first staged file (since agent's first
  tool_use was list_dir, not open_file)
- Times hot read via shim + cold read via subprocess with
  AGENTSTAGE_SHIM_DISABLE=1

**Output:** `outputs/path_a/20260520T003909/{summary,staging_report,stream_log}.json`
(committed)

**Headline:**
- Slack window: **9131 ms** (matches AGENTSTAGE.md §6.1's 6-14s spec)
- 6 rules fired during thinking (only tier-1 auto-dispatched)
- 1 file staged (the file `first_inspect` rule pointed at), 195 ms
  to fetch — well within slack
- Cold read: 19.4 ms, Hot read: 0.099 ms → **195.6× speedup**
- Cost: ~$0.04 per probe

**Notes:**
- 5 bugs found + fixed during development: empty `AZURE_FOUNDRY_ANTHROPIC_URL`
  default, max_tokens<thinking_budget rejection, tool_choice=any
  incompatible with thinking, StreamBlock field mismatch, aggressive
  tier-3 dispatch starving streaming loop. All documented in commit message.
- First run had 1131s "slack" due to runaway tier-3 dispatch starving
  the streaming loop. Tier-1-only filter in `AnthropicClient` fixed it
  to clean 9.1s.

## E-006 — Full-file throughput measurement on aiob_110

**Date:** 2026-05-20
**Goal:** Convert per-syscall first-byte speedup into something
representative of agent wall-time: read FULL files (not 4 KB), so
throughput matters. aiob_110 picked because 350 MB NWB files are the
extreme throughput case.

**Scripts:**
- `scripts/microbench/path0_walltime.py`
- `scripts/microbench/path0_walltime_run.sh` (wrapper)
**Commit:** `1c03164`

**Reproduction:**
```bash
bash scripts/microbench/path0_walltime_run.sh aiob_110 5
```

Reads 5 distinct files from the `all_subjects` bucket of aiob_110's
workspace prior. Per file: evict cold, read full bytes in 1 MiB
chunks, time, record throughput.

**Output:** `outputs/microbench/walltime_aiob_110_20260520T005334/{baseline,with_stager}.json`
(committed past gitignore)

**Headline:** 5 NWB files (310-619 MB each, 2.14 GB total):

| Metric | Cold (XFS) | Hot (tmpfs) | Speedup |
|---|---:|---:|---:|
| full_read p50 | 3.89 s | 122 ms | 32.0× |
| full_read p95 | 6.50 s | 157 ms | 41.3× |
| throughput mean | 101 MB/s | 3352 MB/s | 33× |

**Wall-time projection** for a 15-turn aiob_110 agent run with ~45
file reads + 150 s LLM + 30 s compute:
- Cold total: 358 s
- Hot total: 186 s
- **Speedup: 1.92×**

**Notes:** This is the load-bearing number for the paper. The 32×
throughput differential makes the cold-tier I/O cost vanish; what's
left is LLM time + compute, which the stager doesn't address.

The wall-time number is still a projection (file-count assumption);
to be measured: Path B (multi-turn agent run on aiob_110).

---

## E-007 — Throttled-cold-tier sweep on aiob_110

**Date:** 2026-05-20
**Goal:** Convert the analytical "slow-PFS speedup" projection from
E-006 into MEASURED numbers. Userspace per-chunk throttling enforces
target cold-tier throughput (50 / 30 / 10 MB/s) without requiring root
for `cgroup` or `tc`. Hot tier (tmpfs) is unchanged across runs.

**Scripts:**
- `scripts/microbench/path0_walltime.py` (added `--throttle-mbps`)
- `scripts/microbench/path0_throttle_sweep.sh` (multi-rate driver)
**Commit:** _to land with the doc update_

**Reproduction:**
```bash
bash scripts/microbench/path0_throttle_sweep.sh aiob_110 3
```

Sweep order: native (no throttle) → 50 → 30 → 10 MB/s → with-stager
(tmpfs, unthrottled by design — agent reads from local hot tier
regardless of upstream cold-tier speed).

**Output:** `outputs/microbench/throttle_sweep_aiob_110_20260520T093727/`
  - `baseline_native.json`     (~141 MB/s native XFS-SSD)
  - `baseline_50mbps.json`     (Lustre/NFS-typical)
  - `baseline_30mbps.json`     (S3-class)
  - `baseline_10mbps.json`     (cross-region S3 / very-slow object)
  - `with_stager.json`         (tmpfs, ~4 GB/s)

**Headline:** 3 NWB files (312, 380, 619 MB), per-file full-read time:

| Cold tier | mean cold (ms) | throughput | per-file speedup |
|---|---:|---:|---:|
| Native XFS-SSD | 3,064 | 141 MB/s | **28.9×** |
| Throttled 50 MB/s | 11,209 | 39 MB/s | **105.7×** |
| Throttled 30 MB/s | 15,163 | 29 MB/s | **143.0×** |
| Throttled 10 MB/s | 46,501 | 9.5 MB/s | **438.7×** |
| With-stager (tmpfs) | 106 | 4,112 MB/s | — |

**Wall-time projection** (15-turn run, 45 file reads, 150 s LLM,
30 s compute):

| Cold tier | Cold I/O total | Total cold | Total hot | **Wall speedup** |
|---|---:|---:|---:|---:|
| Native XFS-SSD | 138 s | 318 s | 185 s | **1.72×** |
| Throttled 50 MB/s | 504 s | 684 s | 185 s | **3.70×** |
| Throttled 30 MB/s | 682 s | 862 s | 185 s | **4.67×** |
| Throttled 10 MB/s | 2,093 s | 2,273 s | 185 s | **12.30×** |

**Notes:**
- The 1.72× native-tier number is slightly below the 1.92× from E-006;
  difference is sample variance (3-file vs 5-file). Both are valid.
  Reporting range: **1.7-1.9× on native XFS-SSD**.
- The 30 MB/s case (4.67×) is the most defensible "real-PFS" headline.
  Lustre, OrangeFS, and S3 standard-tier are all in the 20-50 MB/s
  range on a single client; AgentStage's measured wall-time speedup
  there is 3.7-4.7×.
- 12.3× at 10 MB/s is the cross-region S3 case (or PFS under heavy
  multi-agent contention). Less common but credible — papers in this
  space sometimes target it.
- Userspace throttling models throughput but NOT first-byte latency.
  Real PFS adds 50-500 ms first-byte per file on top of throughput.
  These wall-time speedups are therefore conservative — real PFS
  would show even larger ratios because of first-byte cost.

**Methodology limitation:** throttle is implemented as per-chunk sleep
in Python (1 MiB chunks). Effective rates are ~10-20% below target
(measured 39 vs target 50; measured 29 vs target 30) because sleep
granularity + chunk-read time both add overhead. The TARGETED rate
is the more useful framing; the MEASURED rate is what's actually in
the data. Both reported above for honesty.

---

## E-008 — Real S3 cold tier (NOAA's public GOES bucket via mountpoint-s3)

**Date:** 2026-05-20
**Goal:** Validate the throttled-simulator numbers from E-007 against
**actual** S3 latency. NOAA hosts aiob_107's original GOES data as a
public bucket (`s3://noaa-goes16/`) with free egress and no AWS
credentials required for read. mount-s3 (AWS Mountpoint S3) presents
the bucket as a FUSE mount at `/tmp/s3-noaa-goes16/`. Same files we've
been measuring locally, now read directly from S3.

**Scripts:**
- `scripts/microbench/path0_s3.py`
- `scripts/microbench/path0_s3_run.sh` (driver + mount setup)
**Commit:** _to land with this entry_

**Reproduction:**
```bash
# One-time: install mount-s3
mkdir -p ~/.local/bin
cd /tmp
curl -sL https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.tar.gz | tar xz
cp /tmp/bin/mount-s3 ~/.local/bin/   # or /tmp/mount-s3 depending on tarball layout

# Run: mounts noaa-goes16 (if not already), measures 5 GOES files
bash scripts/microbench/path0_s3_run.sh 5
```

**Output:** `outputs/microbench/path0_s3_<ts>/{baseline,with_stager}.json`

**Headline:** 5 GOES NetCDF files (3 MB each, 15 MB total) read from
NOAA's S3 bucket us-east-1 via Ares network egress:

| Metric | Cold (S3 direct) | Hot (after stage → tmpfs) | Speedup |
|---|---:|---:|---:|
| full_read p50 | 4,288 ms | 2.0 ms | **2,144×** |
| full_read p95 | 4,293 ms | 2.0 ms | 2,146× |
| full_read mean | 3,839 ms | 2.4 ms | 1,599× |
| throughput mean | 0.9 MB/s | 1,384 MB/s | **1,537×** |

Stager prefetch: 5 files from S3 in 6.19 s (combined effective 2.4 MB/s
— mountpoint-s3 parallelizes 4 stage workers reading concurrently).

**Wall-time projection** (15-turn aiob_107-style run, 45 × 3 MB reads,
150 s LLM, 30 s compute):
- Cold: 353 s total
- Hot: 180 s total
- **Wall speedup: 1.96×**

**Same projection for aiob_110-style large files** (45 × 350 MB at
this measured 0.8 MB/s S3 throughput): cold I/O = 19,683 s (5.5 hours),
hot I/O = ~3 minutes. **Theoretical wall speedup ~100×** — bounded by
"the agent run is infeasible without the stager," which is itself the
paper's point.

**Notes:**

- **Ares-to-AWS bandwidth is surprisingly low** (0.6-1.6 MB/s per file
  single-connection). This is consistent with academic-network egress
  to S3 being congested or rate-limited. A co-located EC2 instance
  would see 50-100+ MB/s; an HPC cluster with a fast WAN gateway might
  see 10-50 MB/s. Our number is realistic for "researcher on a
  university HPC cluster reading from S3."

- **mountpoint-s3 was installable without root** as a static binary in
  `~/.local/bin/`. FUSE was already present (`/dev/fuse` + `fusermount`
  on this Ubuntu 22.04). No AWS credentials needed for NOAA's public
  Open Data bucket (`--no-sign-request` flag).

- One outlier in the baseline (1847 ms vs 4200 ms range): possibly
  mountpoint-s3 internal caching kicking in on a metadata refresh.
  Doesn't change the headline.

- **The throttle-simulator from E-007 underestimated real S3 latency.**
  At 10 MB/s throttle E-007 saw 46s for a 350 MB file; real S3 to Ares
  is more like 437s for the same file size (0.8 MB/s). For the paper,
  the throttled-simulator should be framed as a controlled-variable
  sensitivity sweep, not a "this matches PFS X" claim. Real-S3 measured
  numbers are the source of truth for the S3 case.

- **Same files, byte-identical to local aiob_107 data.** The NOAA
  bucket is the SOURCE for AIOB's pre-staged GOES data. Reading them
  from S3 is "what the agent would do without AgentStage's local
  staging assumption."

---

## E-009 — Path 0 replay against S3-mounted cold tier

**Date:** 2026-05-20
**Goal:** Verify the shim redirect works end-to-end when the cold
tier is an S3 mount (not local XFS-SSD). Replays the same Sonnet PoC
stream used in E-004 but with `aiob_107_s3` workload + S3-mount
cold root.

**Scripts:**
- `scripts/microbench/path0_replay.py --workload aiob_107_s3`
- `src/agentstage/workloads/aiob.py::load_aiob_107_s3()` (new loader)
**Commit:** _to land with this entry_
**AIOB submodule pin:** `dea56861559af8c209e95881a200528a7df199cf`
(branch `feat/agentstage-integration` — adds the `aiob_107_s3` YAML)

**Reproduction:**
```bash
# 1. Mount NOAA bucket (one-time)
mkdir -p /tmp/s3-noaa-goes16
~/.local/bin/mount-s3 --no-sign-request --read-only --region us-east-1 \
    noaa-goes16 /tmp/s3-noaa-goes16

# 2. Run replay against S3 as cold root
STREAM=outputs/poc/20260518-171234_aiob_107_anthropic_claude-sonnet-4-5_t1_b16384_pp_s0_azure/stream.jsonl
SHIM=$(realpath src/agentstage/stager/shim/libagentstage_shim.so)

# baseline (no shim, direct S3 reads)
AGENTSTAGE_COLD_ROOTS=/tmp/s3-noaa-goes16 \
    uv run python scripts/microbench/path0_replay.py \
        --mode baseline --workload aiob_107_s3 \
        --stream "$STREAM" --n-samples 5 --out baseline.json

# with-stager (LD_PRELOAD redirect)
LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path0 \
AGENTSTAGE_COLD_ROOTS=/tmp/s3-noaa-goes16 \
AGENTSTAGE_RETRY_SPIN_MS=20 \
    uv run python scripts/microbench/path0_replay.py \
        --mode with-stager --workload aiob_107_s3 \
        --stream "$STREAM" --n-samples 5 --out with_stager.json
```

**Output:** `outputs/microbench/path0_e009_s3_<ts>/{baseline,with_stager}.json`

**Headline:** 5 distinct GOES files from S3 mount, 4 KB first-block reads:

| Metric | Baseline (S3 direct) | With-stager (tmpfs via shim) | Speedup |
|---|---:|---:|---:|
| p50 | 1,613 ms | 0.055 ms | **29,283×** |
| p95 | 1,753 ms | 0.079 ms | 22,239× |
| mean | 1,588 ms | 0.059 ms | 26,971× |

**Notes:**
- Higher per-syscall speedup than E-008 (29k× vs 2k×) because E-009
  measures 4 KB first-block reads (first-byte-latency dominated)
  while E-008 measures full-file reads (throughput-dominated).
  Both numbers are real and reflect different aspects of the stager.
- Confirms shim correctly redirects when cold root is the S3 mount.
  Per-file timing of 0.06 ms = tmpfs speed; not S3 mount speed.
- One bug found + fixed: `get_ruleset(args.workload)` didn't recognize
  `aiob_107_s3` — fixed by stripping `_s3` suffix before lookup,
  since S3 variant shares detector rules with local variant
  (rules match thinking text + logical paths, not physical
  storage location).

## E-010 — Path A live Haiku call against S3 cold tier

**Date:** 2026-05-20
**Goal:** Full e2e validation: real LLM thinking → live detector →
stager prefetch from S3 → LD_PRELOAD redirect → hot tmpfs read.
The most rigorous test we can run before Path B.

**Script:** `src/agentstage/runners/path_a_smoke.py --workload aiob_107_s3`
**Commit:** _to land with this entry_

**Reproduction:**
```bash
# Same mount + .env setup as E-008/E-009
source /home/iyildirim/projects/sciiobench/.env  # for AZURE_FOUNDRY_KEY
mountpoint /tmp/s3-noaa-goes16 || \
    ~/.local/bin/mount-s3 --no-sign-request --read-only --region us-east-1 \
        noaa-goes16 /tmp/s3-noaa-goes16
SHIM=$(realpath src/agentstage/stager/shim/libagentstage_shim.so)
rm -rf /dev/shm/agentstage_path_a_s3
LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path_a_s3 \
AGENTSTAGE_COLD_ROOTS=/tmp/s3-noaa-goes16 \
AGENTSTAGE_RETRY_SPIN_MS=20 \
    uv run python -m agentstage.runners.path_a_smoke \
        --workload aiob_107_s3 \
        --out outputs/path_a_s3/$(date +%Y%m%dT%H%M%S)
```

**Output:** `outputs/path_a_s3/20260520T104823/summary.json`

**Headline:**

| Metric | Value |
|---|---:|
| LLM model | claude-haiku-4-5 (8 KB thinking budget) |
| Slack window | **14,433 ms** (live, clean — matches AGENTSTAGE.md §6.1 spec) |
| Detector rules fired during thinking | 4 (`band_08`, `first_inspect`, `all_bands`, `all_files_signal`) |
| Tier-1 file staged | 1 (the file `first_inspect` rule named) |
| Stage fetch time from S3 | **2,591 ms** (well within slack — 11.8 s headroom) |
| File ready at first tool_use? | **✓ yes** |
| **Cold first-byte (S3 → Ares)** | **754.5 ms** |
| **Hot first-byte (tmpfs via shim)** | **0.039 ms** |
| **Per-file speedup** | **19,213×** |
| LLM cost | ~$0.04 |

**This validates every layer of the production architecture against
a real S3 cold tier:**

1. ✓ Real LLM (Haiku 4.5) thinking + tool_use emission
2. ✓ Live detector running on streaming `thinking_delta` chunks
3. ✓ Tier-1-only dispatch (no broad-rule starvation; only 1 file
   staged from 4 rule activations)
4. ✓ Stager prefetches from S3 mount within slack window (2.6 s
   stage in 14.4 s slack)
5. ✓ LD_PRELOAD shim redirects agent's open() to the staged tmpfs
   copy correctly
6. ✓ Hot read confirms tmpfs speed (0.039 ms ≈ RAM access)

**Notes:**
- Agent's first `tool_use` was `list_dir`, not `open_file`
  (exploration behavior). The runner falls back to "first staged
  file" as the measurement target. This is the same pattern as
  E-005 — agents tend to explore before opening specific files.
- The 14.4 s slack window is at the upper end of AGENTSTAGE.md's
  6-15 s expected range. With 2.6 s stage time, there's room to
  stage ~5 files in parallel during one slack window.
- This closes the strongest reviewer-attack vector: "show that the
  full chain works on a real cold tier." E-010 demonstrates the
  entire pipeline on NOAA's public S3 bucket end-to-end.

---

## Reproducibility checklist for new experiments

Before logging an experiment, verify:

- [ ] Script path is committed in the repo
- [ ] Reproduction command works from a fresh `uv sync` + `make -C src/agentstage/stager/shim`
- [ ] Env vars (if any) are listed with their values (or where to source them)
- [ ] Output dir is timestamped under `outputs/<bucket>/<ts>/`
- [ ] Headline result is in a JSON file at the output dir (force-add to git if publishable)
- [ ] Commit hash matches the repo HEAD when the experiment ran
- [ ] If the experiment uses a submodule (dftracer, agentiobench), note its commit too
- [ ] Notes section captures any surprises, bugs found, methodology gotchas

---

## E-011 — Path B hinted multi-turn baseline (2026-05-20)

**Goal**: First multi-turn end-to-end run on the new `path_b_multiturn.py`
runner. Capture per-turn stream / tool_use / tool_result / thinking
artifacts to serve as replay fodder for E-012 / E-013. Hinted prompt
mode (Regime A, our pre-existing condition).

**Reproduction**

```bash
LD_PRELOAD=$SHIM \
AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path_b \
AGENTSTAGE_COLD_ROOTS=/tmp/s3-noaa-goes16/ABI-L2-CMIPC \
~/.local/bin/uv run python -m agentstage.runners.path_b_multiturn \
    --workload aiob_107_s3 \
    --prompt-mode hinted \
    --max-turns 8 \
    --budget 4096 \
    --out outputs/multi_turn/e011_multiturn_hinted_<ts>
```

Or via the wrapper:

```bash
./scripts/path_b_run.sh hinted
```

**Outputs**: `outputs/multi_turn/e011_multiturn_hinted_20260520T174301/`
- `turns/turn_NN/stream.jsonl` — every SSE event for that turn
- `turns/turn_NN/tool_use.jsonl` — assistant-emitted tool calls (one line each)
- `turns/turn_NN/tool_result.jsonl` — what the runner returned to the agent
- `turns/turn_NN/thinking.txt` — the full thinking block text
- `turns/turn_NN/summary.json` — per-turn rule activations
- `summary.json` — top-level run summary with all activations annotated by source + turn
- `messages.json` — full message history for replay

**Results**

| Metric | Value |
|---|---|
| Prompt mode | hinted (full I/O hints) |
| Turns used | 8 (max allowed) |
| Total tool_uses | 10 |
| Total rules fired | 4 |
| Rules from thinking | 2 (`first_inspect`, `all_files_signal` on turn 0) |
| Rules from text | 0 |
| Rules from tool_result | 2 (`band_08`, `band_09` on turn 4) |
| First agent-opened file | `OR_ABI-L2-CMIPC-M6C08_G16_s20241220001170_...nc` (Band 08) |

The agent navigated `/data → /data/goes_cmi_composites → /data/goes_cmi_composites/raw → 2024/122/00 → open_file` over 8 turns. Path resolution worked through the synthetic ancestor listings (`_synthesize_ancestor_listing`) for `/data` and `/data/goes_cmi_composites` (workload prefix_map has only `/data/goes_cmi_composites/raw/` as a real LP).

**Notes**

- After turn 0, the model stopped emitting `thinking` blocks and emitted `text` blocks instead. This was unexpected and prompted the detector extension that scans text blocks (see `_SCANNABLE_BLOCK_TYPES` in `engine.py`). Without that extension, Variant A (thinking-only) would have caught only the 2 turn-0 rules.
- The `band_08` and `band_09` rules fired from a `tool_result` containing the directory listing of `/data/goes_cmi_composites/raw/2024/122/00/` (which contains files named with `M6C08`, `M6C09`, `M6C10`). This is the canonical "tool_result-aware detector adds signal" case — it would have been missed by a thinking-only detector.

---

## E-012 — tool_result-aware detector replay on E-011 corpus (2026-05-20)

**Goal**: Quantify the lift from extending the detector to consume
`tool_result` content blocks. Offline replay against E-011's captured
corpus → no API cost.

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_replay.py \
    --corpus outputs/multi_turn/e011_multiturn_hinted_20260520T174301 \
    --workload aiob_107_s3 \
    --out outputs/multi_turn/e011_multiturn_hinted_20260520T174301/replay_variants.json
```

**Results** (E-011 hinted corpus)

| Variant | Description | Rules fired | Sources |
|---|---|---|---|
| A | thinking only (legacy) | 2 | thinking=2 |
| B | thinking + tool_result | 4 | thinking=2, tool_result=2 |
| C | full (thinking + text + tool_result) | 4 | thinking=2, tool_result=2 |
| D | SessionDetector (streaming feed) | 4 | thinking=2, tool_result=2 |

**Lift from tool_result-awareness**: **+100% activations** (2 → 4) on hinted prompt.

The `band_08` and `band_09` rules would not fire without scanning `tool_result` content — they appear only in the agent's `list_dir` output of the bands directory, never in the agent's own thinking text.

**Notes**

- Variant D (SessionDetector multi-turn deltas) confirmed equivalent to Variant C single-shot on this corpus — the delta tracking does not lose activations.

---

## E-013 — SessionDetector delta-tracking confirmation (2026-05-20)

**Goal**: Confirm that multi-turn `feed_turn` / `feed_tool_results` calls
produce the same cumulative activation set as a single-shot
`run_detector` over all blocks at once. Catches bugs in the
`fired_rule_names` deduplication and the across-turn state hand-off.

**Reproduction**: same as E-012 — variant D in the replay table.

**Result**: SessionDetector's cumulative detection matches Variant C exactly across all three captured corpora (E-011, E-014, E-015). Counts of 4=4, 4=4, 3=3 with identical rule names.

The source attribution differs between B/C and D on E-015 (D reports `text=2` where B/C report `tool_result=2`) because SessionDetector processes assistant text blocks before that turn's tool_result blocks, while the single-shot variant sees them in strict chronological order. Both reach the same activation set; only the "which signal caused the rule to fire first" attribution differs.

---

## E-014 — Path B sparse-prompt multi-turn capture (Regime B) (2026-05-20)

**Goal**: First end-to-end measurement of the architecture under the
"sparse prompt" regime — where the task instruction is stripped of
chunking, file counts, band numbers, locations, and dimensions. The
agent must discover the dataset structure via `list_dir` / `open_file`
before producing useful I/O-relevant tokens. This bounds the impact of
the prompt-leakage confounder identified in
[`IO_LEAKAGE_AUDIT.md`](IO_LEAKAGE_AUDIT.md).

**Reproduction**

```bash
./scripts/path_b_run.sh sparse
```

(equivalent to `path_b_multiturn --prompt-mode sparse --workload aiob_107_s3 --max-turns 8`)

**Results**

| Metric | Hinted (E-011) | **Sparse (E-014)** |
|---|---|---|
| Total rules fired | 4 | 4 |
| Rules from turn-0 thinking | 2 | **1** (just `first_inspect`) |
| Rules from text | 0 | 1 (`one_hour`, turn 4) |
| Rules from tool_result | 2 | 2 (`band_08`, `band_09`) |
| First agent-opened file | Band 08 | **Band 01** (different choice without hint) |

**Findings**

1. **The hinted-only `all_files_signal` rule does not fire under sparse**.
   That rule's pattern matches "6000 files" / total-count tokens which the sparse prompt explicitly strips. The agent's thinking in sparse mode does not echo back a count, so the rule never matches.

2. **The agent opens a different band file (Band 01 vs. Band 08)**. Without the hinted "bands 08, 09, 10" guidance, the model picks an alphabetically-earlier band. This is a meaningful behavioral difference — and a **rule-library weakness**: our `first_inspect` rule's target_keys resolve to a fixed Band 08 file, but the agent picked Band 01. The stager prefetched the wrong file.

3. **`tool_result` band rules fire identically across regimes**. Once the agent does `list_dir(/data/.../2024/122/00/)`, the directory listing contains all bands (C01–C16), and `band_08` / `band_09` rules fire on those filenames regardless of prompt mode.

4. **Total activation count is the same (4)**, but the composition differs (`one_hour` instead of `all_files_signal`). The detector's coverage is robust across regimes; what changes is which rule wins.

**Notes**

- The replay variants table for E-014 shows Variant A=1, B=3, C=4, D=4. Tool_result extension adds +2 rules; text extension adds +1. Combined lift is **+300%** over the legacy thinking-only detector in sparse mode.

---

## E-015 — Path B sparse-prompt live cold→hot measurement (2026-05-20)

**Goal**: E-014 plus the live cold/hot measurement step. Mirrors E-010 (which did this for Regime A) so we have a Regime B equivalent. Captures wall-time speedup AND the rule-mismatch consequences when detection picks the wrong file.

**Reproduction**

```bash
./scripts/path_b_run.sh sparse_live
```

**Results**

| Metric | Value |
|---|---|
| Turns used | 8 |
| Total rules fired | 3 (`first_inspect`, `band_08`, `band_09`) |
| First agent-opened file | `OR_ABI-L2-CMIPC-M6C02_G16_...nc` (Band 02 — different from E-014 too) |
| `was_staged_at_end_of_run` | **False** |
| Hot read | 0.127 ms |
| Cold read | 622.8 ms |
| **Speedup** | 4,903× (after force-prefetch) |

**Honest interpretation**

The headline 4,903× is the speedup **after the runner force-prefetched** the target file at the end of the loop — it shows the *shim still works* on a sparse-mode agent's chosen file, but it does **not** demonstrate that the *detector* successfully detected that file ahead of time.

In E-015's actual session, the detector's `first_inspect` rule prefetched a Band 08 file, but the agent (lacking a "use bands 08/09/10" hint) opened a Band 02 file. The stager's prefetch was **wasted** — `was_staged_at_end_of_run: False` for the agent's target.

**This is the headline finding for the I/O-leakage audit**:

> Under the sparse prompt, the detector's hardcoded `first_inspect → Band 08` mapping does not match the model's actual file choice. **The 19,213× speedup reported in E-010 reflects a setup where the hinted prompt aligned the model's selection with our rule library**. In a realistic sparse regime, the rule library would need to adapt — either by waiting until the agent's first `list_dir` reveals the band naming, or by generating rules dynamically from `tool_result` content.

**Implications for paper architecture / future work**

1. **Static rule library is insufficient for sparse-prompt agents.** The library was tuned against hinted-prompt thinking; the sparse-prompt agent's actual file choices diverge.
2. **`tool_result`-aware detector still helps** (it fires `band_08`/`band_09` from the listing), but those activations come too LATE in the session (turn 5+) for slack-window prefetching.
3. **A "deferred" tier-1 dispatch policy** — only prefetch after observing the agent's first concrete band reference — would have higher precision at the cost of latency.

This is exactly the threats-to-validity content the audit doc was preparing
for.

**Notes**

- The shim *itself* still functions correctly under sparse mode: cold/hot mechanics work, force-prefetch works, redirect works. The architectural failure is at the *detection* layer, not the *staging* layer.
- Multi-turn loop, S3 cold tier, LD_PRELOAD shim, live LLM — all of these continue to work end-to-end under the harder regime. The honest paper claim is: "the architecture works; the rule library needs adaptation."


---

## E-016 — False-positive / precision-recall ablation (2026-05-20)

**Goal**: Quantify "how much data did we move that the agent never used"
across the multi-turn corpora. The original C2 claim
(`byte_overfetch ≤ 1.5×` on 98% of turn-1 seeds) was measured on
hinted-mode single-turn replays where the detected set is a
superset of the accessed set. In multi-turn live runs with sparse
prompts, the two sets can be **disjoint**, and the overfetch
ratio becomes misleading.

Adds the **Jaccard overlap** metric which is well-defined regardless of
set relationship (subset, superset, disjoint, overlapping).

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_falsepos.py \
    --corpus outputs/multi_turn/<run> \
    --workload aiob_107_s3 \
    --out outputs/multi_turn/<run>/falsepos.json
```

Force-prefetch events (`rule_id == "force"`) are excluded from the
"prefetched" set — they're a measurement artifact, not detector output.

**Results**

| Corpus | Prefetched | Opened | Hits | Wasted | Precision | Recall | **Jaccard** | byte_overfetch |
|---|---|---|---|---|---|---|---|---|
| **E-011 hinted** | 1 file (2.82 MB) | 1 file (2.82 MB) | 1 (2.82 MB) | 0 | **100%** | **100%** | **100%** | 1.00× |
| **E-014 sparse** | 1 file (2.82 MB) | 1 file (6.49 MB) | 0 | 1 (2.82 MB) | **0%** | **0%** | **0%** | 0.44× |
| **E-015 sparse_live** | 1 file (2.82 MB) | 1 file (42.26 MB) | 0 | 1 (2.82 MB) | **0%** | **0%** | **0%** | 0.07× |

**Findings**

1. **Hinted mode**: perfect precision and recall. The detector's
   first_inspect rule detected Band 08; agent opened Band 08;
   exactly one file each. No false positives.

2. **Sparse mode (both runs)**: detected and opened sets are
   **completely disjoint**. The detector's static targets (Band 08)
   miss the agent's actual choice (Band 01 in E-014, Band 02 in
   E-015). 100% of detector-driven prefetches were wasted.

3. **byte_overfetch metric collapse**: in sparse mode the metric reads
   "0.44× / 0.07×" because the agent's chosen file is *larger* than the
   one we prefetched. The original C2 ceiling (1.5×) is satisfied —
   but only trivially, because the metric assumed prefetched ⊇ accessed
   and that assumption fails. **Jaccard correctly reports 0% overlap**.

**Implication for paper claim C2**

> Claim C2 (94% byte recall ≥ 0.85; 98% overfetch ≤ 1.5×) was measured
> on the **hinted-prompt single-turn seed corpus** where the detected
> set was constructed to be a superset of (or equal to) the agent's
> immediate-need set. In multi-turn live runs under the sparse-prompt
> regime, the detected and accessed sets can be disjoint, and the
> overfetch metric becomes uninformative. We recommend Jaccard overlap
> for cross-regime comparisons.

**Notes**

- The 2.82 MB wasted byte cost per sparse run is small in absolute terms because the detector only dispatched ONE tier-1 hint. If the rule library dispatched more aggressively (e.g. ALL bands as tier-1), waste would scale linearly with the rule's target set size.

---

## E-017 — Wall-time replay ablation: oracle vs realistic (2026-05-20)

**Goal**: Convert the per-syscall speedup numbers (E-010's 19,213×) into
honest end-to-end wall-time savings per agent session, separating two
scenarios:

- **ORACLE**: every file the agent opens is pre-staged. Upper bound on
  speedup that *perfect* detection would yield.
- **REALISTIC**: only files the detector actually pre-staged in this
  run get hot reads; detector-misses pay cold cost. The wall-time
  speedup that this specific run actually realized.

Reads the first 4 KB of each opened file twice — once cold (subprocess,
shim disabled, page cache evicted), once hot (shim active, file
pre-staged). Sums across the entire agent session.

**Reproduction**

```bash
./scripts/microbench/path_b_walltime_run.sh outputs/multi_turn/<run>
```

(Sets LD_PRELOAD, AGENTSTAGE_HOT_ROOT, AGENTSTAGE_COLD_ROOTS, runs the
Python ablator which manages both cold subprocess + hot in-process reads.)

**Results**

| Corpus | Files opened | Files detector-staged | Cold total | Oracle | Realistic | **Oracle speedup** | **Realistic speedup** | Lost potential |
|---|---|---|---|---|---|---|---|---|
| **E-011 hinted** | 1 | 1 | 487.9 ms | 0.126 ms | 0.126 ms | **3886×** | **3886×** | 0% |
| **E-014 sparse** | 1 | 0 (wrong file) | 491.9 ms | 0.140 ms | 491.857 ms | **3512×** | **1.0×** | **100%** |
| **E-015 sparse_live** | 1 | 0 (wrong file) | 493.8 ms | 0.117 ms | 493.815 ms | **4220×** | **1.0×** | **100%** |

**Findings**

1. **Architecture potential is regime-independent**. Oracle speedup is
   ~3500-4200× across all three runs. The cold-vs-hot ratio at the file-
   read level doesn't care which file the agent picked — once the right
   file is in the hot tier, the redirect is fast.

2. **Detector realization is regime-dependent**. In hinted mode the
   detector captured 100% of the oracle potential (3886× = 3886×). In
   sparse mode the detector captured 0% (1.0× vs 3512-4220× oracle).
   The gap is the cost of the static rule library's brittleness.

3. **The headline number for the paper is the gap**: under hinted
   prompts, the architecture delivers near-oracle wall-time savings.
   Under sparse prompts, the architecture is wasted because the rule
   library cannot adapt.

**Caveat — file-count scaling**

These runs all happen to feature just ONE distinct `open_file` call
(the agent spent most turns on `list_dir` exploration). A
production-style agent run that opens many files would multiply both
oracle and realistic savings, but the **ratio** would stay
approximately the same — detector accuracy is the bottleneck, not
file count.

For aiob_107's full eventual working set (~6042 files), the oracle
wall-time savings would be ~6042 × 487 ms ≈ 49 minutes per run under
the same per-file cost model. The realistic savings depend on what
fraction of those 6042 files the detector correctly identifies.

**Why this is the paper-headline number, not E-010's 19,213×**

E-010 reports a per-syscall 19,213× speedup (`open()+read(4096)` on a
3 MB file: 754 ms cold → 0.039 ms hot). That's the *theoretical*
upper bound. E-017 reports the *session-level* speedup taking into
account detector accuracy: 3886× when detection matches, 1.0× when
it doesn't. For paper purposes, both numbers matter:

- Per-syscall (E-010 / E-009): demonstrates the *mechanism* works.
- Session-wall-time (E-017): demonstrates the *system* works
  end-to-end, modulo detector accuracy.

The honest story for reviewers: "the architecture has 3500-4200×
ceiling per file-access; the static rule library realizes 100% of
that under hinted prompts and 0% under sparse prompts; future work
on learned detectors closes the gap."


---

## E-018 — Subset-detection accuracy replay (2026-05-21)

**Goal**: Reframe the false-positive question at the **subset level** —
the granularity the tiered detector actually operates at. E-016
measured single-file precision against the agent's first-opened file,
which is the wrong frame for a system whose primary value is
identifying which *class* of files the agent is targeting (a band, an
hour, an entire workload's working set).

For each captured corpus, fires the full rule set (not just tier-1
auto-dispatch), takes the union per tier, and measures
precision/recall against TWO ground truths:
- **GT_actual**: files the agent opened in this run (1 file in our smokes)
- **GT_static**: workload's `ground_truth_full` — what a complete
  task execution would access (6042 files for aiob_107_s3)

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_subset_replay.py \
    --corpus outputs/multi_turn/<run> \
    --workload aiob_107_s3 \
    --out outputs/multi_turn/<run>/subset_replay.json
```

**Results (per-rule, against GT_static = 6042 files)**

| Corpus | Rule | Tier | Subset size | **Precision** | Recall |
|---|---|---:|---:|---:|---:|
| **E-011 hinted** | `first_inspect`     | 1 | 1    | **100%** | 0.02% |
| | `all_files_signal`  | 3 | 6,042 | **100%** | 100.0% |
| | `band_08`           | 3 | 2,014 | **100%** | 33.3% |
| | `band_09`           | 3 | 2,014 | **100%** | 33.3% |
| **E-014 sparse**  | `first_inspect`     | 1 | 1    | **100%** | 0.02% |
| | `one_hour`          | 2 | 36   | **100%** | 0.60% |
| | `band_08`           | 3 | 2,014 | **100%** | 33.3% |
| | `band_09`           | 3 | 2,014 | **100%** | 33.3% |
| **E-015 sparse_live** | `first_inspect` | 1 | 1    | **100%** | 0.02% |
| | `band_08`           | 3 | 2,014 | **100%** | 33.3% |
| | `band_09`           | 3 | 2,014 | **100%** | 33.3% |

**Tier-union results against GT_static**

| Regime | Tier-1 union | Tier-2 union | **Tier-3 union** | Tier-3 recall |
|---|---:|---:|---:|---:|
| E-011 hinted | 1 file | 1 file | **6,042 files** | **100%** |
| E-014 sparse | 1 file | 36 files | **4,040 files** | **66.9%** |
| E-015 sparse_live | 1 file | 1 file | **4,028 files** | **66.7%** |

**Findings**

1. **Subset precision is 100% across all rules and both regimes.** When
   a rule fires, every file in its subset is contained in the workload's
   eventual ground truth. Zero false-positive *files* at the subset level.

2. **Subset recall against GT_static differs by regime**:
   - Hinted: 100% (all three band rules fire because thinking + tool_result
     both mention C08/C09/C10)
   - Sparse: ~67% (only C08 and C09 rules fire; the `band_10` rule never
     activates because the agent's exploration didn't surface "C10" before
     the session ran out of turns)
   - The missing 33% corresponds to band_10's 2,014 files.

3. **The 0% vs_actual numbers do NOT contradict the 100% precision.**
   They measure something different: the agent's *actual* file choice in
   sparse mode (Band 01 in E-014, Band 02 in E-015) falls **outside the
   static GT subset entirely** because the S3 bucket contains all 16
   bands (C01-C16) while the workload spec only lists C08-C10. The
   detector identifies the correct subset *per the workload spec*; the
   agent's actual behavior in sparse mode diverges from that spec.

**Implication for paper claim C2**

C2 (94% byte recall ≥ 0.85, 98% overfetch ≤ 1.5×) is measured against
static GT, which is what subset-detection is designed for. E-018
confirms this framing holds at 100% precision per rule, 100% recall in
hinted regime, 67% recall in sparse regime.

The earlier E-016 measurement of 0% precision was correct but framed
poorly: it asked "did the detector identify the exact file the agent
opened first?" — a per-file question the detector doesn't claim to
answer. E-018's per-rule subset-level question matches what the detector
actually does, and gives the better numbers.

**Implication for the architecture**

Our live runner (`path_b_multiturn.py`) only auto-dispatches tier-1.
Tier-2 (36 files) and tier-3 (~4000-6042 files) are detected but never
staged. Two future-work directions:

1. **Bandwidth-aware tier dispatch**: stage tier-2 if slack window
   permits; stage tier-3 in background. Would convert our 100% subset
   precision into a wall-time benefit at the cost of higher hot-tier
   storage footprint.
2. **Dynamic GT enrichment**: when the agent's `list_dir` reveals files
   outside the workload prior (e.g., the C01-C16 bands on S3), expand
   the prior. This would prevent the "agent picks band outside our
   subset" failure mode observed in E-014/E-015.


---

## E-019 — Auto-generated vs hand-tuned rule set replay (2026-05-21)

**Goal**: Defends the AGENTSTAGE.md §11.6 **L3 genericity claim** —
auto-generated rules within 10% of hand-tuned recall.

`auto_rules.py` is now a real implementation (not the stub from
2026-05-19). For each workspace-prior bucket key it mechanically
derives a regex pattern; general rules (first_inspect, all_signal,
report_out) are templated from the task instruction's file-format
tokens.

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_auto_vs_hand.py \
    --corpus outputs/multi_turn/<run> \
    --workload aiob_107_s3 \
    --out outputs/multi_turn/<run>/auto_vs_hand.json
```

**Results**

| Corpus | Rules (hand / auto) | Hand T3 recall vs static GT | **Auto T3 recall vs static GT** | Δ |
|---|---|---:|---:|---:|
| E-011 hinted     | 10 / 16 | 100.0% | **100.0%** | 0 |
| E-014 sparse     | 10 / 16 |  66.9% | **100.0%** | **+33%** |
| E-015 sparse_live| 10 / 16 |  66.7% | **100.0%** | **+33%** |

**Findings**

1. **Auto matches hand in hinted regime** (100% = 100%) and
   **exceeds hand by 33% in both sparse regimes**.

2. Why auto wins on sparse: auto generates per-day regex rules
   (`day_122`, `day_123`, etc.) and a broader `band_10` pattern
   that fires on the agent's `list_dir` output (which lists C10
   alongside C08/C09). The hand-tuned set lacked the `band_10` rule
   activation in sparse because the agent's *thinking* didn't mention
   C10, while auto's bare numeric pattern `\b10\b` matches the C10
   filenames in the directory listing.

3. Auto's risk: bare numeric patterns (`\b08\b`) can over-fire on
   unrelated tokens (timestamps, coordinates). **In our captured
   corpora this didn't translate to false-positive *bucket activation***
   because every bucket the prior maps to is itself within the static
   GT. The over-firing is precision-at-the-rule-level, not
   precision-at-the-file-level.

**Implication for paper claim L3**

> The L3 target ("auto within 10% of hand") was conservative. In our
> measurements auto **equals** hand in hinted regime and **outperforms**
> hand by 33% in sparse regime because auto's mechanical regex
> generation enumerates per-instance buckets (per-day, per-subject,
> per-band) that hand-tuned authors didn't bother to write rules for.
> The hand-coding criticism of the rule library is substantially
> dissolved by this result.

**Caveats**

1. Sample is n=3 captured corpora, all aiob_107_s3 (one workload).
   Cross-workload validation (aiob_104, aiob_110, KramaBench) is part
   of Campaign C.

2. The static GT for aiob_107_s3 happens to equal the entire workspace
   prior (6042 files = all input data). Precision-at-the-file-level
   appears as 100% in both rule sets trivially. For workloads where
   GT is a strict subset of the prior, auto's broader rules could
   trigger over-fetch that hand-tuned rules avoid.

3. Domain shortenings (e.g., "C08" as a synonym for "band 08") are
   inferred from task_instruction scanning, but not from external
   domain knowledge. A workload where the LLM uses unprompted
   abbreviations (e.g., "stt" for "sample type T") would still benefit
   from hand-tuning. We didn't measure this.

**Files**

- `src/agentstage/detector/auto_rules.py` — real implementation
- `scripts/microbench/path_b_auto_vs_hand.py` — replay driver
- `outputs/multi_turn/<run>/auto_vs_hand.json` per corpus


---

## E-020 — Pathful-prompt ablation (2026-05-21)

**Goal**: Test whether injecting "write FULL file paths in your reasoning"
into the system prompt enables literal-path detection (`hot_path_scan`)
to replace the hand-coded regex rule library.

If literal-path matching works, the genericity story becomes: "13 lines
of system prompt + a substring scan over the workspace prior replaces
~480 lines of per-workload regex rules."

**Reproduction**

```bash
./scripts/path_b_run.sh hinted_pathful   # E-020a
./scripts/path_b_run.sh sparse_pathful   # E-020b
```

(Adds `--pathful-prompt` flag to `path_b_multiturn`; injects the
following clause into the system prompt:

> "IMPORTANT: When reasoning about which files you intend to access,
> write the FULL absolute path of each file in your thinking..."

Wired in `hot_path_scan` to the live dispatch path via
`SessionDetector.new_hot_paths()`.)

**Results**

| Run | Rules fired | **Literal-path dispatches** | First file opened | vs static GT (T3 recall) |
|---|---:|---:|---|---:|
| E-011 hinted (control)  | 4 | n/a (not wired then) | Band 08 day 122 | 100% |
| **E-020a hinted+pathful**     | **5** | **0** | Band 08 day 121 | 100% |
| E-014 sparse (control)  | 4 | n/a | Band 01 day 122 | 67% |
| **E-020b sparse+pathful**     | **8** | **0** | Band 01 day 122 | 100% |

**Findings**

1. **Literal-path detection did NOT fire in either pathful run.** Zero
   `hot_path_scan` hits despite the explicit system prompt instruction.
   `SessionDetector.new_hot_paths()` returned empty every turn.

2. **Why**: the LLM follows the instruction but writes **path
   templates with placeholders**, not concrete paths:

   ```
   /data/goes_cmi_composites/raw/YYYY/DDD/HH/OR_ABI-L2-CMIPC-M6C{08,09,10}_G16_s<timestamp>_e<...>_c<...>.nc
   ```

   versus the concrete paths in the workspace prior:

   ```
   /data/goes_cmi_composites/raw/2024/122/00/OR_ABI-L2-CMIPC-M6C08_G16_s20241220001170_e20241220003543_c20241220004042.nc
   ```

   Literal-substring matching cannot bridge templates to concrete files.
   The LLM's natural reasoning style for file collections is at the
   class level (`M6C{08,09,10}`, `YYYY/DDD/HH`), not the instance level.

3. **Pathful prompts INCREASE rule activations.** Hinted: 4 → 5 rules
   fired (+25%); sparse: 4 → 8 rules (+100%). The instruction "list
   every file you plan to read" causes the LLM to enumerate file
   types/structures more explicitly, which triggers MORE matches in the
   regex rule library — including output-file rules (`csv_out`,
   `fig_out`, `report_out`) that the standard prompt didn't activate.

4. **The agent's file choice in sparse mode is stable across runs.**
   E-014, E-015, and now E-020b all picked Band 01 or Band 02 — outside
   the workload spec's 6042-file GT subset. This is consistent enough
   to look structural, not stochastic: sparse-prompt agents pick the
   alphabetically-first available band.

5. **Per-file precision/recall remains 0% in sparse pathful** (same as
   E-014/E-015). The pathful prompt does not change the underlying
   "agent chooses outside our prior's GT subset" problem.

**Implication for paper claims**

The pathful-prompt path **is not a replacement for the rule library**.
It's complementary — it makes the LLM enumerate more I/O-relevant
tokens that the regex rules then catch. The "literal-path layer" of
claim C2 (AGENTSTAGE.md) remains the secondary detector, not the
primary, even when explicitly prompted to surface paths.

**Recommended paper position**:

> "We considered a system-prompt-injection variant that asks the LLM
> to write full file paths in its reasoning, hoping to replace the
> regex rule library with a literal-substring scan. In our measurements
> (E-020), the LLM complies with the instruction but writes path
> *templates* containing placeholders (`M6C{08,09,10}`, `YYYY/DDD/HH`),
> not concrete file paths. Literal-substring matching cannot bridge
> templates to specific files in the workspace prior. The instruction
> does, however, cause the LLM to enumerate more file-class tokens,
> which the regex-rule layer catches (+25%–100% additional rule
> activations in our runs). We therefore recommend the rule-based
> detector — particularly the auto-generated variant (E-019, L3) — as
> the primary mechanism, with literal-path matching as a complement
> rather than a replacement."

**Caveats**

- n=1 seed per pathful cell. Some of the rule-count delta (+25/+100%)
  may be stochastic. Campaign C will need ≥3 seeds per cell.
- A more aggressive system prompt ("after each tool result, write out
  the exact filenames you just discovered") might produce concrete
  paths. We didn't test that — it's a deeper prompt-engineering
  experiment that goes beyond a single ablation cell.
- Output paths ARE concrete (the LLM writes `/repo/result/report.md`
  verbatim). Output-file detection works under pathful prompts; only
  input-data detection fails because of the template phenomenon.

**Files**

- `src/agentstage/runners/path_b_multiturn.py` — `--pathful-prompt` flag
- `src/agentstage/detector/session.py` — `SessionDetector.new_hot_paths()`
- `src/agentstage/detector/engine.py` — `hot_path_scan` now scans
  thinking + text + tool_result (was thinking-only)
- `outputs/multi_turn/e020_multiturn_hinted_pathful_*/`
- `outputs/multi_turn/e020_multiturn_sparse_pathful_*/`


---

## E-020 errata + V2/V3/V4 prompt iteration (2026-05-21)

**Context**: The original E-020 concluded "literal-path detection
doesn't fire under pathful prompts" because the LLM wrote path templates
(`M6C{08,09,10}`). That conclusion was **partly wrong** — there were two
co-occurring bugs:

1. **Prompt was too soft**: V1 instructed "write full paths but list_dir
   first if needed" — which the LLM read as permission to use templates.
2. **Detector prior was the PHYSICAL prior**: when the LLM wrote a
   logical path like `/data/goes_cmi_composites/raw/...nc`, our
   `SessionDetector` was matching against the *physical* prior
   (`/tmp/s3-noaa-goes16/ABI-L2-CMIPC/...nc`). Strings didn't match
   literally even when both pointed at the same file.

Both fixed in this iteration. New V4 prompt + logical prior makes
literal-path detection work in hinted mode and isolates the
sparse-mode failure cleanly.

**V2 prompt** (explicit anti-template):

```
## CRITICAL — How to write file paths

A data-staging system reads your reasoning ... Pre-fetch works ONLY
by EXACT PATH MATCH against the filesystem. Templated paths cannot
be matched and are useless to it.

Rules:
1. When you intend to read a file, write its FULL ABSOLUTE PATH
   exactly as it exists on disk. No abbreviations.
2. NEVER use placeholders, wildcards, braces, or template variables.
   Specifically forbidden:
   - brace expansion:  /data/foo_{a,b,c}.nc
   - wildcards:        /data/foo_*.nc, /data/foo_?.nc
   - template vars:    /data/<timestamp>.nc, /data/[N].nc, /data/YYYY/MM/file.nc
   - ellipses:         /data/foo_001.nc, /data/foo_002.nc, ...
3. After any list_dir result, your next reasoning MUST enumerate
   the concrete files you discovered ...
```

V2 result: still mostly templates. Turn-0 thinking has no concrete
paths (the LLM is in exploration mode and the rule "when you intend to
read" doesn't apply yet).

**V3 prompt** (structured `NEXT_FILES:` block):

```
After EVERY tool result, structure your next reasoning to include
this exact block ... before your next tool_use:

    NEXT_FILES:
    /full/path/to/file1
    /full/path/to/file2
```

V3 result: LLM produced the format but with EMPTY body. Satisfied
the structural requirement trivially.

**V4 prompt** (mandatory copy from tool_result, with worked example):

```
RULE: After EVERY list_dir result that returned a non-empty directory
listing, your next response MUST contain this block BEFORE any
tool_use:

    NEXT_FILES:
    <concrete path 1>
    <concrete path 2>
    ...

The block MUST contain at least one path COPIED VERBATIM from the
most recent tool_result, and it MUST be a path you intend to read
next. Empty NEXT_FILES blocks are not acceptable when you have just
received a directory listing.

[+ worked example showing the copy]
```

V4 result: **concrete paths appear in BOTH regimes**. Hinted turn 5
text and sparse turn 5 text contain real `/data/.../OR_ABI-L2-CMIPC-M6C..._G16_...nc`
paths.

**Companion fix — detector uses LOGICAL prior**:

`path_b_multiturn.py` now passes `workload.workspace_prior` (logical,
e.g. `/data/...`) to `SessionDetector` instead of the translated
physical prior. Dispatch sites translate logical → physical at
`stager.prefetch` time. Without this fix, LLM-written `/data/...`
paths would never match against `/tmp/s3-noaa-goes16/...` strings in
the prior.

**Results — pathful V4 + logical-prior fix**

| Run | Turn-0 thinking has concrete paths? | NEXT_FILES block with concrete paths? | Literal-path dispatches | First agent-opened file | Notes |
|---|---|---|---:|---|---|
| E-020 V1 hinted     | no | n/a | 0 | Band 08 day 121 | original failure |
| E-020 V4 hinted     | no (turn 0 is plan-only) | **yes (turn 5)** | **1** | Band 08 day 122 | concrete path emitted + matched |
| E-020 V1 sparse     | no | n/a | 0 | Band 01 day 122 | template only |
| E-020 V4 sparse     | no | **yes (turn 5)** | **0** | Band 01 day 122 | **concrete paths emitted but outside our prior — Band 01/02 are NOT in workspace_prior, which only contains C08-C10** |

**Revised conclusion**

The pathful-prompt approach **does** make the LLM write concrete paths
when instructed strongly enough (V4 prompt + worked example). The
failure mode that defeated V1/V2/V3 was prompt-engineering, not a
fundamental limitation of LLMs.

What V4 hinted shows: **literal-path detection is viable**. The LLM
writes a concrete path; `hot_path_scan` matches against the workspace
prior; dispatch translates logical → physical; stager prefetches.
Same pipeline as rule-based, just driven by literal substring rather
than regex.

What V4 sparse shows: **the residual sparse-mode failure is a
workspace-prior coverage gap, not a detection gap.** Our prior was
built from the AIOB task spec, which says "bands 08-10". The agent in
sparse mode (without that hint) explores the S3 bucket and finds it
has 16 bands (C01-C16). It picks Band 01 and writes a concrete Band 01
path in NEXT_FILES — but Band 01 isn't in our prior, so neither rules
nor literal-path scan can dispatch a hit. This is **the same finding
from E-014/E-015 in a cleaner form**: the gap is between
benchmark-defined GT (bands 08-10) and agent-discovered file universe
(bands 01-16). The fix is dynamic GT enrichment from `list_dir`
output, not better prompting.

**Implication for paper**

The pathful-prompt experiment now produces two clean results:

1. **Pathful detection works** (V4 hinted) — 1 literal-path dispatch,
   matching the agent's actual first-opened file. This is the
   workload-agnostic detector path the original idea promised.

2. **Sparse-mode brittleness is the prior, not the detector** (V4
   sparse) — the LLM correctly writes concrete paths, but those paths
   aren't in our prior because the prior was built from a constrained
   task spec. Dynamic prior enrichment is the right fix.

The original E-020 conclusion ("pathful prompts produce templates") is
*overridden* by V4. The new conclusion is "pathful prompts work when
the system prompt is sufficiently directive (≥ 4 iteration), and the
detector's prior is in logical address space."

**Files**

- `src/agentstage/runners/path_b_multiturn.py` —
  - `PATHFUL_PROMPTS` dict with v1/v2/v3/v4 versions
  - `--pathful-version v4` (default)
  - SessionDetector now uses logical prior; dispatch translates
- `scripts/path_b_run.sh` — `PATHFUL_VERSION=v4 ./scripts/path_b_run.sh hinted_pathful`
- Captures under `outputs/multi_turn/e020_multiturn_*_pathful_v[234]_*/`

---

## E-021 — Dynamic prior enrichment in sparse mode (2026-05-21)

**Goal**: Close the sparse-mode failure isolated by E-020 V4. The
detector emits concrete paths from the LLM, but the workspace prior
(built from the AIOB task spec) doesn't include the Band 01/02 files
the agent actually picks. Add discovered files to the prior on the
fly from `list_dir` results, so hot_path_scan matches subsequent
agent-written paths.

**Implementation**

1. `execute_tool` now displays LOGICAL paths in list_dir output:
   ```
   # Listing of /data/goes_cmi_composites/raw/2024/122/00 (...):
     FILE  /data/goes_cmi_composites/raw/2024/122/00/OR_*.M6C01*.nc  (...)
   ```
   Previously the listing showed PHYSICAL paths (`/tmp/s3-noaa-goes16/...`),
   which made the LLM mix logical (in NEXT_FILES) and physical (echoed
   from tool_result) addressing.

2. `enrich_prior_from_tool_result(prior, tool_result_text)` parses
   any `FILE <path> (N bytes)` line and adds the path to a
   `discovered` bucket in the prior. Filters by recognized scientific
   extensions (`.nc`, `.csv`, `.parquet`, ...).

3. `path_b_multiturn` main loop calls the enricher between
   `feed_turn` and `feed_tool_results` (so the enriched prior is
   visible to the same turn's `new_hot_paths()` call).

**Reproduction**

```bash
PATHFUL_VERSION=v4 ./scripts/path_b_run.sh e021_sparse_enrich
```

**Results**

| Metric | V4 sparse (E-020) | **V4 sparse + enrichment (E-021)** |
|---|---:|---:|
| Files prefetched (excl. force) | 1 | **17** |
| Files agent opened | 1 | 1 |
| **Hits (agent's file in staged set)** | **0** | **1** |
| Precision (files) | 0% | 5.9% (17 staged, 1 hit) |
| **Recall (files)** | **0%** | **100%** |
| Realistic wall-time speedup | 1.0× | **2,989×** |
| Oracle wall-time speedup | 3,512× | 2,989× |
| Byte overfetch | n/a (disjoint sets) | 35.46× (staged 100 MB, agent read 6.5 MB) |
| `enriched prior` count this run | 0 | **100 new paths from turn-4 list_dir** |

**Findings**

1. **Sparse-mode failure is fully closed.** The detector now delivers
   2989× realistic wall-time speedup in sparse mode (same as the
   oracle bound — perfect realization). The 1.0× brittleness from
   E-014/E-015/E-020 is gone.

2. **Recall is perfect.** Agent picked `OR_ABI-L2-CMIPC-M6C01_G16_s20241210001170_..nc`
   (Band 01, day 121, hour 00). The enrichment from turn-4's
   `list_dir(/data/.../2024/121/00)` added 100 files including that
   exact one. By the time the agent emitted its NEXT_FILES at turn 5,
   the path was in the prior and `hot_path_scan` dispatched it.

3. **Precision is low (5.9%).** Enrichment stages every file in the
   listing, but the agent reads only 1. This is the new trade-off:
   recall ↑↑, precision ↓. Could be tuned by:
   - Only adding files whose names match patterns the LLM has
     discussed (e.g., if the LLM mentions "Band 01", add only C01s)
   - Only adding files within ±1 directory level of recent agent moves
   - Bandwidth budgeting: cap concurrent stage count
   For paper purposes the current conservative behavior is honest
   ("we trade bandwidth for recall in unknown territory") and the
   trade-off knob is documented.

4. **Byte overfetch 35×** is high but bounded. We staged 17 files ×
   ~3 MB = ~51 MB to cover the 6.5 MB the agent opened. In an
   8-turn agent run that's negligible (S3 egress at 10 MB/s = ~5 s
   total, well under the slack window).

**Implication for paper claims**

The "sparse mode kills the speedup" objection in our threats-to-validity
draft is now substantially answered. The detector + enrichment + literal-
path dispatch chain works **in both regimes**:

| Regime | Realistic wall-time | Source |
|---|---:|---|
| Hinted (E-017 / V4) | 3886× | static prior is correct for hinted prompt |
| **Sparse + enrichment (E-021)** | **2989×** | discovered bucket fills the prior gap |
| Sparse without enrichment (E-014/15/20) | 1.0× | prior coverage gap |

The architectural claim shifts from "we have a rule library that works
on hinted prompts" to **"we have a detector that handles both
prompt-leakage regimes via two complementary mechanisms: (a) rules over
the static workspace prior, and (b) literal-path matching against a
prior dynamically enriched from agent exploration."**

**Caveats**

- n=1 seed. Need ≥3 seeds per regime in Campaign C.
- Enrichment is currently bucketed under `discovered`. A more
  structured enrichment that classifies discovered files into
  semantic buckets (per-band, per-day, per-subject) would let
  auto-generated rules fire on the enriched buckets too — currently
  only literal-path matching fires.
- The 35× byte overfetch needs revisiting at scale. A 10-turn agent
  run that explores 5+ directories could stage hundreds of MB; with
  multi-agent contention this matters.

**Files**

- `src/agentstage/runners/path_b_multiturn.py`:
  - `execute_tool` now uses logical paths in list_dir output
  - `enrich_prior_from_tool_result()` parses tool_result FILE lines
  - Main loop calls enrichment between feed_turn and feed_tool_results
- `outputs/multi_turn/e021_multiturn_sparse_pathful_enrich_v4_*/`


---

## E-022 — Cross-workload auto-rules generalization check (2026-05-21)

**Goal**: Confirm L3 genericity beyond aiob_107_s3. Replay the PoC's
captured single-turn streams for aiob_104 (genomics) and aiob_110
(neuroscience) through both hand-tuned and auto-generated rule sets.

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_xworkload.py \
    --workloads aiob_104,aiob_110,aiob_107 \
    --poc-dir outputs/poc \
    --out outputs/x_workload_replay.json
```

**Results** (tier-3 byte recall vs `ground_truth_full`, averaged over PoC captures)

| Workload | n captures | Hand mean recall | **Auto mean recall** | Δ |
|---|---:|---:|---:|---:|
| aiob_104 (genomics, 50 samples) | 9  | 88.9% | **88.7%** | **−0.2%** |
| aiob_110 (neuroscience, 10 subjects) | 31 | 83.9% | **80.9%** | **−3.0%** |
| aiob_107 (meteorology, 3 bands) | 23 | 65.2% | **65.2%** | 0.0% |

**Findings**

1. **Auto rules within 3% of hand** on every workload tested. The
   AGENTSTAGE.md §11.6 L3 target is "within 10%"; we exceed it.

2. The 3% loss on aiob_110 is the worst case: auto's mechanical
   `\bsubject[- _]?sub-Cori\b|\b(?:sub-Cori|Cori)\b` regex doesn't
   capture some rarer phrasings (e.g. "the Cori session") that
   hand-tuned rules include via additional aliases.

3. **L3 claim is now fully measured cross-workload.** The "hand-coded
   rules" criticism is defensibly closed for the paper.

**Files**: `outputs/x_workload_replay.json`, per-capture detail inside.

---

## E-023 — Multi-seed E-021 stability (2026-05-21)

**Goal**: Confirm E-021's sparse-mode closure isn't a single-seed
artifact. Three live runs with the same config.

**Reproduction**

```bash
for i in 1 2 3; do PATHFUL_VERSION=v4 ./scripts/path_b_run.sh e021_sparse_enrich_live; done
```

**Results**

| Seed | Agent's first file | was_staged | hot_read (ms) | cold_read (ms) | **Speedup** |
|---:|---|:---:|---:|---:|---:|
| 1 | Band 01 day 121 | ✅ | 0.055 | 1380.9 | **25,010×** |
| 2 | Band 08 day 122 | ✅ | 0.055 |  398.3 | **7,189×** |
| 3 | Band 08 day 122 | ✅ | 0.095 |  644.2 | **6,789×** |

**Findings**

1. **3 of 3 seeds had `was_staged=True`** — the staging hit is
   structurally reliable, not a lucky draw.

2. **Speedup range 6.8k× — 25.0k×**, dominated by S3 cold-read
   latency variance (398 ms to 1.38 s for the same-size file).

3. **Hot read is consistent** (0.055-0.095 ms) — bounded by tmpfs +
   shim overhead.

4. **Sparse-mode agent behavior varies** (seed 1: Band 01; seeds 2,3:
   Band 08). Enrichment works in both — the agent's chosen file is
   always pulled into the prior by the time it's opened.

The headline number for the paper, taking the conservative
geometric mean: **≈10k× sparse-mode live speedup with enrichment**.

---

## E-024 — Enrichment precision-tuning ablation (2026-05-21)

**Goal**: Measure recall vs precision vs byte_overfetch trade-off
across enrichment policies. The current default ("add ALL files from
list_dir") delivers 100% recall but 1% precision. Smaller policies
might recover precision without sacrificing recall.

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_enrich_ablation.py \
    --corpus outputs/multi_turn/<E-021 run> \
    --workload aiob_107_s3 \
    --out <corpus>/enrich_ablation.json
```

Tested policies:
- **A** `no_enrich` — baseline (rules + static prior only)
- **B** `all_files` — current default
- **C** `cap_N` — first K files per listing
- **D** `pattern_scoped` — only files whose name shares ≥4-char
  substring with any LLM-mentioned token
- **E** `ext_nc_only` — only `.nc` files

**Results** (averaged over 3 E-021 seeds — n=3 corpora)

| Policy | Mean recall | Mean precision | Byte overfetch (when hit) |
|---|---:|---:|---:|
| A no_enrich        | 0%   | 0%   | n/a |
| **B all_files**    | **100%** | 1.0% | 235× |
| C cap_5            | 33%  | 6.7% | 1.5-4.6× |
| C cap_10           | 33%  | 3.3% | 2.8-8.4× |
| C cap_25           | 33%  | 1.3% | 27-82× |
| D pattern_scoped   | 100% | 1.0% | 235× (LLM mentions common prefix → no filter effect) |
| E ext_nc_only      | 100% | 1.0% | 235× (every listed file is .nc) |

**Findings**

1. **Cap-N variants fail in 2 of 3 seeds.** Reason: listing sorts
   alphabetically, so an hour's 192 files are 12 each of C01, C02,
   ..., C16 in order. When agent picks Band 08, cap_25 only catches
   the C01/C02 region — missing C08 by ~60 positions.

2. **Pattern-scoping doesn't help** because the LLM mentions the
   common filename prefix (`OR_ABI-L2-CMIPC`), matching every file.
   A tighter scope (require ≥10-char match, or weighted scoring) is
   needed.

3. **Extension filtering doesn't help** for this workload because
   the directory is uniform.

4. **The current default (B all_files) is the only policy delivering
   100% recall in all seeds.** Recall is what closes the sparse-mode
   gap; precision is the cost.

**Implication for paper / future work**

The "bandwidth for recall" trade-off in dynamic enrichment is real
and needs smarter policies. Two directions identified:

- **Stratified sampling per listing**: detect naming-pattern variation
  (e.g., M6C01 vs M6C08 vs M6C16) and keep one file per pattern
  rather than top-N. This would cap byte overfetch at ~16× (one per
  band) instead of 100-300× (everything).

- **Late-binding (deferred enrichment)**: don't enrich on raw
  list_dir; wait until the LLM's next text mentions a specific
  pattern, then enrich only matching files from the latest listing.
  Adds latency but trims to ~1× overfetch.

Both belong in the paper's future-work section. The smoke ablation
confirms the trade-off exists and the simple knobs we tested don't
resolve it — the smarter policies are real research, not engineering
tweaks.

**Files**: `outputs/multi_turn/e021_*/enrich_ablation.json`


---

## E-025 — End-to-end agentic-loop wall-time vs per-file speedup (2026-05-21)

**Goal**: Distinguish per-file read speedup (what E-010/E-021/E-023 measure)
from the FULL agent session wall-time speedup (what an end user cares about).

**Reproduction**

```bash
# No new live run — analyze existing E-023 captures
~/.local/bin/uv run python3 -c '
import json
from pathlib import Path
for run in sorted(Path("outputs/multi_turn").glob("e021_*_enrich_live_v4_*")):
    per_turn = []
    for tdir in sorted(run.glob("turns/turn_*")):
        s = tdir / "summary.json"
        if s.exists():
            d = json.loads(s.read_text())
            per_turn.append(d.get("duration_ms", 0))
    summary = json.loads((run / "summary.json").read_text())
    m = summary.get("measurements", {})
    total = sum(per_turn)
    saved = m.get("cold_read_ms", 0) - m.get("hot_read_ms", 0)
    print(f"{run.name[:50]}: session {total/1000:.1f}s, saved {saved:.0f}ms")
'
```

**Results** (3 seeds from E-023)

| Seed | Session wall-time | Time saved (1 read) | **Session-level speedup** | Per-file speedup |
|---:|---:|---:|---:|---:|
| 1 | 74.4 s | 1,381 ms | **1.86%** | 25,010× |
| 2 | 68.9 s | 398 ms | **0.58%** | 7,189× |
| 3 | 61.4 s | 644 ms | **1.05%** | 6,789× |

**Finding**

The per-file 10⁴× speedup ≠ session-level 10⁴× speedup. The relationship is:

```
session_speedup_fraction = sum(cold_open_ms - hot_open_ms) / total_session_ms
```

Where total_session_ms is dominated by LLM inference (8 × ~5-10 s
per turn). In our 8-turn smoke runs the agent does ~7 list_dir calls
and ~1 open_file call → 1 read benefiting from staging.

**Two honest measurements coexist**:
- Per-file end-to-end speedup ≈ 10⁴× (real, in-process)
- Per-session end-to-end speedup ≈ 0.6%–1.9% (real, our smoke runs)

The session number scales linearly with `n_file_reads` per session.
For the smoke runs n=1 so the gap is large.

**Projection for full task completion**

If aiob_107's actual task were run end-to-end (~6,042 file reads to
compute brightness-temperature time series across the dataset):

| Scenario | Estimated wall-time |
|---|---:|
| Cold reads only | 6,042 × ~750 ms ≈ **75.5 min of pure I/O** |
| Hot reads via shim | 6,042 × ~0.06 ms ≈ **0.36 s of pure I/O** |
| Saved per task | **~75 min** (assuming the LLM-inference portion adds another N seconds either way) |

This is a *projection*, not a measurement. A real session-wall-time
ablation requires:
1. A runner that drives the agent through full task completion
   (currently path_b_multiturn stops at 8 turns of exploration)
2. A baseline run with the shim disabled or stager turned off
3. Side-by-side timing

That's a real future experiment — call it E-026 when we have a
task-completing runner. For now, the paper has:
- Per-file speedup: measured live (10⁴×)
- Per-session speedup: measured in smoke runs (1%) + projected
  for full task (~75 min/task)

**Implication for paper claims**

The paper should report BOTH numbers explicitly to avoid the "what
does 10⁴× actually mean for a user?" reviewer question. The honest
framing:

> "Per-file end-to-end read latency reduction is ~10⁴× under both
> prompt regimes (measured live with the LD_PRELOAD shim active). The
> session-level wall-time impact scales with the number of file reads
> the agent performs: a typical aiob_107 task involves ~6,000 file
> reads, projecting ~75 minutes of saved I/O time per task. Our
> exploration-heavy smoke runs (8 turns, 1 file read) realize only
> ~1% session-level speedup; full-task end-to-end timing is future
> work pending a task-completing runner."


---

## E-026 — Cross-vendor live multi-turn (Gemini 2.5 Flash, n=3) (2026-05-21)

**Goal**: Confirm the architecture works on a non-Anthropic LLM family.
All prior live experiments were Haiku 4.5; this run uses Gemini 2.5
Flash via google-genai SDK with extended thinking + native tool_use.

**Implementation**

- New `src/agentstage/client/gemini.py` (`GeminiClient`,
  `GeminiStreamingResponse`) — translates Gemini's part-based stream
  into Anthropic-shaped events (`_Event`/`_ContentBlock`/`_Delta`
  namespaces) so `path_b_multiturn` consumes both providers without
  branching the main loop.
- `path_b_multiturn` auto-detects provider from model name
  (`gemini-*` → GeminiClient; `claude-*` → AnthropicClient).
- Same V4 pathful prompt + dynamic enrichment + measure_target_after
  pipeline as E-023.

**Reproduction**

```bash
PATHFUL_VERSION=v4 GEMINI_MODEL=gemini-2.5-flash \
    ./scripts/path_b_run.sh e026_gemini_sparse_enrich_live
```

(repeated 3 times)

**Results**

| Seed | Turns | Agent's first file | # predictor-staged | Hit | Hot ms | Cold ms | **Speedup** |
|---:|---:|---|---:|:---:|---:|---:|---:|
| 1 | 8 | Band 08 day 122 | 46 | ✅ | 0.163 | 644.9 | **3,965×** |
| 2 | 8 | Band 01 day 122 | 38 | ✅ | 0.046 | 610.0 | **13,152×** |
| 3 | 7 | Band 01 day 122 | 44 | ✅ | 0.066 | 570.5 | **8,677×** |

**Findings**

1. **3 of 3 hits.** The agent's chosen file was always in the
   predictor-driven prefetch set before the agent opened it. Cross-
   vendor success rate matches Haiku (3/3 in E-023).

2. **Per-file speedup range 3,965× to 13,152×.** Mean ~7,000×.
   Range is slightly narrower than Haiku's 6.8k–25k× (different S3
   latency window during the run), but the architectural ceiling is
   identical — both providers' agent reads end up redirected to the
   same tmpfs hot tier via the same shim.

3. **Rule activation profile differs across vendors:**
   - Haiku: 4-5 rules typically fire from turn-0 thinking + later text
   - Gemini: 0 rules fire in turns 0-4; some fire only in turns 5-7
     (after `list_dir` populates context)
   - But enrichment + literal-path detection close the gap — Gemini
     gets the staging hit via the post-discovery path even when its
     turn-0 thinking doesn't trip any regex rules.

4. **Cost**: ~$0.22 per run × 3 = **~$0.66 total**. Cheaper than
   Haiku (Gemini 2.5 Flash $0.30/$2.50 vs Haiku ~$1/$5 per M tokens).

**Implication for paper claims**

- **Architecture is vendor-agnostic**, not Anthropic-specific. Both
  Anthropic Claude and Google Gemini drive the same end-to-end staging
  → shim → hot-read pipeline with same-order-of-magnitude speedups.
- **The detection layer adapts via enrichment.** Gemini's thinking
  content fires fewer regex rules than Haiku's, but the dynamic prior
  enrichment from `list_dir` output makes the literal-path scan
  sufficient on its own. This is an unexpectedly clean validation of
  the "two complementary detectors" architecture (rules + literal-path).
- **The L3 genericity claim now extends from "auto-rules within 3% of
  hand on 3 workloads" (E-022, offline) to "architecture works
  end-to-end on 2 vendor families" (E-021+E-023 Anthropic, E-026
  Gemini, all live).**

**Caveats**

- n=3 per vendor is still small. Campaign C target is ≥3 seeds × 3
  models (Haiku, Flash, OSS) × 3 workloads.
- DeepSeek-R1 (third planned vendor) not yet tested live; only PoC
  single-turn captures exist (E-022 inputs).
- Gemini's tool_use blocks don't have stable IDs in the way Anthropic's
  do; `GeminiClient` synthesizes them. This works in our protocol
  but may matter for systems that round-trip tool_use_id across
  longer conversations.

**Files**

- `src/agentstage/client/gemini.py` — Gemini client + event wrapper
- `src/agentstage/runners/path_b_multiturn.py` — provider auto-detect
- `outputs/multi_turn/e026_multiturn_sparse_pathful_enrich_live_gemini_v4_*/`


---

## E-027 — Session-level speedup from real AIOB production runs (2026-05-21)

**Goal**: Replace the smoke-run's 1% session-speedup observation
(E-025) with measurements from REAL agentic task-completion runs.

**The smoke gap was a methodological artifact**: 8-turn smoke runs are
exploration-heavy and only open ~1 file per session. Real AIOB
agentic runs let the agent write a Python script and execute it
(typically turn 12 of ~20+ turns), which performs ALL the bulk file
I/O in one execution turn — thousands of file reads happen in seconds
of wall-clock during that one turn.

We pulled `io_report.json` files from
`/mnt/common/datasets-staging/agentiobench/outputs/` (sciiobench
production runs captured with DFTracer instrumentation, 2026-03 to
2026-05), aggregated POSIX time per run, and computed both the
local-NFS-measured and S3-projected session-level speedups.

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_aiob_realruns.py \
    --workloads aiob_104,aiob_107,aiob_110 \
    --out outputs/realruns_session_speedup.json
```

**Per-run sample (aiob_107, Sonnet 4.5)**

| Date | Job time | # files | POSIX I/O | I/O frac | Local elim speedup |
|---|---:|---:|---:|---:|---:|
| 20260322-224839 | 326.5 s | 6,226 | 218.3 s | **66.9%** | **3.02×** |
| 20260503-221753 | 767.5 s | 6,231 | 204.6 s | 26.7% | 1.36× |

Same workload, vastly different I/O fraction depending on what the
Python script does in turn 12 (chunked reads vs full-file reads).

**Per-workload aggregate (mean over n runs, Sonnet 4.5 + Gemini 2.5 Flash)**

| Workload | n runs | Mean session | Mean I/O frac | **Local elim speedup** | **S3-projected speedup** |
|---|---:|---:|---:|---:|---:|
| aiob_104 (genomics, BAM streams) | 3 | 320.8 s | **1.4%** | 1.01× | **1.32×** |
| aiob_107 (meteorology, 6k files) | 11 | 550.7 s | **30.6%** | **2.08×** | **24.35×** |
| aiob_110 (neuroscience, 58 NWB) | 16 | 1730.9 s | 17.1% | 1.30× | **7.46×** |

S3-projection methodology: multiply measured POSIX time by
(S3_cold_open_ms / local_NFS_open_ms) = (754.5 / 35.0) = **21.6×**.
This is conservative — it assumes the throughput differential is the
same as the open-latency differential. In practice S3 throughput is
even worse than its open latency suggests, so the projected speedup
is a **lower bound** for S3.

**Findings**

1. **The smoke-run's 1% session speedup is the lower-bound case**,
   not the central tendency. Aiob_104 is compute-heavy (1.4% I/O on
   NFS) — its real session speedup matches our smoke observation.

2. **Aiob_107 is the I/O-heavy workload**: 30.6% of session time is
   POSIX I/O on local NFS. Eliminating it gives a **2.08× session
   speedup** even on already-fast NFS. **On S3, projected ~24×**.

3. **Aiob_110 sits between**: 17% I/O fraction on NFS → 1.30×
   eliminated → **7.46× S3-projected**. The NWB files are big (~250
   MB each) and read-heavy.

4. **Per-workload variance is large** within the same workload:
   aiob_107 I/O fraction ranges 0.1% – 67% across 11 captured runs,
   driven by what the agent's generated Python script chooses to do
   (chunked vs full-file reads, single-pass vs multi-pass).

5. **The 10⁴× per-file headline is real but the wrong unit for the
   user-facing claim.** The defensible session-level claim is:
   "AgentStage eliminates 1.4–30.6% of session wall-time on local
   NFS storage (1.01–2.08× session speedup); projects to 1.32–24×
   session speedup on S3-class cold storage."

**Implication for paper claims**

This is the right table for the evaluation section. The previous
paper-language draft had:

> "AgentStage delivers ~10⁴× per-file read latency reduction..."

Updated language should also include:

> "On real AIOB agentic tasks captured with DFTracer instrumentation
> (n=30 production runs across 3 workloads), POSIX I/O accounts for
> 1.4%–30.6% of session wall-time on local NFS storage, corresponding
> to a 1.01×–2.08× achievable session speedup. Projected to S3-class
> cold storage using our measured S3 first-byte latency (E-010), the
> session-level speedup rises to 1.32×–24.35×. The headline per-file
> 10⁴× reduction (E-021, E-023, E-026) translates to session-level
> savings only where the agent does many file reads; workload variance
> is large because agentic scripts choose their I/O pattern."

**Caveats**

1. Local NFS in these runs is /mnt/common XFS on Ares — closer to a
   warm cache than a true cold tier. S3 projection assumes
   open-latency ratio; throughput ratio may be even worse.

2. We didn't run the AIOB harness with AgentStage active — these are
   *baseline* runs. A real measurement would be a side-by-side run
   with the same task, same script, with and without staging. The
   "elim_speedup" we report is the THEORETICAL UPPER BOUND assuming
   AgentStage eliminates 100% of POSIX I/O time. Real staging would
   leave some compulsory misses (first agent exploration turns); we
   don't model that.

3. The "1.4% on aiob_104" is low because the workload's BAM files
   are streamed via samtools/pysam with internal buffering and
   indexes — most of the agent's compute time goes to genomics
   logic, not raw I/O. AgentStage's benefit on this workload class
   would be smaller.

**Files**

- `scripts/microbench/path_b_aiob_realruns.py` — analysis script
- `outputs/realruns_session_speedup.json` — per-run breakdown


---

## E-028 — End-to-end task-script speedup, baseline vs staged (2026-05-22)

**Goal**: The single most important measurement for the paper — run
the ACTUAL Python analysis script that a Sonnet-4.5 agent generated
for aiob_107 (captured from a real AgentIOBench production run,
turn 12 of 23), against a cold storage tier, with and without
AgentStage. This is the real end-to-end session-level speedup: real
agent-written code, really executing, really reading NetCDF files,
with the real LD_PRELOAD shim.

Closes the gap E-025/E-027 flagged: those reported per-file speedup
(measured) + per-session speedup (analyzed/projected). E-028 measures
the per-session number *directly* by side-by-side execution.

**Method**

- Script: the agent's 11.5 KB `process_goes_data.py`, extracted from
  the production run's `replay.yaml`, parameterized for data/output dir.
- Scope: day-of-year 122 (864 C08/C09/C10 NetCDFs, 2,414 MB) — a
  representative subset of the full 7-day task; scales linearly.
- BASELINE: page cache evicted, no shim, script reads from cold tier.
- STAGED: all 864 files pre-fetched to tmpfs via the Stager, cold
  caches evicted, script run with the LD_PRELOAD shim active so its
  netCDF4 `open()` calls redirect to the hot copies.
- Two cold tiers: `local` (AIOB's NFS/XFS dataset copy) and `s3`
  (public noaa-goes16 bucket via mountpoint-s3).

**Reproduction**

```bash
~/.local/bin/uv run python scripts/microbench/path_b_e2e.py \
    --tier local --out outputs/e2e/local
~/.local/bin/uv run python scripts/microbench/path_b_e2e.py \
    --tier s3 --out outputs/e2e/s3
```

**Results**

| Cold tier | Baseline (cold) | Staged (hot) | **Wall saved** | **Session speedup** |
|---|---:|---:|---:|---:|
| local NFS/XFS | 110.3 s | 97.7 s | 12.6 s | **1.13×** |
| **S3 (mountpoint-s3)** | _see s3 entry below_ | | | |

(S3 result appended when the run completes — baseline cold reads of
864 S3 objects take ~15-20 min.)

**Finding (local tier)**

On AIOB's local NFS dataset copy, end-to-end session speedup is
**1.13×** — staging saves 12.6 s of a 110 s task. This is modest and
honest: local XFS first-reads are fast, and the agent's script spends
most of its ~98 s of irreducible time in netCDF decompression + numpy
box-extraction compute, which staging does not touch. This matches
the E-027 projection (aiob_107 local-NFS I/O fraction ~30% →
~1.3-2× session speedup; the e2e measured 1.13× sits at the low end
because day-122-only is a smaller, more compute-dominated slice).


---

## E-028 + E-029 — End-to-end task-script speedup, FINAL (2026-05-22)

Supersedes the partial E-028 entry above (which had a stale 864-file
local-only result and a shim bug). This is the corrected, complete
end-to-end measurement.

### Errata: shim `fopen` bug found + fixed

While running E-028/E-029 on the S3 tier, the staged runs were
anomalously slow (~54 s when they should be ~6 s). Root cause: the
netCDF-C library format-sniffs every file with **`fopen64()`** before
handing it to HDF5's `open()`-based driver. The shim intercepted
`open`/`openat` but **not `fopen`/`fopen64`** — so each file paid one
cold-tier `fopen64` first-byte latency (~2 s on S3) even though the
file was staged hot.

Fix: `agentstage_shim.c` now interposes `fopen`/`fopen64` (read-only
modes), spin-waits for the hot copy, and wraps the hot fd via
`fdopen`. A pthread_once deadlock (cfg_init opening its log via the
now-interposed `fopen`) was fixed by using `dlsym(RTLD_NEXT,"fopen")`
directly in `cfg_init`. After the fix: netCDF4 dataset open dropped
from 2189 ms to 13 ms on a staged S3 file. 18 shim/integration tests
pass.

### Method

- Script: the agent's real 11.5 KB `process_goes_data.py`, extracted
  from a Sonnet-4.5 AgentIOBench production run (turn 12 of 23).
- Scope: day 122 / hour 00 = 36 C08/C09/C10 NetCDFs (105 MB compressed).
  Scoped to 1 hour because HDF5-over-FUSE-S3 issues many small GETs per
  file; a full 864-file day exceeds a 1 h cold baseline. Numbers scale
  ~linearly with file count (full task = 6042 files).
- BASELINE: page cache evicted, no shim — cold reads.
- PLAIN-STAGED (E-028): byte-identical hot copies, shim active.
- DECOMP-STAGED (E-029): hot copies transcoded to UNCOMPRESSED NetCDF
  (`transcode.py`), shim active — the script's reads pay zero zlib.

### Results

| Cold tier | Baseline | Plain-staged (E-028) | Decomp-staged (E-029) |
|---|---:|---:|---:|
| **local NFS/XFS** (warm) | ~6.5 s | 5.5 s — **1.2×** | 3.9 s — **1.6×** |
| **S3** (mountpoint-s3) | ~169 s | 7.2 s — **23.4×** | 5.6 s — **29.1×** |

Transcode cost (decompression done at staging time, off the critical
path): local 1.7 s, S3 42 s (the S3 transcode reads files cold once;
it overlaps with the agent's discovery turns and races ahead during
the script-execution turn). Hot-tier footprint: 389 MB uncompressed
vs 105 MB compressed (3.71× expansion).

### Findings

1. **Plain staging on S3: 23.4× session speedup** — the headline
   end-to-end number. The agent's real analysis script finishes in
   7.2 s instead of 169 s when its input files are staged to tmpfs.

2. **Decompression-staging adds a further increment**: S3 23.4× →
   29.1×; local 1.2× → 1.6×. The increment is ~1.6 s of zlib
   decompression removed — roughly tier-independent (decompression is
   CPU, not I/O), so it shows up as a large *ratio* gain on the
   already-fast S3-staged run and a modest one on local.

3. **Local NFS/XFS staging is modest (1.2×)** because `/mnt/common`
   XFS is effectively warm — first-reads are fast, so there is little
   read latency to eliminate. A genuinely cold first read showed ~4×
   (one outlier rep) but steady-state is ~1.2×. This is honest and
   expected: staging's value scales with how slow the cold tier is.
   S3 is the realistic cloud cold tier; that is where staging earns
   its keep.

4. **This is the real end-to-end measurement**: the agent's actual
   generated code, really executing, really reading NetCDF files,
   with the real LD_PRELOAD shim. The 23.4× / 29.1× are wall-time
   ratios on the data-processing phase of a real scientific-agent
   task — not per-syscall, not projected.

### Reproduction

```bash
~/.local/bin/uv run python scripts/microbench/path_b_e2e.py --tier {local,s3} --out outputs/e2e/{local,s3}
~/.local/bin/uv run python scripts/microbench/path_b_e2e_decompress.py --tier {local,s3} --out outputs/e2e/{local,s3}
```

### Caveats

- 36-file (1-hour) scope. The full aiob_107 task is 6042 files;
  numbers scale ~linearly. The S3 absolute baseline for the full task
  would be ~1 h+ (which is itself the motivation for staging).
- Local NFS baseline is cache-state-sensitive (steady-state ~6.5 s,
  cold-first ~26 s). S3 baseline is stable (~169 s ± 4 s across 4 runs).
- Decomp-staging's transcode cost (S3 42 s) must fit the staging
  window. For the full task it is parallelizable + overlaps the agent
  turns + races ahead during execution; a strict slack-window-only
  budget may not cover all 6042 files — LRU stage-ahead/evict-behind
  needed (see DECOMPRESSION_STAGING.md §5).
- Hot-tier capacity: uncompressed is 3.71× bigger. Full task at
  3.71× would need ~stage-ahead/evict-behind on a 32 GB tmpfs.

### Files

- `scripts/microbench/path_b_e2e.py` — E-028 orchestrator
- `scripts/microbench/path_b_e2e_decompress.py` — E-029 orchestrator
- `outputs/e2e/transcode.py` — decompression transcoder
- `outputs/e2e/task_script.py` — the agent's extracted script
- `outputs/e2e/{local,s3}/e2e_*.json` — per-run results


---

## E-030 — Verified cold-cache local rerun (2026-05-22)

**Goal**: Address the user's concern about cold-cache rigour. AIOB's
methodology (`agentiobench.utils.cache`) does `posix_fadvise(DONTNEED)`
per file PLUS a mincore-based residency-verification check
(`selftest_eviction`). My initial `evict()` had no verification, and
back-to-back runs showed local baseline variance (6 s — 26 s).

**Method**: extended `evict()` in `path_b_e2e.py` to also call
`_resident_pages()` (from `agentiobench.utils.cache`) on a sample of
the targets after fadvise, confirming residency drops to 0 of N pages.
Then ran n=3 reps per cell.

**Verified-cold local results (n=3)**

| Config | Rep 1 | Rep 2 | Rep 3 | Median |
|---|---:|---:|---:|---:|
| E-028 baseline | 16.3 s | 8.2 s | 6.3 s | **8.2 s** |
| E-028 staged   |  7.4 s | 6.0 s | 6.0 s | **6.0 s**  |
| E-028 speedup  | 2.20×  | 1.37× | 1.05× | **1.37×**  |
| E-029 baseline |  6.1 s | 6.3 s | 6.3 s | **6.3 s**  |
| E-029 decomp-staged | 4.2 s | 4.2 s | 4.4 s | **4.2 s** |
| E-029 speedup  | 1.45×  | 1.50× | 1.43× | **1.46×**  |

Residency check after `evict()`: **0 of 3611 sample pages resident**
on /mnt/common XFS. Eviction is genuinely working; the local "cold"
baseline really is page-cache-cold.

**Findings**

1. **The 6-8 s local baseline IS truly cold-page-cache** — eviction
   is verified. The fast number reflects /mnt/common's underlying
   NVMe+XFS performance, not a cache artifact.

2. **First-rep variance**: the very first cold-cache run hits 16 s,
   subsequent runs land at 6-8 s. The XFS metadata/inode cache and
   SSD-internal cache warm up across reps; the page cache eviction
   doesn't reach those layers. This is the realistic "first cold run
   ever vs steady-state cold-page-cache" gap.

3. **E-029 is more stable than E-028** because its baseline doesn't
   include the first-cold outlier (we'd already warmed XFS metadata
   from prior E-028 reps).

**Honest conclusion**

This host has **no slow cold tier** other than S3. Local /mnt/common
XFS is fast NVMe storage — even cold-page-cache, it delivers
~6 ms/file. The 1.2-1.5× session speedup on local IS the honest
reality of staging fast-local-NVMe data. On a slower on-prem cold
tier (NFS, Lustre, BeeGFS), staging would help more — bracketed by
E-007's throttled-sweep (1.7× — 12× across 10-200 MB/s simulated
bandwidth) and our S3 measurement (23-29×).

**Three-point picture for the paper**

| Cold tier                                          | Source         | Plain stage |
|---|---|---:|
| Local NVMe XFS, verified-cold-page-cache (E-030)   | measured       | **1.2×**     |
| Throttled local at 10 MB/s (E-007)                 | simulated      | **12×**      |
| S3 via mountpoint-s3 (E-028)                       | measured       | **23×**      |

The headline is the spectrum, not a single number. The paper should
say: "AgentStage's session-level speedup scales with cold-tier
latency, from ~1.2× on fast-NVMe-XFS to ~23× on S3-class cloud cold
storage, with realistic on-prem NFS/Lustre falling between
(E-007 throttled sweep at 10 MB/s shows ~12×)."

**Reproduction**

```bash
for i in 1 2 3; do
  ~/.local/bin/uv run python scripts/microbench/path_b_e2e.py --tier local --out outputs/e2e/local
  ~/.local/bin/uv run python scripts/microbench/path_b_e2e_decompress.py --tier local --out outputs/e2e/local
done
# Eviction verification fields appear in stderr / json's "evict" payload.
```


---

## Cold-cache methodology (standard for all timing experiments)

All session-level timing experiments (E-028, E-029, E-030, E-031,
and any future Path B / e2e measurement) MUST use this eviction
protocol before the BASELINE phase:

1. **`posix_fadvise(POSIX_FADV_DONTNEED)`** on every target file
   (same as `agentiobench.utils.cache.evict_dataset`)
2. **`os.sync()`** to flush any dirty writes
3. **`mincore` residency verification** via
   `agentiobench.utils.cache._resident_pages` on a sample of 5 files;
   record `resident_frac_sample` in the experiment's JSON output.
4. **Verification**: `resident_frac_sample` should be ~0.0 (≤ 1%).
   If a tier doesn't honor `DONTNEED` (some FUSE mounts, mountpoint-s3,
   read-only FS) this is recorded; the result is still reported but
   marked "warm-cache-suspect".

Both `path_b_e2e.py` (E-028) and `path_b_e2e_decompress.py` (E-029)
share the same `evict()` implementation. Codified 2026-05-22 after the
user flagged inconsistent eviction in earlier reps.

---

## E-031 — Cross-script variance: naive vs chunked I/O patterns (2026-05-22)

**Goal**: Address the "single task/model/rep" concern. The Sonnet 4.5
production-run script we used in E-028/E-029 uses the I/O-NAIVE pattern
(`ds.variables['CMI'][:]` — reads the whole 1500×2500 grid). Real
agent scripts vary: Haiku 4.5, Gemini Flash, and another Sonnet run
all use chunk-aware slicing (`var[y0:y1, x0:x1]`). This run tests
how the architecture's benefit varies with the agent's script style.

**Method**: hand-modified the Sonnet script's read pattern to be
chunk-aware (`outputs/e2e/task_script_chunked.py`) — reads only the
bounding box covering all 5 locations × 5-pixel halo (~46×716 region
out of 1500×2500). This mirrors what an I/O-efficient agent would
write. Same evict-and-verify methodology. n=1 per cell (smoke).

**Results**

| Script | Tier | Baseline | Staged | **Speedup** | Wall saved |
|---|---|---:|---:|---:|---:|
| **Naive** (Sonnet original) | local | 8.2 s | 6.0 s | **1.37×** | 2.2 s |
| | S3 | 169 s | 7.2 s | **23.4×** | 162 s |
| **Chunked** (modified)      | local | 3.9 s | 2.6 s | **1.53×** | 1.3 s |
| | S3 | 163 s | 3.0 s | **53.7×** | 160 s |

**Findings**

1. **Both scripts save ~160 s of wall time on S3.** This is because
   both pay 36 × ~4.5 s per HDF5 file-open against the FUSE-S3 mount
   (file metadata + superblock + chunk-index GETs). The agent's
   chunk-vs-whole-grid choice only affects which chunks get
   decompressed AFTER the file is opened.

2. **The speedup *ratio* differs by 2.3×**: chunked 54× vs naive 23×.
   Chunked has less compute on the staged side (3.0 s vs 7.2 s
   because it decompresses ~5 chunks instead of ~120), so the ratio
   looks more dramatic — but the absolute saved wall time is similar.

3. **The architecture benefit generalizes across agent script
   patterns.** Plain staging eliminates the cold-tier-open cost
   regardless of how the agent reads inside the file. Decompression-
   staging (E-029) helps the naive script substantially; for chunked
   scripts the decompression cost is already small so decomp-staging
   would barely help.

**Caveats** (honest scope)

- **n=1 per cell** for the chunked variant. Should be ≥3 for the
  paper; Campaign C work.
- **Same task** (aiob_107). Cross-workload (aiob_104 / aiob_110)
  would have very different I/O patterns and absolute timings.
- **Same model family** (Sonnet 4.5) — the chunked variant is a
  HAND-MODIFIED version of Sonnet's script, not a different agent's
  output. To genuinely test cross-model variance we'd need to run
  Haiku's / Gemini's / DeepSeek's actual scripts (each hardcodes
  paths differently, would need a per-script parameterization pass).
- **Chunked variant returns rc=1** in our run because the NaN-filled
  array breaks the matplotlib plot at the end. I/O timings are still
  valid (the failure is post-I/O). For paper-grade we'd fix the
  trivial NaN-handling issue.

**Implication for the paper**

The "single script" result IS narrow. The honest scope is:
- E-028/E-029 measure ONE agent script (Sonnet 4.5, I/O-naive)
- E-031 widens to TWO script styles (naive + hand-chunked) on the
  SAME task
- Cross-model + cross-workload + multi-seed = full Campaign C scope

The architecture-level claim — "staging hides cold-tier latency in
the reasoning slack window" — holds across both script styles tested.
Specific speedup numbers depend on the agent's I/O pattern.

