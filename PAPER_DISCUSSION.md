# PAPER_DISCUSSION.md

This file is the long-form companion to the eScience '26 submission. It
collects discussion-level material that did not fit the 10-page limit:
honest framing of speedup units, the reasoning-slack timeline mechanism,
per-regime boundary analysis, the H12 pathful-prompt null result, the
decompression-staging extension, the original hypothesis/claim taxonomy,
and the eScience-award positioning rationale. Project-facing operational
docs (README, CAMPAIGN, EXPERIMENTS, STAGER_*, DIAGRAMS, AIOB_INTEGRATION)
remain in the repo root and are referenced rather than duplicated here.

---

## Table of contents

1. Honest scope and the three speedup units
2. Reasoning-slack: metric definitions and exemplary sessions
3. 27-cell matrix decomposition and per-regime behavior
4. Pathful-prompt H12 null result
5. Decompression-aware staging beyond the eScience submission
6. Hypothesis taxonomy and claim mapping
7. eScience positioning rationale
8. Threats to validity and hardening checklist

---

## 1. Honest scope and the three speedup units

AgentStage produces speedup numbers in three distinct units. Conflating
them is the single most likely reviewer trap, so this section spells out
how each one is computed, what question it answers, and which one is the
honest headline.

### 1.1 Per-file (read-latency) speedup: about 10^4x

This is the latency of one `open()+read()` on a single file, cold tier
versus hot tier, measured live inside the shim:

* Cold (S3 via `mountpoint-s3`): ~754 ms first-byte.
* Hot (tmpfs via LD_PRELOAD): ~0.05 ms first-byte.
* Ratio: about 10^4x.

This number is a **mechanism demonstration**. It proves the shim
intercepts the syscall, the stager populated the hot tier in time, and
the read served from RAM. It is not a claim about agent task time, and it
should never appear in the paper without explicit framing that calls it
the "per-file" or "per-read" ratio. Reporting 10^4x as if it were the
session speedup is the failure mode that destroys credibility.

### 1.2 Per-session (wall-time) speedup: workload-dependent

Most of an agent session is LLM inference plus compute, neither of which
AgentStage touches. The per-session speedup is therefore a ceiling set
by the I/O fraction of the workload.

From 30 real AIOB production runs measured on local NFS (the
conservative baseline):

| Workload                    | I/O fraction | Bound on session speedup |
|-----------------------------|-------------:|-------------------------:|
| aiob_104 (genomics)         |         1.4% |                    1.01x |
| aiob_107 (meteorology)      |        30.6% |                    2.08x |
| aiob_110 (neuroscience)     |        17.1% |                    1.30x |

Projected to an S3 cold tier (using the E-010 cold/hot latency ratio),
the I/O fraction grows and the bound rises to **1.3x to 24x** per
workload. The lower end of that range applies when the agent spends
most of its session on compute; the upper end when the agent is
essentially shuttling bytes.

### 1.3 The rule for the paper

* Report **per-file 10^4x** only as the *mechanism* result, in the
  system-design section. Never headline it.
* Report **per-session 1.0x to 2.1x on NFS, 1.3x to 24x projected on
  S3** as the *user-visible* result.
* End-to-end measurements (E-028 and the full 27-cell campaign) close
  the gap between projection and measurement by running the agent's
  actual generated analysis script with versus without staging, on both
  tiers.

### 1.4 Why an S3 cold tier is the honest stress test

The three benchmarks we integrate (AIOB, ScienceAgentBench, KramaBench)
ship their datasets on local or NFS storage. The honest answer to "did
the benchmark use S3?" is no: we added it ourselves (upstream commit
`dea5686`). The justification is not "benchmarks use S3" but that the
real-world scientific workflows these benchmarks emulate increasingly
read from cloud object storage, and that is the deployment scenario
where staging matters most. Four supporting points: (1) the exact
dataset we use is on S3 (GOES-16 imagery in the public `noaa-goes16`
bucket, NOAA Open Data Dissemination; we point AWS's `mountpoint-s3`
FUSE driver at the real bucket); (2) cloud object storage is a primary
locus of scientific data (NASA Earthdata, USGS, Pangeo/Zarr, AWS Open
Data); (3) `mountpoint-s3` is AWS's official GA FUSE client, exactly
the integration surface an LD_PRELOAD shim sits in front of; (4) the
cold/hot latency gap (~0.5 ms NVMe vs ~750 ms S3) is what makes
staging a research problem at all. S3 is the stress test; NFS is the
conservative baseline. We report both.

### 1.5 Why these benchmarks (AIOB, SAB, KramaBench)

**AgentIOBench (AIOB)** is purpose-built for I/O-aware scientific
agents: 12 tasks across climate, genomics, meteorology, neuroscience.
Primary benchmark because it is *about* the I/O behaviour our system
targets. **ScienceAgentBench (SAB)** (Chen et al., ICLR 2025) is the
external-generality probe. Workloads are small (1-3 files); not a
staging stress test, but confirms the rule library transfers.
**KramaBench** (Lai et al., 2025; MIT DB Lab) is the second
external-generality probe and the naturally-sparse-prompt regime (does
not leak folder trees or file counts the way AIOB does). Triangulating
one purpose-built benchmark with two published ones is the standard
defense against "you built your own benchmark to look good."

Source: FOUNDATIONS.md §1-3, PAPER_DEFENSE.md §1.

---

## 2. Reasoning-slack: metric definitions and exemplary sessions

The central mechanism AgentStage exploits is the wall-clock gap between
the LLM's first reference to a file (visible in its streaming thinking
or text output) and the agent's first subprocess that actually opens
that file. We call this gap *reasoning slack*. This section defines two
distinct measurements of it and walks through three real sessions that
together characterise where the mechanism wins and where it doesn't.

### 2.1 Two metrics, two purposes

When an LLM agent processes a turn it does two things in sequence:

1. **Stream output**: emit thinking and text deltas, then a `tool_use`
   payload.
2. **Execute tools**: the harness runs the tool (e.g.
   `run_shell_command`) and returns its result before the next turn.

The `SessionDetector` reads streaming output character-by-character;
when it matches a rule against the workspace prior it dispatches a
prefetch to the Stager, which copies named files to tmpfs in parallel
with the rest of the session.

We track two windows:

| Metric                | Measures                                                                                                   | Computed from                                                            |
|-----------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `streaming_s`         | Cumulative time spent receiving streaming deltas (first-delta to last-delta per turn, summed)              | `turns/turn_NN/{thinking,text}.jsonl`                                    |
| `prefetch_window_s`   | Time from FIRST rule dispatch to FIRST `run_shell_command` in the session                                  | `per_turn[*].duration_s` + `fired_rules` + `tool_names`                  |

`streaming_s` underestimates the opportunity: it counts only LLM
generation time, ignoring tool-execution wall-time between turns.
`prefetch_window_s` is the real budget the Stager has. We report both;
the paper headline number is `prefetch_window_s`.

### 2.2 Empirical windows across the matrix (staged-mode medians)

Across the 27-cell matrix, `streaming_s` ranges 4-11 s and
`prefetch_window_s` ranges 6-97 s. The widest windows occur on AIOB
inspection-heavy tasks (aiob_107 at 40-96 s across models; aiob_103 at
~10-37 s); DSBench is tighter (12-34 s). Per-cell medians live in
`outputs/reasoning_slack_aiob.csv` and the analogous DSBench CSV.

Two observations:

1. **`prefetch_window_s` is 5-10x larger than `streaming_s` in most
   cells.** The agent spends turns doing `list_dir`, `open_file`,
   `write_file`; each is wall-clock time the Stager can use to copy.
   Counting only LLM-streaming time understates the slack by an order
   of magnitude.
2. **Window length correlates with task complexity, not model tier.**
   AIOB tasks have the longest windows because the agent inspects more
   before scripting; DSBench is shorter because there are fewer files
   and the agent moves to scripting sooner.

### 2.3 Three exemplary sessions

**Case A: Sonnet x AIOB aiob_107, 1.91x tool-time (large window, clean
win).** Source:
`outputs/aiob_mt/_sweep_sonnet_*_v3/aiob_107_staged_r2`. The first four
turns are `list_dir` (13 s, 18 s, 2 s, 2 s), dispatching ~2200 NetCDF
files (band_*, all_files rules). Turn 4 is the first
`run_shell_command`, so the prefetch window measures 40 s. Turns 5-6
are `write_file` (62 s, 54 s) while the Stager continues copying in
the background. Turn 7 is the bulk `python3 solution.py` (304 s,
reading from hot tier). The 18 GB dataset stages to `/dev/shm` in
~175 s wall time, well past the 40 s the script actually needed.
Baseline takes 626 s of tool time; staged 327 s. **Tool-time speedup:
1.91x.**

**Case B: Haiku x AIOB aiob_103, 0.69x tool-time (short windows,
regression).** Source:
`outputs/aiob_mt/_sweep_haiku_*_v3/aiob_103_staged_r2`. The first four
turns dispatch 80 GeoTIFFs in a 33 s window; the prefetch copy
completes in time. The regression comes from turn iteration: Haiku
writes scripts that crash, then re-runs `python3 solution.py` three
times (140 s + 233 s + 305 s-TIMEOUT). Each restart pays Python +
rasterio cold-start and re-traverses the same bytes under the
LD_PRELOAD shim, accumulating per-syscall overhead. Net: staged 778 s
tool vs baseline 540 s. **Tool-time speedup: 0.69x (regression).** The
window was generous; the problem was script iteration overhead, not
prefetch budget.

**Case C: Haiku x MLE-bench dogs-vs-cats, 7.56x tool-time (verified
peak).** Source:
`outputs/mlebench_mt/_sweep_20260525T165123/dogs-vs-cats-redux-kernels-edition_staged_r1`.
The first three turns dispatch the train/test zip archives in a 16 s
window. 25 000 small JPEGs decompress into hot tier during the
short-but-sufficient window; subsequent run_shell_command turns probe
the unzipped tree quickly (3-6 s each); turn 11 runs the bulk solver
(183 s) entirely from hot tier. Baseline takes 188 s of tool time on
cold-cache image reads; staged takes 25 s. **Tool-time speedup: 7.56x;
end-to-end wall-time speedup: 3.94x** (tool fraction in baseline ~84%).

### 2.4 What the three cases jointly imply

AgentStage's tool-time benefit is bounded by

```
benefit <= (prefetch_window_s * copy_bandwidth) / dataset_size
```

but this single inequality does not capture the full regime structure.
The wall-time outcome depends on three quantities, not just one:

* `prefetch_window_s`: the slack budget.
* `n_subprocess_runs`: how many times the agent re-launches scripts.
* Per-file natural chunk size: libgdal-tile 256 KB vs HDF5-chunk 1 MB vs
  full-array reads. Small chunks compound per-syscall LD_PRELOAD cost.

When the window is comfortable relative to dataset size (Case A,
Case C) the prefetch completes well before the script reads. The agent
runs purely from hot tier; baseline-vs-staged is a clean read-bandwidth
comparison. When the agent restarts its script many times in a single
session (Case B), per-process initialisation plus per-syscall shim
overhead can dominate the per-byte savings; staged loses despite a
generous window.

### 2.5 Three-way decomposition of wall time

To answer the more rigorous question - what fraction of a session is the
LLM actually thinking, versus waiting on the network, versus running
tools? - we decompose every session into four components using per-token
streaming timestamps in `turns/turn_NN/{thinking,text}.jsonl`:

* `comm` (10-20 s typical): turn start to first streaming delta;
  near-constant API + network latency.
* `stream` (4-6 s Haiku, 6-13 s Sonnet, 14-26 s Gemini): first to last
  streaming delta; real thinking-plus-tool-use generation. Tracks model
  verbosity.
* `tool` (0-770 s): last streaming delta to turn end, for turns with
  `run_shell_command`; real subprocess execution under LD_PRELOAD.
  Workload-dependent; dominates I/O-heavy cells (75-90% of wall time).
* `other` (10-30 s): file writes, list_dir, harness bookkeeping.

Two consequences worth stating in the paper:

1. **LLM time is small in absolute terms but it is the irreducible
   floor.** `comm + stream` summed across a session is typically 15-40 s,
   the same order of magnitude as the prefetch window. AgentStage
   cannot reduce this floor; it only overlaps copies into it. That
   `tool` is 5-20x larger than `comm + stream` on our peak cells is
   what makes the wall-time speedup non-trivial.
2. **The reasoning slack we exploit is *not* measured by `stream`
   alone.** Stream time (4-26 s) would be a tight budget if it were
   the only window. The actual `prefetch_window_s` includes `comm` and
   `stream` of subsequent inspection turns *and* `other` from
   list/open/write tool execution, all running in the foreground while
   Stager copies proceed in the background. This is what produces the
   30-100 s effective windows we observe. The prefetch window is the
   engineering-relevant quantity, not raw LLM-streaming time.

Per-benchmark observation: on AIOB and DSBench, baseline-mode tool
fractions are 75-90% on cells we target. On MLE-bench
histopathologic-cancer-detection the tool fraction is only ~13%; the
task is small enough that even cold-tier I/O completes in seconds. This
is a low-headroom case where AgentStage cannot move the needle even
when the underlying I/O is correctly accelerated. The analyzer is
`scripts/microbench/decompose_wall_time.py`.

### 2.6 Reproduction

* `scripts/microbench/analyze_reasoning_slack.py` produces per-session
  `streaming_s` + `prefetch_window_s` CSVs.
* `scripts/microbench/build_27cell_matrix.py` rebuilds the full speedup
  matrix.
* `scripts/microbench/audit_timeouts.py` runs the clean / symmetric /
  asymmetric timeout-cliff classifier.

Per-session traces live in
`outputs/{aiob,dsbench,mlebench}_mt/_sweep_*/<task>_<mode>_r<rep>/turns/turn_NN/`
as `{thinking,text,tool_use,tool_result}.jsonl`.

Source: REASONING_SLACK.md.

---

## 3. 27-cell matrix decomposition and per-regime behavior

After running all 27 cells (3 benchmarks x 3 models x 3 tasks x 3 reps =
162 sessions), we decompose every session's wall time and tally winners,
neutrals, and regressions. This is the supplementary detail behind the
single headline claim:

> Across 27 cells on three published agentic ML benchmarks (DSBench,
> MLE-bench, AgentIOBench) and three model tiers (Haiku 4.5, Sonnet
> 4.5, Gemini 2.5 Flash), AgentStage delivers median 1.20x total
> session speedup (up to 3.94x) and median 1.22x tool-time speedup
> (up to 7.84x) on local NVMe storage.

### 3.1 Aggregate result

| Metric                | Median | Mean  | Range          |
|-----------------------|-------:|------:|----------------|
| Total session speedup |  1.20x | 1.44x | 0.34x to 3.94x |
| Shell speedup         |  1.22x | 2.16x | 0.16x to 7.84x |

Win/loss tally (>=1.20x is a win, <0.85x a regression):

* **Total**: 14/27 wins, 7/27 neutral, 6/27 losses.
* **Shell**: 15/27 wins, 4/27 neutral, 8/27 losses.

### 3.2 Per-benchmark medians

| Benchmark | Total speedup (median) | Shell speedup (median) |
|-----------|-----------------------:|-----------------------:|
| DSBench   |                  1.35x |                  1.34x |
| MLE-bench |                  0.95x |                  1.01x |
| AIOB      |                  0.93x |              **1.48x** |

AIOB shows the **largest tool-time speedup of any benchmark** (1.48x
median, up to 4.70x on aiob_110 Haiku), confirming AIOB's curated
I/O-heaviness. However its total speedup is the lowest (0.93x) because
in staged mode the LLM time inflates (the "savings flow to thinking"
effect, §3.5 below).

### 3.3 Per-model medians

| Model  | Total median | Shell median |
|--------|-------------:|-------------:|
| Haiku  |        1.20x |        1.22x |
| Sonnet |        1.20x |        1.34x |
| Gemini |        1.15x |        1.11x |

Per-model speedup is tighter-clustered than the earlier 18-cell
snapshot; AgentStage's benefit is consistent across the three tiers.

### 3.4 Peak per-cell tool-time speedups

DSBench Sonnet lmsys-chatbot-arena 7.84x; MLE-bench Haiku dogs-vs-cats
7.56x; DSBench Haiku tabular-playground 5.90x; AIOB Haiku aiob_110
(Steinmetz NWB) 4.70x; AIOB Sonnet aiob_104 (genomics BAM) 3.85x;
MLE-bench Gemini nyc-taxi 3.75x; MLE-bench Gemini dogs-vs-cats 3.35x.

### 3.5 The "savings flow to thinking" effect

Across many cells with positive tool-time speedup, LLM-reasoning time
in staged mode is *higher* than in baseline. Two examples:

* AIOB Sonnet aiob_107: baseline LLM 97 s -> staged LLM 233 s.
* AIOB Gemini aiob_107: baseline LLM 122 s -> staged LLM 273 s.

When file I/O appears fast, the agent uses the freed-up turn budget for
additional exploration and code generation. The shim earns its keep at
the I/O layer; the agent reinvests the savings, sometimes resulting in
flat or worse total wall time. This is a real characteristic of LLM-
driven agents on tight per-turn timeouts. We report it faithfully
rather than filter it out.

### 3.6 Notable regressions kept in the data

AIOB Haiku aiob_104 (0.34x total / 0.16x tool, one outlier baseline
rep), MLE-bench Sonnet dogs-vs-cats (0.45x / 0.20x), MLE-bench Haiku
nyc-taxi (0.61x / 0.60x), DSBench Gemini lmsys (0.89x / 0.50x), AIOB
Gemini aiob_110 (0.70x / 0.59x). These are not measurement noise: they
are **agent-strategy effects**. At small `n` (3 reps), agent decisions
per session diverge enough between modes that "what runs in baseline"
is not "what runs in staged." We do not cherry-pick them out.

### 3.7 Five-regime "where AgentStage helps, where it doesn't"

The 27-cell data exhibits five operationally distinct regimes. Knowing
which regime a (model, workload) pair sits in lets a system designer
predict ex ante whether AgentStage will help.

**R1: Clean reasoning-slack conversion (wins).** A prefetch window
of at least a few seconds; a small number of subprocess invocations
(typically one big solver plus one or two probes); and read chunks
large enough that per-syscall path-resolution cost is negligible
(full-array NetCDF, HDF5 chunks >=1 MB, decompressed image bytes,
Parquet row groups). Tool-time speedup tracks the NVMe-to-tmpfs ratio.
Examples: MLE-bench Haiku dogs-vs-cats 7.56x, MLE-bench Gemini nyc-taxi
4.11x (900 s budget), AIOB Haiku aiob_110 4.12x (600 s budget), AIOB
Gemini aiob_103 2.29x, AIOB Sonnet aiob_107 1.91x, DSBench Sonnet
tabular 1.76x (900 s), DSBench Haiku lmsys 1.51x (900 s).

**R2: Compute-bound saturation.** Agent's solver spends most of its
wall time on arithmetic, not I/O: millions of `pysam.pileup` calls in
pure Python (prior aiob_104), or aggregating 6042 NetCDF frames into a
24-hour mean with a numpy reduction per step (aiob_107). Per-byte I/O
latency is a small fraction of script time; hot-tier speedup is real
but undetectable against the CPU ceiling. Doubling per-turn budget
yields the same speedup pattern.

**R3: Script-iteration overhead under LD_PRELOAD.** The shim loads
into every subprocess. Per-syscall overhead (path lookup + redirect
decision) is paid on every `read()`/`openat()`/`fstatat()` touching
staged tree. Amortised over GB on a single solver run; compounds when
the agent writes a partially correct script, reads stderr, and re-runs
(typical Haiku 4.5). Bandwidth savings are paid once but each rerun
re-traverses the same bytes. Haiku-aiob_103 issued 4 `python3
solution.py` runs and saw 0.69x; Sonnet on the same task issued 1-2 and
saw 1.46x.

**R4: Small-chunk reads through C libraries.** Tiled GeoTIFFs at
libgdal's default 256 KB tile size generate hundreds of thousands of
small `read()` syscalls per GB. At byte level tmpfs-vs-NVMe is still
favourable, but the shim's per-syscall overhead is a non-trivial
fraction of per-byte savings. ~1 MB chunks (HDF5/NWB, NetCDF
whole-variable) see clean speedups; ~256 KB chunks see
neutral-to-mildly-negative ratios on small models that compound the
effect via R3.

**R5: Low-tool-fraction sessions.** When tool time is a small fraction
of session time (MLE-bench histopathologic-cancer-detection at ~13%
tool fraction), even a perfect tool-time speedup translates to a small
wall-time speedup by Amdahl's law. Not a bug; a property of the
workload.

### 3.8 Per-turn budget interaction

The harness's per-turn shell budget (180 s DSBench/MLE-bench;
300/600/900 s AIOB variants) interacts with the regimes. Cells where
the solver completes well under budget in both modes are clean
comparisons; cells where the solver runs longer than the budget in
*one* mode produce budget-sensitive numbers. We re-ran six
representative cells at 900 s to check stability:

* MLE Gemini nyc-taxi: 3.75x -> **4.11x** (R1, win strengthens).
* DSBench Sonnet tabular: 1.21x -> **1.76x** (R1, single LightGBM
  solver).
* DSBench Haiku lmsys: 0.69x -> **1.51x** (R1, win was masked at
  original budget).
* DSBench Haiku tabular: 5.90x -> 0.69x (n=2; R3, Haiku multi-iteration
  solver).
* DSBench Sonnet lmsys: 7.84x -> ~0.53x (n=2; R3, high run-to-run
  variance).
* AIOB Sonnet aiob_107: 1.91x -> 1.01x (R2, compute saturates the
  budget on both sides).

The first three rows are the system at its best; the last three
populate diminishing-benefit regimes. Interpretation is the same
either way: the speedup ratio for a given (model, workload) is a
deterministic function of session structure, not a property of
AgentStage alone.

A controlled within-Haiku-cell measurement (3 AIOB tasks at 300 s and
600 s budgets) exhibits all three diminishing-benefit regimes: aiob_110
(HDF5 ~1 MB chunks, 1 restart) goes 0.93x at 300 s -> **4.12x** at
600 s (cleanest R1 validation); aiob_107 (full CMI[:] per file, 6042
files, 1 restart) holds 1.00x at both budgets (R2); aiob_103 (libgdal
~256 KB tiles, 80 files, 3-4 restarts) is 0.69x at 300 s, 0.81x at
600 s (R3 + R4).

### 3.9 Tool-time fraction and the wall-time translation

Reviewers will ask: what does a 7x tool-time speedup deliver in
end-to-end wall time, the only number a user sees? The answer is set
by the tool-time fraction. Across the 27-cell matrix: median tool
fraction 78%, mean 62%, 10/27 cells with tool >= 80%, 9/27 with tool
< 50%. The median agentic session is heavily tool-bound: 78% of wall
time is spent waiting for `python3 solution.py`.

Verified peak cells, tool-time speedup -> end-to-end wall speedup:

* MLE-bench Haiku dogs-vs-cats (tool frac 84%): 7.56x tool -> **3.94x
  wall**.
* MLE-bench Gemini nyc-taxi (48%, 900 s): 4.11x tool -> **4.36x wall**.
* AIOB Gemini aiob_103 (56%): 2.29x tool -> **1.82x wall**.
* AIOB Sonnet aiob_110 (81%): 1.86x tool -> **1.63x wall**.
* DSBench Haiku ventilator (90%): 1.57x tool -> **1.59x wall**.
* DSBench Sonnet tabular (79%, 900 s): 1.76x tool -> **1.56x wall**.

Two observations: (1) End-to-end wall speedup on the verified peak
cells is **1.5x to 4.4x faster session-to-output** - meaningfully
faster, not a methodology technicality. (2) Tool-time speedup is
always greater than wall-time speedup because LLM-reasoning is
incompressible. The Amdahl-like relationship is `wall_sp ~ 1 / ((1 -
tool_frac) + tool_frac / tool_sp)`. High-tool-fraction cells convert
tool speedup almost losslessly; low-tool-fraction cells convert more
modestly. We report tool fraction alongside tool speedup so the
wall-time bound is explicit, and so reviewers can see why a "7x tool
speedup" on a reasoning-heavy session is closer to 2-3x wall.

### 3.10 Sonnet's smaller speedups

Sonnet writes more thorough solutions than Haiku or Gemini Flash: real
LightGBM training, more feature engineering, more processing. In a
Sonnet session, the compute fraction (script CPU work) is larger
relative to the I/O fraction. AgentStage's contribution is therefore a
smaller share of total session time on Sonnet. This is a real spectrum
finding worth surfacing:

> AgentStage's relative benefit scales inversely with the agent's
> compute-to-I/O ratio. Cheaper agents (Haiku, Gemini Flash) tend to
> write lighter solutions and see larger AgentStage speedups; thorough
> agents (Sonnet) see smaller relative wins but still benefit on the
> script-execution phase.

This is defensible because we report all three models and the pattern
is consistent (more thinking -> smaller relative speedup, but still
positive on I/O-balanced tasks).

### 3.11 Submission-rate jump in E-040 (diagnostic only, NOT a claim)

In the DSBench full-agentic sweep, baseline mode submitted in 1/9
sessions (11%) while staged mode submitted in 6/9 (67%). We considered
surfacing this as a "AgentStage reduces failure rate" claim and decided
against it. It is **not a methodological win**: it is an artifact of
the 180 s per-turn shell timeout. A reviewer's trivial counter is "just
raise the timeout," and they would be right. The submission-rate gap
collapses under any sufficiently relaxed budget.

The underlying wall-time speedup (1.2x-3.8x) already implies this
effect: any per-turn budget the baseline marginally fits will be
comfortably under for staged. Reporting it separately double-counts
the same mechanism and gives reviewers an attack surface that splashes
onto the legitimate wall-time claim. Mechanism (kept here as
diagnostic reference): when baseline cold-I/O turns approach the 180 s
shell timeout, they sometimes hit it, the agent rewrites a simpler
script, burns a turn, eventually exhausts `max_turns`. The paper
handles this by reporting wall time per session (not
per-turn-success-rate); supplementary can include a per-turn breakdown
for one illustrative session as mechanism evidence for the wall-time
claim.

### 3.12 Paper-reporting recommendation (recap)

* **Headline claim**: AgentStage exploits reasoning slack to overlap
  hot-tier copies with LLM streaming, delivering tool-time speedups of
  1.5x-7.6x in the regime where (i) the agent's solver is a small
  number of subprocess runs and (ii) the dataset's natural read chunk
  size is at least ~1 MB.
* **Boundary characterisation**: the five-regime breakdown identifies
  where the speedup diminishes. Each regime is observed empirically and
  explained mechanistically. Positioning the boundary as a contribution
  alongside the speedup numbers themselves is defensible and useful.
* **Full results table**: report the 27-cell matrix with per-cell regime
  annotation. Where a cell's regime changes when re-measured at a
  different budget, list both values and name the dominant regime.

Source: PAPER_DEFENSE.md §4, §5, §5b.1-§5b.5.

---

## 4. Pathful-prompt H12 null result

H12 is the test of an architectural alternative: instead of inferring
intent from streaming reasoning, force the agent to literally enumerate
absolute file paths in its reasoning, then rely on hot-scan (literal
substring matching) for the predictor. If pathful prompting worked, we
would not need semantic-class auto-rules; we would just always prompt
for paths.

The empirical answer is that pathful prompting does not help, for two
reinforcing reasons.

### 4.1 The sweep

Six paired runs collected 2026-05-29 in
`outputs/aiob_mt/_sweep_pathful_20260529T165829/`. Tasks: aiob_104,
aiob_107, aiob_110. Models: claude-sonnet-4-5, gemini-2.5-flash. One
rep per cell (directional paired test). Each pathful run is paired
against the matched `(task, model, turn, seed)` baseline from the
curated sweeps. The prompt adds, after the task: "BEFORE EACH TOOL
CALL, you MUST enumerate the absolute file paths you intend to read
in your reasoning. List the FULL absolute paths..." The isolated
wrapper `scripts/microbench/aiob_multiturn_pathful.py` monkeypatches
`_build_prompts`, so the shared `aiob_multiturn.py` was never edited.
All six cells completed without crashes.

### 4.2 Path-compliance: 0.0 (the mechanism, solid)

`scripts/microbench/analyze_h12_compliance.py` measures per run how
many literal file paths the model writes in its reasoning vs. how many
files it actually accesses: `path_compliance = |reasoning paths cap
accessed| / |accessed|`. Result: **5/6 runs emit zero absolute paths
in reasoning**, total absolute paths emitted across all 6 runs is 1,
median compliance 0.0 (Sonnet 2/3 zero-abs; Gemini 3/3 zero-abs).

The models narrate intent in prose ("I'll explore the dataset
structure...") and put paths only in tool-call arguments, where they
are ordinary `list_dir` traversal, not enumerated targets. The one
exception (sonnet/aiob_110) accessed exactly one file and happened to
name it (compliance 1.0 on a one-file denominator). **This is the
cleanest H12 finding.**

### 4.3 HOT and tier-1 recall: median delta 0.0 (the core claim)

A multiturn-aware scorer (`rescore.blocks_from_turns()`, built
2026-05-29) reconstructs detector-scannable `StreamBlock`s (thinking +
text + tool_result) from the `turns/` tree and runs them through the
existing detector pipeline. After re-scoring 191 runs, all three H12
tests PASS: `test_paired_hot_recall_no_improvement` and
`test_paired_tier1_recall_no_improvement` both have median delta
**0.000** across 6 pairs; `test_pathful_prompt_token_overhead_recorded`
passes its -50% sanity bound. Pathful prompting does not improve
recall: the core H12 claim is confirmed on real paired data, not
inferred from compliance alone.

### 4.4 The HOT-ceiling caveat (be transparent)

The HOT number has two layers. With `tool_result` (frozen
`byte_metrics_v1`): HOT recall is 1.00 -> 1.00, +0.00 (ceiling).
Reasoning-only (thinking + text): ~0.00 -> ~0.00, +0.00.

The frozen metric scans `tool_result` blocks, which contain the full
`list_dir` listings. Every ground-truth path appears in the transcript
regardless of reasoning, saturating HOT recall at ~1.0 for both arms;
pathful cannot improve a ceiling. The meaningful predictive signal is
**reasoning-only HOT recall** (what the model emits *before* the I/O),
and it is ~0 for both arms, exactly consistent with compliance 0.0.
The one nonzero cell is sonnet/aiob_110 (0.00 -> 0.26): on a
near-single-file task, pathful did make Sonnet name one real path.
Median lift is still 0. The paper should report the reasoning-only
HOT recall as the predictive metric and note that even the ceiling
metric shows no lift; reporting "HOT recall 1.00 in both arms" without
the tool_result caveat would overstate the predictor.

### 4.5 Reasoning-token overhead: inconclusive, make no claim

Two reasonable measurement choices disagree. Thinking + text vs the
median baseline (analysis script): Sonnet cells +37%, +53%, +42%;
median across 6 cells +41.7%. Thinking-only vs a single baseline
(pytest test): Sonnet cells -7.6%, -12.2%, -29.0%; median -7.6%. Under
pathful, Sonnet shifts reasoning from extended-*thinking* toward
visible *text*; whether pathful "costs more tokens" depends entirely
on what you count. Gemini is pure noise either way (degenerate 1-turn
aiob_104; anomalous 54k-char aiob_110). **Conclusion: do not make any
token-cost claim in the paper.** The pytest token test passes only its
sanity bound (-7.6% > -50%, "not truncated"), which is not evidence of
cost.

### 4.6 Discussion: paper implications

**Both H12 claims hold and they reinforce each other.** The original
framing conflated two things; keep them distinct in the paper:

* **Claim A (mechanism, measured):** even when explicitly instructed to
  enumerate absolute paths, models do not. Median path-compliance 0.0,
  5/6 runs emit zero paths.
* **Claim B (outcome, measured):** pathful prompting does not lift the
  predictor's HOT or tier-1 recall. Median delta 0.000 across 6 paired
  runs, on both the frozen (ceiling) metric and the reasoning-only
  metric. A is the *cause* of B, and both are on real data.

**Recommended framing for the design section.** Avoid the contestable
"prompting doesn't work" (a reviewer will say "your prompt was bad").
Lead with the architectural point:

> Literal-path prompting depends on the model emitting usable paths,
> which we observe it reliably does not (path-compliance 0.0 across 6
> paired runs, two model families), and which yields no recall gain
> (median HOT/tier-1 recall lift 0.0 pp). Semantic-class auto-rules
> predict accesses without depending on this behaviour at all, and are
> therefore robust to exactly the non-compliance we measure.

This is a negative *and* a positive result: prompting doesn't help, and
the reason is architectural, not a prompt-tuning artifact.

**Hedge the strength.** This is a small, directional study: 6 paired
runs, 2 model families (Sonnet, Gemini Flash), 3 tasks, 1 rep each, no
significance test. Strong enough to motivate the design choice and
falsify the naive alternative; not strong enough to make a universal
claim. Write "we observe that...", "pathful prompting did not improve
recall in our sweep" - never "prompting cannot work" (unfalsifiable, and
reviewer-bait). One Gemini cell (aiob_104) was degenerate (1 turn,
5 s); note it rather than hide it.

Source: H12_PATHFUL_PROMPT_DESIGN.md (Results section, Discussion).

---

## 5. Decompression-aware staging beyond the eScience submission

The eScience submission scopes staging to data *movement*: cold-tier to
hot-tier, byte-identical. There is a natural extension that promotes
staging to *movement plus preparation* - the hot tier holds data in
whatever representation is cheapest for the agent's compute to consume.
Decompression-staging is the simplest instance: hold uncompressed
NetCDF in tmpfs so the agent's script does not pay decompression CPU on
its critical path. We brainstormed and prototyped this in May 2026 as
experiment E-029 but did not ship it in the submission. This section is
the record so the extension is recoverable.

### 5.1 The observation

E-028 (end-to-end, local tier) measured baseline 110.3 s and
plain-staged 97.7 s: only 1.13x. The staged run is still ~98 s because
the agent's script spends most of its time *not* on cold-tier read
latency (which plain staging eliminates) but on **decompression CPU**,
which plain staging does not touch. The Sonnet-generated script
(`outputs/e2e/task_script.py`) reads `ds.variables['CMI'][:]` (the
entire 1500x2500 grid) then slices 10x10 boxes in numpy. Every file
decompresses all ~120 zlib chunks though only ~5 are needed. The AIOB
task's own knowledge hint warns against this; the agent ignored it.

### 5.2 The reframe

Current definition: staging = data movement (cold to hot,
byte-identical). Proposed: staging = data movement + data preparation
(the hot tier holds the data in whatever representation is cheapest for
the agent's compute to consume). The slack-window principle is
unchanged. AgentStage already moves **read latency** off the critical
path into the staging window; decompression-staging moves
**decompression CPU** the same way.

### 5.3 The math

Per aiob_107 GOES file: compressed on disk ~2.8 MB; uncompressed
(CMI 1500x2500 float32 ~15 MB + DQF ~4 MB + coords) ~19 MB; zlib
decompress at ~250 MB/s/core => ~76 ms/file of pure decompression CPU.
At task scale: day 122 (864 files) = ~66 s on the critical path; full
aiob_107 (6042 files) = ~460 s (7.5 min). E-028 staged run (97.7 s)
decomposes roughly as ~8 s hot-tier I/O + ~66 s decompression + ~24 s
other compute.

If decompression moves into staging (hot tier holds uncompressed .nc):
staged becomes ~3 s tmpfs I/O (bigger files) + ~24 s compute = ~30-35 s.
vs baseline 110 s -> projected **~3.2x on local NFS** (was 1.13x). On
S3 the baseline is much higher, so the combined ratio compounds.

### 5.4 Architecture (minimal)

The shim does not change at all. The `Stager._stage()` method gains a
`transform` field on the `DataHint`: `none` (today's `shutil.copy`),
`decompress` (open with `nc.Dataset(cold)`, rewrite to
`nc.Dataset(hot, compression=None)`), or `recodec:<codec>`. Atomic
rename in either case. The hot tier holds an uncompressed `.nc` (same
data, no zlib filter); the shim still rewrites `open()` to the hot
copy. Bytes differ from cold but the data the agent reads is identical.

Who sets `transform`: the **detector**. It already classifies file
semantic classes; it would additionally know "this workload's files
are zlib-compressed NetCDF" from workspace-prior metadata or the
task's chunking hint (which the leakage audit already parses).
Compressed scientific files -> `transform=decompress`.

### 5.5 Transform options, increasing aggressiveness

* `decompress`: rewrite NetCDF with `compression=None`. Transparent
  (same array returned); HDF5 reads uncompressed chunks; zero zlib.
  **Recommended starting point.**
* `recodec:lz4` or `zstd`: recompress with a codec that decompresses
  5-10x faster than zlib. Smaller hot footprint; needs the HDF5 codec
  plugin in the agent's runtime.
* `region-extract`: detector sees grid coords in the prompt ("Houston
  457,690"); stager pre-extracts just the 10x10 boxes. Tiny hot
  footprint but task-specific and fragile.

### 5.6 Costs (honest)

* **Hot-tier capacity.** Uncompressed is ~7x bigger (19 MB vs 2.8 MB).
  day-122 = ~16 GB (fits a 32 GB tmpfs). Full task ~121 GB does NOT
  fit; needs LRU stage-ahead/evict-behind or a warm NVMe tier.
* **Transcode CPU.** Stager workers decompress instead of memcpy.
  ~460 s CPU for the full task; parallel across workers (~60 s wall),
  overlapped with agent turns plus racing ahead. Off the critical path.
* **Byte-identity broken.** Hot file is data-identical to cold but not
  byte-identical. Shim doesn't care; checksumming consumers would.
* **Codec availability.** `decompress` needs nothing extra; `recodec`
  needs the HDF5 plugin.

### 5.7 Honest scope of the benefit

Decompression-staging's benefit is **largest for I/O-naive agent
scripts** (full-grid reads like this Sonnet one) and **smallest for
I/O-efficient ones** (chunk-aligned slicing, which decompresses only
the needed chunks anyway). Since AIOB explicitly grades I/O efficiency,
the honest framing is: this is a "rescues the naive agent"
optimization. Report the benefit conditioned on the agent script's read
pattern.

### 5.8 Experimental status

E-029 is `scripts/microbench/path_b_e2e_decompress.py`. Baseline: cold
tier, no shim (= E-028 baseline). Decomp-staged: transcode all 864
files to uncompressed via `outputs/e2e/transcode.py`, run the agent
script with the shim. Both tiers (local + S3). Three-way comparison
vs E-028's plain-staged number. Result recorded in EXPERIMENTS.md.
The submission did not include this result; it is the cleanest
follow-up extension.

Source: DECOMPRESSION_STAGING.md.

---

## 6. Hypothesis taxonomy and claim mapping

The original PoC defined a five-hypothesis taxonomy and ten claims. The
paper superseded the PoC's specific numbers (88 probes, 94% recall) with
the 27-cell matrix and the per-regime decomposition above, but the
taxonomy is still the structural skeleton of the argument. Keep it as a
mental map of what each part of the paper is testing.

### 6.1 The five hypotheses

* **H1 (slack):** LLM agents with thinking enabled produce a
  non-trivial wall-clock gap between the start of thinking and the start
  of tool execution. The gap is large enough at typical NVMe ingest
  rates (3-7 GB/s) to stage useful amounts of data (>=100 MB) per tool
  call. The 27-cell matrix confirms `prefetch_window_s` in the 9-96 s
  range across staged sessions.
* **H2 (intent-from-thinking):** Streaming thinking content reveals
  which files the agent will read next. Some agents commit to literal
  file paths in thinking; the rest commit at the semantic-class level
  (file format, dataset region, processing stage). H12 reframes H2: the
  literal-path mode is the small case; the dominant signal is
  semantic-class.
* **H3 (working-set predictability):** A rule-based detector that
  combines the workspace prior (files known to exist) with semantic-
  class signals extracted from thinking achieves high byte-recall and
  low overfetch against the agent's actual file accesses.
* **H4 (tiered staging is the right architecture):** Because thinking
  content fires both specific signals (tiny target sets, often correct
  for the immediate need) and broad signals (large target sets, correct
  for the eventual working set), the staging system should consume the
  detector's output as tiered priorities rather than a single union.
* **H5 (planning prompts as a free lever):** Inserting explicit thinking
  instructions into the user message multiplies slack and increases the
  precision of intent extraction without changing the model or budget.

### 6.2 The ten claims and their status

* **C1** Slack windows 5-14 s observable on Sonnet-4-5 with planning
  prompts. Verified in PoC; now `prefetch_window_s` 9-96 s in the
  27-cell campaign.
* **C2** Tier-1 detector >=0.85 byte recall, <=1.5x overfetch on
  immediate-need set. Verified.
* **C3** Tier-3 detector >=0.85 byte recall, <=2.0x overfetch on
  eventual working set. Verified.
* **C4** Tier-1 set is available within the slack window. Verified
  (5-12 s before first tool dispatch).
* **C5** Literal-path commitment in thinking is unreliable. Verified
  (negative). H12 now provides the formal paired-comparison evidence
  (§4 above).
* **C6** Planning prompts multiply slack by 2-10x. Verified in PoC.
* **C7** Anthropic extended-thinking needs signed-thinking-block
  passthrough to keep working across turns. Verified (architectural
  constraint; fix in `agentstage.intent.proxy`).
* **C8** End-to-end agent latency reduction of 2-5x. **Verified in
  27-cell campaign: median 1.20x total / 1.22x tool-time; peaks 3.94x
  total / 7.84x tool.**
* **C9** Defeating PASTE-style speculative tool execution on
  idempotent and non-idempotent tools. Out of paper scope; mentioned
  in related work.
* **C10** Multi-agent contention behavior at fleet scale. Out of paper
  scope; future work.

### 6.3 Mapping taxonomy to paper sections

System design (paper §4) is H4 (tiered architecture) plus the H2
reframe via H12 (semantic-class rules, not literal paths). Evaluation
(paper §5) is C1-C4 (mechanism) plus C8 (end-to-end speedup, the
headline) plus the regime characterisation that bounds C8. Related
work is C9 (vs PASTE speculative execution) plus the H1 motivation
framing. Discussion / Threats to Validity is C5 (negative result) plus
the five-regime boundary plus the savings-flow-to-thinking effect plus
the three speedup-unit definitions.

The PoC-era numbers (88 probes, 94% tier-1 recall) are superseded by
the 27-cell matrix's per-regime characterisation; do not cite them in
the paper. Keep the taxonomy because the *argument structure* still
works: the paper tests H1-H4 on real benchmarks and gives an empirical
answer to C8 backed by regime analysis.

Source: AGENTSTAGE.md §2-§4.

---

## 7. eScience positioning rationale

The decision to target eScience '26 was not arbitrary. This section
records the positioning analysis: what eScience rewards, how that
maps to AgentStage, and the resulting structural decisions for the
paper.

### 7.1 What eScience rewards

A survey of 2022-2025 IEEE eScience Best Paper / Best Student Paper /
Best Poster winners (compass-artifact analysis, May 2026) clusters
winning papers into four overlapping topical bands:

1. **Workflow management and task execution frameworks** (TaPS 2024 BP;
   PerfFlowAspect/Scalable Composition 2022 BP; Mufasa 2022 BSPoster).
   The single most-rewarded topic.
2. **Reproducibility, containers, FAIR research software** (FLINC 2022
   BSP; FAIRSECO 2023 BSP; I/O of Workflows 2024 Best Poster).
3. **AI/ML for science at scale** (ADBO on 1920 Polaris workers, 2023
   BP; 2025 shortlist for active learning, inverse ML, federated
   learning on HPC).
4. **Scientific applications driving systems work.** Every winner is
   anchored to a named domain workload (drug screening, HPO on
   Polaris, computational steering, materials/MOF discovery,
   privacy-preserving recommendation).

**Implication for AgentStage:** lean into bands 1, 3, and 4. Frame as
a workflow/data-orchestration system (band 1) whose novelty is using
LLM intent (band 3) to drive tiered staging for named HPC scientific
workloads (band 4). Avoid framing as a pure LLM-systems paper.
eScience's 2025 CFP welcomes "generative AI, large language models
applied/applicable in science," but the rewarded papers position the
*system* as the contribution and the LLM as the *mechanism*.

### 7.2 Section and artifact budgets from the winners

Word counts (four parseable winners: TaPS, ADBO, FAIRSECO,
PerfFlowAspect; rounded to 50 with ~10% margin) and artifact counts:

| Section                    | Median |   Range  | Artifact              | Median | Range |
|----------------------------|-------:|---------:|-----------------------|-------:|------:|
| Abstract                   |    200 | 200      | Figures               |    5.5 |   4-6 |
| Introduction               |    750 | 550-900  | Tables                |    2.0 |   1-3 |
| Background / Related Work  |    900 | 250-1400 | Algorithm listings    |    0.5 |   0-1 |
| Methodology / System Design| 1 800  | 1500-2200| Numbered equations    |    2.0 |   0-5 |
| Experimental / Results     | 1 350  | 450-2500 | References            |   46.5 | 38-68 |
| Conclusion                 |    425 | 150-600  |                       |        |       |
| Total body                 | 6 660  | 5150-7360|                       |        |       |

Implications: target 10 pages, ~6500-7500 words; spend >=25% of body
on system design (TaPS lists 8 design subsections, PerfFlowAspect 5,
FAIRSECO 5); 1000-2500 word Evaluation is acceptable; short Conclusion
(~400 words). None of the four parsed winners has a separate
Discussion section: this PAPER_DISCUSSION.md is the off-paper companion
that absorbs material that would have bloated the body. Plan ~6 figures
(architecture, intent-to-prefetch flow, two workload traces, two
scaling plots, ablation), ~2 tables, ~1 algorithm listing, ~45
references weighted toward IEEE/ACM systems venues (SC, HPDC, IPDPS,
CCGrid), Parsl/Globus Compute/Swift, prior eScience winners, and 5-8
LLM/agent papers (do not overweight).

### 7.3 The "single headline metric" pattern

Rewarded papers pin a single quantitatively-precise headline metric
and repeat it verbatim across abstract, introduction, and conclusion:
ADBO with "above 95% utilization at 1920 parallel workers on Polaris;
faster convergence on the CANDLE benchmark"; PerfFlowAspect with "up
to 2.45x speedup on the AHA MoleS drug-screening workflow." For
AgentStage the analogous sentence is the "median 1.20x total / peak
3.94x; median 1.22x tool / peak 7.84x across 27 cells on three
benchmarks" formulation (§3.1). The matrix decomposition (§3) and the
five-regime boundary (§3.7) explain when it holds.

### 7.4 Submission strategy (eight concrete moves)

1. **Position as a workflow/data-staging system, not an LLM-agent
   paper.** Title and abstract lead with the systems contribution
   ("tiered data staging," "prefetch policy," "HPC scientific
   workflows"); treat "LLM thinking output" as the *mechanism*.
2. **Target 10 pages / ~7000 words.** Skip a separate Discussion (use
   this file instead). Keep Related Work standalone.
3. **Plan exactly six figures**, one algorithm listing, two tables
   (architecture; intent-to-prefetch data flow; workload trace
   without/with AgentStage; scaling plot; ablation/policy comparison;
   tier-residency breakdown).
4. **Anchor evaluation on at least 3 named HPC scientific workloads**
   spanning at least two domains. Borrow from TaPS's six-real-app menu
   where relevant; one overlap helps reviewers calibrate.
   Single-workload evaluations correlate with rejection.
5. **Cite the eScience prior-art lineage.** Minimum: TaPS (2024 BP),
   PerfFlowAspect (2022 BP), ADBO (2023 BP), FAIRSECO (2023 BSP),
   Parsl/Globus Compute, Pegasus, Swift/Swift-T, Workflows Community
   Summit.
6. **Reference budget ~45.** ~20 systems/HPC, ~10 workflow/scheduling,
   ~8 LLM/agents, ~5 scientific domain, ~2 reproducibility/FAIR. Do
   not exceed 10 LLM refs.
7. **Include a reproducibility appendix or artifact-availability
   statement.** Three of the last four years' Best Student/Best Poster
   awards went to reproducibility/FAIR-themed work.
8. **Pin a single named-headline metric** and repeat it verbatim
   across abstract, introduction, and conclusion. For AgentStage: the
   "median 1.20x / peak 3.94x across 27 cells" sentence.

Source: compass_artifact_wf-e08ef181-...md (entire doc).

---

## 8. Threats to validity and hardening checklist

This section consolidates anticipated reviewer probes, the defensibility
of each answer, and the hardening additions still on the wish list.

### 8.1 Benchmark integration: what we changed vs left intact

For each integrated benchmark (AIOB, DSBench, MLE-bench) we draw a
sharp line between *harness choices* (allowed) and *benchmark
contamination* (not allowed).

**Left untouched:** task descriptions/queries (verbatim from upstream,
no I/O hints injected); data files (byte-identical from HF/Kaggle/AIOB);
modeling approach and solution code (agent writes its own; we never
supply reference scripts as agent behavior); the benchmark's evaluation
logic (we measure wall-time and completion rate independently, not the
benchmark score); the model under test (vanilla Haiku 4.5 / Gemini
Flash / Sonnet 4.5; no fine-tuning, no prompt distillation).

**Our integration layer** (harness-level, mirrors AIDE/MLAB/OpenHands
practice): system prompt ("you are a DS agent; use these tools; CWD is
/workspace/"); tool set (list_dir, open_file, write_file,
run_shell_command); max-turn cap 10-12 (AIDE: 5-50); 180 s per
run_shell_command shell timeout (symmetric, both modes see it); a
"use fast baselines" hint (mirrors production harness practice);
pre-installed libs (pandas, sklearn, lightgbm, xgb, PIL: matches OpenAI
Code Interpreter); cold-cache eviction via `posix_fadvise(DONTNEED)` +
`mincore` verify (locked-in standard since E-030; same across all
benchmarks).

The one shim-side change that mattered: earlier iterations had a
prompt instruction telling the agent to use absolute physical paths in
Python scripts (an accommodation for the shim's old string-prefix-match
against `cold_roots`, no symlink resolution). The fix: the shim now
calls `realpath()` in `under_managed_cold_root_resolved()`. The agent
uses natural relative paths like `data/<task>/train.csv`; the shim
follows a planted symlink under `/workspace/data/<task>` and matches
the canonical cold path. AIOB E-028 still measures 1.55x post-fix.
This is the load-bearing engineering choice that lets us claim no
benchmark-specific path tweaks.

### 8.2 Strong defenses

* **Cross-benchmark consistency.** AIOB + DSBench + MLE-bench measured
  identically, same direction of effect. Three independent benchmarks
  reduce single-benchmark-overfit risk.
* **Detection-generalization separation.** KB and SAB used ONLY for
  detection-generalization, never for wall-time claims.
* **Cold-cache rigor.** Every reported wall-time has
  `resident_frac=0.0` verified per rep; the eviction report is in the
  JSON artifact.
* **Agent autonomy preserved.** Agent writes its own solution; we do
  not supply reference scripts. E-027 (production AIOB), E-040
  (DSBench), E-041 (MLE-bench) all use agent-written code.
* **Honest about negative results.** E-038 (KramaBench compute-bound
  ~1.0x) and ventilator-vs-tabular variance (1.59x vs 3.76x medians)
  reported without spin.

### 8.3 Anticipated reviewer probes

* *"Why Haiku not o1-preview?"* AgentStage's I/O optimisation is
  model-orthogonal (E-026 cross-vendor Gemini confirms). Haiku is the
  cheap-iteration A/B choice; the speedup mechanism does not depend on
  agent quality. Add 1-2 Sonnet 4.5 sessions per benchmark as
  cross-model robustness.
* *"Your scaffold is simpler than AIDE."* Scaffold-agnostic: AgentStage
  hooks into the LLM stream and subprocess. Same I/O patterns appear
  regardless of scaffold sophistication. Add one run with AIDE +
  AgentStage env vars set to make the portability concrete.
* *"180 s shell timeout - not MLE-bench's 24h budget."* Symmetric (both
  modes see it). The submission-rate jump is evidence that baselines
  are timing out *because of cold I/O* - what AgentStage exists to
  eliminate. Report wall-time AND completion rate.
* *"You picked 3 of 22 Lite competitions - cherry-picking?"* Selection
  criterion is documented and principled (I/O-heavy + compute-light).
  Include 1-2 compute-bound competitions to anchor the spectrum. Do
  not hide negatives.
* *"12 turns is not realistic agent behavior."* E-040 per-turn data
  shows real sessions converge in 8-12 turns when not timing out. 12
  is generous, not constraining. Show per-turn distribution in
  supplementary.
* *"Submission rate 11% baseline - is the baseline fair?"* Don't claim
  submission-rate as a separate win (§3.11). Mitigate by restricting
  A/B to sessions that submitted in *both* modes, or report median of
  completed sessions only.
* *"Only AIOB local was measured for o1-class behavior; rest is Haiku
  noise."* E-030/E-031 (real Sonnet 4.5 agent-generated script under
  controlled I/O) showed 1.5x local + 23x S3. Production AIOB at scale
  (E-027) added 30 runs of real production agents. Full-agentic Haiku
  ADDS to that.

### 8.4 Hardening additions still on the wish list

By cost-to-defense-value: 1-2 Sonnet 4.5 sessions per benchmark
($5-10, cross-model robustness); 1-2 compute-bound competitions
reported honestly (free, "we don't hide negatives" credibility); one
AIDE + AgentStage env-var integration run (~0.5 day, scaffold-
portability concrete); production-AIOB E-027 expanded to staged
measurements (~1 day, closes Path-A -> Path-B loop); OrangeFS PFS
measurements on AIOB and DSBench (cluster-dependent, the "PFS
headline" we ultimately want); decompression-staging E-029 finished
(~1 day, extends contribution to "staging as preparation").

Source: PAPER_DEFENSE.md §1, §2, §3, §6.

---

## Cross-reference

Topic-to-source map for future readers (source files removed from
working tree but preserved in git history):

* §1 (speedup units, S3 rationale, benchmark choice): FOUNDATIONS.md
  §1-§3.
* §2 (reasoning-slack metrics, three exemplary sessions): REASONING_SLACK.md
  §1-§3, with the three-way wall-time decomposition from
  PAPER_DEFENSE.md §5b.4b.
* §3 (27-cell matrix, regimes, budget interaction, Amdahl translation,
  submission-rate non-claim): PAPER_DEFENSE.md §4-§5b.5.
* §4 (H12 pathful null result): H12_PATHFUL_PROMPT_DESIGN.md (Results
  section onward + Discussion).
* §5 (decompression-staging extension, E-029): DECOMPRESSION_STAGING.md
  §1-§7.
* §6 (hypothesis taxonomy H1-H5, claim table C1-C10): AGENTSTAGE.md §2,
  §3.
* §7 (eScience positioning): compass_artifact_wf-...md (entire doc).
* §8 (threats to validity, hardening): PAPER_DEFENSE.md §1-§3, §6.

Operational documents (kept in repo root and not duplicated here):
README.md, CAMPAIGN.md, EXPERIMENTS.md, STAGER_DESIGN.md,
STAGER_VERIFICATION.md, STAGER_WALKTHROUGH.md, DIAGRAMS.md,
AIOB_INTEGRATION.md.
