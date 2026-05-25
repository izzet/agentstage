# Paper Defense — Methodology Notes and Anticipated Reviewer Probes

A running record of methodology decisions, defensibility analysis, and
unexpected findings that should make it into the paper. Captures the
"what's defensible, what reviewers will probe, what to add for hardening"
discussion so the paper text doesn't have to be re-derived from scratch.

---

## 1. Benchmark integration — what we changed vs left intact

For each of the three published benchmarks we integrated (AIOB, DSBench,
MLE-bench), we draw a sharp line between **harness choices** (allowed)
and **benchmark contamination** (not allowed).

### What we left UNTOUCHED in every benchmark

| Item | Why this matters |
|---|---|
| Task descriptions / queries | Verbatim from upstream — never edited, never had I/O hints injected |
| Data files | Byte-identical from upstream sources (HF, Kaggle, AIOB datasets) |
| Modeling approach / solution code | Agent writes its own — we never supply reference scripts as "the agent's behavior" |
| Benchmark's evaluation logic | Untouched (we measure wall-time + completion rate independently, not the benchmark score) |
| Model under test | Vanilla Haiku 4.5 / Gemini Flash, no fine-tuning, no prompt distillation |

### What we DID introduce — our integration layer

These are HARNESS choices, not BENCHMARK changes. Justifiable because every
agentic benchmark requires some scaffold; we use the same shape as the
benchmarks' own bundled scaffolds (AIDE / MLAB / OpenHands / DSBench's
code-interpreter notebook).

| Layer | Choice | Why defensible |
|---|---|---|
| System prompt | "You are a DS agent; use these tools; CWD is /workspace/" | Every agentic harness has one. Mirrors AIDE / MLAB / dummy. |
| Tool set | list_dir / open_file / write_file / run_shell_command | Standard. Same tools SWE-bench, MLE-bench scaffolds, AIDE provide. |
| Max-turn cap | 10–12 | All published scaffolds have one. AIDE: 5–50. Ours within range. |
| Shell timeout | 180s per `run_shell_command` | Necessary because Haiku writes occasionally slow solutions; symmetric (both modes see it). |
| "Use fast baselines" hint | Steer toward sklearn/light LightGBM | Time-budget hint that mirrors what production agentic harnesses do. |
| Pre-installed Python libs | pandas/sklearn/lightgbm/xgb/PIL via system Python | Matches what OpenAI Code Interpreter ships with. |
| Cold-cache eviction | posix_fadvise(DONTNEED) + mincore verify | The locked-in standard since E-030; same for AIOB / KB / DSBench / MLE-bench. |

### The ONE shim-side change that mattered

Earlier iterations had a prompt instruction telling the agent to use
absolute physical paths in its Python scripts. That was a benchmark
accommodation for the shim's old limitation (string prefix-match against
cold_roots, no symlink resolution).

**Fix that made it defensible**: shim now calls `realpath()` in
`under_managed_cold_root_resolved()`. The agent uses natural relative
paths like `data/<task>/train.csv`; the shim follows the symlink we
plant under `/workspace/data/<task>` and matches against the canonical
cold path. AIOB E-028 still measures 1.55× post-fix (no regression).

This is the load-bearing engineering choice that lets us claim:
**no benchmark-specific path tweaks**.

---

## 2. Strong defenses (hardest to attack)

1. **Cross-benchmark consistency.** AIOB + DSBench + MLE-bench all measured
   the same way, all show the same direction of effect. Three independent
   benchmarks reduce single-benchmark-overfit risk substantially.

2. **Detection-generalization separation.** KB + SAB used ONLY for
   detection-generalization (H6), never for wall-time claims. We don't
   conflate the two.

3. **Cold-cache rigor.** Every reported wall-time has `resident_frac=0.0`
   verified per rep. JSON artifacts include the eviction report so
   auditability is built-in.

4. **Agent autonomy preserved.** Agent writes its own solution code; we
   don't supply reference scripts as the workload. Production-AIOB-style
   E-027 plus DSBench E-040 plus MLE-bench E-041 all use agent-written
   code under test.

5. **Honest about negative results.** E-038 (KB compute-bound, ~1.0×)
   and ventilator-vs-tabular variance (1.59× vs 3.76× medians) reported
   without spin. Reviewers see we report what's there.

---

## 3. Anticipated reviewer probes and answers

| Probe | Defense | What to add to paper |
|---|---|---|
| "Why Haiku not o1-preview?" | AgentStage's I/O optimization is model-orthogonal (E-026 cross-vendor Gemini confirms). Haiku is the cheap-iteration choice for A/B; the speedup mechanism doesn't depend on agent quality. | One-paragraph methodology note + ideally 1–2 sessions on Sonnet 4.5 as cross-model robustness. |
| "Your scaffold is simpler than AIDE." | Scaffold-agnostic claim — AgentStage hooks into the LLM stream + the subprocess; either could be replaced. Same I/O patterns appear regardless of scaffold sophistication. | Explicit note: integration is scaffold-portable. Ideally one run with AIDE + AgentStage env vars set. |
| "180s shell timeout — not MLE-bench's 24h budget." | Symmetric (both modes see it). The submission-rate jump is itself evidence that baselines are timing out *because of cold I/O* — exactly what AgentStage exists to eliminate. | Report both raw wall-time AND completion rate. The latter is half the story. |
| "You picked 3 of 22 Lite competitions — cherry-picking?" | Selection criterion is documented and PRINCIPLED: I/O-heavy + compute-light. On compute-bound competitions we showed (E-038) AgentStage helps less by design. | Show the FULL profile: include 1–2 compute-bound competitions to anchor the spectrum. Don't hide negatives. |
| "12 turns isn't realistic agent behavior." | E-040 per-turn data shows real sessions converge in 8–12 turns when not timing out. 12 is generous, not constraining. | Show per-turn distribution in supplementary. |
| "Submission rate 11% baseline — is the baseline a fair comparison?" | That's the honest reality: baseline agents *do* timeout on cold I/O. AgentStage's value IS that more sessions complete. Pretending baseline is healthy would understate AgentStage's value. | Make the failure-mode analysis explicit. "AgentStage reduces session failure rate from X% to Y%" is a *separate* claim from "speedup". |
| "Only AIOB local was measured for o1-class behavior; rest is Haiku noise." | E-030/E-031 (real Sonnet 4.5 agent-generated script under controlled I/O conditions) showed 1.5× local + 23× S3. Production AIOB at scale (E-027) added 30 runs of real production agents. The full-agentic Haiku numbers ADD to that, they don't replace it. | Explicitly stack the evidence — Haiku full-agentic builds ON production-grade lower-stack measurements. |

---

## 4. The unexpected finding — failure-rate reduction

### What we observed (E-040)

```
                            baseline submission rate    staged submission rate
DSBench full agentic sweep            1/9 (11%)                  6/9 (67%)
                                                                 ─────────
                                                                  6× more
```

### Why this happens — mechanism

**Not** a separate "AgentStage makes agents more reliable" property —
it's a downstream consequence of the I/O speedup interacting with the
fixed time budgets agentic harnesses impose:

```
Baseline session timeline:
  Turn 0:  list_dir            (1-2s)
  Turn 1:  read previews       (~5s, mostly LLM)
  Turn 2:  write_file solution (~10s LLM)
  Turn 3:  run python solution.py
              │
              ▼
          pd.read_csv('data/<task>/train.csv')     ← COLD READ
              │ takes 30-50s on local NFS for ~500 MB
              │
          model.fit(...)                            ← compute
              │ another 30-60s
              ▼
          [TOTAL: 60-120s → may approach 180s timeout]

  If timeout fires: rc=-9, stderr="[TIMEOUT]"
  Turn 4:  agent reads timeout, rewrites simpler solution    (~10s)
  Turn 5:  run again — still slow because data still cold    (might timeout again)
  Turn 6:  rewrite again
  Turn 7-9: keep failing/rewriting
  Session ends at max_turns=10 with no submission
```

vs:

```
Staged session timeline:
  Turn 0:  list_dir → detector fires sample_csv / first_inspect rules
             → Stager prefetches train.csv + test.csv to /dev/shm
             (1-2s; parallel with the agent's next reasoning turn)
  Turn 1:  read previews
  Turn 2:  write_file solution
  Turn 3:  run python solution.py
              │
              ▼
          pd.read_csv('data/<task>/train.csv')     ← SHIM REDIRECTS
              │ /dev/shm read: 0.5-2s
              │
          model.fit(...)                            ← compute (same as baseline)
              │ 30-60s
              ▼
          [TOTAL: ~30-65s → comfortably under 180s timeout]

  rc=0, submission.csv written in turn 3
  Turn 4-9: agent says "done" or does minor refinements
  Session completes successfully.
```

### So the failure-rate gain is a TIME-BUDGET interaction effect

Restated for the paper:

> *AgentStage's I/O speedup converts time savings into completion-rate
> gains when the agent has a bounded time budget. With cold I/O, agents
> spend turns retrying timed-out scripts; with staged data, the first
> script run typically succeeds, conserving turns for actual progress.
> An uncapped agent (24h budget, no per-turn timeout) would see the same
> speedup expressed instead as raw wall-time improvement.*

### Is this robust and reproducible?

| Concern | Status |
|---|---|
| Direction consistent across reps? | Yes — every individual session that timed out shows the long-shell-command pattern; never seen staged time out where baseline didn't |
| n=9 per cell — small N | Yes, small N. Caveat in paper or run more reps. |
| Specific to our turn cap + 180s timeout? | Yes — this is contingent on having a bounded budget. AIDE with longer budgets would express this as wall-time, not failure rate. Worth noting explicitly. |
| Could we be triggering the timeout artificially? | Both modes see the SAME timeout; symmetric. The asymmetry comes from baseline's cold-I/O cost being closer to the timeout boundary. |
| Methodologically sound? | Yes if framed honestly as a BUDGET-INTERACTION EFFECT, not as "AgentStage makes agents inherently more reliable" |

### Paper framing recommendation

Treat this as a **separate, complementary** claim alongside wall-time:

```
Claim 1 (wall-time):   AgentStage reduces I/O-bound session wall time
                       by 1.2×–3.8× on representative agentic ML tasks.
Claim 2 (completion):  Under realistic per-turn time budgets, AgentStage
                       raises the agent's session-completion rate by
                       up to 6× on the same tasks, by keeping individual
                       run_shell_command turns within the budget.
```

Don't merge them. Don't pretend completion-rate is the headline. They're
two different consequences of the same underlying I/O reduction, observed
under different framings of "what counts as wall time".

---

## 5. Additions that would harden the story

Listed by cost / value:

| Add | Cost | Defense value |
|---|---|---|
| 1-2 Sonnet 4.5 sessions per benchmark | $5-10 | Cross-model robustness |
| 1-2 compute-bound competitions reported HONESTLY | Free (just data prep) | "We don't hide negatives" credibility |
| One AIDE + AgentStage env-var integration run | ~0.5 day | "Integration is scaffold-portable" |
| Production-AIOB E-027 expanded to staged measurements | ~1 day | Closes the loop on Path-A → Path-B story |
| OrangeFS PFS measurements on AIOB and DSBench | depends on cluster | The "PFS headline" the user wants |

---

*This file accompanies EXPERIMENTS.md (per-experiment log) and is
intended as the input to paper Section 4 (Methodology) + Section 6
(Discussion / Threats to Validity).*
