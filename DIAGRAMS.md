# Diagrams — I/O leakage audit, multi-turn detector, Regime A vs B

Visual companion to [`IO_LEAKAGE_AUDIT.md`](IO_LEAKAGE_AUDIT.md) and the
E-011 — E-015 entries in [`EXPERIMENTS.md`](EXPERIMENTS.md). Each
diagram below is keyed to a specific finding so the paper figures can
inherit the same conceptual layout.

---

## 1. The two-regime comparison (central finding)

Shows why the detector "wins" in hinted mode and "misses" in sparse
mode — the rule library is static, but the agent's file choice depends
on what the prompt told it.

```
                          REGIME A: HINTED PROMPT (what we measured before)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Prompt: "...bands 08, 09, 10... C08 files in /raw/2024/122/00/..."     │
   └──────────────────────────────┬──────────────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │  Turn-0 thinking (1100 chars):        │
              │  "I need to read C08 first, then..."  │  ←─ paraphrases prompt
              └────────────────┬──────────────────────┘
                               │
                               ▼ (rule library scans thinking)
              ┌───────────────────────────────────────┐
              │  Rule library (static, hand-coded):   │
              │    first_inspect  → Band 08 file      │  ←─ tuned for this case
              │    band_08        → Band 08 files     │
              │    all_files_sig  → 6042 files        │
              └────────────────┬──────────────────────┘
                               │
              ┌────────────────▼──────────┐    ┌─────────────────────────┐
              │  Stager prefetches:       │    │  Agent's actual choice: │
              │     Band 08 file ✓        │ ◄══► Band 08 file ✓         │
              └───────────────────────────┘    └─────────────────────────┘
                              ↑                              ↑
                              └────── HIT! 19,213× ──────────┘
                                      speedup (E-010)


                          REGIME B: SPARSE PROMPT (new measurement)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Prompt: "Analyze GOES-16 data under /data/goes_cmi_composites/         │
   │          for several US locations across available bands."              │
   │          (counts, sizes, chunking, band numbers all STRIPPED)           │
   └──────────────────────────────┬──────────────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │  Turn-0 thinking (1100 chars):        │
              │  "I need to explore the data first.   │  ←─ generic, no band
              │   Let me see the structure..."        │
              └────────────────┬──────────────────────┘
                               │
                               ▼ (same static rule library scans thinking)
              ┌───────────────────────────────────────┐
              │  Rule library (same, static):         │
              │    first_inspect  → Band 08 file      │  ←─ still points at C08
              │    band_08        → (no fire yet)     │     (rule hasn't matched)
              └────────────────┬──────────────────────┘
                               │
              ┌────────────────▼──────────┐    ┌─────────────────────────┐
              │  Stager prefetches:       │    │  Agent's actual choice: │
              │     Band 08 file          │ ✗  │  Band 02 file (E-015)   │
              └───────────────────────────┘    └─────────────────────────┘
                              ↑                              ↑
                              └────── MISS! prefetch wasted ─┘
```

**Reading**: the architecture (detector → stager → shim) is identical in
both regimes. What differs is whether the agent's file choice aligns with
the rule library's hard-coded targets. The audit's finding is **not that
the architecture fails** — it's that the *rule library* is brittle to
prompt sparsity.

---

## 2. Multi-turn loop with detection firing points

Shows where in a real agent session the detector catches I/O signal.
Built from the E-011 / E-014 / E-015 captures.

```
   TURN 0           TURN 1          TURN 2          TURN 4          TURN 6
   ━━━━━━           ━━━━━━          ━━━━━━          ━━━━━━          ━━━━━━

  ┌────────┐       ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
  │THINKING│──┐    │ (none) │      │ (none) │      │ (none) │      │ (none) │
  │1100 ch │  │    │        │      │        │      │        │      │        │
  └────────┘  │    └────────┘      └────────┘      └────────┘      └────────┘
  ┌────────┐  │    ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
  │ (none) │  │    │  TEXT  │      │  TEXT  │      │  TEXT  │──┐   │  TEXT  │
  │        │  │    │"Let me │      │"Found  │      │"hourly │  │   │"reading│
  │        │  │    │check..." │   │raw..." │      │dirs..."│  │   │C08..." │
  └────────┘  │    └────────┘      └────────┘      └────────┘  │   └────────┘
  ┌────────┐  │    ┌────────┐      ┌────────┐      ┌────────┐  │   ┌────────┐
  │tool_use│  │    │tool_use│      │tool_use│      │tool_use│  │   │tool_use│
  │ls /data│  │    │ls /raw │      │ls 2024 │      │ls .../00│ │  │open C08│
  └────────┘  │    └────────┘      └────────┘      └────────┘  │   └────────┘
                                                                │
   ▼ (between turns: tool execution happens, results fed back)  │
                                                                │
  ┌────────┐       ┌────────┐      ┌────────┐      ┌────────┐  │
  │tool_   │       │tool_   │      │tool_   │      │tool_   │──┼─┐
  │result  │       │result  │      │result  │      │result  │  │ │
  │"foo,bar│       │"goes_/"│      │"raw/"  │      │"C08 C09│  │ │
  │ baz/"  │       │        │      │        │      │ C10..."│  │ │
  └────────┘       └────────┘      └────────┘      └────────┘  │ │
                                                                │ │
   PREDICTOR FIRING POINTS                                      │ │
   ▼                                                            │ │
   • thinking ──fires "first_inspect" + "all_files_signal" (turn 0)
                                                                │ │
   • text ─────fires "one_hour" (turn 4, sparse mode only) ◄────┘ │
                                                                  │
   • tool_result ─fires "band_08" + "band_09" (turn 4) ◄──────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │  Without our extensions (legacy thinking-only):  2 rules fired │
   │  With tool_result-aware:                         4 rules fired │
   │  With text-aware (sparse mode boost):            +1 more rule  │
   └────────────────────────────────────────────────────────────────┘
```

**Reading**: an agent's I/O-relevant tokens don't all live in turn-0
thinking. In hinted mode they happen to; in sparse mode the agent
discovers them via `list_dir` and surfaces them via `tool_result`
content and visible `text`. Without scanning those, the detector is
deaf to most of the session.

---

## 3. The four detector variants (E-012 / E-013)

The replay study isolates the contribution of each architectural change
by running the same captured corpus through 4 detector configurations.

```
                          INPUT: same captured multi-turn corpus
                                          │
              ┌───────────────┬───────────┴───────────┬───────────────┐
              ▼               ▼                       ▼               ▼
   ╔══════════════╗ ╔═════════════════╗ ╔═══════════════════════╗ ╔══════════════╗
   ║   Variant A   ║ ║    Variant B    ║ ║      Variant C        ║ ║   Variant D  ║
   ║ thinking only ║ ║ thinking +      ║ ║ thinking + text +     ║ ║ SessionPred  ║
   ║   (legacy)    ║ ║ tool_result     ║ ║  tool_result (full)   ║ ║ (streaming   ║
   ║               ║ ║                 ║ ║                       ║ ║  per-turn)   ║
   ╚═══════╤══════╝ ╚════════╤════════╝ ╚══════════╤════════════╝ ╚══════╤══════╝
           │                 │                     │                     │
           ▼                 ▼                     ▼                     ▼
   ┌──────────────┐ ┌──────────────────┐ ┌────────────────────┐ ┌──────────────┐
   │  Sees only:  │ │  Sees:           │ │  Sees:             │ │  Same as C,  │
   │  ┌────────┐  │ │  ┌────────┐      │ │  ┌────────┐        │ │  but yields  │
   │  │THINKING│  │ │  │THINKING│      │ │  │THINKING│        │ │  per-turn    │
   │  └────────┘  │ │  └────────┘      │ │  └────────┘        │ │  DELTAS for  │
   │              │ │  ┌────────┐      │ │  ┌────────┐        │ │  live        │
   │              │ │  │tool_res│      │ │  │  TEXT  │        │ │  dispatch    │
   │              │ │  └────────┘      │ │  └────────┘        │ │              │
   │              │ │                  │ │  ┌────────┐        │ │              │
   │              │ │                  │ │  │tool_res│        │ │              │
   │              │ │                  │ │  └────────┘        │ │              │
   └──────┬───────┘ └────────┬─────────┘ └─────────┬──────────┘ └──────┬───────┘
          │                  │                     │                    │
          ▼                  ▼                     ▼                    ▼
       2 rules            4 rules               4 rules              4 rules
      (E-011 hinted)     (E-011 hinted)       (E-011 hinted)       (E-011 hinted)
                                              4 rules              4 rules
       1 rule             3 rules               (E-014 sparse)      (E-014 sparse)
      (E-014 sparse)     (E-014 sparse)        3 rules              3 rules
                                              (E-015 sparse_live)  (E-015 sparse_live)
       1 rule             3 rules
      (E-015)            (E-015)

                                ────► +100% to +300% lift over legacy
                                       comes from the tool_result extension
```

**Reading**: the load-bearing change is **B over A** (adding
`tool_result`). Adding text (C over B) helps in sparse mode but not in
hinted (text restates what tool_result already had). Session-stateful
(D over C) gives equivalent total activations but produces per-turn
deltas for live dispatch.

---

## 4. End-to-end architecture stack

Reference diagram for the paper's system section.

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                       USER / TASK PROMPT                            │
   │   "Analyze GOES-16 weather satellite data..."                       │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                    Anthropic Haiku 4.5 (Azure Foundry)              │
   │   Generates: thinking + text + tool_use   blocks per turn          │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │ streaming SSE events
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │            AnthropicClient.stream() / StreamingResponse             │
   │   (src/agentstage/client/anthropic.py)                              │
   │   - tees thinking_delta chunks to detector                         │
   │   - records tool_use args                                           │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
   ┌─────────────────┐ ┌────────────────────┐ ┌─────────────────┐
   │ SessionDetector│ │  Tool execution    │ │ Per-turn        │
   │ (multi-turn)    │ │  (list_dir,        │ │ recorder        │
   │                 │ │   open_file)       │ │ → stream.jsonl  │
   │ scans:          │ │                    │ │ → tool_use.jsonl│
   │  ▪ thinking     │ │ sandboxed to:      │ │ → tool_result.. │
   │  ▪ text         │ │  /tmp/s3-noaa-...  │ │ → thinking.txt  │
   │  ▪ tool_result  │ │  /mnt/.../agentio  │ └─────────────────┘
   │                 │ │  /dev/shm/agnst    │
   │ rule library    │ │                    │
   │  (frozen v1,    │ └────────┬───────────┘
   │   sha256 hash)  │          │
   └────────┬────────┘          │
            │ DataHint          │ tool_result content
            │ (tier-1)          │
            ▼                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                          Stager (in-process)                        │
   │   src/agentstage/stager/daemon.py                                   │
   │   - ThreadPoolExecutor                                              │
   │   - prefetch(DataHint) → copies cold → hot via atomic rename        │
   │   - LRU eviction by atime                                           │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
   ┌─────────────────────┐ ┌────────────────┐ ┌─────────────────────┐
   │  COLD TIER          │ │  HOT TIER      │ │  LD_PRELOAD SHIM    │
   │  /tmp/s3-noaa-      │ │  /dev/shm/     │ │  libagentstage_     │
   │   goes16/ABI-L2-... │ │  agentstage_   │ │  shim.so            │
   │  (mountpoint-s3)    │ │  path_b/...    │ │                     │
   │                     │ │                │ │  intercepts:        │
   │  S3: NOAA GOES-16   │ │  tmpfs RAM     │ │   open* stat*       │
   │  bucket             │ │                │ │   access creat      │
   │                     │ │                │ │  → redirects to     │
   │  ~754 ms first-byte │ │  ~0.04 ms      │ │     hot tier IF     │
   └─────────────────────┘ └────────────────┘ │     staged          │
                                              └──────────┬──────────┘
                                                         │
                                                         ▼
                                              ┌────────────────────┐
                                              │  Agent's open()    │
                                              │  call hits hot     │
                                              │  copy transparently│
                                              └────────────────────┘
```

**Reading**: the data plane (right side) operates by filesystem-as-IPC.
The stager places a file at `hot_root/<canonical_path>`; the shim
intercepts the agent's `open()` and rewrites it to the hot copy if
present. The control plane (left side) is the detector extracting
intent from the LLM stream and emitting `DataHint`s to the stager.

---

## 5. The rule-library mismatch (E-015 finding)

The specific architectural failure mode we discovered, and the
forward-work that fixes it.

```
                    HINTED MODE (E-010)                  SPARSE MODE (E-015)
                    ════════════════════                 ════════════════════

   Task prompt          Task prompt
   ───────────          ───────────
   "bands 08,           "Analyze GOES-16
    09, 10..."          data..."
   "C08 first"          (no band hint)
        │                    │
        ▼                    ▼
   ┌─────────┐          ┌─────────┐
   │ THINKING│          │ THINKING│
   │ "C08..."│          │"explore"│
   └────┬────┘          └────┬────┘
        │                    │
        ▼                    ▼
   ┌─────────────────────────────────────┐
   │      STATIC RULE LIBRARY (v1)        │  ◄── SAME rule library
   │                                      │       in both regimes
   │   first_inspect → target = "C08.nc"  │
   │   band_08        → target = "C08.*"  │  ◄── hard-coded to Band 08
   │   band_09        → target = "C09.*"  │
   │   ...                                │
   └────────┬─────────────────┬──────────┘
            │                 │
   First-turn dispatch:       │
            ▼                 ▼
        Band 08 ✓         Band 08 ✓     ◄── detector detects SAME file
            │                 │              in both regimes (static rules)
            │                 │
   ════════════════════ DIVERGENCE ════════════════════
            │                 │
            ▼                 ▼
        ┌───────┐         ┌───────┐
        │ AGENT │         │ AGENT │
        │ opens │         │ opens │
        │ C08 ✓ │         │ C02 ✗ │       ◄── agents pick DIFFERENT bands
        └───────┘         └───────┘            under different prompts
            │                 │
            ▼                 ▼
       ╔════════╗        ╔════════╗
       ║  HIT   ║        ║  MISS  ║
       ║ 19213× ║        ║ wasted ║       ◄── architecture works
       ╚════════╝        ╚════════╝            but rule library is brittle

   ┌─────────────────────────────────────────────────────────────┐
   │  Root cause: rule library was hand-tuned on hinted prompts. │
   │              Static targets don't adapt to agent behavior.  │
   │                                                              │
   │  Fix path:   learned detector (AGENTSTAGE.md §12 future    │
   │              work) trained on (thinking, tool_result,       │
   │              accessed-file) tuples from multi-turn corpora. │
   └─────────────────────────────────────────────────────────────┘
```

**Reading**: this is the threats-to-validity diagram. The architecture
isn't broken; the static rule library is. Future work — learned
detector — fixes this by adapting targets from observed
(thinking, tool_result, accessed-file) tuples. Our multi-turn
capture corpus (E-011 / E-014 / E-015) is the training set.
