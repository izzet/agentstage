# Diagrams — Current Architecture + Reasoning Overlap

Updated 2026-05-22 to reflect the full system after E-021 (dynamic
prior enrichment), E-026 (Gemini cross-vendor), E-028 (end-to-end
demo), E-029 (decompression-staging), and the `fopen`/`fopen64` shim
interception fix.

---

## 1. Architecture — control plane + data plane

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           CONTROL PLANE (per agent session)                 │
│                                                                             │
│   ┌──────────────────────┐                                                 │
│   │   USER PROMPT        │  hinted: explicit file hints                    │
│   │   (task description) │  sparse: I/O hints stripped (Regime B)          │
│   └─────────┬────────────┘                                                 │
│             │  + pathful-prompt clause (V4: "copy paths verbatim from      │
│             │    list_dir into a NEXT_FILES block")                        │
│             ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │  LLM (Anthropic Haiku 4.5 OR Gemini 2.5 Flash, multi-vendor)     │    │
│   │  streams: thinking_delta, text_delta, tool_use, signature_delta │    │
│   └─────────┬────────────────────────────────────────────────────────┘    │
│             │ events                                                       │
│             ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │            SessionDetector (multi-turn, stateful)                 │    │
│   │                                                                   │    │
│   │  Rule library                                       Literal-path  │    │
│   │  ┌─────────────────────────────┐                   ┌────────────┐│    │
│   │  │ hand-tuned OR auto-generated│                   │ hot_path_  ││    │
│   │  │ (E-019: ≥97% recall vs hand)│                   │  scan      ││    │
│   │  │ scans:                       │                   │ scans:     ││    │
│   │  │   thinking + text +          │ ───────►          │ same blks  ││    │
│   │  │   tool_result                │                   │ + dynamic  ││    │
│   │  └─────────────────────────────┘                   │ enrichment ││    │
│   │              │                                      │ from       ││    │
│   │              │ new_acts                             │ list_dir   ││    │
│   │              ▼                                      └─────┬──────┘│    │
│   │      tier-1 (≤10 files)                                  │       │    │
│   │      tier-2 (≤200)                                       │       │    │
│   │      tier-3 (>200, skipped in live)                      │       │    │
│   └──────────┬───────────────────────────────────────────────┼───────┘    │
│              │ DataHint                                       │            │
│              │ (paths, tier, fired_at_ms, rule_id)            │            │
│              └────────────────┬──────────────────────────────┘            │
│                               │  one stream                                │
└───────────────────────────────┼────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              DATA PLANE                                     │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                       Stager (in-process)                         │    │
│   │   ThreadPoolExecutor + atomic rename + LRU eviction               │    │
│   │                                                                   │    │
│   │   transform = none       ──► byte-identical copy (E-028)         │    │
│   │   transform = decompress ──► transcode to uncompressed NetCDF    │    │
│   │                              (E-029, new in this session)        │    │
│   └─────────┬─────────────────────────────────────────────────┬─────┘    │
│             │ read                                              │ write    │
│             ▼                                                   ▼          │
│   ┌────────────────────────┐                       ┌─────────────────────┐│
│   │  COLD TIER             │                       │  HOT TIER (tmpfs)   ││
│   │  (auto-discovered or   │                       │  /dev/shm/agentstage││
│   │   env-configured)      │                       │                     ││
│   │                        │                       │  mirror layout:     ││
│   │  • local NFS/XFS       │                       │  hot_root +         ││
│   │    ~6 ms first-byte    │                       │  abs(cold_path)     ││
│   │    (warm NVMe + SSD)   │                       │                     ││
│   │                        │                       │  ~0.05 ms hot       ││
│   │  • S3 via mountpoint   │                       │  first-byte         ││
│   │    ~750 ms first-byte  │                       │                     ││
│   │    + many small GETs   │                       │  files may be       ││
│   │    per HDF5 file       │                       │  byte-identical OR  ││
│   │                        │                       │  decompressed       ││
│   └───────┬────────────────┘                       └──────────┬──────────┘│
│           │                                                    │           │
│           │ (cold reads only on shim MISS — falls through)     │ hot reads │
│           │                                                    │           │
│           ▼                                                    ▼           │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │              LD_PRELOAD shim  (libagentstage_shim.so)             │   │
│   │                                                                   │   │
│   │   Intercepts:                                                     │   │
│   │     open, open64, openat, openat64                                │   │
│   │     fopen, fopen64       ◄── ADDED in this session (E-028 debug)  │   │
│   │     stat, lstat, fstatat, __xstat family                          │   │
│   │     access, faccessat, creat, creat64                             │   │
│   │                                                                   │   │
│   │   For each open of /COLD/path/file:                               │   │
│   │     1. abs path under managed cold-root?                          │   │
│   │     2. compute hot = HOT_ROOT + abs_cold                          │   │
│   │     3. spin-wait (20 ms) for hot copy to appear                   │   │
│   │     4. if hot exists: open hot, return its fd                     │   │
│   │     5. else: fall through to cold                                 │   │
│   └────────────────────────┬─────────────────────────────────────────┘   │
│                            │ fd points at HOT copy (transparent)         │
│                            ▼                                              │
│              ┌─────────────────────────────────┐                          │
│              │ Agent's tool execution:         │                          │
│              │   • list_dir (turn 0-11)        │                          │
│              │   • open_file (small inspection)│                          │
│              │   • run_shell_command:          │                          │
│              │       python3 process_data.py   │ ◄── the big I/O turn     │
│              │       (turn 12, 6042 file opens │                          │
│              │        via netCDF4/HDF5)        │                          │
│              └─────────────────────────────────┘                          │
└────────────────────────────────────────────────────────────────────────────┘
```

The shim is workload-agnostic. The detector + stager handle the
intent → prefetch mapping. The transcoder (E-029) is an OPTIONAL
data-preparation step that runs at staging time.

---

## 2. What overlaps with reasoning — the slack-window timeline

This is the central claim. Operations to the LEFT of the dashed line
happen on the **critical path** (the user waits). Operations to the
RIGHT happen **inside the agent's natural slack window** (overlapped
with LLM thinking + tool execution that has to happen anyway).

```
            critical-path                               overlapped (hidden)
            ┌──────────────────────────────────────────────────────────────┐
            │                                                              │
            │  WITHOUT AgentStage                                          │
            │  ──────────────────                                          │
            │  ┌── turn 0 ──┐┌─ ... ─┐┌── turn 12: python3 script ──────┐  │
            │  │ LLM think  ││       ││ for f in 6042 files:            │  │
            │  │ + list_dir ││       ││   nc.Dataset(f, 'r')            │  │
            │  │            ││       ││   var[:] ← COLD READ + ZLIB     │  │
            │  └────────────┘└───────┘│   extract box, append           │  │
            │                          │ ...                              │  │
            │                          │ matplotlib, write CSV/PNG       │  │
            │                          └─── ~677 s on local NFS,         │  │
            │                                ~75 min on S3 (projected)   │  │
            │                                                              │
            │  ----------- agent done. user sees output. ----------------  │
            │                                                              │
            │  WITH AgentStage                                             │
            │  ───────────────                                             │
            │  ┌── turn 0 ──┐                                              │
            │  │ LLM think  │  ◄── detector fires rules on streaming       │
            │  │ + list_dir │       thinking_delta                         │
            │  └────┬───────┘                                              │
            │       │  DataHint(tier-1: sample file)                       │
            │       ▼                                                      │
            │       ▼  ▶ ▶ ▶  Stager prefetch  ▶ ▶ ▶  Transcode            │
            │                  (cold → hot)     (zlib → uncompressed)      │
            │                  in 8-worker pool, OVERLAPPED with turn 0    │
            │                                                              │
            │  ┌── turn 1-11 ──────────────────────┐                       │
            │  │ LLM think + list_dir/cat tool     │ ◄── more rules fire   │
            │  │ output reveals more file paths    │     dynamic prior     │
            │  │                                   │     enrichment        │
            │  └──┬────────────────────────────────┘                       │
            │     │  more DataHints (discovered paths)                     │
            │     ▼                                                        │
            │     ▼  ▶ ▶ ▶  Stager + transcode races ahead  ▶ ▶ ▶          │
            │                                                              │
            │  ┌── turn 12: python3 script ──┐                             │
            │  │ for f in 6042 files:        │ ◄── shim intercepts each   │
            │  │   nc.Dataset(f, 'r')        │     open(); fd → tmpfs hot │
            │  │                             │     copy (already           │
            │  │   var[:] ← HOT READ +       │     decompressed if E-029)  │
            │  │   NO ZLIB                   │                             │
            │  │   extract, append           │                             │
            │  └─── ~7 s on staged S3,       │                             │
            │       ~5.6 s with decompress   │                             │
            │       (29× faster than cold)   │                             │
            │                                                              │
            │  ----------- agent done. user sees output. ----------------  │
            │                                                              │
            └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            User-visible savings = (cold script time) - (hot script time)
                                 = 169 s - 5.6 s
                                 = 163 s SAVED PER TASK on S3 (1-hour scope)
                                 ≈ 29× session speedup
            And the staging+transcode work happens in parallel with
            turns 0-11 that the user was waiting for ANYWAY.
```

### Which staging operation overlaps with what

| Phase | Critical path                | Overlapped with                              |
|---|---|---|
| Detection (rules + literal scan) | None | LLM streaming (sub-ms per chunk) |
| Cold → hot copy                  | None | Agent turns 0-11 (discovery slack) |
| Decompression transcode          | None | Same — adds ~CPU during staging window |
| Stage-ahead during script run    | None | The script's I/O loop (race ahead of consumer) |
| Shim redirect (per `open()`)     | μs   | (μs overhead is part of the hot read itself) |

The **only** thing on the critical path is the agent's reasoning + the
hot read itself. Everything else hides in the slack window.

### Why "fopen" matters here

A subtle but load-bearing detail surfaced in E-028 debugging: HDF5/
netCDF-C **format-sniff every file with `fopen64()` before HDF5's
`open()` reader takes over**. Without intercepting `fopen` in the
shim, that sniff goes to the cold tier (one cold-S3 first-byte per
file ≈ 2 s × 36 files = 72 s of unintended critical-path I/O). After
fixing the shim to intercept `fopen`/`fopen64`, the staged S3 run
dropped from 57 s to 7 s — uncovering the real 23× plain-staging
speedup.

The lesson: **anything the agent's code uses on the critical path
must go through the shim**, including stdio (`fopen`), POSIX
(`open`/`openat`), stat (`__xstat`), and access (`access`). Missing
any one of them leaves cold I/O on the critical path.

---

## 3. Detection layers — what catches what

```
                            LLM event stream
                                   │
                                   ▼
        ┌──────────────────────────┴──────────────────────────┐
        │                                                       │
        ▼                                                       ▼
  ┌──────────────┐                                       ┌──────────────┐
  │ Rule library │                                       │ hot_path_scan│
  │              │                                       │ (literal     │
  │ regex on:    │                                       │  substring)  │
  │  thinking    │                                       │              │
  │  text        │                                       │ scans same   │
  │  tool_result │                                       │ blocks for   │
  │              │                                       │ any path in  │
  │ → semantic   │                                       │ the prior    │
  │   class      │                                       │              │
  │   (band_08,  │                                       │ workload     │
  │    sample_X) │                                       │ -agnostic    │
  └──────┬───────┘                                       └──────┬───────┘
         │                                                      │
         │ target_keys → prior[key]                              │
         │                                                      │
         ▼                                                      ▼
  tier-1/2/3 file set                                  concrete file path
  (5 - 6042 files)                                     (1 file)
         │                                                      │
         └──────────────────────┬───────────────────────────────┘
                                ▼
                          DataHint → Stager
                                ▲
                                │
                  ┌─────────────┴──────────────┐
                  │  workspace prior            │
                  │  (auto- or hand-curated)    │
                  │                             │
                  │  + DYNAMIC ENRICHMENT       │
                  │    paths from list_dir      │
                  │    tool_results added on    │
                  │    the fly (E-021)          │
                  └─────────────────────────────┘
```

Two complementary detectors. Rules handle "agent thinks about a
semantic class" (band_08, NWB files). Literal-path scan handles
"agent writes a concrete path". Dynamic enrichment closes the gap
when the agent's chosen file isn't in the original prior.

---

## 4. Current speedup picture

After all of this session's work, the verified per-tier numbers
(n=3 reps each, day-122-hour-00 scope = 36 files, verified cold
cache via posix_fadvise + mincore residency check):

| Cold tier              | Baseline       | Plain staging (E-028) | Decompression staging (E-029) |
|---|---:|---:|---:|
| Local NVMe XFS (warm)  | ~6.5 s         | 5.5 s — **1.2×**       | 4.2 s — **1.5×**               |
| S3 (mountpoint-s3)     | ~169 s         | 7.2 s — **23.4×**      | 5.6 s — **29.1×**              |
| Throttled 10 MB/s (E-007 projection) | – | ~12× — projected       | not yet measured              |

**Honest reading**: the architecture's value scales with cold-tier
slowness. Fast local NVMe XFS already has near-RAM first-byte (~6 ms),
so staging buys only ~1.2×. S3 cold (~750 ms first-byte, plus many
small GETs per HDF5 file) is where staging earns its keep. Real-world
on-prem NFS/Lustre sits somewhere between, depending on network and
filesystem; E-007's throttled-sweep brackets the middle ground.
