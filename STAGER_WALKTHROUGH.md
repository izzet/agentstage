# Stager Walkthrough

Tutorial-style explanation of how the AgentStage staging daemon and
LD_PRELOAD shim actually work in practice.

Three docs in this set:
- This file — the explainer (read first when coming back cold)
- [`STAGER_DESIGN.md`](STAGER_DESIGN.md) — the implementation contract
- [`STAGER_VERIFICATION.md`](STAGER_VERIFICATION.md) — what we tested
  pre-Day-5, what we measured (168-306× speedup, eviction confirmed),
  and which bugs surfaced during testing.

For implementation contracts, go to the spec. For "did it actually
work?" go to the verification doc. Code lives in
`src/agentstage/stager/`.

## Contents

- [The problem](#the-problem)
- [End-to-end architecture](#end-to-end-architecture)
- [Walkthrough — concrete timeline on aiob_107](#walkthrough--concrete-timeline-on-aiob_107)
- [Component-by-component detail](#component-by-component-detail)
  - [1. The LD_PRELOAD shim](#1-the-ld_preload-shim-the-redirect-layer)
  - [2. The stager](#2-the-stager-the-copy-layer)
  - [3. The client library](#3-the-client-library-where-detections-come-from)
  - [4. Filesystem-as-IPC](#4-filesystem-as-ipc)
- [What happens in failure cases](#what-happens-in-failure-cases)
- [Why this design](#why-this-design)

## The problem

LLM agents make tool calls that read files. Each cold-tier first-read
takes 10-500 ms (NFS round-trip + first-byte latency). A 30-turn agent
run cold-reading even small files burns 0.3-15 seconds of pure wait.
Meanwhile, the LLM's thinking phase before each tool call is ~5-10
seconds of *wall-clock slack* on the storage side (`AGENTSTAGE.md` §6.1
documents median 6.9 s, max 14 s on Anthropic/Gemini, 248 s on
DeepSeek-R1).

The stager's job: turn that slack into pre-fetched data so the agent's
first read hits local NVMe (~10 µs) instead of cold storage (~100 ms)
— typically a 10,000× speedup on the first-read latency.

## End-to-end architecture

```
═════════════════════════════════════════════════════════════════════════════
   Single agent process — everything lives here
═════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ┌──────────────────────────┐                                       │
  │  │  Benchmark harness       │       1. agent emits a tool call:     │
  │  │  (AIOB / SAB / Krama)    │          open("/cold/.../foo.nc")     │
  │  │                          │                                       │
  │  │   for turn in range(15): │                                       │
  │  │     resp = client.chat   │                                       │
  │  │     for tool in resp:    │                                       │
  │  │       run_tool(tool) ────┼───── 2. Python's open() →             │
  │  │                          │                glibc openat() →       │
  │  └──────────────────────────┘                                       │
  │             │                                                       │
  │             │ stream                                                │
  │             ▼                                                       │
  │  ┌──────────────────────────┐                                       │
  │  │  AgentStageClient        │   intercepts SSE stream from LLM      │
  │  │  (wraps anthropic /      │   feeds each thinking_delta chunk     │
  │  │   openai / google-genai) │   to the detector                    │
  │  │                          │                                       │
  │  │   ┌─────────────────┐    │                                       │
  │  │   │   Detector     │    │                                       │
  │  │   │   (rule engine) │    │  fires rules on streaming text,       │
  │  │   │                 │    │  emits DataHint(files, tier, ms)      │
  │  │   └────────┬────────┘    │                                       │
  │  │            │             │                                       │
  │  │            ▼             │                                       │
  │  │   ┌─────────────────┐    │                                       │
  │  │   │   Stager        │    │  ThreadPoolExecutor(4 workers)        │
  │  │   │                 │    │                                       │
  │  │   │  prefetch(hint) │────┼──── 3. submit copy job to threadpool  │
  │  │   │  ↓              │    │                                       │
  │  │   │  _stage(path)   │    │                                       │
  │  │   │    cold → tmp   │    │                                       │
  │  │   │    rename → hot │    │                                       │
  │  │   │    log report   │    │                                       │
  │  │   └─────────────────┘    │                                       │
  │  └────────────┬─────────────┘                                       │
  │               │                                                     │
  │      writes files                            ╔═══════════════════╗  │
  │      (atomic rename)                         ║                   ║  │
  │               ▼                              ║   LD_PRELOAD      ║  │
  │  ┌───────────────────────────────────────────╫──── chain ───────╗║  │
  │  │  Filesystem                               ║                  ║║  │
  │  │  ─────────                                ║   libdftracer.so ║║  │
  │  │                                           ║       ↓          ║║  │
  │  │  /scratch/agentstage/         (hot tier)  ║   agentstage.so  ║║  │
  │  │    └── mnt/.../foo.nc       (mirrored)    ║       ↓          ║║  │
  │  │                                           ║   libc → kernel  ║║  │
  │  │  /mnt/datasets/.../foo.nc     (cold tier) ║                  ║║  │
  │  │                                           ╚══════════════════╝║  │
  │  └────────────────────────────────────────────────┬───────────────  │
  │                                                   │                 │
  │                                  4. shim sees openat(/cold/foo.nc)  │
  │                                     → tries openat(/scratch/foo.nc) │
  │                                     → succeeds (stager finished)    │
  │                                     → returns hot fd to libc        │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

The agent never knows the stager exists. The LLM never knows the
detector exists. Both layers are completely transparent.

## Walkthrough — concrete timeline on aiob_107

aiob_107 is the GOES meteorology workload: workspace has 6,042 NetCDF
files (18 GB total), agent's first tool call reads just one (3 MB). PoC
measured 11,328 ms of slack between first thinking chunk and first tool
dispatch (`AGENTSTAGE.md` §6.3). Here's what happens in those 11 seconds:

```
T=0      ms   Benchmark harness sends task prompt to LLM via
              client.messages.create(stream=True)

T=12     ms   SSE chunk arrives:   {"type":"content_block_start",
                                    "content_block":{"type":"thinking"}}
              AgentStageClient routes the chunk to the caller AND
              forks a copy to the Detector.

T=20     ms   SSE chunk:   {"type":"thinking_delta","delta":"Let me look "}
T=24     ms   SSE chunk:   {"type":"thinking_delta","delta":"at the data..."}
T=180    ms   SSE chunk:   {"type":"thinking_delta","delta":"...6042 NetCDFs"}
              Detector's broad_all_files rule fires — emits tier-3 hint
              covering all 6042 paths. Stager starts copying low-priority.

T=1200   ms   SSE chunk:   {"type":"thinking_delta",
                            "delta":"...I'll start by inspecting "
                                    "scene_2024-001-001.nc"}
              Detector's first_inspect_goes rule fires.
              Detector emits DataHint(
                  detected_files=["/cold/.../scene_2024-001-001.nc"],
                  tier=1,
                  fired_at_ms=1200,
                  rule_id="first_inspect_goes",
                  byte_estimate=3_145_728,
              )

T=1201   ms   Client lib calls stager.prefetch(hint).
              Stager checks self.in_flight — not present, submits to executor.
              executor.submit(_stage, "/cold/.../scene_2024-001-001.nc", hint)

T=1202   ms   Worker thread starts _stage():
                hot_path = "/scratch/agentstage/cold/.../scene_2024-001-001.nc"
                hot_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = hot_path + ".tmp.12345.140234"
                shutil.copy("/cold/.../scene_2024-001-001.nc", tmp)
              shutil.copy reads from NFS at ~50 MB/s for 3 MB → ~60 ms

T=1262   ms   tmp is fully written. os.rename(tmp, hot_path) — atomic.
              hot_path now exists with full content. Stager records:
                StageResult(cold_path="...", hot_path="...",
                            size_bytes=3145728, fetch_ms=60.3,
                            tier=1, t_detected=1200, t_completed=1261.3)

T=1262   ms onwards   The hot file is ready. AgentStage is now waiting
              for the agent to open it.

T=1500-8000 ms       LLM continues thinking (planning the analysis steps,
                     considering the data structure, etc.).
                     More chunks stream; more rules may fire (lower-tier
                     hints). Stager copies low-priority files in
                     background.

T=8500   ms   SSE chunk:  {"type":"content_block_start",
                           "content_block":{"type":"tool_use",
                                            "name":"open_file",
                                            "input":{"path":"scene_2024-001-001.nc"}}}
              Tool call dispatched to the harness.

T=8520   ms   Harness's tool implementation calls:
                with open("/cold/.../scene_2024-001-001.nc", "rb") as f:
                  data = f.read()

T=8520.0 ms   Python's open() invokes the C-level os.open() →
              glibc openat(AT_FDCWD, "/cold/.../scene_2024-001-001.nc",
                           O_RDONLY, 0).

T=8520.0 ms   LD_PRELOAD chain enters:
              libdftracer.so's openat wrapper runs first → records
              event: openat(cold_path) — this is the agent's INTENT,
              what DFTracer needs for io_report.json ground truth.
              Then it calls the next openat in the chain.

T=8520.0 ms   libagentstage_shim.so's openat wrapper runs:
                _resolve_absolute(AT_FDCWD, path, abs)
                  → abs = "/cold/.../scene_2024-001-001.nc"
                flags & (O_WRONLY|O_RDWR|...) → false (read-only)
                _under_managed_cold_root(abs) → true
                hot = "/scratch/agentstage/cold/.../scene_2024-001-001.nc"
                fd = _real_openat(AT_FDCWD, hot, O_RDONLY, 0)
                fd ≥ 0 → file exists, return fd.

T=8520.0 ms   libc receives openat(hot_path) request, returns fd.

T=8520.0 ms   Python's f.read() begins reading from the hot file.
              NVMe read at ~3-7 GB/s for 3 MB → ~0.5 ms

T=8520.5 ms   Read complete. Agent has the data.

  RESULT: First-read of foo.nc took 0.5 ms.
  Without AgentStage: would have been 60 ms.
  Saved: 59.5 ms on this one syscall.

  Aggregated over 30 tool calls × 50-200 MB each on a slow cold tier,
  this becomes the multi-second speedup the paper measures (E5).
```

## Component-by-component detail

### 1. The LD_PRELOAD shim (the redirect layer)

Lives at `src/agentstage/stager/shim/agentstage_shim.c`, compiled to
`libagentstage_shim.so`.

When the agent process starts with
`LD_PRELOAD=$DFTRACER:$AGENTSTAGE_SHIM_SO python ...`, the dynamic
linker loads both `.so` files before libc. They register wrappers for
specific syscalls via `dlsym(RTLD_NEXT, "openat")` — meaning their
version of `openat` runs *first*, and they can choose to call the
*next* `openat` in the chain (which eventually reaches glibc).

**What the shim intercepts: only path-taking syscalls.**

| Function | Why intercepted |
|---|---|
| `openat`, `openat2`, `creat` | Primary redirect point. The cold path is rewritten to the hot path here. |
| `statx`, `newfstatat`, `stat`, `lstat`, `fstatat` | Agents stat before open (Python's `os.path.exists`, pandas reading metadata). Stats must match the hot file's size or the agent will allocate the wrong buffer. |
| `access`, `faccessat`, `faccessat2` | Existence checks; must succeed on hot if file is staged. |

**What it does NOT intercept:**

- `read`, `pread`, `pread64`, `readv`, `preadv` — these operate on an
  fd. If openat already returned an fd pointing at the hot file, the
  read hits the hot file by definition.
- `mmap` — also fd-based. Importantly, `madvise(MADV_DONTNEED)` from
  AgentIOBench's cold-cache eviction operates on the hot file's pages,
  which is fine (kernel reloads from hot file on next access).
- `close`, `lseek` — fd-based.

This minimal-set design is a key insight. Most "FS rewriting" shims
intercept dozens of syscalls; we get away with ~10 because the fd is
the natural pivot point.

**Inside the openat wrapper** (pseudocode from §3 of the design doc):

```c
int openat(int dirfd, const char *pathname, int flags, mode_t mode) {
    // 1. Resolve to absolute. Returns false if path is bogus.
    char abs[PATH_MAX];
    if (!_resolve_absolute(dirfd, pathname, abs))
        return _real_openat(dirfd, pathname, flags, mode);

    // 2. Writes always pass through. Never redirect O_WRONLY/O_CREAT.
    if (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND))
        return _real_openat(dirfd, pathname, flags, mode);

    // 3. Only redirect if path is under a managed cold root.
    if (!_under_managed_cold_root(abs))
        return _real_openat(dirfd, pathname, flags, mode);

    // 4. Build hot path: HOT_ROOT + absolute cold path.
    char hot[PATH_MAX];
    snprintf(hot, sizeof(hot), "%s%s", _hot_root, abs);

    // 5. Try hot. If file exists, return its fd.
    int fd = _real_openat(AT_FDCWD, hot, flags, mode);
    if (fd >= 0) { _record_hit(abs); return fd; }
    if (errno != ENOENT) return _real_openat(dirfd, pathname, flags, mode);

    // 6. ENOENT: file might be in-flight. Spin briefly waiting for it.
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (1) {
        nanosleep(&(struct timespec){.tv_nsec = 500000}, NULL);  // 0.5 ms
        fd = _real_openat(AT_FDCWD, hot, flags, mode);
        if (fd >= 0) { _record_late_hit(abs); return fd; }
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed = (now.tv_sec - start.tv_sec) * 1000
                     + (now.tv_nsec - start.tv_nsec) / 1000000;
        if (elapsed >= 20) break;
    }

    // 7. Fall through to cold path.
    _record_miss(abs);
    return _real_openat(dirfd, pathname, flags, mode);
}
```

That's it. The whole shim is roughly 300 lines of C plus a Makefile.

### 2. The stager (the copy layer)

Lives at `src/agentstage/stager/daemon.py`. Despite the filename, it's
NOT a separate process — it runs in-process as a thread pool inside the
agent.

```python
class Stager:
    def __init__(self, hot_root, cold_roots, max_workers=4, capacity=32*GB):
        self.hot_root = hot_root
        self.cold_roots = [p.resolve() for p in cold_roots]
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.in_flight = {}  # cold_path → Future
        self._lock = threading.Lock()

    def prefetch(self, hint: DataHint):
        """Non-blocking. Called by the client lib on each detector firing."""
        for cold_path in hint.detected_files:
            with self._lock:
                if cold_path in self.in_flight:
                    continue  # already staging or staged
                self.in_flight[cold_path] = self.executor.submit(
                    self._stage, cold_path, hint
                )

    def _stage(self, cold_path, hint):
        hot_path = self._hot_for(cold_path)
        if hot_path.exists():
            return StageResult(hit=True, ...)

        size = os.path.getsize(cold_path)
        self._ensure_capacity_for(size)  # may trigger LRU eviction

        tmp = hot_path.with_suffix(
            hot_path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
        )
        hot_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        shutil.copy(cold_path, tmp)
        tmp.rename(hot_path)  # ATOMIC. POSIX rename(2).
        elapsed = (time.monotonic() - start) * 1000

        return StageResult(
            cold_path=cold_path, hot_path=hot_path, size_bytes=size,
            fetch_ms=elapsed, tier=hint.tier, rule_id=hint.rule_id,
            t_detected_ms=hint.fired_at_ms, t_completed_ms=...,
        )
```

Two important properties:

**Atomicity by rename.** The file at `hot_path` either does not exist
(rename hasn't happened) or contains the complete final bytes (rename
completed). There is no intermediate state where reading `hot_path`
would return a truncated file. This is why we need no separate "is it
ready?" IPC mechanism.

**Idempotency.** Calling `prefetch(hint)` repeatedly with the same
`cold_path` is a no-op after the first call — the `in_flight` dict
tracks pending and completed stages. The detector can fire the same
rule on multiple thinking chunks without us double-fetching.

### 3. The client library (where detections come from)

The client wraps the LLM SDK (Anthropic / OpenAI / Google / raw HTTP)
and tees the stream. Pseudocode:

```python
class AnthropicClient:
    def __init__(self, api_key, workspace, stager, rule_library_version):
        self._real_client = anthropic.Anthropic(api_key=api_key)
        self._detector = Detector(workspace, rule_library_version)
        self._stager = stager
        self._data_hints = []

    def messages_create(self, **kwargs):
        # Wraps the underlying streaming API
        for event in self._real_client.messages.create(stream=True, **kwargs):
            # 1. Tee to the caller — they see the original stream unchanged
            yield event

            # 2. Tee to the detector
            if event.type == "content_block_delta" and event.delta.type == "thinking_delta":
                hints = self._detector.feed(event.delta.thinking, event.t_ms)
                for hint in hints:
                    self._data_hints.append(hint)
                    if self._stager is not None:
                        self._stager.prefetch(hint)
```

The key invariant: **the caller's stream is byte-identical to what the
underlying SDK would have returned.** AgentStage adds zero observable
behavior to the LLM call from the caller's perspective. Detector +
stager work is purely a side effect.

### 4. Filesystem-as-IPC

The stager and shim never talk to each other directly. They
communicate only via the filesystem:

| Stager action | Shim observation |
|---|---|
| `tmp = hot_path.tmp...` (in progress) | `openat(hot_path)` returns ENOENT |
| `os.rename(tmp, hot_path)` (complete) | `openat(hot_path)` succeeds |

This is elegant because:

- No socket lifecycle to manage
- No queue overflow conditions
- No daemon-crash-mid-run failure modes
- The shim doesn't care if the stager is even running — it just
  checks the filesystem

The downside: small read latency cost on every openat (one extra
stat-equivalent for hot path lookup). On Linux this is ~3-5 µs from
page cache, dominated by the actual openat. Negligible.

## What happens in failure cases

### Scenario A: Detection wrong (agent reads different file)

Detector emitted `["scene_001.nc"]`, agent actually opens
`"scene_002.nc"`.

```
T=1200ms   Stager copies /cold/scene_001.nc → /hot/scene_001.nc
T=8500ms   Agent calls open("/cold/scene_002.nc")
T=8500ms   Shim tries openat(/hot/scene_002.nc) → ENOENT
T=8500ms   Shim spins 20 ms checking again — nothing arrives
T=8520ms   Shim falls through to openat(/cold/scene_002.nc) → succeeds
T=8580ms   Read completes from cold (60 ms NFS first-byte for 3 MB)
```

Cost of being wrong: **20 ms retry-spin penalty**. Tier-1 byte
overfetch increased (we wasted 3 MB on `scene_001.nc`) but didn't break
correctness. The wrong detected file sits in the hot tier until LRU
eviction.

### Scenario B: Race (file in flight when openat fires)

Detector fires 9.99 seconds before tool call. Stager mid-copy when
openat happens.

```
T=8500ms     Stager 50% done copying scene_001.nc
T=8500ms     Agent calls open("/cold/scene_001.nc")
T=8500ms     Shim tries openat(/hot/scene_001.nc) → ENOENT (rename hasn't happened yet)
T=8500.5ms   Shim spins, checks again → still ENOENT
T=8510ms     Stager rename(tmp, hot) completes
T=8510ms     Shim's next check during spin: openat(/hot/scene_001.nc) → succeeds!
T=8510ms     Returns hot fd (10 ms late hit)
```

Cost of the race: **10 ms penalty** (vs 0 ms on hit, 60 ms on cold
fallthrough). Still better than cold by 50 ms. The 20 ms retry-spin
budget is sized to catch races up to that long.

### Scenario C: Capacity pressure (hot tier full)

Stager has filled 32 GB hot tier. New prefetch arrives for an 8 GB NWB
file.

```python
_ensure_capacity_for(8 * GB):
  current_used = 32 GB (scan or counter)
  needed = 8 GB
  walk hot dir, sort by atime ascending
  for path, size in oldest_first:
    if path in self.in_flight_paths:
      continue  # protect in-flight files from eviction
    path.unlink()
    freed += size
    if freed >= 8 GB: break
  if freed < 8 GB: raise StagerOutOfSpace
```

Files protected: anything currently being staged (in
`self.in_flight_paths`). Files evicted: the LRU-oldest files no longer
in use. For the eScience paper, capacity is sized so eviction never
triggers on a single workload — this scenario is the multi-agent or
multi-task case, which is future work.

### Scenario D: Agent writes a file (intermediate result)

Agent's tool call: `open("/cold/.../output/result.json", "w")`.

```
T=12000ms   Agent calls open(path, "w")
T=12000ms   Shim checks flags: O_WRONLY | O_CREAT — write detected
T=12000ms   Shim immediately passes through to libc: openat(cold path)
T=12000ms   Write goes to the actual cold-tier output dir, NOT the hot mirror
```

The hot tier is read-only from the agent's perspective. Intermediate
writes always land where the benchmark expects them (e.g., AIOB's
`task.output_fname` directory) — which is what the soft-stop predicate
watches for.

### Scenario E: DFTracer + shim interaction

Both are LD_PRELOAD'd. Order:
`LD_PRELOAD="$DFTRACER:$AGENTSTAGE_SHIM"`.

```
Agent: openat("/cold/.../foo.nc", O_RDONLY)
  ↓
libdftracer.so::openat:
  records event { syscall: "openat", path: "/cold/.../foo.nc", t: 8500ms }
  calls next_openat (= agentstage's)
  ↓
libagentstage.so::openat:
  rewrites path to "/scratch/.../foo.nc"
  calls next_openat (= libc's)
  ↓
libc → kernel
  ↓
returns fd pointing at /scratch/.../foo.nc

io_report.json records the COLD path (agent's intent) — used by paper_evals
for ground-truth scoring.
staging_report.json records the redirect to HOT — used for performance
measurement.
```

DFTracer sees the agent's intent (what file it MEANT to read). The
shim redirects to the staged copy. The agent sees the data either way.
We get both signals.

## Why this design

Total cost: ~300 lines of C for the shim + ~200 lines of Python for the
stager + 32 GB of scratch space = single-digit-millisecond first-read
latency on files that would otherwise take 60-500 ms cold.

The cleverness isn't in any one component — it's in keeping the
components decoupled. The shim doesn't know the detector exists. The
detector doesn't know the shim exists. They synchronize through the
most boring possible primitive: **a file exists or it doesn't.**

This decoupling buys us:

- **Testability.** Each layer can be tested in isolation: shim with
  hand-placed hot files (no stager); stager with mock cold paths (no
  shim); detector with recorded SSE streams (no LLM).
- **Failure isolation.** If the stager crashes, the shim's ENOENT
  fallthrough means the agent still gets correct (slow) reads. If the
  detector produces nonsense, the stager just copies useless files —
  no correctness violation. If the shim has a bug, the agent reads
  cold paths directly. There's no scenario where a component failure
  produces a wrong answer.
- **Composability.** The HTTP proxy (for SWE-bench-Docker-style
  isolated harnesses) is a thin wrapper around the client library's
  detector + stager. Same components, different transport.
- **Reviewer-defensibility.** "Why isn't there a daemon?" — because we
  don't need one. "Why isn't there a custom protocol?" — because we
  don't need one. "What if the stager is slow?" — bounded by the 20 ms
  retry spin; falls through to cold otherwise. Every design choice has
  a one-sentence justification.

The bet AgentStage makes is that LLM thinking time is the right slot
to do this work — that the slack is reliable, that the detector's
hit rate is high enough to be useful, and that prestaging from cold to
local NVMe is the right transformation. The PoC numbers
(`AGENTSTAGE.md` §6) say yes on all three. The stager is the
infrastructure that turns those numbers into real wall-clock speedup
the paper can measure (E5, E6, E9, E10).
