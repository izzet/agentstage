# Stager Design

Engineering design for the AgentStage staging daemon and LD_PRELOAD
path-rewriting shim. Locks the implementation contract before the Day-5
build (T29-T34 in `TASKS.md`).

> **Companions:**
> - [`STAGER_WALKTHROUGH.md`](STAGER_WALKTHROUGH.md) — tutorial-style
>   explainer with concrete timelines, scenario walkthroughs, and
>   component-by-component reasoning. Read first if coming back cold.
> - [`STAGER_VERIFICATION.md`](STAGER_VERIFICATION.md) — what we tested
>   pre-Day-5, what we measured (168-306× speedup, eviction confirmed),
>   and which bugs surfaced during testing.
> Read this doc here for the implementation contract.

Authoritative research context: `AGENTSTAGE.md` §4.3 (staging daemon),
§11.5 (E5 / E7 evaluations), §11.9 (risk register — stager is the
critical-path component).

Decisions locked 2026-05-20 (T14).

## 1. Goals and non-goals

### Goals

- **G1.** Decouple application-data fetch from tool execution. When the
  client library emits a `DataHint` from streaming-thinking intent, copy
  the detected files from cold tier (NFS / PFS / object store) to a
  local-NVMe hot tier *before* the agent's tool call dispatches its
  `openat`.
- **G2.** Transparent to the agent code. No modifications to agent
  scripts, benchmark harnesses, or tool implementations. The shim does
  the redirection at the libc boundary.
- **G3.** Honest fallback when detection is wrong. If a file isn't
  staged when the agent opens it, fall through to the cold path with
  bounded extra latency.
- **G4.** Compatible with DFTracer. The agent's "intent" (cold-path
  openat) must be visible to DFTracer for ground-truth capture, while
  the actual syscall hits the hot path for measurement.
- **G5.** Single-agent, single-node. Multi-agent contention (C10 in
  AGENTSTAGE.md §7.2) is explicitly out of scope.

### Non-goals

- **NG1.** Write-back. The stager is a read-path optimization. Writes
  always pass through to the original cold path. We do not buffer or
  redirect writes.
- **NG2.** Cache coherency with mutating cold tiers. We assume the cold
  data is read-only for the duration of a run (the case for all 11
  workloads in `CAMPAIGN.md`). If the cold file mutates mid-run, the
  hot copy goes stale; we do not detect this.
- **NG3.** Cross-process daemon for the eScience paper. The stager runs
  in-process inside the AgentStage client. A persistent daemon for
  multi-agent cache sharing is documented in §10 as future work.
- **NG4.** Replacing the kernel page cache. The hot tier is a curated
  subset; the kernel page cache still operates on cold reads that fall
  through.

## 2. Architecture

```
                        ┌──────────────────────────────────────────┐
                        │ agent process (uv-run python ...)        │
                        │                                          │
   tool call            │   openat("/cold/.../file.nc", O_RDONLY)  │
                        │      │                                   │
   LD_PRELOAD chain  →  │      ├── libdftracer.so   logs intent    │
                        │      │     (cold path; for io_report)    │
                        │      ├── libagentstage_shim.so           │
                        │      │     redirects → hot path          │
                        │      └── libc → kernel                   │
                        │                                          │
                        │   ┌──────────────────────────────────┐   │
                        │   │ AgentStage client library        │   │
                        │   │  (anthropic / openai / gemini)   │   │
                        │   │                                  │   │
                        │   │  ◄── SSE chunks ── upstream LLM  │   │
                        │   │  ── thinking_delta ──► detector │   │
                        │   │  ── DataHint(files,tier) ──► ↓   │   │
                        │   │                              ↓   │   │
                        │   │  ┌────────────────────────────┐  │   │
                        │   │  │ Stager (thread pool, 4 wks)│  │   │
                        │   │  │  cold → hot_path.tmp       │  │   │
                        │   │  │  rename atomically         │  │   │
                        │   │  │  log staging_report.json   │  │   │
                        │   │  │  evict LRU on ENOSPC       │  │   │
                        │   │  └────────────────────────────┘  │   │
                        │   └──────────────────────────────────┘   │
                        └──────────────────────────────────────────┘

                          $HOT_ROOT (/scratch/agentstage by default)
                          mirrors cold paths under absolute prefix:
                            /cold/.../file.nc
                              → $HOT_ROOT/cold/.../file.nc
```

The filesystem is the IPC. There is no socket, no shared-memory
ringbuffer, no daemon RPC. The stager copies cold → `<hot>.tmp` → atomic
rename to `<hot>`. The shim's `openat(hot)` either succeeds (file is
ready) or returns ENOENT (not staged yet) — in the latter case it spins
briefly and then opens the cold path.

## 3. LD_PRELOAD shim

### Why LD_PRELOAD over alternatives

Decision rationale (already in AGENTSTAGE.md §11.9 risk row, formalized
here):

| Mechanism | Pro | Con |
|---|---|---|
| **LD_PRELOAD** ✓ | no kernel module; no root; debuggable from Python harness; transparent to agent code; lowest overhead | bypassable via direct `syscall(SYS_openat, …)`; lost on `execve` without env propagation |
| FUSE | works for any process; survives execve | requires kernel module / privileged mount; per-syscall round-trip overhead; harder to debug |
| Bind-mount | kernel-level redirect | requires mount privilege (or user namespaces with restrictions); not per-process |
| Application-level patching | most control | requires modifying every benchmark harness; not transparent |

LD_PRELOAD's bypassability is acceptable for our workloads: Python
agents using `open()`, `np.load()`, `pandas.read_csv()`,
`h5py.File()`, `netCDF4.Dataset()`, etc. all go through libc.

### Syscall interception set

**Minimal interception is sufficient.** The shim intercepts only the
syscalls that take a *path* and either open a file descriptor or
report metadata. Once a file descriptor points at the hot file, every
subsequent `read`, `pread`, `mmap`, `lseek`, `close` operates on that
descriptor — no path involved, no further interception needed.

Intercept:

| Function | Reason |
|---|---|
| `openat`, `openat2` | Primary redirect point. `open()` in modern glibc is `openat(AT_FDCWD, …)`. |
| `creat` | Legacy `open()` shorthand. Pass-through (write). |
| `statx`, `newfstatat` | Agents stat before open (`os.path.exists`, `pandas` reading metadata). Returned size must match the hot file's actual size. |
| `stat`, `lstat`, `fstatat` | Legacy variants of the above. |
| `access`, `faccessat`, `faccessat2` | Existence checks must succeed on hot if the file is staged. |

Do **not** intercept:

| Function | Why not |
|---|---|
| `read`, `pread`, `pread64`, `readv`, `preadv` | Operates on an fd. If the fd points at the hot file, reads hit hot. No redirection logic needed. |
| `mmap`, `mmap2` | Operates on an fd. Same reasoning. (Important: `MADV_DONTNEED` from `agentiobench.utils.cache.evict_dataset` operates on pages — it'll drop hot-file pages from the page cache, which is fine; the kernel reloads them from the hot file.) |
| `close` | Cleanup; doesn't need redirection. |
| `lseek` | fd-based. |

### Path mapping

Single mapping rule. For a cold path `P` that lies under any directory
listed in `AGENTSTAGE_COLD_ROOTS` (colon-separated):

```
hot_path(P) = $AGENTSTAGE_HOT_ROOT + os.path.realpath(P)
            = /scratch/agentstage<realpath of P>
```

The hot root mirrors the absolute path. Examples:

```
/mnt/common/datasets-staging/agentiobench/aiob_107/scene1.nc
  → /scratch/agentstage/mnt/common/datasets-staging/agentiobench/aiob_107/scene1.nc

/scratch/sab_tasks/task_0042/data.csv
  → /scratch/agentstage/scratch/sab_tasks/task_0042/data.csv
```

This preserves uniqueness across multiple cold roots without hashing
(debuggable: `ls $AGENTSTAGE_HOT_ROOT/<cold_path>` works). Symlinks are
resolved at stage time via `realpath()` so the shim and stager agree on
canonical paths.

### Atomicity model

**Rename-into-place.** The stager:

1. Computes `hot_path = hot_for(cold_path)`
2. Creates `hot_path.parent` (mkdir -p)
3. Opens `tmp = hot_path.with_suffix(hot_path.suffix + f".tmp.{pid}.{tid}")`
4. Copies bytes from cold to tmp
5. `os.rename(tmp, hot_path)` — atomic on local filesystems (XFS, ext4, tmpfs)
6. Updates `staging_report.json`

**Existence at `hot_path` is the readiness signal.** No sentinel files,
no separate IPC for readiness. The shim's `openat(hot_path)` either
succeeds (atomic rename completed) or fails with ENOENT (not yet
staged). There is no intermediate state where the file appears
truncated or partially-written — the rename is atomic from POSIX.

Note: We do **not** fsync the staged file before rename. Reads against
the hot path don't require persistence to disk; if the system crashes,
the run is lost anyway.

### Shim behavior on openat

Pseudocode for the redirect:

```c
int openat(int dirfd, const char *pathname, int flags, mode_t mode) {
    // Resolve to absolute path
    char abs[PATH_MAX];
    if (!_resolve_absolute(dirfd, pathname, abs)) {
        return _real_openat(dirfd, pathname, flags, mode);  // bad path; pass-through
    }

    // Writes pass through to cold path unchanged
    if (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND)) {
        return _real_openat(dirfd, pathname, flags, mode);
    }

    // Read-only opens: check if cold path is under a managed cold root
    if (!_under_managed_cold_root(abs)) {
        return _real_openat(dirfd, pathname, flags, mode);
    }

    // Build hot path
    char hot[PATH_MAX];
    snprintf(hot, sizeof(hot), "%s%s", _hot_root, abs);

    // Try hot, retry briefly, fall through to cold
    int fd = _real_openat(AT_FDCWD, hot, flags, mode);
    if (fd >= 0) {
        _record_hit(abs);
        return fd;
    }
    if (errno != ENOENT) {
        return _real_openat(dirfd, pathname, flags, mode);  // unexpected; cold path
    }

    // ENOENT: file might be in-flight. Spin up to AGENTSTAGE_RETRY_SPIN_MS.
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int retry_ms = _retry_spin_ms;  // 20 ms default
    while (1) {
        struct timespec sleep_for = {.tv_sec = 0, .tv_nsec = 500000};  // 0.5 ms
        nanosleep(&sleep_for, NULL);
        fd = _real_openat(AT_FDCWD, hot, flags, mode);
        if (fd >= 0) {
            _record_late_hit(abs);
            return fd;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed_ms = (now.tv_sec - start.tv_sec) * 1000
                        + (now.tv_nsec - start.tv_nsec) / 1000000;
        if (elapsed_ms >= retry_ms) break;
    }

    // Stage didn't land in time; fall through to cold
    _record_miss(abs);
    return _real_openat(dirfd, pathname, flags, mode);
}
```

Retry spin = **20 ms** (`AGENTSTAGE_RETRY_SPIN_MS=20`). Bounded latency
injection. Rationale: a 50 MB/s cold-tier read of a typical 3 MB file
(aiob_107) takes ~60 ms; injecting 20 ms of retry-spin gains the hot
read on borderline races without materially hurting the cold-path P95.

The `stat`/`access` family use the same redirect path (try hot,
ENOENT-retry, fall through to cold). `access(F_OK)` on a cold path must
succeed if the hot copy exists, since the agent then expects `openat`
to succeed.

### Write pass-through

Any `openat` call with `O_WRONLY`, `O_RDWR`, `O_CREAT`, `O_TRUNC`, or
`O_APPEND` in flags is passed through to the original cold path. This
ensures:

- Agent intermediate writes (e.g., aiob task `output_fname`) land in
  the dataset's `output/` subdirectory as expected
- We never accidentally redirect a write into the read-only hot mirror
- The soft-stop predicate (per `CAMPAIGN.md`) sees the actual write
  to the task output target

### DFTracer load order

```bash
LD_PRELOAD="$DFTRACER_SO:$AGENTSTAGE_SHIM_SO" python -m agentstage.runners.aiob ...
```

`dftracer` is loaded **first**, so its `openat` wrapper runs first.
DFTracer logs the agent's *intent* (the cold path) before our shim
performs the redirect. The shim then issues `openat(hot_path)`, which
DFTracer also sees (since DFTracer's wrapper invokes the next symbol in
the LD_PRELOAD chain, which is ours).

This gives us both signals in the `io_report.json`:
- Cold-path opens for ground-truth scoring (was this file accessed?)
- Hot-path opens for performance measurement (what was actually read?)

Implementation note: DFTracer's instrumentation labels each event with
the path the wrapper saw. Distinguishing intent-vs-redirect in the
io_report is straightforward — cold paths and hot paths have disjoint
prefixes (`/cold/...` vs `$AGENTSTAGE_HOT_ROOT/...`).

### Subprocess inheritance

LD_PRELOAD propagates to child processes via the standard `LD_PRELOAD`
environment variable. Children inherit unless they explicitly clear the
env (e.g., `subprocess.run([...], env={...})` with no `LD_PRELOAD`
key).

**Convention for benchmark harnesses:** when spawning subprocesses,
propagate the full parent env. AgentIOBench's runner already does this;
SAB and KramaBench harnesses to be verified during integration (T41,
T47 in `TASKS.md`).

If a benchmark cannot be made to propagate, we fall back to the HTTP
proxy mode (see `proxy/server.py`).

## 4. Stager process model

**In-process thread pool inside the AgentStage client process.** No
separate daemon, no socket, no Unix-pipe IPC. Justification:

- The eScience paper measures single-agent runs (E5, E6, E7, E9, E10
  are all single-agent)
- The stager's only producer is the client library in the same process
- The stager's only consumer (the shim) talks to the stager via the
  filesystem, not a direct API
- IPC ceremony adds bug surface (socket lifecycle, queue overflow,
  daemon-crash-mid-run) for zero benefit at this scale

```python
# src/agentstage/stager/daemon.py (in-process despite the filename)

class Stager:
    def __init__(
        self,
        hot_root: Path,
        cold_roots: list[Path],
        max_workers: int = 4,
        capacity_bytes: int = 32 * 1024**3,
        report_path: Path | None = None,
    ):
        self.hot_root = hot_root
        self.cold_roots = [p.resolve() for p in cold_roots]
        self.capacity_bytes = capacity_bytes
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="agentstage-stage"
        )
        self.in_flight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._report = StagingReport(path=report_path)

    def prefetch(self, hint: DataHint) -> None:
        """Called by AgentStageClient on each detector rule firing."""
        for cold_path in hint.detected_files:
            with self._lock:
                if cold_path in self.in_flight:
                    continue
                future = self.executor.submit(self._stage, cold_path, hint)
                self.in_flight[cold_path] = future

    def _stage(self, cold_path: str, hint: DataHint) -> StageResult:
        hot_path = self._hot_for(cold_path)
        if hot_path.exists():
            return self._report.record_hit(cold_path, hint)

        size = os.path.getsize(cold_path)
        self._ensure_capacity_for(size)

        tmp = hot_path.with_suffix(
            hot_path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            hot_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            shutil.copy(cold_path, tmp)
            tmp.rename(hot_path)
            elapsed_ms = (time.monotonic() - start) * 1000
            return self._report.record_stage(
                cold_path, hot_path, size, elapsed_ms, hint
            )
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            return self._report.record_error(cold_path, hint, exc)
```

The `StagingReport` writes `outputs/<run_id>/staging_report.json` at
run end with per-stage records:

```json
{
  "stages": [
    {
      "cold_path": "/mnt/.../aiob_107/scene1.nc",
      "hot_path": "/scratch/agentstage/mnt/.../scene1.nc",
      "size_bytes": 3145728,
      "fetch_ms": 62.3,
      "tier": 1,
      "rule_id": "first_inspect_goes",
      "t_detected_ms": 8537,
      "t_completed_ms": 8599.3,
      "outcome": "staged"
    },
    ...
  ],
  "summary": {
    "n_stages_attempted": 47,
    "n_hits_before_first_open": 41,
    "n_late_hits_in_retry_spin": 3,
    "n_misses_fell_through_to_cold": 3,
    "total_bytes_staged": 18874368,
    "p50_fetch_ms": 58.0,
    "p95_fetch_ms": 142.0
  }
}
```

These records feed `paper_evals/test_h8_staging_effectiveness.py` and
`test_h10_proxy_overhead.py`.

## 5. Cache eviction policy

### Capacity

Default `AGENTSTAGE_HOT_CAPACITY_BYTES = 32 * 1024**3` (32 GB). Sized
to fit any single AIOB workload's full working set without eviction:

| Workload | Total bytes | Fits in 32 GB? |
|---|---:|---|
| aiob_101 ERA5 | 4.9 GB | ✓ |
| aiob_104 IGSR | 10.7 GB | ✓ |
| aiob_107 GOES | 18.0 GB | ✓ |
| aiob_110 NWB | 14.7 GB | ✓ |
| code_repo | 27 MB | ✓ |

For the eScience paper's headline E5 numbers, no eviction occurs. This
isolates the staging-effectiveness signal from eviction artifacts.
Eviction behavior gets a dedicated micro-experiment if needed (backlog
item, not on the critical path).

### Eviction algorithm

Opportunistic LRU triggered on capacity pressure during `_stage`:

```python
def _ensure_capacity_for(self, incoming_size: int) -> None:
    with self._lock:
        current_used = self._scan_used_bytes()
        if current_used + incoming_size <= self.capacity_bytes:
            return
        # Need to evict
        to_free = (current_used + incoming_size) - self.capacity_bytes
        candidates = self._list_hot_files_by_atime()  # oldest first
        freed = 0
        for path, size in candidates:
            if path in self.in_flight_paths:
                continue  # don't evict files being staged
            try:
                path.unlink()
                freed += size
            except FileNotFoundError:
                pass
            if freed >= to_free:
                break
        if freed < to_free:
            raise StagerOutOfSpace(needed=to_free, freed=freed)
```

LRU is `atime`-based when available (file system mounted with `relatime`
or `strictatime`); falls back to `mtime` on `noatime` mounts. The
default `/scratch` on most HPC nodes is `relatime`, which gives us
useful LRU signal at low cost.

Files currently in-flight (`self.in_flight_paths`) are protected from
eviction.

## 6. Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `AGENTSTAGE_HOT_ROOT` | `/scratch/agentstage` | Hot tier root. Must be on local NVMe. Fall-back `/tmp/agentstage` if `/scratch` is absent. |
| `AGENTSTAGE_COLD_ROOTS` | `$AGENTIOBENCH_DATA_ROOT` | Colon-separated list of cold-tier path prefixes the shim manages. |
| `AGENTSTAGE_HOT_CAPACITY_BYTES` | `34359738368` (32 GB) | Eviction triggers above this. |
| `AGENTSTAGE_RETRY_SPIN_MS` | `20` | Shim's ENOENT retry budget on `openat(hot_path)`. |
| `AGENTSTAGE_MAX_WORKERS` | `4` | Stager thread pool size. |
| `AGENTSTAGE_SHIM_LOG` | unset | If set to a file path, the shim writes per-syscall events for debugging. Off by default — non-trivial overhead. |
| `LD_PRELOAD` | (none) | Must include `libagentstage_shim.so`. With DFTracer: `LD_PRELOAD="$DFTRACER_SO:$AGENTSTAGE_SHIM_SO"`. |

The Python runner sets these before invoking the agent process; the
shim reads them at the first `openat` call.

## 7. Testing approach

### Unit tests (`tests/test_stager.py`)

- `prefetch_dispatches_to_executor`: calling `Stager.prefetch(hint)`
  submits the right number of `_stage` tasks
- `stage_is_atomic`: while `_stage` is mid-copy, a concurrent
  `openat(hot_path)` either returns the final fd or ENOENT — never
  a partial file
- `re_prefetch_is_idempotent`: calling `prefetch` twice with the same
  `cold_path` produces one stage, not two
- `eviction_on_enospc`: filling the hot tier to capacity and then
  prefetching one more file triggers LRU eviction
- `in_flight_protection`: a file being staged cannot be evicted
- `write_passthrough`: stager never observes `O_WRONLY` opens (shim
  short-circuits those before reaching the stager)

### Shim tests (`tests/shim/`)

Built with a C test harness that loads the shim and exercises specific
syscalls:

- `redirects_openat_under_cold_root`: `openat(/cold/x)` returns an fd
  with content from `$HOT_ROOT/cold/x`
- `falls_through_on_enoent`: `openat(/cold/x)` when hot is missing,
  with no stager activity, returns an fd with content from `/cold/x`
- `retry_spin_catches_late_stage`: stager finishes staging after 10ms;
  shim returns the hot fd
- `retry_spin_falls_through_after_budget`: stager doesn't finish within
  20ms; shim returns the cold fd
- `writes_pass_through`: `openat(/cold/x, O_WRONLY)` opens the cold
  path, not the hot mirror

### Integration test (`tests/integration/test_end_to_end_staging.py`)

A miniature end-to-end smoke run on a synthetic 5-file workload:

1. Set `AGENTSTAGE_COLD_ROOTS=/tmp/synthetic_cold`, `AGENTSTAGE_HOT_ROOT=/tmp/synthetic_hot`
2. Place 5 known files under `/tmp/synthetic_cold/`
3. Spawn a Python subprocess with `LD_PRELOAD=libagentstage_shim.so`
4. Detector (mock) emits a `DataHint` for 3 of the 5 files
5. After 100 ms, the subprocess opens all 5 files
6. Assert: 3 files were served from hot (per `staging_report.json`),
   2 fell through to cold

This is the closest test to E5's actual workload shape.

## 8. Edge cases and gotchas

| Case | Behavior | Notes |
|---|---|---|
| Cold file modified mid-run | Stale hot copy served | Out of scope (NG2). Datasets are read-only. |
| Symlinks in cold root | Resolved via `realpath()` at stage time and at shim openat time | Both shim and stager call `realpath` so they agree on canonical paths |
| `O_PATH` opens (descriptor without read) | Redirect to hot if exists | The agent typically `fstatat`'s through O_PATH descriptors — needs consistent metadata |
| `O_DIRECTORY` opens | Redirect to hot if exists | `os.scandir(/cold/dir)` should see the hot directory if mirrored |
| `/proc`, `/sys`, `/dev` paths | Never redirect | Outside `AGENTSTAGE_COLD_ROOTS` by construction |
| Hot file owned by stage process, agent runs as different user | Fail-open with ENOENT → falls through to cold | Both run as same user in our setup; document the assumption |
| Shim called with relative path + AT_FDCWD | Resolve to absolute via `realpath` | Standard glibc behavior |
| Shim called with relative path + a real dirfd | `_real_fstatat(dirfd, "")` to get the directory's absolute path, then concat | Less common but legal |
| Agent uses raw `syscall(SYS_openat, ...)` | Bypasses shim | Document as known limitation; none of our benchmarks do this |
| mmap of staged file, then `madvise(MADV_DONTNEED)` | Drops hot pages from page cache | Fine; next read repopulates from hot file |
| Shim crashes mid-openat | Process aborts (no signal handler) | Acceptable for a research artifact; document for production hardening |
| Capacity full + stager out of memory + agent opens unstaged file | Falls through to cold with retry-spin penalty | The 20 ms penalty is the worst-case overhead from stager failure |

## 9. Implementation layout

```
src/agentstage/stager/
├── __init__.py
├── daemon.py                  # Stager class (in-process despite the name)
├── report.py                  # StagingReport dataclass + JSON serialization
├── eviction.py                # LRU sweep on capacity pressure
└── shim/
    ├── Makefile               # builds libagentstage_shim.so
    ├── agentstage_shim.c      # the LD_PRELOAD wrapper
    ├── path_mapping.c         # cold→hot path translation
    ├── recursion_guard.c      # thread-local guard against shim-recursion
    └── tests/                 # C-level shim tests
        └── test_openat.c
```

Build:

```bash
make -C src/agentstage/stager/shim
# produces src/agentstage/stager/shim/libagentstage_shim.so
```

The Makefile uses `gcc -shared -fPIC -ldl -lpthread`. No external
dependencies beyond libc.

## 10. Future work (out of scope for eScience)

- **Multi-agent daemon.** A separate process keeping a shared hot tier
  for N co-located agents. Required for C10 (multi-agent contention)
  but deferred. Would need Unix-socket IPC and reference-counting on
  staged files.
- **Tier-aware admission control.** Currently any file in a DataHint is
  admitted equally. A richer policy would admit tier-1 detections
  preferentially and stage tier-3 only if capacity permits.
- **Hot tier sharding across multiple NVMe devices.** For nodes with
  multiple NVMe SSDs, distribute staged files across devices to
  parallelize the cold→hot copy.
- **Stage from PFS without LD_PRELOAD on the source.** Currently the
  stager `shutil.copy()`s; for PFS sources (Lustre / OrangeFS / GPFS)
  a tuned read (e.g., libdaos for DAOS, posix-aio for Lustre) could be
  faster.
- **Predicted-but-unread eviction priority.** Track which staged files
  the agent actually opened vs. which remained unread; bias eviction
  against unread files first (they were detector false positives).
- **In-flight cold reads.** If the agent opens a cold file *while* the
  stager is mid-copy of that same file, current behavior falls through
  to cold (the rename hasn't happened yet). A more aggressive
  implementation could splice the in-progress copy.

## 11. Open questions (resolved 2026-05-20 via AskUserQuestion)

- **Hot root location**: NVMe (`/scratch/agentstage`), not tmpfs. Matches
  AGENTSTAGE.md §4.3 framing.
- **Hot capacity**: 32 GB default. Fits any single AIOB workload
  without eviction; isolates the E5 signal.
- **Atomicity retry**: 20 ms retry-spin then fallthrough to cold.
  Bounded latency injection; covers detection-race window.

These are configurable via env vars (§6) for sensitivity sweeps if
reviewers ask.
