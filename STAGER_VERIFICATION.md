# Stager Verification

Pre-Day-5 verification of the staging stack. Captures what we tested,
what we measured, and what we found before any production stager runs
took place. Companion to:

- [`STAGER_DESIGN.md`](STAGER_DESIGN.md) — the implementation contract
- [`STAGER_WALKTHROUGH.md`](STAGER_WALKTHROUGH.md) — the tutorial-style
  explainer
- This doc — what we verified and how

When this work happened: 2026-05-19 (Day 2), on the `task14` branched
conversation. Full suite ran in **17.63 s** on a single Ares node.

## Headline

**52 tests passing + 1 deferred-to-Day-5; 0 failures across the stack.**

The stager + LD_PRELOAD shim contract is now validated end-to-end on a
synthetic workload without any LLM. If the Day-7 manual smoke (T32:
aiob_107 + Haiku + real DFTracer) fails, it will be for predictor /
harness / DFTracer-chain reasons rather than stager bugs.

| Layer | Code | Tests | Status |
|---|---|---:|---|
| L0 microbench | `scripts/microbench/stager_baseline.py` | 3 measurements | ✓ headroom confirmed |
| L1 stager unit | `tests/test_stager.py` | 10 | ✓ all pass |
| L2 shim unit | `tests/test_shim.py` | 11 (10 + 1 skip) | ✓ all pass; 1 deferred |
| L3 integration | `tests/integration/test_end_to_end_staging.py` | 2 | ✓ all pass |

## Layer 0 — Environment microbenchmarks

The whole AgentStage proposition hinges on three environment assumptions
about the testbed. Before writing any stager code, we verified them.

### A. Cold-tier first-read P95 is large enough to justify staging

Measured `open() + read(4096)` per file, after evicting each file's page
cache via `POSIX_FADV_DONTNEED`. Three workload-shaped buckets:

| Bucket | n | File sizes | Cold P95 |
|---|---:|---|---:|
| `small_goes_3mb` | 30 | 2.6-3.4 MB | **19 ms** |
| `medium_era5_50mb` | 12 | 130-139 MB | **17 ms** |
| `large_nwb_350mb` | 15 | 273-498 MB | **23 ms** |

### B. Hot-tier (tmpfs) P95 is small enough to be the ceiling

Same files copied to `/dev/shm/agentstage/` (RAM-backed tmpfs), evicted,
measured the same way:

| Bucket | Hot P95 | Speedup |
|---|---:|---:|
| `small_goes_3mb` | **0.06 ms** | **306×** |
| `medium_era5_50mb` | **0.10 ms** | **168×** |
| `large_nwb_350mb` | **0.11 ms** | **216×** |

The 4 KB first-block measurement understates the real speedup because
the agent reads the WHOLE file, not just the first block. Full-file
reads on 372 MB NWB files would take ~3-7 seconds on cold (at typical
sequential read rate of 50-100 MB/s on XFS-on-SSD) versus ~70 ms on
tmpfs — a 50-100× ratio on top of the first-byte speedup.

The cold tier here is **XFS on local SSD**, not NFS/PFS/S3. Real cold
tiers would show 5-10× higher cold latency and correspondingly larger
speedup. These numbers establish a conservative lower bound.

### C. POSIX_FADV_DONTNEED actually drops residency on XFS

Critical: without working eviction, every "with vs without stager"
comparison is meaningless because the baseline runs would silently hit
warm page cache.

```
Before evict: 3626 pages resident / 3626 total  (100%)
evict_dataset() called
After evict:  0 pages resident / 3626 total  (0%)
```

Eviction confirmed via direct `mincore(2)` per file using
`agentiobench.utils.cache._resident_pages`.

### Initial false-negative in the eviction check

First run of the microbench reported `eviction_works: False`. Diagnosis:
the verification logic used `measure_temperature(bucket_root)` which
samples 8 random files from the bucket. The bucket had ~75 files but we
only warmed 8 (and only 4 KB each), so the bucket-wide resident
fraction was 0.2% — below the 5% threshold even before eviction.
Eviction actually worked, but the test couldn't see it.

Fixed by using `_resident_pages(file)` directly on each of the warmed
files. Re-run showed 100% → 0% residency drop.

The artifact is at
`outputs/microbench/stager_baseline_<ts>.json` for reproducibility.

## Layer 1 — Python stager unit tests

`tests/test_stager.py` — 10 tests pinning the Stager class invariants.
No subprocess, no shim, no LLM. Runs in **0.46 s**.

| # | Test | What it pins |
|---|---|---|
| 1 | `test_prefetch_dispatches_one_stage_per_file` | Dispatch fan-out: N predicted files → N stage tasks |
| 2 | `test_prefetch_skips_paths_outside_managed_cold_roots` | Cold-root filter: prefetch silently skips paths outside configured roots |
| 3 | **`test_stage_is_atomic_under_concurrent_open`** | The atomicity contract (see below) |
| 4 | `test_re_prefetch_returns_same_future` | Idempotency: re-prefetch returns the existing Future, no second copy |
| 5 | `test_re_prefetch_after_completion_yields_hit` | The `_stage` hit-path when called after the file is already staged |
| 6 | `test_eviction_frees_lru_when_capacity_exceeded` | atime-based LRU sweep on capacity pressure |
| 7 | `test_eviction_raises_when_freeing_impossible` | Oversize file records `skip_oversize` event, doesn't crash |
| 8 | `test_in_flight_files_protected_from_eviction` | Files with futures `not done()` are never evicted |
| 9 | `test_stager_only_sees_predicted_files_not_writes` | Contract test: Stager API has no write-mode methods |
| 10 | `test_staged_file_is_byte_identical_to_source` | sha256 check: hot copy bytes == cold source bytes |

### The atomicity test (#3)

The most interesting test in L1. We need to guarantee that no reader
ever sees a partial file during a stage. The atomic-rename strategy
should give us this, but the test verifies it concretely:

```python
SIZE = 16 * 1024 * 1024  # 16 MB so the copy takes meaningful time
cold = make_cold_file(cold_dir, "big.bin", SIZE)
expected = cold.read_bytes()

hot_path = stager.hot_path_for(cold)
partial_reads = []
stop = threading.Event()

def watcher():
    while not stop.is_set():
        try:
            with open(hot_path, "rb") as f:
                content = f.read()
            if len(content) != SIZE or content != expected:
                partial_reads.append(len(content))
            else:
                hit_count[0] += 1
        except FileNotFoundError:
            enoent_count[0] += 1

threading.Thread(target=watcher, daemon=True).start()
stager.prefetch(make_hint(cold))
# wait for stage to complete
assert not partial_reads
```

The watcher thread hammers openat on the hot path while the main thread
runs the stager. During the race window (16 MB copy takes a few ms on
tmpfs), the watcher should see only:
- **ENOENT** (rename hasn't happened yet — file genuinely doesn't exist)
- **Full 16 MB byte-identical to source** (rename completed)

Never:
- A partial file (length < 16 MB)
- A 16 MB file with mismatched bytes (somehow corrupted)

Result: 0 partial reads observed across the race. POSIX `rename(2)` is
atomic on local filesystems; this confirms our implementation respects
that.

## Layer 2 — LD_PRELOAD shim tests

`tests/test_shim.py` — 11 tests spawning subprocesses with the shim
loaded via `LD_PRELOAD`. Built `libagentstage_shim.so` first, then
exercised it through real `python3` and `cat` processes. Runs in
**0.62 s** including subprocess startup.

| # | Test | What it pins |
|---|---|---|
| 1 | `test_open_redirects_to_hot` | Python `open()` resolves to our shim for files under managed cold roots |
| 2 | `test_cat_also_redirects` | `cat` (which uses `open()` not `openat()`) also redirects — verifies the `open` alias |
| 3 | `test_falls_through_when_hot_missing` | ENOENT on hot → fall through to cold |
| 4 | `test_unmanaged_cold_root_passes_through` | Paths outside managed cold roots are never touched |
| 5 | **`test_retry_spin_catches_late_rename`** | Race window: stager finishes after openat starts; shim catches it (see below) |
| 6 | `test_retry_spin_falls_through_after_budget` | Bounded latency injection on miss |
| 7 | `test_writes_pass_through_to_cold` | O_WRONLY/O_CREAT lands in cold dir, not hot mirror |
| 8 | **`test_stat_returns_hot_size_when_redirected`** | Critical: stat must return hot file's size or buffer allocations break |
| 9 | `test_shim_disable_makes_passthrough` | AGENTSTAGE_SHIM_DISABLE=1 escape hatch |
| 10 | `test_redirect_preserves_byte_identity` | sha256 end-to-end through subprocess |
| — | `test_shim_works_with_dftracer_before_it` | **Deferred to Day 5** (needs real dftracer.so) |

### The retry-spin race test (#5)

This test exercises the prediction-race scenario: predictor fires X ms
before the agent's openat, but the stager hasn't finished the copy
when openat hits. Without retry-spin, the shim would fall through to
cold. With retry-spin, it should catch the late rename.

```python
# Cold file exists; tmp file at hot.tmp; renamer thread will move it
# into place 10 ms after openat starts. retry_spin_ms=50.

script = f"""
import os, threading, time, sys

def renamer():
    time.sleep(0.010)  # 10 ms — within 50 ms retry-spin budget
    os.rename({str(hot_tmp)!r}, {str(hot_path)!r})

threading.Thread(target=renamer).start()
content = open({str(cold_file)!r}).read()
sys.stdout.write(content)
"""

# Subprocess runs with LD_PRELOAD=shim.so
r = run_python(env, script)
assert r.stdout == "HOT_LATE"  # got hot version, not cold fallback
```

The renamer races with the agent's openat. Without retry-spin, the
openat would see ENOENT and fall through to cold (returning the cold
content). With retry-spin enabled, the shim polls every 0.5 ms for up
to 50 ms — catches the rename and returns the hot fd.

This is the failure mode that "20 ms retry-spin then fall through to
cold" in `STAGER_DESIGN.md` §3 is specifically designed to handle.

### The stat redirect test (#8)

A subtle correctness invariant. When the shim redirects opens, it must
also redirect stats — otherwise an agent that does:

```python
size = os.path.getsize(path)        # → returns cold file size
buf = bytearray(size)
with open(path, "rb") as f:
    f.readinto(buf)                  # → reads hot file (different size!)
```

would corrupt its read or allocate the wrong buffer. Test verifies:

```python
cold_file.write_bytes(b"x" * 100)      # 100 bytes
place_hot_copy(hot_dir, cold_file, b"y" * 200)  # 200 bytes

st = os.stat(cold_file)
assert st.st_size == 200  # got hot's size, not cold's
```

Confirmed: stat/lstat/fstatat/access all redirect when the hot file
exists.

## Layer 3 — End-to-end integration

`tests/integration/test_end_to_end_staging.py` — full stack on a
synthetic 5-file workload, no LLM. The closest analog of E5 we can
run pre-Day-5. Runs in **~16 s** (subprocess startup dominates).

### Test 1: synthetic 5-file workload

```
files:  file1=1 MB    file2=5 MB    file3=10 MB    file4=25 MB    file5=50 MB
        └─────────── stager pre-stages ───────────┘    └── NOT staged ──┘

agent subprocess (LD_PRELOAD=shim.so) opens all 5 files
```

After the run, we verify three things:

1. **Shim log shows correct HIT/MISS pattern:**
   ```
   HIT  .../cold/file1_1MB.bin  -> .../hot/.../file1_1MB.bin
   HIT  .../cold/file2_5MB.bin  -> .../hot/.../file2_5MB.bin
   HIT  .../cold/file3_10MB.bin -> .../hot/.../file3_10MB.bin
   MISS .../cold/file4_25MB.bin (errno=2)
   MISS .../cold/file5_50MB.bin (errno=2)
   ```

2. **Byte identity:** Each file's sha256 read through the shim equals
   the cold source's sha256 exactly. (No corruption from the redirect.)

3. **StagingReport consistent:** Stager-side report shows 3 staged,
   16 MB total, p50 fetch ~9 ms, p95 fetch ~16 ms.

### Test 2: latency-signal direction

Two 10 MB files, one pre-staged, one not. After cold-cache eviction on
the non-staged one, the subprocess reads both and reports elapsed time.

```
staged read: 10.8 ms
cold read:   13.0 ms
```

The 20% gap is modest because "cold" here is `/tmp` XFS — not a real
PFS/NFS/S3 backend. The L0 microbench's 100-300× speedup numbers are
what we'd actually see when the cold tier is genuinely cold-tier-shaped.

What matters for this test: **direction is correct.** Staged < cold,
not the other way around. The stager is genuinely shortening I/O time
even at this synthetic scale.

## Bugs found and fixed

Three real bugs and one false-alarm test design issue surfaced during
verification.

### Bug 1 — `_in_flight` accumulating forever, blocking eviction

**Symptom:** `test_eviction_frees_lru_when_capacity_exceeded` failed.
The new file couldn't be staged because every previously-staged file
was protected from eviction.

**Diagnosis:** I was adding entries to `self._in_flight[cold_path] =
future` at submission time but never cleaning them up. After staging
N files, all N entries remained, so the eviction protect-set was
"everything we've ever staged" instead of "files currently being
copied."

**Fix:** in `_ensure_capacity_for`, filter on `not fut.done()`:

```python
in_flight_hot_paths = {
    str(self.hot_path_for(c))
    for c, fut in self._in_flight.items()
    if not fut.done()
}
```

Completed futures still live in `_in_flight` (for idempotency on
re-prefetch), but their files are eviction-eligible.

**Severity:** Production-critical. Without this fix, a real campaign
would fill the hot tier and then fail to stage anything new — the
stager would silently degrade to "no-op after first N files."

### Bug 2 — Shim only intercepted unversioned symbols

**Symptom:** Shim loaded fine (LD_DEBUG=libs confirmed; constructor
ran), but `python3 -c "open('cold/foo')"` returned the cold content.
Redirect wasn't happening.

**Diagnosis path:**

1. `objdump -T $(which python3) | grep openat` →
   `0000... DF *UND* 0000... (GLIBC_2.4) openat64`
   Python references `openat64@GLIBC_2.4`, not plain `openat`.

2. `nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep openat` showed libc
   exports `openat`, `openat64`, `__openat_2`, `__openat64_2` (the
   fortify variants when compiled with `-D_FORTIFY_SOURCE=2`).

3. My shim only exported plain `openat`. The dynamic linker resolved
   Python's `openat64` reference to libc's openat64 directly, never
   touching my shim.

**Fix:** added alias declarations for every variant:

```c
int openat64(int dirfd, const char *pathname, int flags, ...)
    __attribute__((alias("openat")));

int open(const char *pathname, int flags, ...) {
    /* ... unpack varargs ... */
    return openat(AT_FDCWD, pathname, flags, mode);
}
int open64(...) __attribute__((alias("open")));

int __open_2(const char *pathname, int flags) {
    return openat(AT_FDCWD, pathname, flags, 0);
}
int __open64_2(...) __attribute__((alias("__open_2")));
int __openat_2(int dirfd, const char *pathname, int flags) {
    return openat(dirfd, pathname, flags, 0);
}
int __openat64_2(...) __attribute__((alias("__openat_2")));
```

Plus `__xstat`, `__lxstat`, `__fxstatat` for legacy glibc (and their
`64` variants).

**Verification:** `nm -D libagentstage_shim.so | grep T` after rebuild
showed all 22 exported function symbols.

**Severity:** Would have been a Day-5 catastrophe. The shim would have
loaded successfully but done nothing on every modern Linux system. We
would have spent hours debugging why E5 speedup was 0%.

### Bug 3 — `stat64` alias broke compilation

**Symptom:** `make` failed:

```
agentstage_shim.c:518:5: error: conflicting types for 'stat64';
  have 'int(const char *, struct stat *)'
note: previous declaration of 'stat64' with type
  'int(const char * restrict, struct stat64 * restrict)'
```

**Diagnosis:** glibc's `<sys/stat.h>` declares `stat64()` with a
`struct stat64 *` parameter (technically distinct from `struct stat *`
even though they have identical layouts on 64-bit Linux). The
`__attribute__((alias("stat")))` requires matching signatures.

**Fix:** dropped the `stat64`/`lstat64`/`fstatat64` aliases. The
`__xstat64`/`__lxstat64`/`__fxstatat64` legacy variants cover what
older callers reference. Modern code uses the unversioned names which
we already intercept.

**Severity:** Build-blocking. Couldn't ship the shim with this error.

### Test design issue — chain compatibility stand-in

**Symptom:** `test_shim_works_with_another_ld_preload_before_it`
failed when using libc.so.6 as a stand-in for dftracer in the
LD_PRELOAD chain.

**Diagnosis:** libc.so.6 itself **defines** openat (it's the actual
glibc implementation). Putting libc first in LD_PRELOAD just makes
Python's openat call resolve to libc directly, bypassing our shim.

A real tracer like dftracer doesn't define openat — it **wraps**
openat via `dlsym(RTLD_NEXT, "openat")`. When the agent calls openat,
dftracer's wrapper runs first, then calls the next openat in the chain
(our shim), which redirects and then calls the next openat (libc's).

**Fix:** marked the test `@pytest.mark.skip` with a clear explanation;
re-enable on Day 5 when real dftracer is in `external/dftracer/`.

**Severity:** Test-only — not a real shim bug. But worth flagging so
we run the real chain check on Day 5 (T29).

## What remains unverified

Day-7 manual smoke (T32 in `TASKS.md`) is the last stager-related
risk. Specifically:

1. **DFTracer + agentstage LD_PRELOAD chain order in practice.** Our
   shim is designed to be loaded after DFTracer
   (`LD_PRELOAD="$DFTRACER:$AGENTSTAGE_SHIM"`). Verified in theory;
   needs the real chain test on Day 5.

2. **Hot/cold ratio on actual aiob_107 workload through real agent
   I/O patterns.** The synthetic workload's 5 files don't capture the
   6042-file fanout, the deep directory hierarchy, or the
   small-file syscall amplification that aiob_107 specifically
   exercises.

3. **Real DFTracer io_report.json correctness with the shim active.**
   The shim redirects opens to hot paths, but DFTracer should still
   log the cold-path intent (it runs first in the chain). Need
   end-to-end verification that the cold paths appear in the
   io_report's `file_name_view`.

4. **NVMe vs tmpfs.** This Ares node has only `/dev/shm` (tmpfs) for
   the hot tier; production deployments would use NVMe. Performance
   characteristics differ (tmpfs is faster, lower capacity).

5. **Real cold-tier (NFS/PFS) latency.** The XFS-on-local-SSD cold
   tier here gives 17-23 ms first-read P95. A real PFS would give
   100-500 ms, making AgentStage's speedup story even stronger.

## Reproducibility

```bash
# L0: environment microbench
uv run python scripts/microbench/stager_baseline.py
# → outputs/microbench/stager_baseline_<ts>.json

# L1+L2+L3: every layer at once
make -C src/agentstage/stager/shim
uv run pytest tests/ -v

# expected: 52 passed, 1 skipped in ~17 s
```

Both `.so` build and `.venv` are reproducible from the pinned
`uv.lock` + `pyproject.toml` + standard glibc on Ubuntu 22.04.

## Commits

- `5527268` — Phase 1: microbench
- `d25e6fa` — Phases 2-3: Python stager + L1 tests
- `f61938e` — Phases 4-5: LD_PRELOAD shim + L2 tests
- `2e483b6` — Phase 6: L3 integration
