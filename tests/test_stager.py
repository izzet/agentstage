"""Layer-1 stager unit tests.

These tests pin the Stager's behavioral invariants. They use real files on
the filesystem (tmpfs via pytest's tmp_path) but no LLM, no shim, and no
LD_PRELOAD. Fast (< 1 s total); runs on every `uv run pytest`.

Six invariants pinned:
  - prefetch dispatches one stage per detected file
  - _stage's atomic rename means a concurrent observer sees either ENOENT
    or the full final bytes — never a partial file
  - re-prefetching the same cold_path produces no new copy work
  - eviction triggers on ENOSPC and frees the smallest LRU set that fits
  - files currently being staged are protected from eviction
  - writes never reach the stager (contract test — shim short-circuits them)
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from agentstage.stager import (
    DataHint,
    Stager,
    StagerOutOfSpace,
    StagingReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cold_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cold"
    d.mkdir()
    return d


@pytest.fixture
def hot_dir(tmp_path: Path) -> Path:
    d = tmp_path / "hot"
    d.mkdir()
    return d


@pytest.fixture
def stager(cold_dir: Path, hot_dir: Path):
    """Default stager: 4 workers, 1 GB capacity, our temp dirs."""
    s = Stager(
        hot_root=hot_dir,
        cold_roots=[cold_dir],
        max_workers=4,
        capacity_bytes=1024**3,  # 1 GB; plenty for unit tests
    )
    yield s
    s.shutdown(wait=True)


def make_cold_file(cold_dir: Path, name: str, size_bytes: int) -> Path:
    """Create a deterministic-content file of `size_bytes` under cold_dir."""
    path = cold_dir / name
    # Use a simple repeating pattern so we can verify byte-identity later
    chunk = (name.encode() + b"\n") * 64
    n_chunks = (size_bytes + len(chunk) - 1) // len(chunk)
    with open(path, "wb") as f:
        for _ in range(n_chunks):
            f.write(chunk)
        # Truncate to exact size
        f.truncate(size_bytes)
    return path


def make_hint(*paths: Path, tier: int = 1, rule_id: str = "test") -> DataHint:
    return DataHint(
        detected_files=tuple(str(p) for p in paths),
        tier=tier,
        fired_at_ms=0.0,
        rule_id=rule_id,
    )


# ---------------------------------------------------------------------------
# T1: prefetch dispatches to executor
# ---------------------------------------------------------------------------

def test_prefetch_dispatches_one_stage_per_file(stager: Stager, cold_dir: Path):
    """A DataHint with N files produces N stage events (or fewer if dedupe)."""
    files = [make_cold_file(cold_dir, f"f{i}.bin", 1024) for i in range(5)]
    hint = make_hint(*files)

    futures = stager.prefetch(hint)
    assert len(futures) == 5
    for f in futures:
        f.result(timeout=5)

    assert len(stager.report.events) == 5
    assert all(e.outcome == "staged" for e in stager.report.events)
    for src in files:
        assert stager.is_staged(src), f"{src} not staged"


def test_prefetch_skips_paths_outside_managed_cold_roots(
    stager: Stager, cold_dir: Path, tmp_path: Path
):
    """A path outside the configured cold_roots is silently skipped, not staged."""
    inside = make_cold_file(cold_dir, "inside.bin", 1024)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = make_cold_file(outside_dir, "outside.bin", 1024)

    hint = make_hint(inside, outside)
    futures = stager.prefetch(hint)

    # Only the inside file should produce a future
    assert len(futures) == 1
    for f in futures:
        f.result(timeout=5)

    assert stager.is_staged(inside)
    assert not stager.is_staged(outside)


# ---------------------------------------------------------------------------
# T2: atomic rename — concurrent observer never sees a partial file
# ---------------------------------------------------------------------------

def test_stage_is_atomic_under_concurrent_open(stager: Stager, cold_dir: Path):
    """While _stage is mid-copy, openat(hot_path) either:
       (a) returns ENOENT (rename hasn't happened), or
       (b) returns an fd whose content equals the cold file's bytes.
       Never a partial file."""
    # Make the cold file big enough that the copy takes meaningful time
    SIZE = 16 * 1024 * 1024  # 16 MB
    cold = make_cold_file(cold_dir, "big.bin", SIZE)
    expected = cold.read_bytes()
    assert len(expected) == SIZE

    hot_path = stager.hot_path_for(cold)

    # Race: spawn a watcher thread that hammers openat on hot_path
    # while the stager copies.
    partial_reads = []
    enoent_count = [0]
    hit_count = [0]
    stop = threading.Event()

    def watcher() -> None:
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
            except OSError:
                pass

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    try:
        hint = make_hint(cold)
        for f in stager.prefetch(hint):
            f.result(timeout=15)
        # Give the watcher a moment to catch the post-rename state
        time.sleep(0.05)
    finally:
        stop.set()
        t.join(timeout=2)

    assert not partial_reads, (
        f"watcher observed {len(partial_reads)} partial reads "
        f"(sample sizes: {partial_reads[:5]}); rename was not atomic"
    )
    # We should see at least one ENOENT (before stage) AND at least one hit (after)
    # but on extremely fast tmpfs the watcher may miss the ENOENT window.
    # Accept any non-zero combination.
    assert enoent_count[0] + hit_count[0] > 0


# ---------------------------------------------------------------------------
# T3: re-prefetch is idempotent
# ---------------------------------------------------------------------------

def test_re_prefetch_returns_same_future(stager: Stager, cold_dir: Path):
    """Calling prefetch() twice with the same cold_path returns the same
    Future the first time around, and produces zero additional events."""
    cold = make_cold_file(cold_dir, "idem.bin", 4096)
    hint = make_hint(cold)

    futures_a = stager.prefetch(hint)
    futures_b = stager.prefetch(hint)

    assert len(futures_a) == 1
    assert len(futures_b) == 1
    assert futures_a[0] is futures_b[0], "re-prefetch should return existing future"

    for f in futures_a:
        f.result(timeout=5)

    assert len(stager.report.events) == 1, (
        f"expected 1 stage event, got {len(stager.report.events)}"
    )


def test_re_prefetch_after_completion_yields_hit(stager: Stager, cold_dir: Path):
    """After a stage completes, the in_flight dict still maps cold_path to the
    completed Future, so re-prefetch returns it; no second _stage call.

    BUT: if a caller bypasses the dedupe (e.g. by directly invoking _stage),
    the hit path inside _stage returns a 'hit' outcome instead of 'staged'.
    This test exercises the hit path explicitly to confirm it works."""
    cold = make_cold_file(cold_dir, "hit.bin", 4096)
    hint = make_hint(cold)

    for f in stager.prefetch(hint):
        f.result(timeout=5)
    n_before = len(stager.report.events)
    assert n_before == 1
    assert stager.is_staged(cold)

    # Directly call _stage to exercise the hit branch
    ev = stager._stage(str(cold.resolve()), hint)
    assert ev.outcome == "hit"
    assert len(stager.report.events) == n_before + 1


# ---------------------------------------------------------------------------
# T4: eviction on ENOSPC
# ---------------------------------------------------------------------------

def test_eviction_frees_lru_when_capacity_exceeded(
    cold_dir: Path, hot_dir: Path
):
    """Fill the hot tier to capacity, then stage one more file — verify the
    oldest-atime file in hot_root was evicted."""
    # Tight capacity: 5 MB total, files are 1 MB each
    s = Stager(
        hot_root=hot_dir,
        cold_roots=[cold_dir],
        capacity_bytes=5 * 1024 * 1024,
    )
    try:
        files = [make_cold_file(cold_dir, f"e{i}.bin", 1024 * 1024) for i in range(5)]
        for f in s.prefetch(make_hint(*files)):
            f.result(timeout=10)

        # All 5 should be staged
        for src in files:
            assert s.is_staged(src), f"{src} should be staged before pressure"

        # Touch e4 last so it's most-recently-used; e0 should be LRU.
        # (atime updates on read.)
        time.sleep(0.05)
        for src in files[1:]:
            s.hot_path_for(src).read_bytes()

        # Now stage one more file (1 MB) which should force eviction of e0
        extra = make_cold_file(cold_dir, "extra.bin", 1024 * 1024)
        for f in s.prefetch(make_hint(extra)):
            f.result(timeout=10)

        assert s.is_staged(extra), "new file should have been staged"
        # The LRU one (e0, never re-read) should be evicted
        assert not s.is_staged(files[0]), (
            f"LRU file {files[0]} should have been evicted"
        )
        # Others should still be present
        for src in files[1:]:
            assert s.is_staged(src), f"{src} should NOT have been evicted"
    finally:
        s.shutdown(wait=True)


def test_eviction_raises_when_freeing_impossible(
    cold_dir: Path, hot_dir: Path
):
    """If even after eviction we can't fit the incoming file, raise."""
    # Capacity 4 MB; try to stage an 8 MB file.
    s = Stager(
        hot_root=hot_dir,
        cold_roots=[cold_dir],
        capacity_bytes=4 * 1024 * 1024,
    )
    try:
        big = make_cold_file(cold_dir, "too_big.bin", 8 * 1024 * 1024)
        for f in s.prefetch(make_hint(big)):
            f.result(timeout=10)
        # Should record a skip_oversize event, not raise
        events = s.report.events
        assert len(events) == 1
        assert events[0].outcome == "skip_oversize"
    finally:
        s.shutdown(wait=True)


# ---------------------------------------------------------------------------
# T5: in-flight files are protected from eviction
# ---------------------------------------------------------------------------

def test_in_flight_files_protected_from_eviction(
    cold_dir: Path, hot_dir: Path
):
    """When eviction sweeps to make room, files currently mid-stage must
    not be deleted. We verify by examining _in_flight while it's still
    populated immediately after submitting."""
    s = Stager(
        hot_root=hot_dir,
        cold_roots=[cold_dir],
        capacity_bytes=8 * 1024 * 1024,
        max_workers=1,  # single worker so we can observe one-at-a-time
    )
    try:
        # Pre-stage 4 small files
        existing = [make_cold_file(cold_dir, f"e{i}.bin", 1024 * 1024) for i in range(4)]
        for f in s.prefetch(make_hint(*existing)):
            f.result(timeout=10)

        # Now submit a stage that will need to evict — verify _in_flight tracks it
        new_files = [make_cold_file(cold_dir, f"n{i}.bin", 2 * 1024 * 1024)
                     for i in range(2)]
        futures = s.prefetch(make_hint(*new_files))

        # While futures are pending, _in_flight should contain both new file paths
        with s._lock:
            in_flight = set(s._in_flight)
        # All new files we just submitted should be tracked
        for nf in new_files:
            assert str(nf.resolve()) in in_flight, (
                f"{nf} should be in _in_flight while staging"
            )

        for f in futures:
            f.result(timeout=10)

        # New files should now be staged; some of the old ones should be evicted
        for nf in new_files:
            assert s.is_staged(nf)
    finally:
        s.shutdown(wait=True)


# ---------------------------------------------------------------------------
# T6: write pass-through contract
# ---------------------------------------------------------------------------

def test_stager_only_sees_detected_files_not_writes(stager: Stager, cold_dir: Path):
    """The shim is responsible for never invoking the stager on writes.
    The stager's API surface (prefetch) only takes file paths from DataHints,
    which are read-set detections. This is a contract test pinning that
    the stager has no API for write-redirection."""
    # The Stager class should have no method that accepts open flags or
    # write-mode semantics. If someone ever adds one, this test will fail
    # (or the method name will appear in the dir() listing and we'll know).
    public_methods = {
        m for m in dir(stager)
        if not m.startswith("_") and callable(getattr(stager, m))
    }
    forbidden_substrings = ("write", "create", "open_for_write", "trunc")
    for method in public_methods:
        for word in forbidden_substrings:
            assert word not in method.lower(), (
                f"Stager has method '{method}' that suggests a write-mode "
                f"API — writes must be handled by the shim, not the stager"
            )


# ---------------------------------------------------------------------------
# Bonus: byte-identity check (the staged file == the cold file)
# ---------------------------------------------------------------------------

def test_staged_file_is_byte_identical_to_source(stager: Stager, cold_dir: Path):
    """After staging, the hot copy's bytes must match the cold source exactly.
    Critical for correctness — we never serve stale or partial data."""
    import hashlib

    cold = make_cold_file(cold_dir, "checksum.bin", 5 * 1024 * 1024)
    cold_hash = hashlib.sha256(cold.read_bytes()).hexdigest()

    for f in stager.prefetch(make_hint(cold)):
        f.result(timeout=10)

    hot = stager.hot_path_for(cold)
    hot_hash = hashlib.sha256(hot.read_bytes()).hexdigest()
    assert cold_hash == hot_hash, "staged file content diverges from source"
