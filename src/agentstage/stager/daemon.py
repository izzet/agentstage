"""In-process stager.

Despite the filename, this is NOT a separate daemon process. It runs as a
ThreadPoolExecutor inside the AgentStageClient's address space. The
filesystem (atomic rename of <hot_path>.tmp -> <hot_path>) is the only IPC
between the stager and the LD_PRELOAD shim.

Lifecycle:
  client_lib.prefetch(hint) -> Stager.prefetch(hint) -> executor.submit(_stage)
  _stage(cold_path, hint):
    1. compute hot_path
    2. ensure capacity (LRU sweep if needed)
    3. shutil.copy(cold_path, tmp_path)
    4. os.rename(tmp_path, hot_path)  # atomic
    5. record StageEvent in StagingReport
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .report import DataHint, StageEvent, StagingReport, now_ms


GB = 1024 ** 3


class StagerOutOfSpace(RuntimeError):
    """Raised when LRU eviction can't free enough capacity for an incoming file."""


class Stager:
    """In-process file pre-stager.

    Thread-safe. Designed for single-agent use; the executor is sized for
    that case (default 4 workers — enough to pipeline 3-4 stages during
    one slack window without saturating cold-tier bandwidth).

    Invariants:
      - prefetch(hint) is non-blocking; copies happen in background threads.
      - Files in self._in_flight cannot be evicted by LRU sweep.
      - _stage's atomic rename means hot_path either does not exist or
        contains the full final bytes — never a partial file.
      - Re-prefetching a cold_path is a no-op (idempotency).
    """

    def __init__(
        self,
        hot_root: str | Path,
        cold_roots: list[str | Path],
        *,
        max_workers: int = 4,
        capacity_bytes: int = 32 * GB,
        report: StagingReport | None = None,
    ) -> None:
        self.hot_root = Path(hot_root).resolve()
        self.cold_roots = tuple(Path(r).resolve() for r in cold_roots)
        self.capacity_bytes = capacity_bytes
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agentstage-stage",
        )
        self.report = report if report is not None else StagingReport()
        self._in_flight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self.hot_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def prefetch(self, hint: DataHint) -> list[Future]:
        """Submit copy jobs for each predicted file. Returns Futures so
        callers can wait() in tests; production callers fire-and-forget."""
        futures: list[Future] = []
        for cold_path in hint.predicted_files:
            f = self._submit_one(cold_path, hint)
            if f is not None:
                futures.append(f)
        return futures

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting new work; optionally block until in-flight stages finish."""
        self._executor.shutdown(wait=wait)

    def hot_path_for(self, cold_path: str | Path) -> Path:
        """Return the hot mirror path for a cold path. Uses absolute-path
        mirroring under `self.hot_root`."""
        abs_cold = str(Path(cold_path).resolve())
        # Strip leading "/" so we get hot_root/mnt/.../foo.nc not double-rooted
        return self.hot_root / abs_cold.lstrip("/")

    def is_staged(self, cold_path: str | Path) -> bool:
        """True if the file is fully staged (rename completed)."""
        return self.hot_path_for(cold_path).exists()

    def wait_for_all(self, timeout: float | None = None) -> None:
        """Block until every currently-known stage future has completed.
        Test helper; production code should not call this."""
        with self._lock:
            futures = list(self._in_flight.values())
        for f in futures:
            f.result(timeout=timeout)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _submit_one(self, cold_path: str, hint: DataHint) -> Future | None:
        """Submit one stage job; return the Future (or None if no-op)."""
        cold_abs = str(Path(cold_path).resolve())

        # Only stage files under managed cold roots.
        if not self._under_managed_cold_root(cold_abs):
            return None

        with self._lock:
            existing = self._in_flight.get(cold_abs)
            if existing is not None:
                return existing
            future = self._executor.submit(self._stage, cold_abs, hint)
            self._in_flight[cold_abs] = future
        return future

    def _under_managed_cold_root(self, abs_path: str) -> bool:
        return any(
            abs_path.startswith(str(root) + os.sep) or abs_path == str(root)
            for root in self.cold_roots
        )

    def _stage(self, cold_path: str, hint: DataHint) -> StageEvent:
        """Worker function: copy cold -> tmp -> rename to hot. Records event.
        Idempotent: if hot already exists, returns a 'hit' event."""
        t_completed = lambda: now_ms()  # noqa: E731
        hot_path = self.hot_path_for(cold_path)

        # Idempotency: another worker may have completed already.
        if hot_path.exists():
            ev = StageEvent(
                cold_path=cold_path,
                hot_path=str(hot_path),
                size_bytes=hot_path.stat().st_size,
                fetch_ms=0.0,
                tier=hint.tier,
                rule_id=hint.rule_id,
                t_predicted_ms=hint.fired_at_ms,
                t_completed_ms=t_completed(),
                outcome="hit",
            )
            self.report.record(ev)
            return ev

        try:
            size = os.path.getsize(cold_path)
        except OSError as exc:
            return self._record_error(cold_path, hot_path, hint, exc)

        if size > self.capacity_bytes:
            ev = StageEvent(
                cold_path=cold_path,
                hot_path=str(hot_path),
                size_bytes=size,
                fetch_ms=0.0,
                tier=hint.tier,
                rule_id=hint.rule_id,
                t_predicted_ms=hint.fired_at_ms,
                t_completed_ms=t_completed(),
                outcome="skip_oversize",
                error=f"file size {size} > capacity {self.capacity_bytes}",
            )
            self.report.record(ev)
            return ev

        try:
            self._ensure_capacity_for(size)
        except StagerOutOfSpace as exc:
            return self._record_error(cold_path, hot_path, hint, exc)

        hot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hot_path.with_suffix(
            hot_path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
        )

        t_start = time.monotonic_ns()
        try:
            shutil.copyfile(cold_path, tmp)
            os.rename(tmp, hot_path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return self._record_error(cold_path, hot_path, hint, exc)
        fetch_ms = (time.monotonic_ns() - t_start) / 1e6

        ev = StageEvent(
            cold_path=cold_path,
            hot_path=str(hot_path),
            size_bytes=size,
            fetch_ms=fetch_ms,
            tier=hint.tier,
            rule_id=hint.rule_id,
            t_predicted_ms=hint.fired_at_ms,
            t_completed_ms=t_completed(),
            outcome="staged",
        )
        self.report.record(ev)
        return ev

    def _record_error(
        self,
        cold_path: str,
        hot_path: Path,
        hint: DataHint,
        exc: BaseException,
    ) -> StageEvent:
        ev = StageEvent(
            cold_path=cold_path,
            hot_path=str(hot_path),
            size_bytes=0,
            fetch_ms=0.0,
            tier=hint.tier,
            rule_id=hint.rule_id,
            t_predicted_ms=hint.fired_at_ms,
            t_completed_ms=now_ms(),
            outcome="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        self.report.record(ev)
        return ev

    # -----------------------------------------------------------------
    # Capacity / eviction
    # -----------------------------------------------------------------

    def _ensure_capacity_for(self, incoming_size: int) -> None:
        """Evict LRU files (by atime) until incoming_size fits. Files
        currently being copied (futures not yet done) are protected from
        eviction; already-completed stages are eligible."""
        with self._lock:
            in_flight_hot_paths = {
                str(self.hot_path_for(c))
                for c, fut in self._in_flight.items()
                if not fut.done()
            }

        used = self._scan_used_bytes()
        if used + incoming_size <= self.capacity_bytes:
            return

        to_free = (used + incoming_size) - self.capacity_bytes
        freed = 0

        # Gather (atime, path, size) of all files in hot_root, sorted oldest first
        candidates: list[tuple[float, Path, int]] = []
        for f in self.hot_root.rglob("*"):
            if not f.is_file():
                continue
            if str(f) in in_flight_hot_paths:
                continue
            if f.name.endswith(".tmp") or ".tmp." in f.name:
                # Don't evict in-flight tmp files
                continue
            try:
                st = f.stat()
                candidates.append((st.st_atime, f, st.st_size))
            except OSError:
                continue
        candidates.sort(key=lambda t: t[0])

        for _atime, path, size in candidates:
            try:
                path.unlink()
                freed += size
            except OSError:
                continue
            if freed >= to_free:
                return

        if freed < to_free:
            raise StagerOutOfSpace(
                f"could not free {to_free} bytes (freed {freed}); "
                f"capacity={self.capacity_bytes}, used~={used}"
            )

    def _scan_used_bytes(self) -> int:
        """Walk hot_root and sum file sizes. O(n_files); cheap on tmpfs/NVMe."""
        total = 0
        for f in self.hot_root.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
        return total
