"""Staging report — JSON-serializable per-stage records + summary statistics."""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


StageOutcome = Literal["staged", "hit", "error", "skip_oversize"]


@dataclass(frozen=True)
class StageEvent:
    """One staging attempt — emitted when _stage runs.

    Field semantics:
      - cold_path / hot_path: absolute paths of source and destination.
      - size_bytes: source file size at stage time (may differ from final
        hot size if a write race interleaves; we capture the cold size).
      - fetch_ms: wall-clock time from start of shutil.copy to atomic
        rename. Zero on `hit` (file already staged before this call).
      - tier: 1/2/3 from the DataHint that triggered the stage.
      - rule_id: predictor rule that fired.
      - t_predicted_ms: monotonic time when the predictor emitted the hint.
      - t_completed_ms: monotonic time when rename finished.
      - outcome: "staged" (fresh copy), "hit" (already present), "error"
        (exception during copy/rename), "skip_oversize" (file larger than
        capacity even after eviction).
      - error: stringified exception if outcome == "error", else "".
    """

    cold_path: str
    hot_path: str
    size_bytes: int
    fetch_ms: float
    tier: int
    rule_id: str
    t_predicted_ms: float
    t_completed_ms: float
    outcome: StageOutcome
    error: str = ""


@dataclass
class StagingReport:
    """Thread-safe accumulator for stage events. Serializable to JSON."""

    events: list[StageEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, event: StageEvent) -> None:
        with self._lock:
            self.events.append(event)

    def summary(self) -> dict:
        with self._lock:
            events = list(self.events)
        if not events:
            return {"n_events": 0}

        staged = [e for e in events if e.outcome == "staged"]
        hits = [e for e in events if e.outcome == "hit"]
        errors = [e for e in events if e.outcome == "error"]

        out: dict = {
            "n_events": len(events),
            "n_staged": len(staged),
            "n_hits": len(hits),
            "n_errors": len(errors),
            "total_bytes_staged": sum(e.size_bytes for e in staged),
        }
        if staged:
            fetch_times = [e.fetch_ms for e in staged]
            out["fetch_ms"] = {
                "p50": round(statistics.median(fetch_times), 3),
                "p95": round(_percentile(fetch_times, 0.95), 3),
                "max": round(max(fetch_times), 3),
                "mean": round(statistics.mean(fetch_times), 3),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "events": [asdict(e) for e in self.events],
            "summary": self.summary(),
        }

    def write(self, path: Path) -> None:
        """Atomically write report to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        tmp.rename(path)


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# DataHint — re-exported here to keep the stager module self-contained.
# When the client library lands (T18), it'll import DataHint from here too.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataHint:
    """Predictor output: a set of files predicted to be read next, with
    metadata about why and when. Consumed by Stager.prefetch().
    """

    predicted_files: tuple[str, ...]
    tier: int
    fired_at_ms: float
    rule_id: str
    byte_estimate: int = 0
    signature: str = ""

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1/2/3, got {self.tier}")


def now_ms() -> float:
    """Monotonic time in milliseconds. Stager uses time.monotonic_ns
    internally; this is a convenience for tests + the client lib."""
    return time.monotonic_ns() / 1e6
