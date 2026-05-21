"""Byte-level recall and overfetch — the load-bearing scoring functions
for AgentStage's detector evaluation.

`byte_recall` = bytes of (detected ∩ ground_truth) / bytes of ground_truth.
`byte_overfetch` = bytes of detected / bytes of ground_truth.

Both use **logical** file paths (the paths the rule library and
detector emit, e.g. `/data/igsr_coverage_qc/raw/data/HG00096/...`); a
`prefix_map` translates logical → real on-disk paths for `os.path.getsize`
lookups. Sizes are cached per-process to avoid repeated stat calls when
the same workload is scored against many seeds.

Ported from `poc/probe_reasoning_slack.py::byte_score` on 2026-05-19.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PrefixMap = tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# File-size resolution
# ---------------------------------------------------------------------------

def resolve_path(logical_path: str, prefix_map: PrefixMap) -> str:
    """Translate a logical path through the prefix_map. First matching
    prefix wins; if nothing matches, the path is returned unchanged."""
    for logical_prefix, real_prefix in prefix_map:
        if logical_path.startswith(logical_prefix):
            return real_prefix + logical_path[len(logical_prefix):]
    return logical_path


@lru_cache(maxsize=200_000)
def _cached_getsize(real_path: str) -> int:
    """`os.path.getsize` with process-wide caching. Returns 0 on missing
    files (matching the PoC's behavior — output/synthetic paths in the
    workspace prior may not exist on disk yet)."""
    p = Path(real_path)
    try:
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def file_size(logical_path: str, prefix_map: PrefixMap = ()) -> int:
    """Bytes on disk for a logical path. 0 if the file doesn't exist
    (output paths, missing data, etc.)."""
    real = resolve_path(logical_path, prefix_map)
    return _cached_getsize(real)


def clear_size_cache() -> None:
    """Drop the size cache. Tests + long-running campaigns should call
    this between workloads if filesystem state may have changed."""
    _cached_getsize.cache_clear()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ByteScore:
    """Byte-level recall + overfetch for a (detected, ground_truth) pair.

    The "byte" framing matters because file sizes vary by orders of
    magnitude on the workloads we care about — counting file *count*
    recall on aiob_107 would mark a one-file detection as 1/6042 recall
    even when that one file IS the 3 MB immediate need.
    """

    n_detected: int
    n_ground_truth: int
    n_overlap: int
    bytes_detected: int
    bytes_ground_truth: int
    bytes_overlap: int

    @property
    def byte_recall(self) -> float:
        return (
            self.bytes_overlap / self.bytes_ground_truth
            if self.bytes_ground_truth
            else 0.0
        )

    @property
    def byte_overfetch(self) -> float:
        return (
            self.bytes_detected / self.bytes_ground_truth
            if self.bytes_ground_truth
            else float("inf")
        )

    @property
    def file_recall(self) -> float:
        return self.n_overlap / self.n_ground_truth if self.n_ground_truth else 0.0

    @property
    def file_precision(self) -> float:
        return self.n_overlap / self.n_detected if self.n_detected else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_detected": self.n_detected,
            "n_ground_truth": self.n_ground_truth,
            "n_overlap": self.n_overlap,
            "bytes_detected": self.bytes_detected,
            "bytes_ground_truth": self.bytes_ground_truth,
            "bytes_overlap": self.bytes_overlap,
            "byte_recall": self.byte_recall,
            "byte_overfetch": self.byte_overfetch,
            "file_recall": self.file_recall,
            "file_precision": self.file_precision,
        }


def byte_score(
    detected: Iterable[str],
    ground_truth: Iterable[str],
    prefix_map: PrefixMap = (),
) -> ByteScore:
    """Compute byte recall + overfetch for one (detected, ground_truth) pair.

    Both sets are de-duped before scoring. `prefix_map` is used to resolve
    logical paths for `os.path.getsize`; pass `()` for already-resolved
    real paths.
    """
    p_set = set(detected)
    g_set = set(ground_truth)
    overlap = p_set & g_set

    bytes_pred = sum(file_size(p, prefix_map) for p in p_set)
    bytes_gt = sum(file_size(p, prefix_map) for p in g_set)
    bytes_overlap = sum(file_size(p, prefix_map) for p in overlap)

    return ByteScore(
        n_detected=len(p_set),
        n_ground_truth=len(g_set),
        n_overlap=len(overlap),
        bytes_detected=bytes_pred,
        bytes_ground_truth=bytes_gt,
        bytes_overlap=bytes_overlap,
    )
