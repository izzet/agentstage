"""Empirical ground-truth loader — parses AgentIOBench `io_report.json`
files into structured records describing what files the agent actually read.

For E2's re-score of the PoC corpus against the frozen rule library, the
empirical GT replaces the static "expert thinks the agent should read X"
ground truth with the empirical "the agent did read X with N POSIX bytes"
ground truth. The byte-recall denominator becomes the agent's actual
read volume instead of an expert's enumeration.

io_report.json schema (sampled from sciiobench outputs on 2026-05-19):

    {
      "raw_stats": {...},
      "file_name_view": [
        {
          "file_name": "/data/.../somefile.nc",
          "posix_count_sum": 12.0,        # number of POSIX ops on this file
          "posix_read_size_sum": 524288,  # bytes read via POSIX
          "posix_metadata_count_sum": 3.0,
          "posix_time_sum": 0.001234,
          ...
        },
        ...
      ],
      "step_view": [...],   # per-turn breakdown
      "decomposition": {...}  # solution_io vs exploration_io
    }

We extract `file_name_view[*]` entries with `posix_count_sum > 0` AND
`posix_read_size_sum > 0` (filtering out files that were only stat'd
but not read) and filter to input paths (no `/output/`, `/result/`
prefixes — those are write targets).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


# Output-path prefixes excluded from empirical reads — these are write
# targets, not staging candidates. Matches the PoC's _OUTPUT_PREFIX_PATTERNS.
_OUTPUT_PREFIXES = ("/output/", "/repo/result/")


@dataclass(frozen=True)
class EmpiricalRead:
    """One file's empirical I/O profile from an io_report.json."""

    path: str
    posix_count: int       # number of POSIX ops (reads + opens)
    bytes_read: int        # POSIX read bytes (load-bearing for byte_recall)
    metadata_count: int    # POSIX metadata ops (stat, fstat, getxattr, ...)
    posix_time_s: float    # wall time spent in POSIX I/O on this file

    @property
    def is_output_path(self) -> bool:
        return any(self.path.startswith(p) for p in _OUTPUT_PREFIXES)


def load_empirical_reads(
    io_report_path: str | Path,
    include_outputs: bool = False,
    min_bytes: int = 1,
    min_posix_count: int = 1,
) -> list[EmpiricalRead]:
    """Parse an io_report.json into per-file empirical-read records.

    Args:
        io_report_path: Path to the io_report.json file.
        include_outputs: If False (default), drop files under /output/
            and /repo/result/ — these are write targets, not staging
            candidates.
        min_bytes: Minimum `posix_read_size_sum` to include (default 1 —
            drops files that were stat'd but never read).
        min_posix_count: Minimum `posix_count_sum` to include (default 1).

    Returns:
        List of EmpiricalRead, sorted by `bytes_read` descending.
    """
    p = Path(io_report_path)
    if not p.is_file():
        raise FileNotFoundError(f"io_report.json not found: {p}")

    blob = json.loads(p.read_text())
    file_view = blob.get("file_name_view") or []
    if not isinstance(file_view, list):
        raise ValueError(
            f"{p}: file_name_view is {type(file_view).__name__}, expected list"
        )

    reads: list[EmpiricalRead] = []
    for entry in file_view:
        if not isinstance(entry, dict):
            continue
        path = entry.get("file_name")
        if not isinstance(path, str):
            continue
        posix_count = int(entry.get("posix_count_sum") or 0)
        bytes_read = int(entry.get("posix_read_size_sum") or 0)
        if posix_count < min_posix_count or bytes_read < min_bytes:
            continue
        rec = EmpiricalRead(
            path=path,
            posix_count=posix_count,
            bytes_read=bytes_read,
            metadata_count=int(entry.get("posix_metadata_count_sum") or 0),
            posix_time_s=float(entry.get("posix_time_sum") or 0.0),
        )
        if not include_outputs and rec.is_output_path:
            continue
        reads.append(rec)

    reads.sort(key=lambda r: r.bytes_read, reverse=True)
    return reads


def empirical_paths(reads: Iterable[EmpiricalRead]) -> tuple[str, ...]:
    """Just the paths from a list of EmpiricalRead — for byte_score()."""
    return tuple(r.path for r in reads)


def total_bytes_read(reads: Iterable[EmpiricalRead]) -> int:
    """Sum of POSIX read bytes across all records."""
    return sum(r.bytes_read for r in reads)


def find_io_report(
    outputs_root: str | Path,
    task_id: str,
    model: str | None = None,
    seed: int | None = None,
) -> Path | None:
    """Locate an io_report.json under an outputs root by (task, model, seed).

    The search is best-effort: AIOB historical outputs use a flat
    `<mode>_<knowledge>_<model>_<task>_rep<N>/` naming convention, while
    AgentStage's new campaign uses `<task>_<model>_<config>_s<seed>/`.
    Returns the first matching directory's io_report.json, or None if
    no match is found.
    """
    root = Path(outputs_root)
    if not root.is_dir():
        return None
    candidates = sorted(root.glob(f"*{task_id}*"))
    for cand in candidates:
        if not cand.is_dir():
            continue
        if model is not None and model not in cand.name:
            continue
        if seed is not None and f"s{seed}" not in cand.name and f"rep{seed}" not in cand.name:
            continue
        report = cand / "io_report.json"
        if report.is_file():
            return report
    return None
