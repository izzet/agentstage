"""Scoring primitives for AgentStage's predictor + stager.

`byte_metrics` is the load-bearing scorer: byte recall and overfetch
against static or empirical ground truth.

`empirical_gt` parses AgentIOBench-format `io_report.json` files into
empirical ground-truth records (what files the agent actually read,
with POSIX byte counts), used in E2's re-score of the PoC corpus
against the frozen rule library.
"""

from agentstage.metrics.byte_metrics import (
    ByteScore,
    PrefixMap,
    byte_score,
    clear_size_cache,
    file_size,
    resolve_path,
)
from agentstage.metrics.empirical_gt import (
    EmpiricalRead,
    empirical_paths,
    find_io_report,
    load_empirical_reads,
    total_bytes_read,
)

__all__ = [
    "ByteScore",
    "EmpiricalRead",
    "PrefixMap",
    "byte_score",
    "clear_size_cache",
    "empirical_paths",
    "file_size",
    "find_io_report",
    "load_empirical_reads",
    "resolve_path",
    "total_bytes_read",
]
