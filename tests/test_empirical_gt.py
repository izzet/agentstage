"""Unit tests for the io_report.json → EmpiricalRead loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentstage.metrics.empirical_gt import (
    EmpiricalRead,
    empirical_paths,
    find_io_report,
    load_empirical_reads,
    total_bytes_read,
)


def _write_io_report(path: Path, entries: list[dict]) -> None:
    """Write a minimal io_report.json with just the file_name_view we care about."""
    blob = {"file_name_view": entries, "raw_stats": {}}
    path.write_text(json.dumps(blob))


def test_load_filters_to_files_with_actual_reads(tmp_path: Path):
    _write_io_report(tmp_path / "r.json", [
        {"file_name": "/data/a.nc", "posix_count_sum": 5,  "posix_read_size_sum": 1_000_000, "posix_metadata_count_sum": 1, "posix_time_sum": 0.01},
        {"file_name": "/data/b.nc", "posix_count_sum": 3,  "posix_read_size_sum": 0,        "posix_metadata_count_sum": 3, "posix_time_sum": 0.001},  # stat'd, not read — drop
        {"file_name": "/data/c.nc", "posix_count_sum": 0,  "posix_read_size_sum": 0,        "posix_metadata_count_sum": 0, "posix_time_sum": 0.0},    # drop
    ])
    reads = load_empirical_reads(tmp_path / "r.json")
    assert [r.path for r in reads] == ["/data/a.nc"]
    assert reads[0].bytes_read == 1_000_000
    assert reads[0].posix_count == 5


def test_load_drops_output_paths_by_default(tmp_path: Path):
    _write_io_report(tmp_path / "r.json", [
        {"file_name": "/data/a.nc",      "posix_count_sum": 1, "posix_read_size_sum": 1024},
        {"file_name": "/output/r.parquet", "posix_count_sum": 1, "posix_read_size_sum": 1024},  # written-then-read
        {"file_name": "/repo/result/x.md", "posix_count_sum": 1, "posix_read_size_sum": 1024},  # written-then-read
    ])
    reads = load_empirical_reads(tmp_path / "r.json")
    assert [r.path for r in reads] == ["/data/a.nc"]


def test_load_includes_outputs_when_requested(tmp_path: Path):
    _write_io_report(tmp_path / "r.json", [
        {"file_name": "/data/a.nc",      "posix_count_sum": 1, "posix_read_size_sum": 1024},
        {"file_name": "/output/r.parquet", "posix_count_sum": 1, "posix_read_size_sum": 1024},
    ])
    reads = load_empirical_reads(tmp_path / "r.json", include_outputs=True)
    assert {r.path for r in reads} == {"/data/a.nc", "/output/r.parquet"}


def test_load_respects_min_bytes_threshold(tmp_path: Path):
    _write_io_report(tmp_path / "r.json", [
        {"file_name": "/data/big.nc",   "posix_count_sum": 1, "posix_read_size_sum": 100_000},
        {"file_name": "/data/small.nc", "posix_count_sum": 1, "posix_read_size_sum": 100},
    ])
    reads = load_empirical_reads(tmp_path / "r.json", min_bytes=10_000)
    assert [r.path for r in reads] == ["/data/big.nc"]


def test_load_sorts_by_bytes_read_descending(tmp_path: Path):
    _write_io_report(tmp_path / "r.json", [
        {"file_name": "/data/small.nc", "posix_count_sum": 1, "posix_read_size_sum": 100},
        {"file_name": "/data/big.nc",   "posix_count_sum": 1, "posix_read_size_sum": 10_000},
        {"file_name": "/data/med.nc",   "posix_count_sum": 1, "posix_read_size_sum": 1_000},
    ])
    reads = load_empirical_reads(tmp_path / "r.json")
    assert [r.path for r in reads] == ["/data/big.nc", "/data/med.nc", "/data/small.nc"]


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_empirical_reads(tmp_path / "nope.json")


def test_load_handles_missing_file_name_view(tmp_path: Path):
    (tmp_path / "r.json").write_text(json.dumps({"raw_stats": {}}))
    assert load_empirical_reads(tmp_path / "r.json") == []


def test_empirical_paths_helper():
    reads = [
        EmpiricalRead("/data/a", 1, 100, 0, 0.0),
        EmpiricalRead("/data/b", 1, 200, 0, 0.0),
    ]
    assert empirical_paths(reads) == ("/data/a", "/data/b")


def test_total_bytes_read_helper():
    reads = [
        EmpiricalRead("/data/a", 1, 100, 0, 0.0),
        EmpiricalRead("/data/b", 1, 200, 0, 0.0),
    ]
    assert total_bytes_read(reads) == 300


def test_find_io_report_flat_layout(tmp_path: Path):
    (tmp_path / "aiob_104_haiku_pp_s0").mkdir()
    (tmp_path / "aiob_104_haiku_pp_s0" / "io_report.json").write_text("{}")
    hit = find_io_report(tmp_path, "aiob_104", model="haiku", seed=0)
    assert hit is not None
    assert hit.name == "io_report.json"


def test_find_io_report_returns_none_when_missing(tmp_path: Path):
    assert find_io_report(tmp_path, "nonexistent") is None
