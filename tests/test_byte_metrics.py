"""Unit tests for the byte-recall + overfetch scorer."""

from __future__ import annotations

import os
from pathlib import Path

from agentstage.metrics.byte_metrics import (
    ByteScore,
    byte_score,
    clear_size_cache,
    file_size,
    resolve_path,
)


def test_resolve_path_first_prefix_wins(tmp_path: Path):
    prefix_map = (
        ("/data/foo/", str(tmp_path / "foo") + "/"),
        ("/data/", str(tmp_path) + "/"),
    )
    # Most-specific prefix wins because it's listed first
    assert resolve_path("/data/foo/x.nc", prefix_map) == str(tmp_path / "foo") + "/x.nc"
    assert resolve_path("/data/bar/y.nc", prefix_map) == str(tmp_path) + "/bar/y.nc"


def test_resolve_path_no_match_returns_unchanged():
    assert resolve_path("/other/path", (("/data/", "/real/"),)) == "/other/path"


def test_file_size_missing_returns_zero(tmp_path: Path):
    clear_size_cache()
    assert file_size(str(tmp_path / "nonexistent.bin")) == 0


def test_file_size_with_prefix_map(tmp_path: Path):
    clear_size_cache()
    real = tmp_path / "data" / "f.bin"
    real.parent.mkdir()
    real.write_bytes(b"x" * 1024)
    prefix_map = (("/data/", str(tmp_path / "data") + "/"),)
    assert file_size("/data/f.bin", prefix_map) == 1024


def test_byte_score_perfect_recall(tmp_path: Path):
    clear_size_cache()
    for name, n in (("a.nc", 100), ("b.nc", 200), ("c.nc", 300)):
        (tmp_path / name).write_bytes(b"x" * n)
    prefix_map = (("/data/", str(tmp_path) + "/"),)
    pred = ("/data/a.nc", "/data/b.nc", "/data/c.nc")
    gt = pred
    score = byte_score(pred, gt, prefix_map)
    assert score.n_detected == 3
    assert score.n_overlap == 3
    assert score.bytes_detected == 600
    assert score.bytes_ground_truth == 600
    assert score.byte_recall == 1.0
    assert score.byte_overfetch == 1.0


def test_byte_score_goes_collapse_arithmetic(tmp_path: Path):
    """Synthetic version of the §6.3 GOES collapse: 1 small file as gt,
    1000 small files in the naive baseline → ~1000× overfetch."""
    clear_size_cache()
    for i in range(1000):
        (tmp_path / f"f{i}.nc").write_bytes(b"x" * 1000)  # 1 KB each
    prefix_map = (("/data/", str(tmp_path) + "/"),)
    gt = ("/data/f0.nc",)
    naive = tuple(f"/data/f{i}.nc" for i in range(1000))
    score = byte_score(naive, gt, prefix_map)
    assert score.byte_recall == 1.0
    assert score.byte_overfetch == 1000.0
    assert score.n_detected == 1000


def test_byte_score_partial_recall(tmp_path: Path):
    clear_size_cache()
    sizes = {"a.nc": 1000, "b.nc": 2000, "c.nc": 4000}
    for name, n in sizes.items():
        (tmp_path / name).write_bytes(b"x" * n)
    prefix_map = (("/data/", str(tmp_path) + "/"),)
    gt = ("/data/a.nc", "/data/b.nc", "/data/c.nc")
    pred = ("/data/a.nc", "/data/b.nc")  # missing c.nc
    score = byte_score(pred, gt, prefix_map)
    assert score.bytes_detected == 3000
    assert score.bytes_ground_truth == 7000
    assert score.bytes_overlap == 3000
    assert abs(score.byte_recall - 3000 / 7000) < 1e-9
    assert abs(score.byte_overfetch - 3000 / 7000) < 1e-9


def test_byte_score_dedupes_inputs(tmp_path: Path):
    clear_size_cache()
    (tmp_path / "a.nc").write_bytes(b"x" * 500)
    prefix_map = (("/data/", str(tmp_path) + "/"),)
    score = byte_score(
        ("/data/a.nc", "/data/a.nc", "/data/a.nc"),
        ("/data/a.nc", "/data/a.nc"),
        prefix_map,
    )
    assert score.n_detected == 1
    assert score.n_ground_truth == 1


def test_byte_score_empty_ground_truth_yields_zero_recall_inf_overfetch(tmp_path: Path):
    clear_size_cache()
    (tmp_path / "a.nc").write_bytes(b"x" * 100)
    score = byte_score(("/data/a.nc",), (), (("/data/", str(tmp_path) + "/"),))
    assert score.byte_recall == 0.0
    assert score.byte_overfetch == float("inf")


def test_byte_score_to_dict_round_trip():
    score = ByteScore(
        n_detected=2, n_ground_truth=3, n_overlap=1,
        bytes_detected=2000, bytes_ground_truth=6000, bytes_overlap=1000,
    )
    d = score.to_dict()
    assert d["byte_recall"] == 1000 / 6000
    assert d["byte_overfetch"] == 2000 / 6000
    assert d["file_recall"] == 1 / 3
    assert d["file_precision"] == 1 / 2
