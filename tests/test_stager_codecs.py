"""Tests for compression-aware staging and companion-index co-staging.

The governing invariant: the hot copy is byte-identical to what the tool
expects **at the path it opens**. Expansion happens only when the target path
differs from the cold artifact holding its bytes.
"""

from __future__ import annotations

import gzip
import lzma
from pathlib import Path

import pytest

from agentstage.stager import DataHint, Stager
from agentstage.stager.codecs import (
    FALLBACK_RATIO,
    companions,
    materialise,
    resolve,
)

PLAIN = b"sample,value\nHG00096,0.42\n" * 200


@pytest.fixture()
def cold(tmp_path: Path) -> Path:
    d = tmp_path / "cold"
    d.mkdir()
    return d


@pytest.fixture()
def stager(tmp_path: Path, cold: Path) -> Stager:
    s = Stager(hot_root=tmp_path / "hot", cold_roots=[cold], max_workers=2)
    yield s
    s.shutdown(wait=True)


def _stage_and_wait(stager: Stager, *targets: Path) -> None:
    stager.prefetch(DataHint(
        detected_files=tuple(str(t) for t in targets),
        tier=1, fired_at_ms=0.0, rule_id="test",
    ))
    stager.wait_for_all(timeout=30)


class TestResolve:
    def test_plain_file_resolves_to_itself(self, cold: Path):
        target = cold / "x.csv"
        target.write_bytes(PLAIN)
        plan = resolve(target)
        assert plan is not None
        assert plan.source == target
        assert not plan.expands
        assert plan.size_bytes == len(PLAIN)

    def test_compressed_only_target_resolves_through_a_codec(self, cold: Path):
        src = cold / "x.csv.gz"
        src.write_bytes(gzip.compress(PLAIN))
        plan = resolve(cold / "x.csv")
        assert plan is not None
        assert plan.source == src
        assert plan.expands
        assert plan.codec.suffix == ".gz"

    def test_target_wins_over_compressed_variant(self, cold: Path):
        """A tool opening x.csv.gz means to decompress it itself. Staging the
        expanded bytes under that name would break it, so a present target is
        always a plain copy."""
        (cold / "x.csv").write_bytes(PLAIN)
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        plan = resolve(cold / "x.csv")
        assert not plan.expands

    def test_compressed_target_is_copied_not_expanded(self, cold: Path):
        src = cold / "x.csv.gz"
        src.write_bytes(gzip.compress(PLAIN))
        plan = resolve(src)
        assert plan.source == src
        assert not plan.expands

    def test_missing_everything_is_a_miss(self, cold: Path):
        assert resolve(cold / "absent.csv") is None

    def test_gzip_size_comes_from_the_isize_trailer(self, cold: Path):
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        assert resolve(cold / "x.csv").size_bytes == len(PLAIN)

    def test_zstd_target_resolves_and_round_trips(self, cold: Path, tmp_path: Path):
        """zstandard is an optional extra. When present the .zst path must
        behave like every other codec; when absent resolve must skip it
        rather than raise."""
        zstd = pytest.importorskip("zstandard")
        src = cold / "x.csv.zst"
        src.write_bytes(zstd.ZstdCompressor().compress(PLAIN))

        plan = resolve(cold / "x.csv")
        assert plan is not None and plan.expands
        assert plan.codec.suffix == ".zst"

        dest = tmp_path / "out"
        materialise(plan, dest)
        assert dest.read_bytes() == PLAIN

    def test_zstd_source_is_skipped_when_the_codec_is_unavailable(
        self, cold: Path, monkeypatch
    ):
        """Without the optional dependency a .zst-only target is a miss, so
        the tool falls through to cold instead of hitting an ImportError."""
        (cold / "x.csv.zst").write_bytes(b"irrelevant")
        monkeypatch.setattr(
            "agentstage.stager.codecs.Codec.available",
            lambda self: self.suffix != ".zst",
        )
        assert resolve(cold / "x.csv") is None

    def test_codec_without_a_size_trailer_uses_the_ratio_estimate(self, cold: Path):
        src = cold / "x.csv.xz"
        src.write_bytes(lzma.compress(PLAIN))
        plan = resolve(cold / "x.csv")
        assert plan.codec.suffix == ".xz"
        assert plan.size_bytes == src.stat().st_size * FALLBACK_RATIO


class TestMaterialise:
    def test_expansion_round_trips(self, cold: Path, tmp_path: Path):
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        dest = tmp_path / "out"
        materialise(resolve(cold / "x.csv"), dest)
        assert dest.read_bytes() == PLAIN

    def test_plain_copy_round_trips(self, cold: Path, tmp_path: Path):
        (cold / "x.csv").write_bytes(PLAIN)
        dest = tmp_path / "out"
        materialise(resolve(cold / "x.csv"), dest)
        assert dest.read_bytes() == PLAIN


class TestStagerExpansion:
    def test_hot_copy_lands_at_the_path_the_tool_opens(self, stager, cold):
        """The whole point: the shim maps cold->hot by path, so the expanded
        bytes must appear at the target's mirror, not the archive's."""
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        target = cold / "x.csv"

        _stage_and_wait(stager, target)

        hot = stager.hot_path_for(target)
        assert hot.is_file()
        assert hot.read_bytes() == PLAIN
        assert stager.is_staged(target)

    def test_shim_needs_no_codec_knowledge(self, stager, cold):
        """hot_path_for is a pure path mapping; expansion must not perturb it."""
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        target = cold / "x.csv"
        expected = stager.hot_root / str(target.resolve()).lstrip("/")

        _stage_and_wait(stager, target)

        assert stager.hot_path_for(target) == expected

    def test_corrupt_archive_leaves_nothing_at_the_hot_path(self, stager, cold):
        """A truncated archive must not publish a partial file; the tool falls
        through to cold instead."""
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN)[: 40])
        target = cold / "x.csv"

        _stage_and_wait(stager, target)

        assert not stager.hot_path_for(target).exists()
        assert not stager.is_staged(target)
        assert any(e.outcome == "error" for e in stager.report.events)

    def test_no_temp_files_survive(self, stager, cold):
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        (cold / "bad.csv.gz").write_bytes(b"not gzip at all")

        _stage_and_wait(stager, cold / "x.csv", cold / "bad.csv")

        leftovers = [p for p in stager.hot_root.rglob("*") if ".tmp." in p.name]
        assert leftovers == []

    def test_missing_target_records_an_error_and_stages_nothing(self, stager, cold):
        _stage_and_wait(stager, cold / "absent.csv")
        assert not stager.is_staged(cold / "absent.csv")
        assert any(e.outcome == "error" for e in stager.report.events)


class TestCompanions:
    def test_bam_pulls_its_index_both_conventions(self, cold: Path):
        (cold / "a.bam").write_bytes(b"bam")
        (cold / "a.bam.bai").write_bytes(b"idx")
        (cold / "b.bam").write_bytes(b"bam")
        (cold / "b.bai").write_bytes(b"idx")

        assert companions(cold / "a.bam") == (cold / "a.bam.bai",)
        assert companions(cold / "b.bam") == (cold / "b.bai",)

    def test_absent_index_yields_nothing(self, cold: Path):
        (cold / "a.bam").write_bytes(b"bam")
        assert companions(cold / "a.bam") == ()

    def test_compressed_vcf_keys_off_the_inner_suffix(self, cold: Path):
        (cold / "v.vcf.gz").write_bytes(b"vcf")
        (cold / "v.vcf.gz.tbi").write_bytes(b"idx")
        assert companions(cold / "v.vcf.gz") == (cold / "v.vcf.gz.tbi",)

    def test_formats_without_an_index_touch_the_filesystem_zero_times(
        self, cold: Path, monkeypatch
    ):
        """Gating on the suffix with string ops is what keeps a 22,500-image
        dispatch from paying metadata RPCs it cannot use."""
        calls = []
        real = Path.is_file
        monkeypatch.setattr(Path, "is_file",
                            lambda self: calls.append(self) or real(self))

        assert companions(cold / "photo.jpg") == ()
        assert calls == []

    def test_staging_a_bam_co_stages_its_index(self, stager, cold):
        (cold / "a.bam").write_bytes(b"bam-bytes")
        (cold / "a.bam.bai").write_bytes(b"index-bytes")

        _stage_and_wait(stager, cold / "a.bam")

        assert stager.is_staged(cold / "a.bam")
        assert stager.is_staged(cold / "a.bam.bai")
        assert stager.hot_path_for(cold / "a.bam.bai").read_bytes() == b"index-bytes"


class TestExistingBehaviourPreserved:
    def test_plain_staging_is_unchanged(self, stager, cold):
        (cold / "plain.nc").write_bytes(PLAIN)
        _stage_and_wait(stager, cold / "plain.nc")
        assert stager.hot_path_for(cold / "plain.nc").read_bytes() == PLAIN

    def test_restaging_is_idempotent(self, stager, cold):
        (cold / "x.csv.gz").write_bytes(gzip.compress(PLAIN))
        target = cold / "x.csv"

        _stage_and_wait(stager, target)
        first = stager.hot_path_for(target).stat().st_mtime_ns
        _stage_and_wait(stager, target)

        # Same Stager: deduped by _in_flight, so _stage never re-runs.
        assert stager.hot_path_for(target).stat().st_mtime_ns == first

        # Fresh Stager on the same hot root: _stage runs and takes the
        # already-present branch instead of recopying.
        other = Stager(hot_root=stager.hot_root, cold_roots=list(stager.cold_roots),
                       max_workers=1)
        try:
            _stage_and_wait(other, target)
            assert stager.hot_path_for(target).stat().st_mtime_ns == first
            assert any(e.outcome == "hit" for e in other.report.events)
        finally:
            other.shutdown(wait=True)

    def test_files_outside_managed_cold_roots_are_ignored(self, stager, tmp_path):
        outside = tmp_path / "elsewhere.csv"
        outside.write_bytes(PLAIN)
        _stage_and_wait(stager, outside)
        assert not stager.is_staged(outside)
