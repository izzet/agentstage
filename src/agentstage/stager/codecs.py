"""Compression codecs and companion-file rules for staging.

The stager is handed the path the tool will open. This module answers two
questions about that path:

1. What on the cold tier can produce it, and how? The target may be present
   verbatim, or only as a compressed artifact that has to be expanded.
2. What else does the tool need beside it? Formats with an external index are
   unreadable for random access without their companion file.

The invariant throughout is that the hot copy is byte-identical to whatever
the tool expects **at the path it opens**. Expansion happens only when the
target path differs from the cold artifact holding its bytes, i.e. the tool
opens `x.csv` and the cold tier holds `x.csv.gz`. A tool that opens `x.csv.gz`
itself gets a plain copy, because it means to do its own decompression.

That invariant is what keeps the `LD_PRELOAD` shim ignorant of compression:
the hot copy always lands at the mirrored target path, so the shim's existing
cold-to-hot prefix mapping resolves it with no codec awareness at all.

Single-file codecs only. Archives (`.tar.gz`) are deliberately unsupported:
one archive expanding to many files breaks the one-target-one-file model that
the capacity accounting and the shim's per-file mapping both rely on.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

#: Used when a codec cannot report its expanded size cheaply. Deliberately
#: generous: under-reserving capacity means an eviction sweep mid-stage.
FALLBACK_RATIO = 4

#: Read/write chunk for streaming expansion. Large enough to keep syscall
#: count low on network cold tiers, small enough not to balloon RSS when
#: several workers expand concurrently.
CHUNK_BYTES = 1 << 20


def _ratio_estimate(path: Path) -> int:
    try:
        return path.stat().st_size * FALLBACK_RATIO
    except OSError:
        return 0


def _gzip_expanded_size(path: Path) -> int:
    """gzip records the expanded size in the trailing 4 bytes (ISIZE).

    ISIZE is the size modulo 2^32, so it is only trustworthy below 4 GiB. A
    value smaller than the compressed size is proof of wraparound, in which
    case fall back to the ratio estimate.
    """
    try:
        compressed = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            isize = int.from_bytes(fh.read(4), "little")
    except OSError:
        return 0
    if isize <= compressed:
        return compressed * FALLBACK_RATIO
    return isize


def _zstd_open(path: Path) -> IO[bytes]:
    # zstandard is an optional dependency; import lazily so its absence only
    # affects .zst targets rather than breaking the stager at import time.
    import zstandard

    return zstandard.open(path, "rb")


@dataclass(frozen=True)
class Codec:
    """How to expand one compression suffix."""

    suffix: str
    opener: Callable[[Path], IO[bytes]]
    expanded_size: Callable[[Path], int]

    def available(self) -> bool:
        """False when the codec's backing library is not installed."""
        if self.suffix != ".zst":
            return True
        try:
            import zstandard  # noqa: F401
        except ImportError:
            return False
        return True


CODECS: tuple[Codec, ...] = (
    Codec(".gz", gzip.open, _gzip_expanded_size),
    Codec(".zst", _zstd_open, _ratio_estimate),
    Codec(".bz2", bz2.open, _ratio_estimate),
    Codec(".xz", lzma.open, _ratio_estimate),
)


@dataclass(frozen=True)
class StagePlan:
    """How to materialise one target path in the hot tier.

    `source` is what to read on the cold tier. `codec` is None for a plain
    copy, otherwise the codec to expand `source` through. `size_bytes` is the
    expected size of the resulting hot artifact, used for capacity reservation
    before any bytes move.
    """

    target: Path
    source: Path
    codec: Codec | None
    size_bytes: int

    @property
    def expands(self) -> bool:
        return self.codec is not None


def resolve(target: str | Path) -> StagePlan | None:
    """Find what on the cold tier can materialise `target`.

    Returns None when neither the target nor any compressed variant exists,
    which the caller treats as a miss: nothing is staged and the shim lets the
    tool's open fall through to the cold tier unchanged.

    The target is probed first, so the common case costs exactly one stat and
    a metadata-bound workload never pays for the codec probes.
    """
    target = Path(target)
    try:
        if target.is_file():
            return StagePlan(target, target, None, target.stat().st_size)
    except OSError:
        return None

    for codec in CODECS:
        candidate = target.with_name(target.name + codec.suffix)
        try:
            if not candidate.is_file():
                continue
        except OSError:
            continue
        if not codec.available():
            continue
        return StagePlan(target, candidate, codec, codec.expanded_size(candidate))
    return None


def materialise(plan: StagePlan, dest: Path) -> None:
    """Write the target's bytes to `dest`, expanding if the plan says so.

    `dest` is a temporary path; the caller renames it into place so a reader
    never observes a partial file. On a corrupt or truncated archive the
    codec raises and the caller unlinks `dest`, leaving nothing at the hot
    path so the tool falls through to cold.
    """
    if plan.codec is None:
        shutil.copyfile(plan.source, dest)
        return
    with plan.codec.opener(plan.source) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, CHUNK_BYTES)


#: Formats whose random access needs an external index. Suffixes are tried
#: both appended (`x.bam.bai`) and substituted (`x.bai`), since both
#: conventions are in the wild.
SIDECAR_SUFFIXES: dict[str, tuple[str, ...]] = {
    ".bam": (".bai",),
    ".cram": (".crai",),
    ".vcf": (".tbi", ".csi"),
    ".bcf": (".csi",),
    ".fasta": (".fai",),
    ".fa": (".fai",),
    ".fastq": (".fai",),
}


def companions(primary: str | Path) -> tuple[Path, ...]:
    """Index files the tool needs alongside `primary`, if they exist on cold.

    Staging a BAM without its BAI leaves the tool doing indexed random access
    against a cold index, which is the read pattern staging exists to remove.
    """
    primary = Path(primary)
    # `.vcf.gz` and friends: the index keys off the inner format suffix.
    stem_suffix = primary.suffix.lower()
    if stem_suffix in {c.suffix for c in CODECS}:
        stem_suffix = Path(primary.stem).suffix.lower()

    found: list[Path] = []
    for suffix in SIDECAR_SUFFIXES.get(stem_suffix, ()):
        for candidate in (
            primary.with_name(primary.name + suffix),
            primary.with_suffix(suffix),
        ):
            try:
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
            except OSError:
                continue
    return tuple(found)
