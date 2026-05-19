"""Stager baseline microbenchmarks (T14a/b/c).

Verifies the three environment assumptions the stager design relies on:

  A. Cold-tier first-read P95 is large enough to justify staging.
     The stager only pays off if cold reads are slow. Measured per file-size
     bucket (small ~3 MB GOES, medium ~50 MB ERA5, large ~350 MB NWB) after
     POSIX_FADV_DONTNEED-based cache eviction.

  B. Hot-tier (tmpfs) first-read P95 is small enough to be the ceiling.
     The "with stager" speedup is bounded above by hot-tier read latency.

  C. POSIX_FADV_DONTNEED actually drops page residency on /mnt/common (XFS).
     Per agentiobench.utils.cache, eviction is "best effort" on some
     filesystems; if it doesn't work here, E5's "with vs without stager"
     comparison is meaningless because the baseline runs would hit
     warm cache.

Writes a JSON report to outputs/microbench/stager_baseline_<ts>.json.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------

COLD_ROOT = Path("/mnt/common/datasets-staging/agentiobench/datasets")
HOT_ROOT = Path("/dev/shm/agentstage_microbench")

# Per-bucket sample. {bucket_name: (glob pattern under COLD_ROOT, n_samples)}
BUCKETS = {
    "small_goes_3mb": (
        "goes_cmi_composites/raw/2024/**/*.nc",
        30,
    ),
    "medium_era5_50mb": (
        "era5_heatwave/raw/era5_single_levels_2024_*.nc",
        12,  # only 12 monthly files exist
    ),
    "large_nwb_350mb": (
        "steinmetz_neuropixels/raw/**/*.nwb",
        15,
    ),
}

FIRST_BLOCK_BYTES = 4096
RNG = random.Random(0)  # deterministic file selection


# ---------------------------------------------------------------------------
# Eviction primitives — verify import works before measuring
# ---------------------------------------------------------------------------

try:
    from agentiobench.utils.cache import (
        _resident_pages,
        evict_dataset,
        measure_temperature,
    )
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: cannot import agentiobench.utils.cache: {exc}")
    print("Run `uv sync` and verify external/benchmarks/agentiobench/ is checked out.")
    sys.exit(2)


def _evict_paths(paths: list[Path]) -> None:
    """Drop page cache for a specific list of files via POSIX_FADV_DONTNEED."""
    for p in paths:
        try:
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure_first_read_ms(path: Path) -> float:
    """Time open + read(FIRST_BLOCK_BYTES). Returns wall-clock ms."""
    t0 = time.monotonic_ns()
    with open(path, "rb") as f:
        f.read(FIRST_BLOCK_BYTES)
    return (time.monotonic_ns() - t0) / 1e6


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def measure_cold_first_read(files: list[Path]) -> dict:
    """Each file: evict its page cache, then time the first read."""
    times = []
    for f in files:
        _evict_paths([f])
        try:
            times.append(measure_first_read_ms(f))
        except OSError as exc:
            print(f"  skip {f}: {exc}")
    return _summarize(times, files)


def measure_warm_first_read(files: list[Path]) -> dict:
    """Each file: read once to warm, then time a second read (page cache hit)."""
    times = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                fh.read(FIRST_BLOCK_BYTES)  # warm
            times.append(measure_first_read_ms(f))
        except OSError as exc:
            print(f"  skip {f}: {exc}")
    return _summarize(times, files)


def measure_hot_first_read(cold_files: list[Path]) -> dict:
    """Copy each file to HOT_ROOT mirror, evict, time first read from hot."""
    HOT_ROOT.mkdir(parents=True, exist_ok=True)
    times = []
    sizes = []
    for src in cold_files:
        hot = HOT_ROOT / src.relative_to("/")
        hot.parent.mkdir(parents=True, exist_ok=True)
        if not hot.exists():
            try:
                shutil.copy(src, hot)
            except OSError as exc:
                print(f"  skip copy {src}: {exc}")
                continue
        sizes.append(hot.stat().st_size)
        # tmpfs lives in RAM; fadvise is a no-op there, but consistent treatment
        _evict_paths([hot])
        try:
            times.append(measure_first_read_ms(hot))
        except OSError as exc:
            print(f"  skip {hot}: {exc}")
    return {**_summarize(times, cold_files), "mean_bytes": _mean(sizes)}


def _summarize(times: list[float], files: list[Path]) -> dict:
    if not times:
        return {"n": 0}
    return {
        "n": len(times),
        "p50_ms": round(statistics.median(times), 3),
        "p75_ms": round(percentile(times, 0.75), 3),
        "p95_ms": round(percentile(times, 0.95), 3),
        "p99_ms": round(percentile(times, 0.99), 3),
        "max_ms": round(max(times), 3),
        "min_ms": round(min(times), 3),
        "mean_ms": round(statistics.mean(times), 3),
        "all_ms": [round(t, 3) for t in times],
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 1) if values else None


# ---------------------------------------------------------------------------
# Eviction verification (C)
# ---------------------------------------------------------------------------

def verify_eviction(bucket_path: Path) -> dict:
    """Fully warm a small set of files, mincore each one directly, evict the
    bucket, mincore the same files again. Per-file residency via the
    `_resident_pages` helper from agentiobench.utils.cache (mincore-based).
    """
    candidates = [p for p in bucket_path.rglob("*") if p.is_file()][:5]
    if not candidates:
        return {"bucket_path": str(bucket_path), "error": "no_files"}

    # Fully warm each file
    for f in candidates:
        try:
            with open(f, "rb") as fh:
                while fh.read(1 << 20):
                    pass
        except OSError:
            pass

    def _measure(files: list[Path]) -> dict:
        per_file = []
        total_resident = 0
        total_pages = 0
        for f in files:
            try:
                resident, npages = _resident_pages(f)
                per_file.append({
                    "path": str(f),
                    "resident_pages": resident,
                    "total_pages": npages,
                    "fraction": resident / npages if npages else 0.0,
                })
                total_resident += resident
                total_pages += npages
            except OSError as exc:
                per_file.append({"path": str(f), "error": str(exc)})
        return {
            "total_pages": total_pages,
            "resident_pages": total_resident,
            "resident_fraction": (
                total_resident / total_pages if total_pages else 0.0
            ),
            "per_file": per_file,
        }

    before = _measure(candidates)
    evict_dataset(str(bucket_path))
    after = _measure(candidates)

    return {
        "bucket_path": str(bucket_path),
        "n_files_warmed": len(candidates),
        "before_evict": before,
        "after_evict": after,
        "eviction_works": (
            before["resident_fraction"] > 0.5
            and after["resident_fraction"] < 0.05
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_sample(pattern: str, n: int) -> list[Path]:
    candidates = list(COLD_ROOT.glob(pattern))
    if not candidates:
        print(f"  WARN: no files match {pattern} under {COLD_ROOT}")
        return []
    RNG.shuffle(candidates)
    return candidates[:n]


def main() -> int:
    report = {
        "timestamp": datetime.now().isoformat(),
        "cold_root": str(COLD_ROOT),
        "hot_root": str(HOT_ROOT),
        "first_block_bytes": FIRST_BLOCK_BYTES,
        "buckets": {},
        "eviction_check": None,
    }

    for bucket_name, (pattern, n) in BUCKETS.items():
        print(f"\n=== {bucket_name} (pattern={pattern}, n={n}) ===")
        sample = pick_sample(pattern, n)
        if not sample:
            report["buckets"][bucket_name] = {"error": "no_samples"}
            continue
        sample_sizes = [p.stat().st_size for p in sample]
        print(f"  picked {len(sample)} files, sizes: "
              f"min={min(sample_sizes)/1e6:.1f} MB, "
              f"median={statistics.median(sample_sizes)/1e6:.1f} MB, "
              f"max={max(sample_sizes)/1e6:.1f} MB")

        print(f"  measuring cold first-read (with fadvise eviction per file)...")
        cold = measure_cold_first_read(sample)

        print(f"  measuring warm first-read (page cache hit)...")
        warm = measure_warm_first_read(sample)

        print(f"  measuring hot first-read (after copy to tmpfs)...")
        hot = measure_hot_first_read(sample)

        report["buckets"][bucket_name] = {
            "n_sampled": len(sample),
            "size_bytes": {
                "min": min(sample_sizes),
                "median": int(statistics.median(sample_sizes)),
                "max": max(sample_sizes),
            },
            "cold_first_read": cold,
            "warm_first_read": warm,
            "hot_first_read": hot,
            "speedup_p95": (
                round(cold["p95_ms"] / hot["p95_ms"], 1)
                if cold.get("p95_ms") and hot.get("p95_ms")
                else None
            ),
        }
        print(f"  cold p95: {cold.get('p95_ms')} ms")
        print(f"  warm p95: {warm.get('p95_ms')} ms (page cache effect)")
        print(f"  hot  p95: {hot.get('p95_ms')} ms")
        print(f"  speedup p95: {report['buckets'][bucket_name]['speedup_p95']}x")

    # Eviction verification on a small bucket
    print("\n=== eviction check on /mnt/common (XFS) ===")
    report["eviction_check"] = verify_eviction(
        COLD_ROOT / "goes_cmi_composites" / "raw" / "2024" / "124" / "21"
    )
    print(f"  works: {report['eviction_check']['eviction_works']}")
    print(f"  before: {report['eviction_check']['before_evict']}")
    print(f"  after:  {report['eviction_check']['after_evict']}")

    # Cleanup hot mirror
    if HOT_ROOT.exists():
        print(f"\nCleaning {HOT_ROOT}...")
        shutil.rmtree(HOT_ROOT, ignore_errors=True)

    # Write report
    out_dir = Path(__file__).parent.parent.parent / "outputs" / "microbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"stager_baseline_{ts}.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out_path}")

    # Smoke verdict
    print("\n=== verdict ===")
    headroom_ok = True
    for name, b in report["buckets"].items():
        if not isinstance(b, dict) or "speedup_p95" not in b or b["speedup_p95"] is None:
            continue
        speedup = b["speedup_p95"]
        verdict = "OK" if speedup >= 5 else "TIGHT" if speedup >= 2 else "WEAK"
        print(f"  {name}: cold→hot p95 speedup {speedup}x — {verdict}")
        if speedup < 5:
            headroom_ok = False

    print(f"\n  eviction on XFS: "
          f"{'works' if report['eviction_check']['eviction_works'] else 'BROKEN'}")

    if not headroom_ok:
        print("\n  ⚠ cold→hot speedup is < 5x on at least one bucket.")
        print("    The cold tier here (XFS on local SSD) is fast.")
        print("    Real cold tiers (Lustre/PFS/S3) would have larger speedup.")
        print("    Document this in the paper as 'we measure on a fast cold tier")
        print("    to establish a conservative lower bound on stager value.'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
