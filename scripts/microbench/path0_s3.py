"""Path 0 — real S3 cold tier measurement.

Reads GOES NetCDF files directly from NOAA's public S3 bucket
(noaa-goes16) mounted via mountpoint-s3. Same files aiob_107 was
originally built from. Closes the "throttled simulator vs real S3"
gap from E-007.

Mount NOAA's bucket first:
  mkdir -p /tmp/s3-noaa-goes16
  mount-s3 --no-sign-request --read-only --region us-east-1 \
    noaa-goes16 /tmp/s3-noaa-goes16

Usage:
  ./scripts/microbench/path0_s3_run.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "with-stager"], required=True)
    parser.add_argument("--s3-mount", type=Path,
                        default=Path("/tmp/s3-noaa-goes16"),
                        help="mountpoint-s3 mount of noaa-goes16")
    parser.add_argument("--prefix", default="ABI-L2-CMIPC/2024/122/00",
                        help="Subdirectory under the mount to sample from")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of distinct files to measure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from agentstage.stager import DataHint, Stager

    cold_root = args.s3_mount  # the mount IS the cold root
    if not (cold_root / args.prefix).is_dir():
        print(json.dumps({"error": f"prefix {args.prefix} not visible "
                                   f"under {cold_root}; is mount-s3 running?"}))
        return 1

    # List candidate files (matches GOES NetCDF naming)
    candidates = sorted((cold_root / args.prefix).glob("OR_ABI-L2-CMIPC-M6C08_G16_*.nc"))
    if len(candidates) < args.n_samples:
        # Fall back to any *.nc
        candidates = sorted((cold_root / args.prefix).glob("*.nc"))
    if not candidates:
        print(json.dumps({"error": "no GOES files found under prefix"}))
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = [str(p) for p in candidates[: args.n_samples]]

    # Sizes via stat() (this also triggers S3 HEAD requests through the mount)
    sample_sizes_mb = []
    for p in sample:
        try:
            sample_sizes_mb.append(Path(p).stat().st_size / 1e6)
        except OSError as exc:
            print(f"# stat failed for {p}: {exc}", file=sys.stderr)
    print(f"# s3_mount={cold_root} prefix={args.prefix} "
          f"sample_n={len(sample)} "
          f"size_mb_med={statistics.median(sample_sizes_mb):.1f}",
          file=sys.stderr)

    hot_root = Path(os.environ.get("AGENTSTAGE_HOT_ROOT", "/dev/shm/agentstage_s3"))
    hot_root.mkdir(parents=True, exist_ok=True)

    stager = Stager(
        hot_root=hot_root,
        cold_roots=[cold_root],
        max_workers=4,
        capacity_bytes=8 * 1024**3,
    )

    # Note: POSIX_FADV_DONTNEED is a no-op on FUSE mounts (no page cache for
    # mountpoint-s3 contents — each open() triggers a fresh S3 HEAD/GET).
    # That makes each cold read here genuinely cold from the S3 perspective.

    # Prefetch in with-stager mode (this is the cold-tier read; stager
    # copies from S3 mount to /dev/shm)
    if args.mode == "with-stager":
        t_prefetch = time.monotonic_ns()
        hint = DataHint(
            predicted_files=tuple(sample),
            tier=1,
            fired_at_ms=0.0,
            rule_id="path0_s3",
        )
        for f in stager.prefetch(hint):
            f.result(timeout=300)
        prefetch_s = (time.monotonic_ns() - t_prefetch) / 1e9
        print(f"# prefetched {len(sample)} files from S3 in {prefetch_s:.2f}s "
              f"(combined throughput {(sum(sample_sizes_mb)/prefetch_s):.1f} MB/s)",
              file=sys.stderr)

    # Per-file measurement: open + read full file
    per_file = []
    all_ms = []
    all_throughput = []
    chunk_size = 1 << 20

    for phys in sample:
        size = Path(phys).stat().st_size
        t0 = time.monotonic_ns()
        with open(phys, "rb") as f:
            while f.read(chunk_size):
                pass
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        throughput_mbps = (size / 1e6) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        all_ms.append(elapsed_ms)
        all_throughput.append(throughput_mbps)
        per_file.append({
            "path": phys,
            "size_mb": round(size / 1e6, 1),
            "full_read_ms": round(elapsed_ms, 1),
            "throughput_mbps": round(throughput_mbps, 1),
        })
        print(f"  {Path(phys).name[:55]:55s} {size/1e6:6.1f} MB  "
              f"{elapsed_ms:8.1f} ms  {throughput_mbps:6.1f} MB/s",
              file=sys.stderr)

    stager.shutdown(wait=True)

    summary = {
        "mode": args.mode,
        "cold_backend": "noaa-goes16 (mountpoint-s3, us-east-1)",
        "s3_mount": str(cold_root),
        "prefix": args.prefix,
        "n_samples": len(sample),
        "ld_preload_set": bool(os.environ.get("LD_PRELOAD")),
        "shim_in_ld_preload": "agentstage_shim" in os.environ.get("LD_PRELOAD", ""),
        "aggregate": {
            "full_read_ms": {
                "p50": round(statistics.median(all_ms), 1),
                "p95": round(sorted(all_ms)[int(0.95 * (len(all_ms) - 1))], 1),
                "mean": round(statistics.mean(all_ms), 1),
                "min": round(min(all_ms), 1),
                "max": round(max(all_ms), 1),
            },
            "throughput_mbps": {
                "p50": round(statistics.median(all_throughput), 1),
                "mean": round(statistics.mean(all_throughput), 1),
            },
            "total_bytes": sum(int(p["size_mb"] * 1e6) for p in per_file),
        },
        "per_file": per_file,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
