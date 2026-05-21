"""Path 0 wall-time projection — full-file read measurement.

The 4 KB first-block measurements in path0_replay.py give first-byte
latency, but the wall-time speedup story depends on whole-file
throughput. This script reads the ENTIRE file for each sample (not just
4 KB) so the measurement reflects what an agent actually pays.

Two modes: baseline (no shim, no prefetch) and with-stager
(LD_PRELOAD shim + prefetch). Both sample N distinct files.

Usage:
  ./scripts/microbench/path0_walltime_run.sh aiob_110 [N_SAMPLES]

For aiob_110's 350 MB NWB files this is slower per-sample than the
4 KB measurement (each cold read takes seconds), so default N is 5.
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
    parser.add_argument("--workload",
                        choices=["aiob_104", "aiob_107", "aiob_110"], required=True)
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of DISTINCT files to sample.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for deterministic file selection.")
    parser.add_argument("--throttle-mbps", type=float, default=None,
                        help="Throttle cold reads to this max throughput (MB/s) "
                        "to simulate a slower cold tier (e.g. real PFS, S3). "
                        "Inserts per-chunk sleep to enforce. Only applies to "
                        "the baseline (cold) measurement; with-stager mode "
                        "reads from tmpfs (unthrottled).")
    args = parser.parse_args()

    from agentiobench.utils.cache import _resident_pages
    from agentstage.stager import DataHint, Stager
    from agentstage.workloads.aiob import (
        load_aiob_104, load_aiob_107, load_aiob_110,
    )

    loaders = {
        "aiob_104": load_aiob_104,
        "aiob_107": load_aiob_107,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()

    # Pick the largest bucket in the workspace prior (most representative)
    prefix_map = workload.prefix_map

    def to_physical(logical: str) -> str:
        for lp, rp in prefix_map:
            if logical.startswith(lp):
                return rp + logical[len(lp):]
        return logical

    # Take all files from the largest bucket
    largest_bucket = max(
        workload.workspace_prior.items(), key=lambda kv: len(kv[1])
    )
    bucket_name, bucket_files = largest_bucket
    all_phys = [to_physical(p) for p in bucket_files]
    all_phys = [p for p in all_phys if Path(p).is_file()]
    if not all_phys:
        print(json.dumps({"error": "no physical files", "bucket": bucket_name}))
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(all_phys)
    sample = all_phys[: args.n_samples]
    sample_sizes_mb = [Path(p).stat().st_size / 1e6 for p in sample]
    print(f"# workload={args.workload} bucket={bucket_name} "
          f"sample_n={len(sample)} "
          f"size_mb_med={statistics.median(sample_sizes_mb):.1f} "
          f"size_mb_max={max(sample_sizes_mb):.1f}",
          file=sys.stderr)

    # Stager setup
    cold_root = Path(os.environ.get(
        "AGENTSTAGE_COLD_ROOTS",
        "/mnt/common/datasets-staging/agentiobench/datasets",
    ).split(":")[0])
    hot_root = Path(os.environ.get("AGENTSTAGE_HOT_ROOT", "/dev/shm/agentstage_walltime"))
    hot_root.mkdir(parents=True, exist_ok=True)

    stager = Stager(
        hot_root=hot_root,
        cold_roots=[cold_root],
        max_workers=4,
        capacity_bytes=64 * 1024**3,  # 64 GB — fits aiob_110's biggest files
    )

    # Strict eviction helper
    def evict_strict(path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            os.sync()
        except OSError:
            pass

    # Prefetch ALL sampled files in with-stager mode
    if args.mode == "with-stager":
        t_prefetch_start = time.monotonic_ns()
        hint = DataHint(
            detected_files=tuple(sample),
            tier=1,
            fired_at_ms=0.0,
            rule_id="path0_walltime",
        )
        for f in stager.prefetch(hint):
            f.result(timeout=600)  # large files need time
        prefetch_total_s = (time.monotonic_ns() - t_prefetch_start) / 1e9
        print(f"# prefetched {len(sample)} files in {prefetch_total_s:.1f}s",
              file=sys.stderr)

    # Measurement: per file, evict cold, read FULL file
    per_file_results: list[dict] = []
    all_full_read_ms: list[float] = []
    all_throughput_mbps: list[float] = []

    # Throttling: enforce a per-chunk minimum time so cold reads don't
    # exceed the target throughput. Only applies to baseline; with-stager
    # reads from tmpfs (unthrottled, because the agent reads from local
    # hot tier — that's the whole point of the design).
    chunk_size = 1 << 20  # 1 MiB
    apply_throttle = (args.mode == "baseline" and args.throttle_mbps is not None)
    target_chunk_s = chunk_size / (args.throttle_mbps * 1e6) if apply_throttle else 0

    for phys in sample:
        size = Path(phys).stat().st_size
        evict_strict(phys)
        try:
            resident, total = _resident_pages(Path(phys))
            resident_frac = resident / total if total else 0.0
        except OSError:
            resident_frac = -1.0

        t0 = time.monotonic_ns()
        with open(phys, "rb") as f:
            if apply_throttle:
                while True:
                    cs = time.monotonic_ns()
                    data = f.read(chunk_size)
                    if not data:
                        break
                    chunk_read_s = (time.monotonic_ns() - cs) / 1e9
                    sleep_for = target_chunk_s - chunk_read_s
                    if sleep_for > 0:
                        time.sleep(sleep_for)
            else:
                while f.read(chunk_size):
                    pass
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        throughput_mbps = (size / 1e6) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        all_full_read_ms.append(elapsed_ms)
        all_throughput_mbps.append(throughput_mbps)

        per_file_results.append({
            "path": phys,
            "size_bytes": size,
            "size_mb": round(size / 1e6, 1),
            "cold_resident_frac_before": round(resident_frac, 4),
            "full_read_ms": round(elapsed_ms, 1),
            "throughput_mbps": round(throughput_mbps, 1),
        })
        print(f"  {Path(phys).name[:60]:60s} {size/1e6:6.1f} MB  "
              f"{elapsed_ms:9.1f} ms  {throughput_mbps:6.1f} MB/s",
              file=sys.stderr)

    stager.shutdown(wait=True)

    summary = {
        "mode": args.mode,
        "workload": args.workload,
        "bucket": bucket_name,
        "n_samples": len(sample),
        "throttle_mbps": args.throttle_mbps,
        "ld_preload_set": bool(os.environ.get("LD_PRELOAD")),
        "shim_in_ld_preload": "agentstage_shim" in os.environ.get("LD_PRELOAD", ""),
        "aggregate": {
            "full_read_ms": {
                "p50": round(statistics.median(all_full_read_ms), 1),
                "p95": round(
                    sorted(all_full_read_ms)[int(0.95 * (len(all_full_read_ms) - 1))], 1),
                "mean": round(statistics.mean(all_full_read_ms), 1),
                "min": round(min(all_full_read_ms), 1),
                "max": round(max(all_full_read_ms), 1),
            },
            "throughput_mbps": {
                "p50": round(statistics.median(all_throughput_mbps), 1),
                "p95": round(
                    sorted(all_throughput_mbps)[int(0.95 * (len(all_throughput_mbps) - 1))], 1),
                "mean": round(statistics.mean(all_throughput_mbps), 1),
            },
            "total_bytes": sum(p["size_bytes"] for p in per_file_results),
        },
        "per_file": per_file_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
