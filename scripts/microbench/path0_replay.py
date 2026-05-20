"""Path 0 — replay smoke for the stager.

Replays a recorded PoC stream.jsonl through the frozen rule library
against a real Stager + real cold files. Measures first-read latency
on the predicted tier-1 files with vs without the stager.

This is the cheapest path to a real-data, real-files speedup number.
No LLM call (we use a recorded thinking stream instead) but everything
downstream of the LLM — predictor, stager, shim, file I/O — is real.

Modes:
  --mode baseline:    no Stager prefetch; opens cold paths directly
  --mode with-stager: Stager prefetches each predicted tier-1 file,
                      waits, then opens (shim redirects to hot)

The shell wrapper at path0_run.sh runs both modes and prints
the comparison.

Run from outside via:
    LD_PRELOAD=$SHIM \
    AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path0 \
    AGENTSTAGE_COLD_ROOTS=/mnt/common/datasets-staging/agentiobench/datasets \
    python scripts/microbench/path0_replay.py --mode with-stager --stream <path>

For baseline mode, omit LD_PRELOAD.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "with-stager"], required=True)
    parser.add_argument("--stream", type=Path, required=True,
                        help="Path to the PoC stream.jsonl to replay")
    parser.add_argument("--workload", default="aiob_107",
                        choices=["aiob_101", "aiob_104", "aiob_107", "aiob_110"])
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Number of DISTINCT files to sample (each gives "
                        "one guaranteed-cold reading). Sampling distinct files "
                        "is critical because SSD device-level cache keeps "
                        "previously-accessed files fast even after kernel "
                        "page-cache eviction.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Lazy imports so the import path doesn't trip the shim before we want it
    from agentiobench.utils.cache import _resident_pages
    from agentstage.predictor.engine import parse_anthropic_stream, run_predictor
    from agentstage.predictor.rules import get_ruleset
    from agentstage.stager import DataHint, Stager
    from agentstage.workloads.aiob import (
        load_aiob_101, load_aiob_104, load_aiob_107, load_aiob_110,
    )

    loaders = {
        "aiob_101": load_aiob_101,
        "aiob_104": load_aiob_104,
        "aiob_107": load_aiob_107,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    ruleset = get_ruleset(args.workload)

    # 1. Replay the stream → tier-1 logical paths
    blocks = parse_anthropic_stream(args.stream)
    prediction = run_predictor(blocks, workload.workspace_prior, ruleset)

    tier1_logical = list(prediction.tier_1.predicted_files)
    tier3_logical = list(prediction.tier_3.predicted_files)

    # 2. Translate logical → physical via the workload's prefix_map
    def to_physical(logical: str) -> str:
        for lp, rp in workload.prefix_map:
            if logical.startswith(lp):
                return rp + logical[len(lp):]
        return logical

    # Pick the sample set: tier-1 if it has enough files; otherwise extend
    # from tier-3 (the predictor's broader set). All samples must be predicted
    # by SOMETHING — we're measuring "files the stager would prefetch."
    candidate_logical = tier1_logical + [
        p for p in tier3_logical if p not in tier1_logical
    ]
    candidate_physical = [to_physical(p) for p in candidate_logical]
    candidate_physical = [p for p in candidate_physical if Path(p).is_file()]
    if not candidate_physical:
        print(json.dumps({
            "error": "no physical files matched",
            "mode": args.mode,
            "tier1_logical_sample": tier1_logical[:3],
            "tier3_logical_sample": tier3_logical[:3],
        }))
        return 1

    sample = candidate_physical[: args.n_samples]
    # The predicted tier-1 set we actually stage (might be smaller than sample)
    tier1_physical = [to_physical(p) for p in tier1_logical
                      if Path(to_physical(p)).is_file()]

    cold_root_str = os.environ.get(
        "AGENTSTAGE_COLD_ROOTS",
        "/mnt/common/datasets-staging/agentiobench/datasets",
    )
    cold_root = Path(cold_root_str.split(":")[0])

    # 3. Set up the Stager (only used in with-stager mode, but we configure it
    #    consistently to keep the import surface the same)
    hot_root = Path(os.environ.get("AGENTSTAGE_HOT_ROOT", "/dev/shm/agentstage_path0"))
    hot_root.mkdir(parents=True, exist_ok=True)

    stager = Stager(
        hot_root=hot_root,
        cold_roots=[cold_root],
        max_workers=4,
        capacity_bytes=32 * 1024**3,
    )

    # 4. Strict eviction helper: fadvise(DONTNEED) + verify residency dropped.
    # Without verification, page cache can stay resident from the previous
    # read; cold-read timing then measures a warm-cache hit (~0.2 ms instead
    # of the true ~20 ms cold first-read).
    def evict_strict(paths: list[str], max_attempts: int = 5) -> None:
        for p in paths:
            for _ in range(max_attempts):
                try:
                    fd = os.open(p, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    return
                # Sync to force any dirty pages to flush before eviction
                # takes effect (best-effort on read-only files; harmless)
                os.sync()
                # Verify
                try:
                    resident, total = _resident_pages(Path(p))
                except OSError:
                    return
                if total == 0 or resident / total < 0.01:
                    break  # eviction worked

    evict_files = evict_strict  # noqa: F811

    # 5. Optional: warm the hot tier via Stager for ALL sampled files
    # (We stage the same set we'll measure, so the comparison is per-file.)
    if args.mode == "with-stager":
        hint = DataHint(
            predicted_files=tuple(sample),
            tier=1,
            fired_at_ms=0.0,
            rule_id="path0_replay",
        )
        futures = stager.prefetch(hint)
        for f in futures:
            f.result(timeout=120)
        print(f"# prefetched {len(sample)} files into {hot_root}",
              file=sys.stderr)

    # 6. Measure: one sample per file (each is guaranteed-cold from the
    # storage's perspective because we never accessed it before)
    per_file_results: list[dict] = []
    all_first_reads: list[float] = []

    for phys in sample:
        # Evict ONLY this cold-tier file; hot copy (if any) stays in tmpfs
        evict_files([phys])
        try:
            resident, total = _resident_pages(Path(phys))
            resident_frac = resident / total if total else 0.0
        except OSError:
            resident_frac = -1.0

        # Measurement: open + read 4096 bytes. With LD_PRELOAD shim and a hot
        # copy present, this redirects to hot; otherwise hits cold.
        t0 = time.monotonic_ns()
        with open(phys, "rb") as f:
            f.read(4096)
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        all_first_reads.append(elapsed_ms)

        per_file_results.append({
            "path": phys,
            "size_bytes": Path(phys).stat().st_size,
            "cold_resident_frac_before": resident_frac,
            "first_read_ms": elapsed_ms,
        })

    stager.shutdown(wait=True)

    summary = {
        "mode": args.mode,
        "ld_preload_set": "LD_PRELOAD" in os.environ and bool(os.environ["LD_PRELOAD"]),
        "shim_in_ld_preload": "agentstage_shim" in os.environ.get("LD_PRELOAD", ""),
        "workload": args.workload,
        "stream": str(args.stream),
        "rule_library_version": prediction.rule_library_version,
        "n_tier1_files_predicted": len(tier1_logical),
        "n_tier3_files_predicted": len(tier3_logical),
        "n_samples_measured": len(sample),
        "hot_root": str(hot_root),
        "cold_root": str(cold_root),
        "aggregate": {
            "n_samples": len(all_first_reads),
            "p50_ms": statistics.median(all_first_reads),
            "p95_ms": sorted(all_first_reads)[int(0.95 * (len(all_first_reads) - 1))],
            "mean_ms": statistics.mean(all_first_reads),
            "min_ms": min(all_first_reads),
            "max_ms": max(all_first_reads),
        },
        "per_file": per_file_results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    print(f"# mode={args.mode} p50={summary['aggregate']['p50_ms']:.3f}ms "
          f"p95={summary['aggregate']['p95_ms']:.3f}ms "
          f"(n={summary['aggregate']['n_samples']} distinct files)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
