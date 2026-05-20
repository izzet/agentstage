"""Path B wall-time replay ablator.

For a captured multi-turn run, replays each agent file-access in two
configurations and reports the cumulative wall-clock saved:

  COLD:  no LD_PRELOAD shim, page cache evicted before each read.
         Simulates "no staging at all" — every read pays cold latency.
  HOT:   file is pre-staged via the stager (force-prefetch if needed),
         shim active, page cache evicted from cold tier first so the
         shim must redirect to hot tier for the read to be fast.

Computes:
  - Total session I/O wall time, cold
  - Total session I/O wall time, hot
  - Speedup ratio
  - Per-file breakdown

Distinguishes two "hot" scenarios:
  - ORACLE: every file the agent opened is treated as if the predictor
    had pre-staged it. Upper bound on the speedup that perfect
    prediction would yield.
  - REALISTIC: only files the predictor ACTUALLY pre-staged in this
    run get hot reads; predictor-misses still pay cold cost. Honest
    speedup for this particular run.

Reads each file's first 4 KB (sufficient to expose page-cache vs.
shim-redirect divergence; matches earlier microbench methodology).

Usage:
    LD_PRELOAD=$SHIM AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path_b \\
    AGENTSTAGE_COLD_ROOTS=/tmp/s3-noaa-goes16/ABI-L2-CMIPC \\
    python scripts/microbench/path_b_walltime.py \\
        --corpus outputs/multi_turn/e011_multiturn_hinted_<ts> \\
        --workload aiob_107_s3 \\
        --out <corpus>/walltime_replay.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agentstage.runners.path_b_multiturn import _resolve_logical_to_physical
from agentstage.stager import DataHint, Stager
from agentstage.workloads.aiob import (
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)


def collect_opened_paths(
    corpus: Path,
    prefix_map: tuple[tuple[str, str], ...],
    cold_root: str,
) -> list[str]:
    """Return physical paths in agent open order (de-duplicated)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for turn_dir in sorted((corpus / "turns").glob("turn_*")):
        tu_path = turn_dir / "tool_use.jsonl"
        if not tu_path.exists():
            continue
        for line in tu_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("name") not in ("open_file", "read_file"):
                continue
            logical = (d.get("parsed_input") or {}).get("path", "")
            if not logical:
                continue
            phys = _resolve_logical_to_physical(
                logical, prefix_map, cold_root=cold_root)
            if not Path(phys).is_file():
                continue
            if phys not in seen:
                seen.add(phys)
                ordered.append(phys)
    return ordered


def collect_predictor_staged(staging_report_path: Path) -> set[str]:
    """Files the predictor ACTUALLY staged (rule_id != 'force' / 'path_a_force')."""
    if not staging_report_path.exists():
        return set()
    data = json.loads(staging_report_path.read_text())
    return {
        ev["cold_path"]
        for ev in data.get("events", [])
        if ev.get("outcome") in ("staged", "hit")
        and ev.get("rule_id") not in ("force", "path_a_force")
    }


def measure_cold_via_subprocess(path: str) -> float:
    """Evict page cache for `path`, open, read 4 KB, return ms.
    Uses subprocess so LD_PRELOAD shim is bypassed entirely."""
    env_no_shim = os.environ.copy()
    env_no_shim.pop("LD_PRELOAD", None)
    env_no_shim["AGENTSTAGE_SHIM_DISABLE"] = "1"
    r = subprocess.run(
        ["python3", "-c",
         f"import os, time; "
         f"fd=os.open({path!r}, os.O_RDONLY); "
         f"os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); "
         f"t0=time.monotonic_ns(); "
         f"open({path!r},'rb').read(4096); "
         f"print((time.monotonic_ns()-t0)/1e6)"],
        env=env_no_shim, capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"cold read failed: {r.stderr}")
    return float(r.stdout.strip())


def measure_hot_via_shim(path: str) -> float:
    """Read 4 KB through the current process's LD_PRELOAD shim.
    Assumes the hot copy already exists for redirect to fire."""
    # Evict the cold copy first so the timing is not page-cache assisted
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    os.sync()
    t0 = time.monotonic_ns()
    with open(path, "rb") as f:
        f.read(4096)
    return (time.monotonic_ns() - t0) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workload",
                        choices=["aiob_107", "aiob_107_s3", "aiob_110"],
                        default="aiob_107_s3")
    parser.add_argument("--cold-root", default="/tmp/s3-noaa-goes16/ABI-L2-CMIPC")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaders = {
        "aiob_107": load_aiob_107,
        "aiob_107_s3": load_aiob_107_s3,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    prefix_map = workload.prefix_map

    opened = collect_opened_paths(args.corpus, prefix_map, args.cold_root)
    predictor_staged = collect_predictor_staged(
        args.corpus / "staging_report.json")

    if not opened:
        print(f"FATAL: no opened files in {args.corpus}", file=sys.stderr)
        return 2

    print(f"Corpus: {args.corpus}")
    print(f"  Agent opened {len(opened)} distinct file(s)")
    print(f"  Predictor staged {len(predictor_staged)} distinct file(s) (excl. force)")

    # Set up a stager for the ORACLE scenario (pre-stage every opened file)
    hot_root = Path(os.environ.get("AGENTSTAGE_HOT_ROOT",
                                   "/dev/shm/agentstage_walltime"))
    hot_root.mkdir(parents=True, exist_ok=True)
    cold_root = Path(args.cold_root)
    stager = Stager(
        hot_root=hot_root,
        cold_roots=[cold_root],
        max_workers=4,
        capacity_bytes=64 * 1024**3,
    )

    # Pre-stage every opened file (for ORACLE) — predictor_staged ⊆ this set
    print(f"  Pre-staging {len(opened)} file(s) for ORACLE scenario...")
    futures = stager.prefetch(DataHint(
        predicted_files=tuple(opened),
        tier=1, fired_at_ms=0.0, rule_id="walltime_oracle",
    ))
    for f in futures:
        f.result(timeout=300)

    per_file: list[dict] = []
    cold_total_ms = 0.0
    realistic_total_ms = 0.0  # cold for files predictor missed, hot for files staged
    oracle_total_ms = 0.0     # hot for everything

    for phys in opened:
        size = Path(phys).stat().st_size
        cold_ms = measure_cold_via_subprocess(phys)
        # Hot read (file is staged for ORACLE, and may or may not be for predictor)
        hot_ms = measure_hot_via_shim(phys)
        per_file.append({
            "path": phys,
            "size_bytes": size,
            "cold_ms": round(cold_ms, 3),
            "hot_ms": round(hot_ms, 3),
            "speedup_per_file": round(cold_ms / hot_ms, 1) if hot_ms > 0 else None,
            "was_predictor_staged": phys in predictor_staged,
        })
        cold_total_ms += cold_ms
        oracle_total_ms += hot_ms
        if phys in predictor_staged:
            realistic_total_ms += hot_ms
        else:
            realistic_total_ms += cold_ms

    speedup_oracle = (cold_total_ms / oracle_total_ms) if oracle_total_ms > 0 else None
    speedup_realistic = (cold_total_ms / realistic_total_ms) if realistic_total_ms > 0 else None

    result = {
        "corpus": str(args.corpus),
        "workload": args.workload,
        "n_files_opened": len(opened),
        "n_files_predictor_staged": len(predictor_staged & set(opened)),
        "cold_total_ms": round(cold_total_ms, 3),
        "oracle_total_ms": round(oracle_total_ms, 3),
        "realistic_total_ms": round(realistic_total_ms, 3),
        "speedup_oracle": round(speedup_oracle, 1) if speedup_oracle else None,
        "speedup_realistic": round(speedup_realistic, 1) if speedup_realistic else None,
        "savings_oracle_ms": round(cold_total_ms - oracle_total_ms, 3),
        "savings_realistic_ms": round(cold_total_ms - realistic_total_ms, 3),
        "per_file": per_file,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print()
    print(f"  COLD total:       {cold_total_ms:.1f} ms  ({len(opened)} reads)")
    print(f"  HOT/ORACLE total: {oracle_total_ms:.3f} ms  "
          f"({speedup_oracle:.0f}× speedup if predictor were perfect)")
    print(f"  HOT/REALISTIC:    {realistic_total_ms:.3f} ms  "
          f"({speedup_realistic:.1f}× speedup with actual predictor staging)")
    print(f"  Oracle savings:   {cold_total_ms - oracle_total_ms:.1f} ms")
    print(f"  Realistic save:   {cold_total_ms - realistic_total_ms:.1f} ms")
    print(f"\nWrote {args.out}")

    stager.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
