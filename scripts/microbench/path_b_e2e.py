"""E-028 — End-to-end task-script speedup: baseline vs staged, local + S3.

Runs the ACTUAL Python script a Sonnet-4.5 agent generated for the
aiob_107 task (captured from a real AgentIOBench production run,
turn 12 of 23) against a cold storage tier, twice:

  BASELINE  — page cache evicted, no LD_PRELOAD shim. Every file the
              script opens pays cold-tier latency.
  STAGED    — every file the script reads is pre-fetched into the
              tmpfs hot tier by the Stager; the script runs with the
              LD_PRELOAD shim active so its open() calls redirect to
              the hot copies.

This is the end-to-end measurement: real agent-written analysis code,
really executing, really reading NetCDF files, with the real shim.
The difference between the two runs is the wall-time AgentStage saves
on the data-processing phase of a real scientific-agent task.

Run for two cold tiers:
  - local : AgentIOBench's local NFS/XFS dataset copy
  - s3    : the public noaa-goes16 bucket via mountpoint-s3

Usage:
    python scripts/microbench/path_b_e2e.py --tier local --out outputs/e2e/...
    python scripts/microbench/path_b_e2e.py --tier s3    --out outputs/e2e/...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agentstage.stager import DataHint, Stager

TASK_SCRIPT = Path("outputs/e2e/task_script.py")

# Cold-tier roots (scoped to day-of-year 122 to keep the run tractable;
# the full task is days 122-128. Day 122 has 864 C08/C09/C10 files.)
COLD_ROOTS = {
    "local": "/mnt/common/datasets-staging/agentiobench/datasets/goes_cmi_composites/raw/2024/122",
    "s3": "/tmp/s3-noaa-goes16/ABI-L2-CMIPC/2024/122",
}

# Managed cold-root ancestors for the Stager + shim. Must be a real
# directory prefix of the data (NOT "/" — the Stager's prefix check
# turns "/" into "//" and matches nothing).
COLD_ROOT_ANCESTORS = {
    "local": "/mnt/common/datasets-staging/agentiobench/datasets",
    "s3": "/tmp/s3-noaa-goes16",
}

SHIM = Path("src/agentstage/stager/shim/libagentstage_shim.so").resolve()


def enumerate_target_files(data_dir: str) -> list[str]:
    """Replicate the agent script's glob: C08/C09/C10 NetCDFs under data_dir."""
    files: list[str] = []
    files += glob.glob(os.path.join(data_dir, "**", "*C0[8-9]*.nc"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**", "*C10*.nc"), recursive=True)
    return sorted(set(files))


def evict(paths: list[str]) -> None:
    """Drop the OS page cache for every file (best-effort)."""
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass
    os.sync()


def run_script(data_dir: str, output_dir: str, *,
               ld_preload: str | None, hot_root: str | None,
               cold_roots: str | None) -> dict:
    """Run the agent's task script in a subprocess. Returns timing + status."""
    env = os.environ.copy()
    # The agent's script needs netCDF4/numpy/pandas/matplotlib, which live
    # in the SYSTEM python site-packages, not the uv venv. Strip venv env
    # vars so /usr/bin/python3 uses the system site-packages.
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(var, None)
    env["E2E_DATA_DIR"] = data_dir
    env["E2E_OUTPUT_DIR"] = output_dir
    env["MPLBACKEND"] = "Agg"  # headless matplotlib
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload
        env["AGENTSTAGE_HOT_ROOT"] = hot_root or ""
        env["AGENTSTAGE_COLD_ROOTS"] = cold_roots or ""
        env["AGENTSTAGE_RETRY_SPIN_MS"] = "20"
    else:
        env.pop("LD_PRELOAD", None)
        env["AGENTSTAGE_SHIM_DISABLE"] = "1"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    # Explicitly use the SYSTEM python3 — it has netCDF4/numpy/pandas/matplotlib
    r = subprocess.run(
        ["/usr/bin/python3", str(TASK_SCRIPT)],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.monotonic() - t0
    return {
        "elapsed_s": round(elapsed, 3),
        "returncode": r.returncode,
        "stdout_tail": r.stdout[-500:],
        "stderr_tail": r.stderr[-800:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["local", "s3"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_e2e")
    args = parser.parse_args()

    if not TASK_SCRIPT.is_file():
        print(f"FATAL: {TASK_SCRIPT} not found — extract it first", file=sys.stderr)
        return 2
    if not SHIM.is_file():
        print(f"FATAL: shim not built at {SHIM}", file=sys.stderr)
        return 2

    cold_dir = COLD_ROOTS[args.tier]
    if not Path(cold_dir).is_dir():
        print(f"FATAL: cold dir not found: {cold_dir}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    hot_root = Path(args.hot_root)
    if hot_root.exists():
        shutil.rmtree(hot_root)
    hot_root.mkdir(parents=True, exist_ok=True)

    targets = enumerate_target_files(cold_dir)
    total_bytes = sum(Path(f).stat().st_size for f in targets if Path(f).is_file())
    print(f"E-028 end-to-end | tier={args.tier}")
    print(f"  cold dir:   {cold_dir}")
    print(f"  task files: {len(targets)} C08/C09/C10 NetCDFs "
          f"({total_bytes/1024/1024:.0f} MB)")
    print(f"  script:     {TASK_SCRIPT}")
    print()

    # ── BASELINE: evict, run with no shim ──────────────────────────────
    print("[1/2] BASELINE — evicting page cache, running script cold...")
    evict(targets)
    baseline = run_script(
        data_dir=cold_dir,
        output_dir=str(args.out / "baseline_output"),
        ld_preload=None, hot_root=None, cold_roots=None,
    )
    print(f"      baseline elapsed: {baseline['elapsed_s']:.1f} s "
          f"(rc={baseline['returncode']})")
    if baseline["returncode"] != 0:
        print(f"      stderr: {baseline['stderr_tail']}", file=sys.stderr)

    # ── STAGED: pre-fetch all files, run with shim ─────────────────────
    print(f"[2/2] STAGED — pre-fetching {len(targets)} files into {hot_root}...")
    cold_root_for_stager = COLD_ROOT_ANCESTORS[args.tier]
    stager = Stager(
        hot_root=hot_root,
        cold_roots=[Path(cold_root_for_stager)],
        max_workers=8,
        capacity_bytes=64 * 1024**3,
    )
    t_stage0 = time.monotonic()
    futures = stager.prefetch(DataHint(
        detected_files=tuple(targets),
        tier=3, fired_at_ms=0.0, rule_id="e2e_prestage",
    ))
    n_staged = 0
    for f in futures:
        try:
            f.result(timeout=1800)
            n_staged += 1
        except Exception as e:
            print(f"      stage error: {e!r}", file=sys.stderr)
    stage_elapsed = time.monotonic() - t_stage0
    print(f"      staged {n_staged}/{len(targets)} files in {stage_elapsed:.1f} s")

    # Evict cold copies so the script MUST hit the hot tier via the shim
    evict(targets)
    staged = run_script(
        data_dir=cold_dir,
        output_dir=str(args.out / "staged_output"),
        ld_preload=str(SHIM),
        hot_root=str(hot_root),
        cold_roots=cold_root_for_stager,
    )
    print(f"      staged elapsed:   {staged['elapsed_s']:.1f} s "
          f"(rc={staged['returncode']})")
    if staged["returncode"] != 0:
        print(f"      stderr: {staged['stderr_tail']}", file=sys.stderr)

    stager.shutdown(wait=True)

    speedup = (baseline["elapsed_s"] / staged["elapsed_s"]
               if staged["elapsed_s"] > 0 else None)
    result = {
        "experiment": "E-028",
        "tier": args.tier,
        "cold_dir": cold_dir,
        "n_task_files": len(targets),
        "total_bytes": total_bytes,
        "n_staged": n_staged,
        "stage_elapsed_s": round(stage_elapsed, 3),
        "baseline": baseline,
        "staged": staged,
        "session_speedup": round(speedup, 2) if speedup else None,
        "wall_time_saved_s": round(baseline["elapsed_s"] - staged["elapsed_s"], 1),
    }
    (args.out / f"e2e_{args.tier}.json").write_text(json.dumps(result, indent=2))

    print()
    print(f"  ===== E-028 {args.tier.upper()} RESULT =====")
    print(f"  baseline (cold):  {baseline['elapsed_s']:>9.1f} s")
    print(f"  staged   (hot):   {staged['elapsed_s']:>9.1f} s")
    print(f"  wall-time saved:  {result['wall_time_saved_s']:>9.1f} s")
    print(f"  session speedup:  {result['session_speedup']}x")
    print(f"\n  wrote {args.out / f'e2e_{args.tier}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
