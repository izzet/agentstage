"""E-029 — Decompression-staging end-to-end.

Extends E-028: instead of staging byte-identical copies into the hot
tier, the stager TRANSCODES each zlib-compressed NetCDF into an
uncompressed copy (data-identical, no compression filter). The agent's
analysis script then runs with the LD_PRELOAD shim active, so its
netCDF4 `open()` calls redirect to the uncompressed hot copies — and
HDF5 reads chunks with zero zlib decompression cost.

This moves decompression CPU off the critical path (turn-12 script
execution) and into the staging window — the same latency-hiding
principle AgentStage already applies to cold-tier read latency, now
applied to decompression CPU.

Compares three configurations:
  BASELINE       — cold tier, no shim (= E-028 baseline)
  PLAIN-STAGED   — byte-identical hot copies, shim (= E-028 staged)
  DECOMP-STAGED  — uncompressed hot copies, shim (NEW)

Usage:
    python scripts/microbench/path_b_e2e_decompress.py --tier local --out ...
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

TASK_SCRIPT = Path("outputs/e2e/task_script.py")
TRANSCODER = Path("outputs/e2e/transcode.py")
SHIM = Path("src/agentstage/stager/shim/libagentstage_shim.so").resolve()

COLD_ROOTS = {
    "local": "/mnt/common/datasets-staging/agentiobench/datasets/goes_cmi_composites/raw/2024/122",
    "s3": "/tmp/s3-noaa-goes16/ABI-L2-CMIPC/2024/122",
}
COLD_ROOT_ANCESTORS = {
    "local": "/mnt/common/datasets-staging/agentiobench/datasets",
    "s3": "/tmp/s3-noaa-goes16",
}


def enumerate_target_files(data_dir: str) -> list[str]:
    files: list[str] = []
    files += glob.glob(os.path.join(data_dir, "**", "*C0[8-9]*.nc"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**", "*C10*.nc"), recursive=True)
    return sorted(set(files))


def evict(paths: list[str]) -> None:
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


def run_script(data_dir: str, output_dir: str, *, ld_preload: str | None,
               hot_root: str | None, cold_roots: str | None) -> dict:
    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(var, None)
    env["E2E_DATA_DIR"] = data_dir
    env["E2E_OUTPUT_DIR"] = output_dir
    env["MPLBACKEND"] = "Agg"
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
    r = subprocess.run(["/usr/bin/python3", str(TASK_SCRIPT)],
                       env=env, capture_output=True, text=True, timeout=3600)
    return {
        "elapsed_s": round(time.monotonic() - t0, 3),
        "returncode": r.returncode,
        "stderr_tail": r.stderr[-800:],
    }


def transcode(cold_dir: str, hot_root: str, n_workers: int = 8) -> dict:
    """Run the decompression transcoder under system python3."""
    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(var, None)
    t0 = time.monotonic()
    r = subprocess.run(
        ["/usr/bin/python3", str(TRANSCODER), cold_dir, hot_root, str(n_workers)],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    return {
        "elapsed_s": round(time.monotonic() - t0, 3),
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr_tail": r.stderr[-800:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["local", "s3"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_e2e_decomp")
    args = parser.parse_args()

    for need in (TASK_SCRIPT, TRANSCODER, SHIM):
        if not Path(need).exists():
            print(f"FATAL: missing {need}", file=sys.stderr)
            return 2

    cold_dir = COLD_ROOTS[args.tier]
    args.out.mkdir(parents=True, exist_ok=True)
    hot_root = Path(args.hot_root)
    if hot_root.exists():
        shutil.rmtree(hot_root)
    hot_root.mkdir(parents=True, exist_ok=True)

    targets = enumerate_target_files(cold_dir)
    cold_bytes = sum(Path(f).stat().st_size for f in targets if Path(f).is_file())
    print(f"E-029 decompression-staging | tier={args.tier}")
    print(f"  cold dir:   {cold_dir}")
    print(f"  task files: {len(targets)} ({cold_bytes/1024/1024:.0f} MB compressed)")
    print()

    # ── BASELINE (cold, no shim) ───────────────────────────────────────
    print("[1/2] BASELINE — evict, run cold...")
    evict(targets)
    baseline = run_script(cold_dir, str(args.out / "baseline_output"),
                          ld_preload=None, hot_root=None, cold_roots=None)
    print(f"      baseline: {baseline['elapsed_s']:.1f} s (rc={baseline['returncode']})")

    # ── DECOMP-STAGED (transcode to uncompressed, run with shim) ───────
    print(f"[2/2] DECOMP-STAGED — transcoding {len(targets)} files to "
          f"uncompressed in {hot_root}...")
    tr = transcode(cold_dir, str(hot_root), n_workers=8)
    for line in tr["stdout"].splitlines():
        print(f"      {line}")
    if tr["returncode"] != 0:
        print(f"      transcode stderr: {tr['stderr_tail']}", file=sys.stderr)
    hot_bytes = sum(
        f.stat().st_size for f in hot_root.rglob("*.nc") if f.is_file())
    print(f"      transcode wall: {tr['elapsed_s']:.1f} s   "
          f"hot footprint: {hot_bytes/1024/1024:.0f} MB "
          f"({hot_bytes/max(1,cold_bytes):.2f}x compressed size)")

    evict(targets)  # force the script to hit hot tier via shim
    staged = run_script(cold_dir, str(args.out / "decomp_staged_output"),
                        ld_preload=str(SHIM), hot_root=str(hot_root),
                        cold_roots=COLD_ROOT_ANCESTORS[args.tier])
    print(f"      decomp-staged: {staged['elapsed_s']:.1f} s "
          f"(rc={staged['returncode']})")
    if staged["returncode"] != 0:
        print(f"      stderr: {staged['stderr_tail']}", file=sys.stderr)

    speedup = (baseline["elapsed_s"] / staged["elapsed_s"]
               if staged["elapsed_s"] > 0 else None)
    result = {
        "experiment": "E-029",
        "tier": args.tier,
        "cold_dir": cold_dir,
        "n_task_files": len(targets),
        "cold_bytes": cold_bytes,
        "hot_bytes_uncompressed": hot_bytes,
        "transcode": tr,
        "baseline": baseline,
        "decomp_staged": staged,
        "session_speedup": round(speedup, 2) if speedup else None,
        "wall_time_saved_s": round(baseline["elapsed_s"] - staged["elapsed_s"], 1),
    }
    (args.out / f"e2e_decomp_{args.tier}.json").write_text(
        json.dumps(result, indent=2))

    print()
    print(f"  ===== E-029 {args.tier.upper()} RESULT =====")
    print(f"  baseline (cold):         {baseline['elapsed_s']:>9.1f} s")
    print(f"  decomp-staged (hot):     {staged['elapsed_s']:>9.1f} s")
    print(f"  transcode (staging-time, off critical path): {tr['elapsed_s']:>6.1f} s")
    print(f"  wall-time saved:         {result['wall_time_saved_s']:>9.1f} s")
    print(f"  session speedup:         {result['session_speedup']}x")
    print(f"\n  wrote {args.out / f'e2e_decomp_{args.tier}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
