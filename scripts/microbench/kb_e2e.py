"""E-038 — KramaBench end-to-end speedup (full AgentStage stack).

Runs KramaBench reference solutions baseline-vs-staged. Mirrors
path_b_e2e.py but for KB tasks. Real script + real input files + real
stager + real LD_PRELOAD shim.

Local NFS-class storage is the HEADLINE. Optional --throttle-mbps adds
a tc-rate-limited measurement for PFS-class regimes.

Usage:
    python scripts/microbench/kb_e2e.py --task biomedical-hard-1 \\
        --out outputs/kb_e2e/bh1_native
    python scripts/microbench/kb_e2e.py --task wildfire-easy-2 \\
        --out outputs/kb_e2e/we2_native
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agentstage.stager import DataHint, Stager

REPO = Path(__file__).resolve().parents[2]
KB_ROOT = REPO / "external" / "benchmarks" / "kramabench"
SHIM = (REPO / "src" / "agentstage" / "stager" / "shim"
        / "libagentstage_shim.so").resolve()


# Per-task: (domain, solution_relpath, [input_glob_patterns])
TASKS: dict[str, tuple[str, str, list[str]]] = {
    "biomedical-hard-1": (
        "biomedical",
        "biomedical/biomedical-hard-1.py",
        ["1-s2.0-S0092867420301070-mmc1.xlsx",
         "1-s2.0-S0092867420301070-mmc2.xlsx"],
    ),
    "wildfire-easy-2": (
        "wildfire",
        "wildfire/wildfire-easy-2.py",
        ["usa.gpkg", "nifc_geographic_areas.gpkg"],
    ),
    "wildfire-easy-3": (
        "wildfire",
        "wildfire/wildfire-easy-3.py",
        ["usa.gpkg", "nifc_geographic_areas.gpkg"],
    ),
    "astronomy-hard-7": (
        "astronomy",
        "astronomy/astronomy-hard-7.py",
        ["STORM-AI/warmup/v2/OMNI2/*.csv",
         "STORM-AI/warmup/v2/GOES/*.csv",
         "STORM-AI/warmup/v2/Sat_Density/*.csv"],
    ),
}


def enumerate_input_files(task_name: str) -> list[Path]:
    """Resolve input glob patterns to actual file paths."""
    domain, _, patterns = TASKS[task_name]
    base = KB_ROOT / "data" / domain / "input"
    out: list[Path] = []
    for pat in patterns:
        for p in base.glob(pat):
            if p.is_file():
                out.append(p.resolve())
    return sorted(set(out))


def evict(paths: list[Path], *, verify: bool = True) -> dict:
    """AIOB-style: posix_fadvise(DONTNEED) per file + mincore verify on sample."""
    n_files = n_bytes = 0
    for p in paths:
        try:
            st = p.stat()
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n_files += 1
                n_bytes += st.st_size
            finally:
                os.close(fd)
        except OSError:
            pass
    os.sync()
    if not verify:
        return {"files": n_files, "bytes": n_bytes}
    try:
        from agentiobench.utils.cache import _resident_pages
    except ImportError:
        return {"files": n_files, "bytes": n_bytes}
    resident = total = 0
    for p in paths[:5]:
        try:
            r, t = _resident_pages(p)
            resident += r; total += t
        except OSError:
            continue
    return {"files": n_files, "bytes": n_bytes,
            "resident_pages_sample": resident,
            "total_pages_sample": total,
            "resident_frac_sample": resident / total if total else 0.0}


def run_solution(task_name: str, *, ld_preload: str | None, hot_root: str | None,
                 cold_roots: str | None) -> dict:
    """Run a KB reference solution from external/benchmarks/kramabench/ as CWD.

    The script's data_path "./data/<domain>/input/" resolves to the
    cold-tier path; shim redirects opens to the staged hot copies."""
    _, solution_relpath, _ = TASKS[task_name]
    script = KB_ROOT / "solutions" / solution_relpath

    env = os.environ.copy()
    # Strip uv-venv vars — KB references pandas/scipy from system site-packages
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                "UV_PROJECT_ENVIRONMENT"):
        env.pop(var, None)
    env["MPLBACKEND"] = "Agg"
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload
        env["AGENTSTAGE_HOT_ROOT"] = hot_root or ""
        env["AGENTSTAGE_COLD_ROOTS"] = cold_roots or ""
        env["AGENTSTAGE_RETRY_SPIN_MS"] = "20"
    else:
        env.pop("LD_PRELOAD", None)
        env["AGENTSTAGE_SHIM_DISABLE"] = "1"

    t0 = time.monotonic()
    r = subprocess.run(["/usr/bin/python3", str(script)],
                       cwd=str(KB_ROOT), env=env,
                       capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0
    return {"elapsed_s": round(elapsed, 3), "returncode": r.returncode,
            "stdout_tail": r.stdout[-600:],
            "stderr_tail": r.stderr[-600:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_kb")
    parser.add_argument("--reps", type=int, default=3,
                        help="Repetitions per condition for median (default 3)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not SHIM.is_file():
        print(f"FATAL: shim missing at {SHIM}", file=sys.stderr)
        return 2

    domain, _, _ = TASKS[args.task]
    cold_root_anc = str((KB_ROOT / "data" / domain).resolve())  # ancestor for shim

    targets = enumerate_input_files(args.task)
    if not targets:
        print(f"FATAL: no input files resolved for {args.task}", file=sys.stderr)
        return 2
    total_mb = sum(p.stat().st_size for p in targets) / 1024 / 1024
    print(f"E-038 KB end-to-end | task={args.task} domain={domain}")
    print(f"  reference solution: {TASKS[args.task][1]}")
    print(f"  input files: {len(targets)} ({total_mb:.1f} MB)")
    print(f"  cold-root ancestor: {cold_root_anc}")
    print()

    hot_root = Path(args.hot_root)
    if hot_root.exists():
        shutil.rmtree(hot_root)
    hot_root.mkdir(parents=True, exist_ok=True)

    # ── BASELINE reps ─────────────────────────────────────────────────
    print(f"[1/2] BASELINE — {args.reps} reps cold...")
    baseline_runs: list[dict] = []
    for i in range(args.reps):
        ev = evict([Path(p) for p in targets])
        r = run_solution(args.task, ld_preload=None,
                          hot_root=None, cold_roots=None)
        baseline_runs.append({**r, "evict": ev})
        print(f"  baseline rep {i+1}: {r['elapsed_s']:.2f}s (rc={r['returncode']})  "
              f"resident_frac_after_evict={ev.get('resident_frac_sample','?')}")

    # ── STAGED: prefetch into hot tier ────────────────────────────────
    print(f"\n[2/2] STAGED — prestaging {len(targets)} files into {hot_root}...")
    stager = Stager(
        hot_root=hot_root,
        cold_roots=[Path(cold_root_anc)],
        max_workers=4, capacity_bytes=32 * 1024**3,
    )
    t_stage0 = time.monotonic()
    futures = stager.prefetch(DataHint(
        detected_files=tuple(str(p) for p in targets),
        tier=3, fired_at_ms=0.0, rule_id="kb_e2e_prestage",
    ))
    n_staged = 0
    for f in futures:
        try:
            f.result(timeout=600)
            n_staged += 1
        except Exception as e:
            print(f"  stage error: {e!r}", file=sys.stderr)
    stage_elapsed = time.monotonic() - t_stage0
    print(f"  staged {n_staged}/{len(targets)} in {stage_elapsed:.2f}s")

    staged_runs: list[dict] = []
    for i in range(args.reps):
        ev = evict([Path(p) for p in targets])  # force shim path (cold tier MUST be evicted)
        r = run_solution(args.task, ld_preload=str(SHIM),
                          hot_root=str(hot_root), cold_roots=cold_root_anc)
        staged_runs.append({**r, "evict": ev})
        print(f"  staged rep {i+1}:   {r['elapsed_s']:.2f}s (rc={r['returncode']})")
    stager.shutdown(wait=True)

    # ── Aggregate ─────────────────────────────────────────────────────
    base_times = sorted(r["elapsed_s"] for r in baseline_runs if r["returncode"] == 0)
    stag_times = sorted(r["elapsed_s"] for r in staged_runs if r["returncode"] == 0)
    base_med = base_times[len(base_times)//2] if base_times else None
    stag_med = stag_times[len(stag_times)//2] if stag_times else None
    speedup = (base_med / stag_med) if (base_med and stag_med) else None

    result = {
        "experiment": "E-038",
        "task": args.task,
        "domain": domain,
        "n_input_files": len(targets),
        "input_total_mb": round(total_mb, 2),
        "n_staged": n_staged,
        "stage_elapsed_s": round(stage_elapsed, 3),
        "reps": args.reps,
        "baseline": {"times_s": base_times,
                     "median_s": base_med,
                     "runs": baseline_runs},
        "staged":   {"times_s": stag_times,
                     "median_s": stag_med,
                     "runs": staged_runs},
        "session_speedup": round(speedup, 3) if speedup else None,
        "wall_time_saved_s": round((base_med or 0) - (stag_med or 0), 2)
                              if (base_med and stag_med) else None,
    }
    (args.out / f"kb_e2e_{args.task}.json").write_text(json.dumps(result, indent=2))

    print()
    print(f"  ===== E-038 {args.task.upper()} RESULT =====")
    print(f"  baseline median: {base_med:>9.2f}s  (n={len(base_times)})")
    print(f"  staged median:   {stag_med:>9.2f}s  (n={len(stag_times)})")
    print(f"  wall-time saved: {result['wall_time_saved_s']:>9.2f}s")
    print(f"  session speedup: {result['session_speedup']}×")
    print(f"\n  wrote {args.out}/kb_e2e_{args.task}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
