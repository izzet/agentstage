"""E-041e — MLE-bench script-only end-to-end A/B (no LLM in the loop).

Mirrors dsbench_e2e.py and kb_e2e.py. Runs an AGENT-WRITTEN reference
solution (lifted verbatim from a successful E-041 full-agentic session
in outputs/mlebench_e2e/scripts/) baseline-vs-staged with verified
cold cache. This removes the LLM-non-determinism noise that dominated
E-041's full-agentic measurements.

The reference solution is the actual Python code Haiku wrote in a
real agentic session — same I/O pattern any real agent solution
would exercise. We run it identically under two conditions:

  BASELINE  — cold-tier reads via XFS. No LD_PRELOAD shim.
  STAGED    — Stager prefetches all input files into /dev/shm; the
              same script reads via LD_PRELOAD shim, which redirects
              reads under the cold root to the hot copies.

The script_only A/B isolates AgentStage's wall-time effect from
agent-strategy variance.

Usage:
    python scripts/microbench/mlebench_e2e.py \\
        --task new-york-city-taxi-fare-prediction \\
        --out outputs/mlebench_e2e/nyc-taxi_e2e
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
from agentstage.workloads.mlebench import (
    load_mle_competition, MLEBENCH_DATA_ROOT,
)

REPO = Path(__file__).resolve().parents[2]
SHIM = (REPO / "src" / "agentstage" / "stager" / "shim"
        / "libagentstage_shim.so").resolve()

# Per-task reference solution (lifted from real E-041 sessions).
# Each script reads the competition's input files using the same relative
# paths the agent natively used ('data/<comp>/<file>'); the script is run
# from a working directory that contains a symlink 'data/<comp>' →
# .../prepared/public/ so paths resolve naturally.
SOLUTIONS = {
    # Lifted: the actual Haiku-written solutions from real E-041 sessions.
    # These are valid but I/O-light (agent uses nrows=500_000 sampling or
    # reads from unpacked dirs we don't stage).
    "lifted": {
        "new-york-city-taxi-fare-prediction":
            REPO / "outputs" / "mlebench_e2e" / "scripts" / "nyc-taxi_solution.py",
        "dogs-vs-cats-redux-kernels-edition":
            REPO / "outputs" / "mlebench_e2e" / "scripts" / "dogs-vs-cats_solution.py",
    },
    # Thorough: production-realistic baselines using fast Arrow-backed
    # parsers (polars for CSV, raw-bytes for zips). Exercises the full
    # I/O path of the staged data. Still parse-bound on local NVMe with
    # modern parsers (polars CSV: ~600 MB/s CPU-bound).
    "thorough": {
        "new-york-city-taxi-fare-prediction":
            REPO / "outputs" / "mlebench_e2e" / "scripts" / "nyc-taxi_thorough.py",
        "dogs-vs-cats-redux-kernels-edition":
            REPO / "outputs" / "mlebench_e2e" / "scripts" / "dogs-vs-cats_thorough.py",
    },
    # Streaming: I/O-bound baselines (chunked raw read or streaming CSV
    # split). Mirrors production ETL patterns (Spark/Beam streaming,
    # polars-streaming, custom DataLoaders). I/O dominates here, so
    # AgentStage's contribution is direct.
    "streaming": {
        "new-york-city-taxi-fare-prediction":
            REPO / "outputs" / "mlebench_e2e" / "scripts" / "nyc-taxi_streaming.py",
        "dogs-vs-cats-redux-kernels-edition":
            REPO / "outputs" / "mlebench_e2e" / "scripts" / "dogs-vs-cats_thorough.py",
    },
}


def list_task_files(workload) -> list[Path]:
    """Physical files we'll evict + stage. Uses the same workspace_prior
    that the agent saw (loader already filters out huge subdirs)."""
    prefix_map = workload.prefix_map
    log_root = prefix_map[0][0].rstrip("/")
    phys_root = prefix_map[0][1].rstrip("/")
    out: list[Path] = []
    for k, paths in workload.workspace_prior.items():
        if k in ("all_files", "output_submission"):
            continue
        for p in paths:
            rel = p[len(log_root) + 1:] if p.startswith(log_root + "/") else p
            phys = Path(phys_root) / rel
            if phys.is_file():
                out.append(phys.resolve())
    return sorted(set(out))


def evict(paths: list[Path], *, verify: bool = True) -> dict:
    """Cold-cache methodology (same as E-030/E-039/E-040)."""
    n_files = n_bytes = 0
    for p in paths:
        try:
            st = p.stat()
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n_files += 1; n_bytes += st.st_size
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


def run_solution(*, solution_path: Path, workspace_dir: Path,
                 task_id: str, prefix_map,
                 ld_preload: str | None, hot_root: str | None,
                 cold_root: str | None) -> dict:
    """Run the reference solution as a subprocess, with the agent's natural
    cwd setup (workspace_dir contains data/<task> symlink → physical root).
    Times the subprocess wall-clock."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    data_dir = workspace_dir / "data"
    data_dir.mkdir(exist_ok=True)
    data_link = data_dir / task_id
    if not data_link.exists():
        data_link.symlink_to(prefix_map[0][1].rstrip("/"))

    # Stage the solution script into the workspace as 'solution.py'.
    # Patch the agent's hard-coded '/workspace/submission.csv' to a
    # cwd-relative path so the script writes into our workspace_dir.
    target_script = workspace_dir / "solution.py"
    src = solution_path.read_text()
    src = src.replace("'/workspace/submission.csv'", "'submission.csv'")
    src = src.replace('"/workspace/submission.csv"', '"submission.csv"')
    target_script.write_text(src)

    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                "UV_PROJECT_ENVIRONMENT"):
        env.pop(var, None)
    path_parts = env.get("PATH", "/usr/bin:/bin").split(":")
    env["PATH"] = ":".join(p for p in path_parts
                            if "/.venv/" not in p
                            and "/agentstage/.venv" not in p)
    env["MPLBACKEND"] = "Agg"
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload
        env["AGENTSTAGE_HOT_ROOT"] = hot_root or ""
        env["AGENTSTAGE_COLD_ROOTS"] = cold_root or ""
        # Script-only A/B always pre-stages to completion BEFORE the
        # subprocess runs, so the shim's per-miss retry-spin can never
        # gain anything — disable it (default 20ms × 25k missing files
        # in dogs-vs-cats train/ = ~500s of pure overhead otherwise).
        env["AGENTSTAGE_RETRY_SPIN_MS"] = "0"
    else:
        env.pop("LD_PRELOAD", None)
        env["AGENTSTAGE_SHIM_DISABLE"] = "1"

    t0 = time.monotonic()
    r = subprocess.run(["/usr/bin/python3", str(target_script.resolve())],
                       cwd=str(workspace_dir.resolve()), env=env,
                       capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0
    return {"elapsed_s": round(elapsed, 3), "returncode": r.returncode,
            "stdout_tail": (r.stdout or "")[-600:],
            "stderr_tail": (r.stderr or "")[-600:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=list(SOLUTIONS["lifted"].keys()))
    parser.add_argument("--solution", choices=["lifted", "thorough", "streaming"],
                        default="thorough",
                        help="lifted = agent-written script from E-041 session; "
                             "thorough = generic full-read baseline (default)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_mlee2e")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not SHIM.is_file():
        print(f"FATAL: shim missing at {SHIM}", file=sys.stderr)
        return 2

    workload = load_mle_competition(args.task)
    solution = SOLUTIONS[args.solution][args.task]
    if not solution.is_file():
        print(f"FATAL: solution {solution} not found", file=sys.stderr)
        return 2

    prefix_map = workload.prefix_map
    targets = list_task_files(workload)
    total_mb = sum(p.stat().st_size for p in targets) / 1024 / 1024
    cold_root_anc = str(Path(prefix_map[0][1].rstrip("/")).parent.parent.parent)
    # cold_root_anc = .../mlebench-data (so AGENTSTAGE_COLD_ROOTS covers any
    # comp's prepared/public subtree)

    print(f"E-041e MLE-bench script-only | task={args.task}")
    print(f"  staging targets: {len(targets)} files ({total_mb:.1f} MB)")
    print(f"  reference solution: {solution}")
    print(f"  cold root ancestor: {cold_root_anc}")
    print()

    hot_root = Path(args.hot_root)
    if hot_root.exists():
        shutil.rmtree(hot_root)
    hot_root.mkdir(parents=True, exist_ok=True)

    # ── BASELINE reps ─────────────────────────────────────────────────
    print(f"[1/2] BASELINE — {args.reps} reps cold...")
    baseline_runs: list[dict] = []
    for i in range(args.reps):
        ev = evict(targets)
        ws = args.out / f"baseline_workspace_{i}"
        if ws.exists():
            shutil.rmtree(ws)
        r = run_solution(
            solution_path=solution, workspace_dir=ws,
            task_id=args.task, prefix_map=prefix_map,
            ld_preload=None, hot_root=None, cold_roots=None
            if False else None,
        ) if False else run_solution(
            solution_path=solution, workspace_dir=ws,
            task_id=args.task, prefix_map=prefix_map,
            ld_preload=None, hot_root=None, cold_root=None,
        )
        baseline_runs.append({**r, "evict": ev})
        print(f"  baseline rep {i+1}: {r['elapsed_s']:.2f}s (rc={r['returncode']})  "
              f"resident_frac_after_evict={ev.get('resident_frac_sample','?')}")
        if r["returncode"] != 0:
            print(f"     stderr: {r['stderr_tail'][:300]}", file=sys.stderr)

    # ── STAGED: prefetch into hot tier ────────────────────────────────
    print(f"\n[2/2] STAGED — prestaging {len(targets)} files into {hot_root}...")
    stager = Stager(
        hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
        max_workers=4, capacity_bytes=64 * 1024**3,
    )
    t_stage0 = time.monotonic()
    futures = stager.prefetch(DataHint(
        detected_files=tuple(str(p) for p in targets),
        tier=3, fired_at_ms=0.0, rule_id="mle_e2e_prestage",
    ))
    n_staged = 0
    for f in futures:
        try:
            f.result(timeout=1800)
            n_staged += 1
        except Exception as e:
            print(f"  stage error: {e!r}", file=sys.stderr)
    stage_elapsed = time.monotonic() - t_stage0
    print(f"  staged {n_staged}/{len(targets)} in {stage_elapsed:.2f}s")

    staged_runs: list[dict] = []
    for i in range(args.reps):
        ev = evict(targets)
        ws = args.out / f"staged_workspace_{i}"
        if ws.exists():
            shutil.rmtree(ws)
        r = run_solution(
            solution_path=solution, workspace_dir=ws,
            task_id=args.task, prefix_map=prefix_map,
            ld_preload=str(SHIM), hot_root=str(hot_root),
            cold_root=cold_root_anc,
        )
        staged_runs.append({**r, "evict": ev})
        print(f"  staged rep {i+1}:   {r['elapsed_s']:.2f}s (rc={r['returncode']})")
        if r["returncode"] != 0:
            print(f"     stderr: {r['stderr_tail'][:300]}", file=sys.stderr)
    stager.shutdown(wait=True)

    base_times = sorted(r["elapsed_s"] for r in baseline_runs if r["returncode"] == 0)
    stag_times = sorted(r["elapsed_s"] for r in staged_runs if r["returncode"] == 0)
    base_med = base_times[len(base_times)//2] if base_times else None
    stag_med = stag_times[len(stag_times)//2] if stag_times else None
    speedup = (base_med / stag_med) if (base_med and stag_med) else None

    result = {
        "experiment": "E-041e",
        "task": args.task,
        "solution_mode": args.solution,
        "reference_solution": str(solution),
        "n_input_files": len(targets),
        "input_total_mb": round(total_mb, 2),
        "n_staged": n_staged,
        "stage_elapsed_s": round(stage_elapsed, 3),
        "reps": args.reps,
        "baseline": {"times_s": base_times, "median_s": base_med,
                      "runs": baseline_runs},
        "staged":   {"times_s": stag_times, "median_s": stag_med,
                      "runs": staged_runs},
        "session_speedup": round(speedup, 3) if speedup else None,
        "wall_time_saved_s": round((base_med or 0) - (stag_med or 0), 2)
                              if (base_med and stag_med) else None,
    }
    (args.out / f"mle_e2e_{args.task}.json").write_text(json.dumps(result, indent=2))

    print()
    print(f"  ===== E-041e {args.task.upper()} RESULT =====")
    print(f"  baseline median: {base_med:>9.2f}s  (n={len(base_times)})")
    print(f"  staged median:   {stag_med:>9.2f}s  (n={len(stag_times)})")
    print(f"  wall-time saved: {result['wall_time_saved_s']:>9.2f}s")
    print(f"  session speedup: {result['session_speedup']}×")
    print(f"\n  wrote {args.out}/mle_e2e_{args.task}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
