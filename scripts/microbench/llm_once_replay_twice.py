"""LLM-once + replay-twice mechanism evaluation campaign.

For each (task, model, rep):
  1. Run a fresh LIVE BASELINE session via aiob_multiturn.py (one LLM cost).
  2. Trajectory-replay the recorded baseline in COLD mode.
  3. Trajectory-replay the recorded baseline in STAGED mode.
  4. Record mechanism speedup = cold_replay / staged_replay.

Halves LLM cost vs. live baseline+staged, and ensures apples-to-apples
mechanism measurement by holding the trajectory fixed.

Output: outputs/replay/<campaign-name>/results.json with per-cell records.

Usage:
    uv run python scripts/microbench/llm_once_replay_twice.py \
        --campaign-name aiob_xxh_main \
        --tasks aiob_201 aiob_202 aiob_203 \
        --models claude-haiku-4-5 claude-sonnet-4-5 gemini-2.5-flash \
        --reps 2
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

REPO = Path(__file__).resolve().parents[2]
MULTITURN = REPO / "scripts" / "microbench" / "aiob_multiturn.py"
REPLAY = REPO / "scripts" / "microbench" / "replay_session.py"


def run_live_baseline(task: str, model: str, out_dir: Path,
                       hot_root: Path) -> dict:
    """Run a fresh live baseline LLM session."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    proc = subprocess.run(
        ["uv", "run", "python", str(MULTITURN),
         "--task", task,
         "--model", model,
         "--mode", "baseline",
         "--out", str(out_dir),
         "--hot-root", str(hot_root),
         "--max-turns", "12"],
        cwd=str(REPO),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"error": "live failed", "rc": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}
    if not (out_dir / "summary.json").exists():
        return {"error": "no summary",
                "stderr_tail": proc.stderr[-2000:]}
    return json.loads((out_dir / "summary.json").read_text())


def run_replay(session_dir: Path, mode: str, out_path: Path,
               hot_root: Path) -> dict:
    """Trajectory-replay a baseline session in cold or staged mode."""
    free = shutil.disk_usage("/dev/shm").free
    if mode == "staged" and free < 20 * 1024**3:
        return {"error": f"/dev/shm only {free/1e9:.1f}GB free, need >=20GB"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["uv", "run", "python", str(REPLAY),
         "--session", str(session_dir),
         "--mode", mode,
         "--out", str(out_path),
         "--hot-root", str(hot_root)],
        cwd=str(REPO),
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        return {"error": "replay failed", "rc": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}
    return json.loads(out_path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-name", required=True,
                    help="Subdir name under outputs/replay/")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="Task IDs (e.g., aiob_201 aiob_202)")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model names (e.g., claude-haiku-4-5)")
    ap.add_argument("--reps", type=int, default=2,
                    help="Reps per (task, model) cell")
    ap.add_argument("--live-out-root", default="outputs/aiob_mt",
                    help="Root for live baseline sessions")
    ap.add_argument("--replay-out-root", default="outputs/replay",
                    help="Root for replay outputs")
    ap.add_argument("--hot-root-live",
                    default="/dev/shm/agentstage_live_baseline",
                    help="Hot root for live baseline (cold-mode, so unused)")
    ap.add_argument("--hot-root-replay",
                    default="/dev/shm/agentstage_replay",
                    help="Hot root for replay")
    ap.add_argument("--skip-existing-baseline", action="store_true",
                    help="Reuse existing baseline dirs if present (faster reruns)")
    args = ap.parse_args()

    live_root = Path(args.live_out_root) / f"_sweep_{args.campaign_name}"
    replay_root = Path(args.replay_out_root) / args.campaign_name
    live_root.mkdir(parents=True, exist_ok=True)
    replay_root.mkdir(parents=True, exist_ok=True)

    # Roster
    cells = [(t, m, r+1)
             for t in args.tasks for m in args.models
             for r in range(args.reps)]
    print(f"=== CAMPAIGN {args.campaign_name} ===", flush=True)
    print(f"Cells: {len(cells)} = "
          f"{len(args.tasks)} tasks x {len(args.models)} models x "
          f"{args.reps} reps", flush=True)
    print(f"Live root: {live_root}", flush=True)
    print(f"Replay root: {replay_root}", flush=True)
    print(flush=True)

    results = []
    t_start = time.monotonic()
    for i, (task, model, rep) in enumerate(cells, 1):
        cell_id = f"{task}_{model.replace('/','_').replace('-','_')}_r{rep}"
        baseline_dir = live_root / cell_id
        print(f"[{i}/{len(cells)}] {task}/{model}/rep{rep}", flush=True)

        # 1. Live baseline
        if args.skip_existing_baseline and (baseline_dir/"summary.json").exists():
            print(f"  [baseline] skipping (exists)", flush=True)
            baseline_summary = json.loads((baseline_dir/"summary.json").read_text())
        else:
            print(f"  [baseline] running live LLM session...", flush=True)
            t0 = time.monotonic()
            baseline_summary = run_live_baseline(
                task, model, baseline_dir, Path(args.hot_root_live))
            wall = time.monotonic() - t0
            if "error" in baseline_summary:
                print(f"  [baseline] ERROR: {baseline_summary['error']}", flush=True)
                results.append({"cell": cell_id, "task": task, "model": model,
                                "rep": rep, "baseline_error": baseline_summary['error']})
                Path(args.replay_out_root, args.campaign_name, "results.json").write_text(json.dumps(results, indent=2))
                continue
            print(f"  [baseline] {baseline_summary['session_elapsed_s']:.1f}s "
                  f"(wall {wall:.0f}s), submitted={baseline_summary.get('submitted')}", flush=True)

        rec = {
            "cell": cell_id, "task": task, "model": model, "rep": rep,
            "baseline_elapsed_s": baseline_summary.get("session_elapsed_s"),
            "baseline_submitted": baseline_summary.get("submitted"),
            "baseline_n_turns": baseline_summary.get("n_turns"),
            "baseline_n_prefetched": baseline_summary.get("n_prefetched_files"),
            "baseline_dir": str(baseline_dir),
        }

        # 2. & 3. Cold and staged replay
        for mode in ("cold", "staged"):
            replay_path = replay_root / f"{cell_id}_{mode}.json"
            print(f"  [replay-{mode}] running...", flush=True)
            t0 = time.monotonic()
            r = run_replay(baseline_dir, mode, replay_path,
                           Path(args.hot_root_replay))
            wall = time.monotonic() - t0
            if "error" in r:
                print(f"  [replay-{mode}] ERROR: {r['error']}", flush=True)
                rec[f"{mode}_error"] = r["error"]
                continue
            print(f"  [replay-{mode}] {r['session_elapsed_s']:.1f}s "
                  f"(wall {wall:.0f}s), fidelity={r.get('fidelity_vs_baseline_minus_kills')}, "
                  f"prefetched={r.get('n_prefetched_files',0)}", flush=True)
            rec[f"{mode}_elapsed_s"] = r["session_elapsed_s"]
            rec[f"{mode}_fidelity"] = r.get("fidelity_vs_baseline_minus_kills")
            shio = r.get("shell_io_aggregate", {})
            rec[f"{mode}_shell_elapsed_s"] = shio.get("total_shell_elapsed_s")
            rec[f"{mode}_read_bytes"] = shio.get("total_read_bytes")
            rec[f"{mode}_syscr"] = shio.get("total_syscr")
            if mode == "staged":
                rec["n_prefetched_staged"] = r.get("n_prefetched_files", 0)

        # Compute mechanism speedup
        if "cold_elapsed_s" in rec and "staged_elapsed_s" in rec:
            rec["mechanism_speedup"] = round(
                rec["cold_elapsed_s"] / rec["staged_elapsed_s"], 3)
            print(f"  ==> mechanism speedup: {rec['mechanism_speedup']}×", flush=True)

        results.append(rec)
        # Checkpoint after each cell
        (replay_root / "results.json").write_text(json.dumps(results, indent=2))
        print(flush=True)

    elapsed_min = (time.monotonic() - t_start) / 60
    print(f"=== DONE in {elapsed_min:.1f} min ===", flush=True)

    # Summary table
    print(f"\n{'cell':50s} {'baseline':>9s} {'cold':>8s} {'staged':>8s} "
          f"{'mech':>6s}  fid_cold")
    for r in results:
        if r.get("mechanism_speedup") is None:
            continue
        print(f"{r['cell'][:50]:50s} {r['baseline_elapsed_s']:>9.1f} "
              f"{r['cold_elapsed_s']:>8.1f} {r['staged_elapsed_s']:>8.1f} "
              f"{r['mechanism_speedup']:>5.2f}× {r['cold_fidelity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
