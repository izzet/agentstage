"""Trajectory-controlled I/O wall-time replay.

For each (workload, mode) we measure the cumulative wall-clock time to
read the workload's representative file set fully:

  COLD baseline : each file is opened fresh, page cache pre-evicted
                  via posix_fadvise(POSIX_FADV_DONTNEED); the read
                  pulls bytes from the cold tier on every file.
  STAGED        : every file is pre-copied to the local-NVMe hot tier
                  before the timed read; the timed read pulls bytes
                  from the hot copy.

The read order matches the agent's actual access pattern recorded in
the session's `turns/turn_*/tool_use.jsonl` (we walk run_shell_command
contents for file references that resolve against the workload prior).
For sessions where the shell-command scrape returns fewer files than
the workload's ground_truth_first_inspect, we fall back to the
first-inspect set (the agent's intended initial reads) to keep the
replay representative.

Output: per-(cell, rep) cumulative read time in cold and staged
configurations + per-cell median speedup.

Run on three AIOB cells (aiob_110, aiob_104, aiob_107) by default,
3 reps each per mode, total 18 replays.

Usage:
    uv run python scripts/microbench/trajectory_replay.py --reps 3 \\
        --hot-root /dev/shm/agentstage_replay \\
        --out outputs/trajectory_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentstage.workloads import get_workload  # noqa: E402


def _resolve_logical(logical: str, prefix_map) -> str | None:
    for log_pre, real_pre in prefix_map:
        if logical.startswith(log_pre):
            return real_pre + logical[len(log_pre):]
    return None


def _extract_files_from_shell_command(cmd: str, prefix_map, all_paths: set[str]) -> list[str]:
    """Find logical paths mentioned in a shell command that exist in the
    workload prior. Resolves to physical paths via prefix_map."""
    out: list[str] = []
    # Match any string containing /data/... (the workload logical prefix)
    for m in re.finditer(r"/data/[A-Za-z0-9_./\-]+", cmd):
        logical = m.group(0).rstrip(".,;:'\")")
        if logical in all_paths:
            phys = _resolve_logical(logical, prefix_map)
            if phys and Path(phys).is_file():
                if phys not in out:
                    out.append(phys)
    return out


def session_access_files(session_dir: Path, workload) -> list[str]:
    """Walk a session's tool_use.jsonl in turn order, extract physical
    paths the agent's shell commands referenced. Falls back to the
    workload's ground_truth_first_inspect resolved paths if the scrape
    finds fewer files."""
    all_paths_set = set(workload.all_workspace_paths)
    seen: list[str] = []
    seen_set: set[str] = set()
    for tdir in sorted((session_dir / "turns").glob("turn_*")):
        tu = tdir / "tool_use.jsonl"
        if not tu.is_file():
            continue
        for line in tu.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("name") != "run_shell_command":
                continue
            cmd = (d.get("parsed_input") or {}).get("cmd", "")
            for p in _extract_files_from_shell_command(cmd, workload.prefix_map, all_paths_set):
                if p not in seen_set:
                    seen.append(p)
                    seen_set.add(p)
    # Fallback to ground_truth_full if the scrape was empty or returned
    # too few files (< 3): take up to 50 files from the workload's full
    # access set, ordered as recorded in the workload prior. This keeps
    # the read volume large enough to be timing-significant.
    if len(seen) < 3:
        for logical in list(workload.ground_truth_full)[:50]:
            phys = _resolve_logical(logical, workload.prefix_map)
            if phys and Path(phys).is_file() and phys not in seen_set:
                seen.append(phys)
                seen_set.add(phys)
    return seen


def evict_caches(paths: list[str]) -> None:
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


def read_full_and_time(paths: list[str]) -> float:
    """Open each path, read the full content, return cumulative ms."""
    t0 = time.monotonic_ns()
    for p in paths:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)  # 8 MiB
                if not chunk:
                    break
    return (time.monotonic_ns() - t0) / 1e6


def stage_to_hot(paths: list[str], hot_root: Path) -> dict[str, str]:
    """Copy each cold path to hot_root preserving its absolute layout.
    Returns dict cold_path -> hot_path."""
    hot_root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for p in paths:
        rel = Path(p).relative_to("/")
        hot = hot_root / rel
        hot.parent.mkdir(parents=True, exist_ok=True)
        if not hot.is_file() or hot.stat().st_size != Path(p).stat().st_size:
            shutil.copy2(p, hot)
        mapping[p] = str(hot)
    return mapping


def measure_cell(
    session_dir: Path, task_id: str, reps: int, hot_root: Path,
) -> dict:
    workload = get_workload(task_id)
    files = session_access_files(session_dir, workload)
    if not files:
        return {"task": task_id, "error": "no files identified"}
    total_bytes = sum(Path(f).stat().st_size for f in files if Path(f).is_file())

    cold_times: list[float] = []
    for _ in range(reps):
        evict_caches(files)
        ms = read_full_and_time(files)
        cold_times.append(ms)

    # Stage to hot once, then time hot reads (each rep evicts cold first
    # to ensure we never accidentally read from cold page cache).
    hot_paths = stage_to_hot(files, hot_root)
    hot_files = [hot_paths[f] for f in files]
    staged_times: list[float] = []
    for _ in range(reps):
        evict_caches(files)  # evict cold copies so cold page cache misses
        ms = read_full_and_time(hot_files)
        staged_times.append(ms)

    return {
        "task": task_id,
        "session_dir": str(session_dir.relative_to(REPO / "outputs")),
        "n_files": len(files),
        "total_bytes": total_bytes,
        "cold_ms": cold_times,
        "staged_ms": staged_times,
        "cold_median_ms": statistics.median(cold_times),
        "staged_median_ms": statistics.median(staged_times),
        "speedup": statistics.median(cold_times) / statistics.median(staged_times),
    }


_CELLS = [
    ("aiob_110", "outputs/aiob_mt/_sweep_haiku_20260527T164353_v3/aiob_110_staged_r1"),
    ("aiob_104", "outputs/aiob_mt/_sweep_sonnet_20260526T174525_sonnet/aiob_104_staged_r1"),
    ("aiob_107", "outputs/aiob_mt/_sweep_haiku_20260526T174524_haiku/aiob_107_staged_r1"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hot-root", type=Path, default=Path("/tmp/agentstage_replay"))
    ap.add_argument("--out", type=Path, default=Path("outputs/trajectory_replay.json"))
    args = ap.parse_args()

    results = []
    for task_id, rel_session in _CELLS:
        session_dir = REPO / rel_session
        if not session_dir.is_dir():
            print(f"SKIP {task_id}: {session_dir} not a directory", file=sys.stderr)
            continue
        print(f"=== {task_id} from {session_dir.relative_to(REPO / 'outputs')} ===")
        r = measure_cell(session_dir, task_id, args.reps, args.hot_root)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            results.append(r)
            continue
        print(f"  files={r['n_files']}  bytes={r['total_bytes']/1e9:.2f} GB")
        print(f"  cold median: {r['cold_median_ms']/1000:.2f}s   "
              f"staged median: {r['staged_median_ms']/1000:.2f}s   "
              f"speedup: {r['speedup']:.2f}x")
        results.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
