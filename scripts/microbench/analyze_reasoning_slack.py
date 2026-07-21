"""Compute per-turn reasoning slack across sessions.

For each turn we have:
  - duration_s: total turn wall time (LLM streaming + tool execution)
  - turns/turn_NN/thinking.jsonl: timestamped thinking-delta stream
  - turns/turn_NN/text.jsonl: timestamped visible-text-delta stream
  - turns/turn_NN/tool_use.jsonl: tool invocations

Reasoning slack = the window between the FIRST streaming delta and the
LAST streaming delta in a turn (before tool execution begins). This is
the time the LLM was generating output, during which the Stager can
prefetch in parallel.

For each session we report:
  - total streaming time across all turns
  - shell execution time
  - llm-only turn time (turns with no shell command)
  - reasoning slack fraction = streaming_time / total_session_time

For the paper this lets us claim: "AgentStage overlaps X seconds of
prefetch with Y seconds of LLM streaming per session, yielding a
shell-time speedup of Z×."

Usage:
    python scripts/microbench/analyze_reasoning_slack.py \\
        --sweep "outputs/aiob_mt/_sweep_haiku_*_v3"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def turn_streaming_ms(turn_dir: Path) -> float:
    """Compute LLM streaming time for one turn from text/thinking jsonl logs.

    Returns the time (ms) from the first to the last streaming delta —
    i.e., the LLM-generation window. Tool execution time is NOT included
    (it happens after streaming ends within the turn).
    """
    t_min: float | None = None
    t_max: float | None = None
    for stream in ("thinking.jsonl", "text.jsonl"):
        p = turn_dir / stream
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("t_ms")
            if t is None:
                continue
            if t_min is None or t < t_min:
                t_min = t
            if t_max is None or t > t_max:
                t_max = t
    if t_min is None or t_max is None:
        return 0.0
    return max(0.0, t_max - t_min)


def analyze_session(session_dir: Path) -> dict | None:
    """Return per-session metrics dict, or None if no summary.

    Two slack metrics are reported:
      streaming_s : cumulative LLM-streaming time (first-to-last delta
                    per turn, summed). Underestimates the prefetch
                    opportunity because it excludes inter-turn time.
      prefetch_window_s : time from the FIRST rule dispatch to the FIRST
                    run_shell_command. This is the real budget the
                    Stager has to complete copies before the agent's
                    script opens the staged files.
    """
    sf = session_dir / "summary.json"
    if not sf.is_file():
        return None
    s = json.loads(sf.read_text())
    if s.get("crash"):
        return None
    total_s = s.get("session_elapsed_s")
    if total_s is None:
        return None

    streaming_ms = 0.0
    shell_s = 0.0
    llm_only_s = 0.0  # turns with no shell command
    turns = s.get("per_turn", [])

    for t in turns:
        idx = t.get("turn", 0)
        turn_dir = session_dir / "turns" / f"turn_{idx:02d}"
        if turn_dir.is_dir():
            streaming_ms += turn_streaming_ms(turn_dir)
        dur = t.get("duration_s", 0.0)
        if "run_shell_command" in t.get("tool_names", []):
            shell_s += dur
        else:
            llm_only_s += dur

    streaming_s = streaming_ms / 1000.0

    # Effective prefetch window: first rule dispatch -> first shell call
    first_dispatch_t: float | None = None
    cum = 0.0
    for t in turns:
        if t.get("fired_rules") or t.get("dispatched_prefetches"):
            first_dispatch_t = cum
            break
        cum += t.get("duration_s", 0.0)

    first_shell_t: float | None = None
    cum = 0.0
    for t in turns:
        if "run_shell_command" in t.get("tool_names", []):
            first_shell_t = cum
            break
        cum += t.get("duration_s", 0.0)

    prefetch_window_s: float | None = None
    if first_dispatch_t is not None and first_shell_t is not None:
        prefetch_window_s = round(first_shell_t - first_dispatch_t, 2)

    return {
        "task": s.get("task"),
        "model": s.get("model"),
        "mode": s.get("mode"),
        "total_s": total_s,
        "shell_s": round(shell_s, 2),
        "llm_only_s": round(llm_only_s, 2),
        "streaming_s": round(streaming_s, 2),
        "slack_frac": round(streaming_s / total_s, 3) if total_s else None,
        "prefetch_window_s": prefetch_window_s,
        "n_turns": s.get("n_turns", 0),
        "n_tool_uses": s.get("n_tool_uses", 0),
        "n_prefetched": s.get("n_prefetched_files", 0),
        "n_outputs": s.get("n_workspace_outputs", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True,
                        help="Glob for sweep dirs (multiple OK)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional CSV output path")
    args = parser.parse_args()

    rows: list[dict] = []
    for sweep_glob in sorted(glob.glob(args.sweep)):
        sweep = Path(sweep_glob)
        if not sweep.is_dir():
            continue
        for sess in sorted(sweep.iterdir()):
            if not sess.is_dir():
                continue
            r = analyze_session(sess)
            if r is None:
                continue
            r["sweep"] = sweep.name
            r["session"] = sess.name
            rows.append(r)

    if not rows:
        print(f"No sessions found matching {args.sweep}")
        return 2

    print(f"{'sweep':<40} {'session':<40} {'mode':<9} "
          f"{'total_s':>8} {'shell_s':>8} {'stream_s':>9} "
          f"{'pre_win':>9} {'slack_f':>8}")
    print("-" * 130)
    for r in rows:
        pw = r.get("prefetch_window_s")
        pw_str = f"{pw:>7.1f}s" if pw is not None else "    n/a"
        print(f"{r['sweep'][:39]:<40} {r['session'][:39]:<40} {r['mode']:<9} "
              f"{r['total_s']:>7.1f}s {r['shell_s']:>7.1f}s "
              f"{r['streaming_s']:>8.1f}s {pw_str:>9} {r['slack_frac']:>7.3f}")

    # Aggregate per (task, mode)
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["model"], r["mode"])].append(r)

    print()
    print(f"{'task':<40} {'model':<24} {'mode':<9} "
          f"{'med_total':>10} {'med_shell':>10} {'med_stream':>10} "
          f"{'med_win':>9} {'med_slack':>10}")
    print("-" * 125)
    for (task, model, mode), rs in sorted(by_cell.items()):
        totals = [r["total_s"] for r in rs]
        shells = [r["shell_s"] for r in rs]
        streams = [r["streaming_s"] for r in rs]
        windows = [r["prefetch_window_s"] for r in rs
                   if r.get("prefetch_window_s") is not None]
        slacks = [r["slack_frac"] for r in rs if r["slack_frac"] is not None]
        win_med = statistics.median(windows) if windows else None
        win_str = f"{win_med:>7.1f}s" if win_med is not None else "    n/a"
        print(f"{task[:39]:<40} {model[:23]:<24} {mode:<9} "
              f"{statistics.median(totals):>9.1f}s "
              f"{statistics.median(shells):>9.1f}s "
              f"{statistics.median(streams):>9.1f}s "
              f"{win_str:>9} "
              f"{statistics.median(slacks):>10.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
