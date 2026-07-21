"""Three-way decomposition of agentic session wall time.

For each turn, we split the wall time into:
  comm_s   : time from turn start to first streaming delta
             (API latency + network round-trip; LLM hasn't emitted yet)
  stream_s : time from first to last streaming delta
             (the LLM is actively generating tokens — this is the
              window AgentStage's reasoning-slack prefetch exploits)
  tool_s   : on shell turns, time from last streaming delta to end of
             turn (the subprocess running python3 solution.py)
  other_s  : same residual on non-shell turns (file write, list_dir,
             harness overhead).

Per-session: wall_s = comm_s + stream_s + tool_s + other_s.

Aggregated per cell (median over reps), this answers two reviewer
questions:
  (a) How much of a session is the LLM actually "thinking" vs
      waiting on the network? (stream_s vs comm_s)
  (b) How much of a session is genuinely tool execution that
      AgentStage can accelerate? (tool_s)

Usage:
    python scripts/microbench/decompose_wall_time.py \\
        --sweep "outputs/aiob_mt/_sweep_*_v3"
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def turn_first_last_ms(turn_dir: Path) -> tuple[float | None, float | None]:
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
    return t_min, t_max


def decompose(session_dir: Path) -> dict | None:
    sf = session_dir / "summary.json"
    if not sf.is_file():
        return None
    s = json.loads(sf.read_text())
    if s.get("crash"):
        return None
    total = s.get("session_elapsed_s")
    if total is None:
        return None
    comm = stream = tool = other = 0.0
    for t in s.get("per_turn", []):
        idx = t.get("turn", 0)
        dur_ms = t.get("duration_s", 0) * 1000.0
        td = session_dir / "turns" / f"turn_{idx:02d}"
        t_first, t_last = (None, None)
        if td.is_dir():
            t_first, t_last = turn_first_last_ms(td)
        if t_first is None or t_last is None:
            # No streaming deltas — Haiku often returns a tool_use payload
            # with no thinking/text. Treat the entire turn as tool/other
            # based on which tool ran.
            if "run_shell_command" in t.get("tool_names", []):
                tool += dur_ms / 1000.0
            else:
                other += dur_ms / 1000.0
            continue
        comm += t_first / 1000.0
        stream += (t_last - t_first) / 1000.0
        rest = (dur_ms - t_last) / 1000.0
        if "run_shell_command" in t.get("tool_names", []):
            tool += rest
        else:
            other += rest
    return {
        "task": s.get("task"),
        "model": s.get("model"),
        "mode": s.get("mode"),
        "total_s": round(total, 2),
        "comm_s": round(comm, 2),
        "stream_s": round(stream, 2),
        "tool_s": round(tool, 2),
        "other_s": round(other, 2),
    }


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True,
                        help="Glob for sweep dirs (multiple OK)")
    args = parser.parse_args()

    rows: list[dict] = []
    for sweep_glob in sorted(glob.glob(args.sweep)):
        sweep = Path(sweep_glob)
        if not sweep.is_dir():
            continue
        for sess in sorted(sweep.iterdir()):
            if not sess.is_dir():
                continue
            r = decompose(sess)
            if r is None:
                continue
            r["sweep"] = sweep.name
            r["session"] = sess.name
            rows.append(r)

    if not rows:
        print(f"No sessions found matching {args.sweep}")
        return 2

    print(f"{'session':<60} {'mode':<9} {'total':>7} "
          f"{'comm':>6} {'stream':>7} {'tool':>7} {'other':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r['session'][:59]:<60} {r['mode']:<9} "
              f"{r['total_s']:>6.0f}s {r['comm_s']:>5.0f}s "
              f"{r['stream_s']:>6.0f}s {r['tool_s']:>6.0f}s "
              f"{r['other_s']:>6.0f}s")

    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["model"], r["mode"])].append(r)

    print()
    print("=== Medians per cell ===")
    print(f"{'task':<38} {'model':<22} {'mode':<9} "
          f"{'total':>7} {'comm':>6} {'stream':>7} {'tool':>7} "
          f"{'tool%':>6} {'stream%':>7} {'comm%':>6}")
    print("-" * 122)
    for k in sorted(by_cell):
        rs = by_cell[k]
        totals = [r["total_s"] for r in rs]
        comms = [r["comm_s"] for r in rs]
        streams = [r["stream_s"] for r in rs]
        tools = [r["tool_s"] for r in rs]
        t_med = med(totals)
        c_med = med(comms)
        s_med = med(streams)
        tl_med = med(tools)
        print(f"{k[0][:37]:<38} {k[1][:21]:<22} {k[2]:<9} "
              f"{t_med:>6.0f}s {c_med:>5.0f}s {s_med:>6.0f}s {tl_med:>6.0f}s "
              f"{tl_med/t_med:>5.0%} {s_med/t_med:>6.1%} {c_med/t_med:>5.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
