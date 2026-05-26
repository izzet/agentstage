"""Per-turn analyzer for agentic-loop sweeps (E-040 DSBench / E-041 MLE-bench).

Decomposes each session's wall time into:

  shell_s     — sum of turn durations where run_shell_command was the tool
                used in that turn. THIS is where AgentStage's I/O speedup
                lives — the LD_PRELOAD shim only affects subprocess reads.
  llm_s       — sum of turn durations where no run_shell_command was used
                (pure thinking / file-preview / write_file turns). This is
                LLM reasoning + tool-execute overhead; AgentStage cannot
                change it.
  total_s     — session wall time (sum of all turns).

The shell-only speedup isolates the system's contribution; the total
speedup is what the agent's user actually experiences. Both belong in
the paper — together they bound the regime where AgentStage helps and
explain why a thoughtful agent (Sonnet) with long LLM-reasoning
fractions sees smaller TOTAL speedup than a lightweight agent.

Usage:
    python scripts/microbench/analyze_agentic_sweep.py \\
        --sweep outputs/mlebench_mt/_sweep_sonnet_*
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def decompose_session(summary: dict) -> dict:
    """Returns {total_s, shell_s, llm_s} for one session."""
    total_s = summary.get("session_elapsed_s")
    if total_s is None:
        return {"total_s": None, "shell_s": None, "llm_s": None}
    shell_s = 0.0
    llm_s = 0.0
    for t in summary.get("per_turn", []):
        d = t.get("duration_s", 0.0)
        if "run_shell_command" in t.get("tool_names", []):
            shell_s += d
        else:
            llm_s += d
    return {"total_s": total_s,
            "shell_s": round(shell_s, 3),
            "llm_s": round(llm_s, 3)}


def aggregate_sweep(sweep_dir: str) -> dict:
    """Walks a sweep directory, decomposes every session, returns aggregates."""
    runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in sorted(glob.glob(f"{sweep_dir}/*/")):
        sf = os.path.join(d, "summary.json")
        if not os.path.isfile(sf):
            continue
        s = json.load(open(sf))
        if s.get("crash") or s.get("session_elapsed_s") is None:
            continue
        runs[(s["task"], s["mode"])].append({
            "session_elapsed_s": s["session_elapsed_s"],
            "submitted": s.get("submitted", False),
            "decomp": decompose_session(s),
        })
    return runs


def median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return round(statistics.median(xs), 2)


def fmt_speedup(b: float | None, s: float | None) -> str:
    if not b or not s:
        return "—"
    return f"{b/s:.2f}×"


def report(label: str, runs: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"{'task':<46} {'mode':<9} {'n':>3} "
          f"{'total_s':>10} {'shell_s':>10} {'llm_s':>10} {'sub':>5}")
    print("-" * 105)
    agg: dict[tuple[str, str], dict] = {}
    for (task, mode), rs in sorted(runs.items()):
        totals = [r["decomp"]["total_s"] for r in rs]
        shells = [r["decomp"]["shell_s"] for r in rs]
        llms = [r["decomp"]["llm_s"] for r in rs]
        m_total = median(totals)
        m_shell = median(shells)
        m_llm = median(llms)
        subs = sum(1 for r in rs if r["submitted"])
        agg[(task, mode)] = {"total": m_total, "shell": m_shell, "llm": m_llm,
                              "subs": subs, "n": len(rs)}
        print(f"{task[:46]:<46} {mode:<9} {len(rs):>3} "
              f"{m_total:>9.1f}s {m_shell:>9.1f}s {m_llm:>9.1f}s "
              f"{subs}/{len(rs):<3}")
    print()
    print(f"{'task':<46} {'total speedup':>16} {'shell speedup':>16} "
          f"{'llm time stable?':>16}")
    print("-" * 95)
    for t in sorted(set(t for (t, _) in runs.keys())):
        b = agg.get((t, "baseline"))
        s = agg.get((t, "staged"))
        if not (b and s):
            continue
        tot_sp = fmt_speedup(b["total"], s["total"])
        sh_sp = fmt_speedup(b["shell"], s["shell"])
        llm_b = b["llm"]; llm_s = s["llm"]
        llm_consist = "(same)" if (llm_b and llm_s and
                                    abs(llm_b - llm_s) / max(llm_b, llm_s) < 0.30) \
            else f"({llm_b:.0f} vs {llm_s:.0f}s)"
        print(f"{t[:46]:<46} {tot_sp:>16} {sh_sp:>16} {llm_consist:>16}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True,
                        help="Glob for sweep dirs (multiple OK)")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    matches = sorted(glob.glob(args.sweep))
    if not matches:
        print(f"FATAL: no sweeps matched {args.sweep}")
        return 2

    for d in matches:
        label = args.label or Path(d).name
        runs = aggregate_sweep(d)
        report(label, runs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
