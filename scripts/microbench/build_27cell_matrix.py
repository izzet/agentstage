"""Build the full 27-cell paper matrix: 3 benchmarks × 3 models × 3 tasks.

For each canonical sweep, decompose every session into (total_s, shell_s,
llm_s), take 3-rep medians, and emit a single CSV + markdown table.

Canonical sweeps (most recent per benchmark x model):

    DSBench
      haiku   outputs/dsbench_mt/_sweep_20260525T113623
      sonnet  outputs/dsbench_mt/_sweep_sonnet_20260526T150919
      gemini  outputs/dsbench_mt/_sweep_gemini_20260526T151730

    MLE-bench
      haiku   outputs/mlebench_mt/_sweep_20260525T165123
      sonnet  outputs/mlebench_mt/_sweep_sonnet_20260526T150922_mle
      gemini  outputs/mlebench_mt/_sweep_gemini_20260526T151731_mle

    AIOB (v2 — post-prompt-fix, post-pysam-install, post-crash-retry)
      haiku   outputs/aiob_mt/_sweep_haiku_20260526T192754_v2_haiku
      sonnet  outputs/aiob_mt/_sweep_sonnet_20260526T192755_v2_sonnet
      gemini  outputs/aiob_mt/_sweep_gemini_20260526T192756_v2_gemini
"""

from __future__ import annotations
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/common/iyildirim/projects/agentstage")

SWEEPS = {
    ("DSBench", "haiku"):  ROOT / "outputs/dsbench_mt/_sweep_20260525T113623",
    ("DSBench", "sonnet"): ROOT / "outputs/dsbench_mt/_sweep_sonnet_20260526T150919",
    ("DSBench", "gemini"): ROOT / "outputs/dsbench_mt/_sweep_gemini_20260526T151730",
    ("MLE-bench", "haiku"):  ROOT / "outputs/mlebench_mt/_sweep_20260525T165123",
    ("MLE-bench", "sonnet"): ROOT / "outputs/mlebench_mt/_sweep_sonnet_20260526T150922_mle",
    ("MLE-bench", "gemini"): ROOT / "outputs/mlebench_mt/_sweep_gemini_20260526T151731_mle",
    ("AIOB", "haiku"):  ROOT / "outputs/aiob_mt/_sweep_haiku_20260527T164353_v3",
    ("AIOB", "sonnet"): ROOT / "outputs/aiob_mt/_sweep_sonnet_20260527T164353_v3",
    ("AIOB", "gemini"): ROOT / "outputs/aiob_mt/_sweep_gemini_20260527T164353_v3",
}

# Canonical 3 tasks per benchmark — these match how the sweeps were launched.
TASKS = {
    "DSBench": ["lmsys-chatbot-arena", "tabular-playground-series-may-2022",
                "ventilator-pressure-prediction"],
    "MLE-bench": ["dogs-vs-cats-redux-kernels-edition",
                  "histopathologic-cancer-detection",
                  "new-york-city-taxi-fare-prediction"],
    "AIOB": ["aiob_103", "aiob_107", "aiob_110"],
}


def decompose(s):
    total_s = s.get("session_elapsed_s")
    if total_s is None or s.get("crash"):
        return None
    shell_s = sum(t.get("duration_s", 0) for t in s.get("per_turn", [])
                  if "run_shell_command" in t.get("tool_names", []))
    llm_s = total_s - shell_s
    return {"total": total_s, "shell": shell_s, "llm": llm_s}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def speedup(b, s):
    if b is None or s is None or s == 0:
        return None
    return round(b / s, 2)


def load_sweep(sweep_dir: Path):
    """Returns {(task, mode): [decomp, ...]}."""
    runs = defaultdict(list)
    for f in sorted(sweep_dir.glob("*/summary.json")):
        s = json.load(open(f))
        d = decompose(s)
        if d is None:
            continue
        runs[(s["task"], s["mode"])].append(d)
    return runs


def main():
    rows = []
    out_csv = ROOT / "outputs/27cell_matrix.csv"

    print(f"{'Benchmark':<10} {'Model':<7} {'Task':<40} "
          f"{'total_b':>8} {'total_s':>8} {'tot_sp':>7} "
          f"{'shell_b':>8} {'shell_s':>8} {'sh_sp':>7}")
    print("-" * 110)

    for (bench, model), sweep_dir in SWEEPS.items():
        runs = load_sweep(sweep_dir)
        for task in TASKS[bench]:
            b = runs.get((task, "baseline"), [])
            s = runs.get((task, "staged"), [])
            tot_b = med([r["total"] for r in b])
            tot_s = med([r["total"] for r in s])
            sh_b = med([r["shell"] for r in b])
            sh_s = med([r["shell"] for r in s])
            tot_sp = speedup(tot_b, tot_s)
            sh_sp = speedup(sh_b, sh_s)
            rows.append({
                "benchmark": bench, "model": model, "task": task,
                "n_baseline": len(b), "n_staged": len(s),
                "total_baseline_s": tot_b, "total_staged_s": tot_s,
                "shell_baseline_s": sh_b, "shell_staged_s": sh_s,
                "total_speedup": tot_sp, "shell_speedup": sh_sp,
            })
            print(f"{bench:<10} {model:<7} {task[:39]:<40} "
                  f"{tot_b or 0:>7.1f}s {tot_s or 0:>7.1f}s "
                  f"{(f'{tot_sp:.2f}x' if tot_sp else '-'):>7} "
                  f"{sh_b or 0:>7.1f}s {sh_s or 0:>7.1f}s "
                  f"{(f'{sh_sp:.2f}x' if sh_sp else '-'):>7}")

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"wrote {out_csv}")
    print()
    # Aggregates
    tot_sp = [r["total_speedup"] for r in rows if r["total_speedup"]]
    sh_sp = [r["shell_speedup"] for r in rows if r["shell_speedup"]]
    print(f"Across all 27 cells:")
    print(f"  total speedup  median={statistics.median(tot_sp):.2f}x  "
          f"mean={statistics.mean(tot_sp):.2f}x  "
          f"range=[{min(tot_sp):.2f}x .. {max(tot_sp):.2f}x]")
    print(f"  shell speedup  median={statistics.median(sh_sp):.2f}x  "
          f"mean={statistics.mean(sh_sp):.2f}x  "
          f"range=[{min(sh_sp):.2f}x .. {max(sh_sp):.2f}x]")
    print()
    # Per-benchmark
    for bench in ["DSBench", "MLE-bench", "AIOB"]:
        sub = [r for r in rows if r["benchmark"] == bench]
        t = [r["total_speedup"] for r in sub if r["total_speedup"]]
        sh = [r["shell_speedup"] for r in sub if r["shell_speedup"]]
        print(f"  {bench:<10}  n={len(sub)}  "
              f"total median={statistics.median(t):.2f}x  "
              f"shell median={statistics.median(sh):.2f}x")
    # Per-model
    print()
    for model in ["haiku", "sonnet", "gemini"]:
        sub = [r for r in rows if r["model"] == model]
        t = [r["total_speedup"] for r in sub if r["total_speedup"]]
        sh = [r["shell_speedup"] for r in sub if r["shell_speedup"]]
        print(f"  {model:<7}  n={len(sub)}  "
              f"total median={statistics.median(t):.2f}x  "
              f"shell median={statistics.median(sh):.2f}x")
    # Wins
    print()
    tot_wins = sum(1 for r in rows if (r["total_speedup"] or 0) >= 1.20)
    tot_losses = sum(1 for r in rows if 0 < (r["total_speedup"] or 0) < 0.85)
    sh_wins = sum(1 for r in rows if (r["shell_speedup"] or 0) >= 1.20)
    sh_losses = sum(1 for r in rows if 0 < (r["shell_speedup"] or 0) < 0.85)
    print(f"  Total: {tot_wins}/27 wins (>=1.20x), "
          f"{tot_losses}/27 losses (<0.85x), "
          f"{27 - tot_wins - tot_losses} neutral")
    print(f"  Shell: {sh_wins}/27 wins (>=1.20x), "
          f"{sh_losses}/27 losses (<0.85x), "
          f"{27 - sh_wins - sh_losses} neutral")


if __name__ == "__main__":
    main()
