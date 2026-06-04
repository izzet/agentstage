"""Two-axis Amdahl analysis on existing speedup-campaign data.

Computes per cell:
  - observed session speedup (= cold/staged)
  - shell speedup r = cold_shell / staged_shell
  - shell share = cold_shell_elapsed / cold_elapsed
  - I/O share = (I/O time of cold shell) / cold_shell_elapsed
       (approximated as 1.0 - 1/r, i.e. all shell-time savings were I/O)
  - Amdahl-predicted session speedup:
        session_sp_ceiling = 1 / (1 - shell_share * io_share * (1 - 1/r))
  - efficiency = observed_session_sp / session_sp_ceiling
  - aggregate stats per (bench), per (model)
  - scatter of observed_session_sp vs predicted_session_sp_ceiling

This is the analysis that supports the "two-axis Amdahl framework"
paper framing: if observed tracks predicted (high R^2), the framework
explains when AgentStage wins and the lower per-cell numbers become
confirmations of the model rather than failures.

Usage:
  uv run python scripts/microbench/amdahl_analysis.py
Output:
  - prints per-cell table
  - writes outputs/amdahl_analysis.json + amdahl_analysis.csv
  - prints aggregate efficiency per bench + grand R^2
"""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Bench classification from task id prefix
def bench_of(task: str) -> str:
    if task.startswith("aiob_"):
        return "AIOB"
    if task.startswith("mle_") or "integrity" in task or task == "integrity_manifest":
        return "MLE"
    if task.startswith("kb_"):
        return "KB"
    return "DSB"


# Model normalization
MODEL_LABEL = {
    "claude-haiku-4-5": "Haiku",
    "claude-sonnet-4-5": "Sonnet",
    "gemini-2.5-flash": "Flash",
    "Qwen/Qwen3.6-27B": "Qwen",
}


# Paper-grade canonical campaigns (n=3 cells we cite in the paper).
# Each cell appears in only ONE of these to avoid double-counting.
PAPER_GRADE_CAMPAIGNS = {
    "AIOB": ["qwen_v2_aiob", "aiob_haiku_sonnet_fill", "flash_aiob_full",
             "aiob_full_27cell"],
    "MLE":  ["qwen_v2_mle", "mle_integrity_3model", "flash_mle_integrity"],
    "KB":   ["qwen_v2_kb", "kb_full_3model", "flash_kb_inventory"],
    "DSB":  ["qwen_v2_dsb", "dsbench_tabular_full", "flash_dsb_tabular"],
}


def collect_cells():
    """Walk outputs/replay/*/results.json and yield records that have
    BOTH cold and staged data. Returns list of dicts."""
    rows = []
    for f in sorted(glob.glob("outputs/replay/*/results.json")):
        camp = Path(f).parent.name
        if camp.startswith("_"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d:
            if "cold_elapsed_s" not in r or "staged_elapsed_s" not in r:
                continue
            csh = r.get("cold_shell_elapsed_s")
            ssh = r.get("staged_shell_elapsed_s")
            if csh is None or ssh is None or ssh <= 0:
                continue
            cold = r["cold_elapsed_s"]
            stg = r["staged_elapsed_s"]
            session_sp = cold / stg
            shell_sp = csh / ssh
            shell_share = csh / cold if cold > 0 else 0.0
            # I/O share within shell time — proxy: fraction of shell time
            # that disappears under prefetch. Assumes all the time saved is
            # I/O (which is the prefetch model's assumption).
            io_share = max(0.0, 1.0 - 1.0 / shell_sp) if shell_sp > 1 else 0.0
            # Two-axis Amdahl ceiling
            denom = 1.0 - shell_share * io_share * (1.0 - 1.0 / shell_sp) \
                    if shell_sp > 1 else 1.0
            ceiling = 1.0 / denom if denom > 0 else float("inf")
            efficiency = session_sp / ceiling if ceiling > 0 and not math.isinf(ceiling) else 0.0

            rows.append({
                "campaign": camp,
                "cell": r["cell"],
                "bench": bench_of(r["task"]),
                "task": r["task"],
                "model": MODEL_LABEL.get(r["model"], r["model"][:15]),
                "rep": r.get("rep"),
                "baseline_elapsed_s": r.get("baseline_elapsed_s"),
                "cold_elapsed_s": cold,
                "staged_elapsed_s": stg,
                "cold_shell_elapsed_s": csh,
                "staged_shell_elapsed_s": ssh,
                "session_sp": session_sp,
                "shell_sp": shell_sp,
                "shell_share": shell_share,
                "io_share_proxy": io_share,
                "amdahl_ceiling": ceiling,
                "efficiency_vs_ceiling": efficiency,
                "n_prefetched_staged": r.get("n_prefetched_staged", 0),
            })
    return rows


def main():
    rows = collect_cells()
    if not rows:
        print("No cells found.")
        return

    # Deduplicate: prefer the v2/canonical campaign per (bench,task,model,rep).
    # Skip exploratory cells with no clean rep numbering.
    seen = {}
    for r in rows:
        key = (r["bench"], r["task"], r["model"], r["rep"])
        # Prefer v2 campaigns or those starting with most-recent
        priority = (
            0 if r["campaign"].startswith("qwen_v2_") else
            1 if "fill" in r["campaign"] or "full" in r["campaign"] else
            2
        )
        if key not in seen or priority < seen[key][0]:
            seen[key] = (priority, r)
    canonical = [v[1] for v in seen.values()]

    # Print per-cell table
    canonical.sort(key=lambda r: (r["bench"], r["task"], r["model"], r["rep"] or 0))
    print(f"{'bench':5s} {'task':36s} {'model':6s} rep "
          f"{'session_sp':>9s} {'shell_sp':>8s} {'sh_share':>8s} "
          f"{'io_proxy':>8s} {'ceiling':>7s} {'eff':>6s}")
    for r in canonical:
        print(f"{r['bench']:5s} {r['task'][:36]:36s} {r['model']:6s} "
              f"r{r['rep']}  {r['session_sp']:>8.3f}× {r['shell_sp']:>7.2f}× "
              f"{r['shell_share']:>7.2%} {r['io_share_proxy']:>7.2%} "
              f"{r['amdahl_ceiling']:>6.2f}× {r['efficiency_vs_ceiling']:>5.1%}")

    # Aggregate per bench
    print("\n--- Per-bench aggregates ---")
    print(f"{'bench':5s} {'n':>3s} {'sess AM':>8s} {'shell AM':>9s} "
          f"{'ceil AM':>8s} {'eff AM':>7s}")
    for bench in ["AIOB", "MLE", "KB", "DSB"]:
        sub = [r for r in canonical if r["bench"] == bench]
        if not sub:
            continue
        sess = [r["session_sp"] for r in sub]
        shell = [r["shell_sp"] for r in sub]
        ceil = [r["amdahl_ceiling"] for r in sub
                if not math.isinf(r["amdahl_ceiling"])]
        eff = [r["efficiency_vs_ceiling"] for r in sub
               if not math.isinf(r["amdahl_ceiling"])]
        print(f"{bench:5s} {len(sub):>3d} "
              f"{statistics.mean(sess):>7.3f}× "
              f"{statistics.mean(shell):>8.2f}× "
              f"{statistics.mean(ceil):>7.2f}× "
              f"{statistics.mean(eff):>6.1%}")

    # Per model
    print("\n--- Per-model aggregates ---")
    print(f"{'model':6s} {'n':>3s} {'sess AM':>8s} {'shell AM':>9s} {'eff AM':>7s}")
    for model in ["Haiku", "Sonnet", "Flash", "Qwen"]:
        sub = [r for r in canonical if r["model"] == model]
        if not sub:
            continue
        sess = [r["session_sp"] for r in sub]
        shell = [r["shell_sp"] for r in sub]
        eff = [r["efficiency_vs_ceiling"] for r in sub
               if not math.isinf(r["amdahl_ceiling"])]
        print(f"{model:6s} {len(sub):>3d} "
              f"{statistics.mean(sess):>7.3f}× "
              f"{statistics.mean(shell):>8.2f}× "
              f"{statistics.mean(eff):>6.1%}")

    # Correlation: observed_session_sp vs amdahl_ceiling
    pairs = [(r["amdahl_ceiling"], r["session_sp"]) for r in canonical
             if not math.isinf(r["amdahl_ceiling"])]
    if len(pairs) >= 3:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
        den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
        if den_x > 0 and den_y > 0:
            pearson_r = num / (den_x * den_y)
            r_squared = pearson_r ** 2
            print(f"\n--- Predicted (Amdahl ceiling) vs observed correlation ---")
            print(f"  n={len(pairs)}  Pearson r={pearson_r:.3f}  R^2={r_squared:.3f}")
            print(f"  observed AM: {statistics.mean(ys):.3f}×")
            print(f"  ceiling  AM: {statistics.mean(xs):.3f}×")
            print(f"  observed/ceiling avg efficiency: "
                  f"{statistics.mean([y/x for x,y in pairs if x>0]):.1%}")

    # Save
    out_json = REPO / "outputs" / "amdahl_analysis.json"
    out_csv = REPO / "outputs" / "amdahl_analysis.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"cells": canonical}, indent=2, default=str))
    if canonical:
        keys = list(canonical[0].keys())
        with out_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(canonical)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
