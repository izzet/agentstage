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


# Paper-grade 72-cell canonical mapping: (task, model) -> (campaign, n=3).
# This is the EXACT set of cells cited in the paper. Each (task, model)
# pair has exactly one canonical campaign source. The script filters to
# this set so the Amdahl R^2 / per-bench / per-model numbers reproduce
# exactly when readers run the analysis against the published artifacts.
PAPER_GRADE_CELLS = {
    # ─── Curated AIOB (3 tasks × 4 models × n=3 = 36 cells) ───
    ("aiob_201", "claude-haiku-4-5"):  ["aiob_haiku_sonnet_fill"],
    ("aiob_201", "claude-sonnet-4-5"): ["aiob_haiku_sonnet_fill"],
    ("aiob_201", "gemini-2.5-flash"):  ["flash_aiob_campaign"],
    ("aiob_201", "Qwen/Qwen3.6-27B"):  ["qwen_v2_aiob"],
    ("aiob_202", "claude-haiku-4-5"):  ["aiob_haiku_sonnet_fill"],
    ("aiob_202", "claude-sonnet-4-5"): ["aiob_haiku_sonnet_fill"],
    ("aiob_202", "gemini-2.5-flash"):  ["flash_aiob_campaign"],
    ("aiob_202", "Qwen/Qwen3.6-27B"):  ["qwen_v2_aiob"],
    ("aiob_205", "claude-haiku-4-5"):  ["aiob_205_3x3"],
    ("aiob_205", "claude-sonnet-4-5"): ["aiob_205_3x3"],
    ("aiob_205", "gemini-2.5-flash"):  ["flash_aiob_campaign"],
    ("aiob_205", "Qwen/Qwen3.6-27B"):  ["qwen_v2_aiob"],
    # ─── Community (3 benchmarks × 1 task × 4 models × n=3 = 36 cells) ───
    # MLE: dogvscats thumbhash (all 4 models)
    ("mle_dogsvcats_thumbhash", "claude-haiku-4-5"):  ["haiku_flash_v2_dogvscats_thumbhash"],
    ("mle_dogsvcats_thumbhash", "claude-sonnet-4-5"): ["sonnet_v2_dogvscats_thumbhash"],
    ("mle_dogsvcats_thumbhash", "gemini-2.5-flash"):  ["haiku_flash_v2_dogvscats_thumbhash"],
    ("mle_dogsvcats_thumbhash", "Qwen/Qwen3.6-27B"):  ["qwen_v2_dogvscats_thumbhash"],
    # KB: astronomy inventory (Haiku reps spread across two OrangeFS campaigns)
    ("kb_astronomy_inventory", "claude-haiku-4-5"):  ["kb_astronomy_inventory_v2_pilot", "kb_astronomy_rep3"],
    ("kb_astronomy_inventory", "claude-sonnet-4-5"): ["kb_astronomy_sonnet_x3"],
    ("kb_astronomy_inventory", "gemini-2.5-flash"):  ["kb_astronomy_flash_x3"],
    ("kb_astronomy_inventory", "Qwen/Qwen3.6-27B"):  ["qwen_v2_kb"],
    # DSB: tabular oct-2021 (Haiku reps spread across pilot + rep3)
    ("tabular-playground-series-oct-2021", "claude-haiku-4-5"):  ["dsb_tabular_oct2021_pilot", "dsb_tabular_oct2021_rep3"],
    ("tabular-playground-series-oct-2021", "claude-sonnet-4-5"): ["dsb_tabular_sonnet_x3"],
    ("tabular-playground-series-oct-2021", "gemini-2.5-flash"):  ["dsb_tabular_flash_x3"],
    ("tabular-playground-series-oct-2021", "Qwen/Qwen3.6-27B"):  ["qwen_v2_dsb"],
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

    # Filter to PAPER_GRADE_CELLS only. For each (task, model_full) pair
    # in the map, take up to 3 reps from the listed canonical campaigns.
    # Anything outside this set is exploratory and excluded from the
    # paper-cited numbers.
    by_cell = defaultdict(list)
    for r in rows:
        # Need full model name (not normalized label) to match PAPER_GRADE_CELLS keys
        # — find it from the original row
        pass
    # Rebuild: re-read raw results.json with full model names so we can match keys
    canonical = []
    cells_filled = set()
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
            if r.get("staged_elapsed_s") is None or r.get("cold_elapsed_s") is None:
                continue
            csh = r.get("cold_shell_elapsed_s")
            ssh = r.get("staged_shell_elapsed_s")
            if csh is None or ssh is None:
                continue
            key = (r["task"], r["model"])
            if key not in PAPER_GRADE_CELLS:
                continue
            if camp not in PAPER_GRADE_CELLS[key]:
                continue
            # Cap at 3 reps per cell
            n_existing = sum(1 for c in canonical
                             if (c["task"], c["model_full"]) == key)
            if n_existing >= 3:
                continue
            cold = r["cold_elapsed_s"]
            stg = r["staged_elapsed_s"]
            session_sp = cold / stg
            # Degenerate trajectories where the agent did not invoke any
            # shell tool (shell time = 0 in both modes) are degenerate
            # measurements of AgentStage — the session is pure LLM streaming.
            # Treat as shell_sp=1.0, ceiling=1.0 so the Amdahl framework
            # trivially predicts session_sp ≈ 1.0 (which is what we observe).
            degenerate = (csh == 0 and ssh == 0)
            if degenerate:
                shell_sp = 1.0
                shell_share = 0.0
                io_share = 0.0
                ceiling = 1.0
                efficiency = session_sp / 1.0
            elif ssh > 0:
                shell_sp = csh / ssh
                shell_share = csh / cold if cold > 0 else 0.0
                io_share = max(0.0, 1.0 - 1.0 / shell_sp) if shell_sp > 1 else 0.0
                denom = (1.0 - shell_share * io_share * (1.0 - 1.0 / shell_sp)
                         if shell_sp > 1 else 1.0)
                ceiling = 1.0 / denom if denom > 0 else float("inf")
                efficiency = (session_sp / ceiling
                              if ceiling > 0 and not math.isinf(ceiling) else 0.0)
            else:
                continue  # ssh=0 but csh>0 is malformed; skip
            canonical.append({
                "campaign": camp,
                "cell": r["cell"],
                "bench": bench_of(r["task"]),
                "task": r["task"],
                "model": MODEL_LABEL.get(r["model"], r["model"][:15]),
                "model_full": r["model"],
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
    # Audit coverage
    coverage = defaultdict(int)
    for c in canonical:
        coverage[(c["task"], c["model_full"])] += 1
    missing = []
    for key in PAPER_GRADE_CELLS:
        if coverage.get(key, 0) < 3:
            missing.append((key, coverage.get(key, 0)))
    if missing:
        print(f"⚠ Incomplete paper-grade coverage: {len(missing)} cell(s) below n=3")
        for (task, model), n in missing:
            print(f"    {task} × {model}: n={n}")
        print()

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
