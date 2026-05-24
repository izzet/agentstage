"""E-032 — AutoRuleGenerator runtime cost microbench.

Measures the wall-clock cost of generating an auto-rule set from
(task_instruction, workspace_prior_keys). Backs the workflow-hook claim:
auto-rule generation is fast enough to run as a pre-task hook with no
perceptible overhead, OR after the agent's first `list_dir` tool_result.

Method:
  - For each workload, time AutoRuleGenerator(...).generate() N times
    (default N=1000) and report p50/p95/max.
  - Also break out the cost contribution per workspace-prior-key count
    so we can extrapolate to larger workloads.

Run:
    python scripts/microbench/auto_rules_cost.py \
        --out outputs/microbench/auto_rules_cost.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agentstage.detector.auto_rules import AutoRuleGenerator
from agentstage.workloads.aiob import (
    load_aiob_101,
    load_aiob_104,
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)


def bench_one(wl, n: int) -> dict:
    """Time AutoRuleGenerator.generate() n times on a workload."""
    task_instruction = wl.task.task_inst
    keys = tuple(wl.workspace_prior.keys())

    # Warm-up — compile any module-level regexes once before timing.
    AutoRuleGenerator(wl.task_id, task_instruction, keys).generate()

    samples_us: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        rs = AutoRuleGenerator(wl.task_id, task_instruction, keys).generate()
        elapsed_us = (time.perf_counter_ns() - t0) / 1_000.0
        samples_us.append(elapsed_us)

    return {
        "workload": wl.task_id,
        "n_prior_keys": len(keys),
        "n_prior_paths": len(wl.all_workspace_paths),
        "n_rules_generated": len(rs.rules),
        "n_samples": n,
        "p50_us": round(statistics.median(samples_us), 2),
        "p95_us": round(statistics.quantiles(samples_us, n=20)[-1], 2),
        "mean_us": round(statistics.mean(samples_us), 2),
        "min_us": round(min(samples_us), 2),
        "max_us": round(max(samples_us), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-n", "--samples", type=int, default=1000)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    workloads = [
        load_aiob_101(),
        load_aiob_104(),
        load_aiob_107(),
        load_aiob_107_s3(),
        load_aiob_110(),
    ]
    results = [bench_one(wl, args.samples) for wl in workloads]

    # Sanity-cross-comparison numbers
    all_p50 = [r["p50_us"] for r in results]
    all_p95 = [r["p95_us"] for r in results]
    summary = {
        "experiment": "E-032",
        "n_samples_per_workload": args.samples,
        "n_workloads": len(results),
        "p50_us_across_workloads": round(statistics.mean(all_p50), 2),
        "p95_us_across_workloads": round(max(all_p95), 2),
        "max_us_observed": round(max(r["max_us"] for r in results), 2),
        "per_workload": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))

    print("E-032 AutoRuleGenerator microbench")
    print(f"  samples per workload: {args.samples}")
    print()
    print(f"  {'workload':<14} {'keys':>5} {'rules':>6} {'p50':>8} {'p95':>8} {'max':>8}")
    for r in results:
        print(f"  {r['workload']:<14} {r['n_prior_keys']:>5} "
              f"{r['n_rules_generated']:>6} "
              f"{r['p50_us']:>7.1f}µs "
              f"{r['p95_us']:>7.1f}µs "
              f"{r['max_us']:>7.1f}µs")
    print()
    print(f"  cross-workload p50:  {summary['p50_us_across_workloads']:.1f} µs")
    print(f"  cross-workload p95:  {summary['p95_us_across_workloads']:.1f} µs")
    print(f"  max ever observed:   {summary['max_us_observed']:.1f} µs")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
