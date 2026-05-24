"""E-034 — H7 leave-one-out rule sufficiency.

For each captured multi-turn corpus, replays it through the SessionDetector
multiple times, each time DROPPING one auto-generated rule. Compares the
resulting cumulative subset recall against the full-set baseline.

A rule is "load-bearing" if removing it drops recall by ≥ 10 percentage
points. A ruleset is "robust" if no single rule's removal drops recall
below 80%.

Output goes both to a JSON artifact and to a pytest-consumable structure
(per-corpus per-rule recall deltas).

Usage:
    python scripts/microbench/path_b_h7_loo.py \\
        --corpus outputs/multi_turn/<run> \\
        --workload aiob_107_s3 \\
        --out <corpus>/h7_loo.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import sys

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.rules import RuleSet  # noqa: E402
from agentstage.detector.session import SessionDetector  # noqa: E402
from agentstage.runners.path_b_multiturn import _resolve_logical_to_physical  # noqa: E402
from agentstage.workloads.aiob import (  # noqa: E402
    load_aiob_107, load_aiob_107_s3, load_aiob_110,
)
from path_b_subset_replay import (  # noqa: E402
    collect_agent_opens, reconstruct_blocks_per_turn,
)


def replay_with_ruleset(rs: RuleSet, blocks_chunks, prior) -> set[str]:
    """Return the cumulative LOGICAL path set the detector would prefetch."""
    sd = SessionDetector(prior=prior, ruleset=rs)
    for _turn, blocks in blocks_chunks:
        assistant = [b for b in blocks if b.type in ("thinking", "text")]
        tool_results = [b for b in blocks if b.type == "tool_result"]
        if assistant:
            sd.feed_turn(assistant)
        if tool_results:
            sd.feed_tool_results(tool_results)
    pred = sd.cumulative_detection()
    return set(pred.tier_1.detected_files + pred.tier_2.detected_files
               + pred.tier_3.detected_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workload",
                        choices=["aiob_107", "aiob_107_s3", "aiob_110"],
                        default="aiob_107_s3")
    parser.add_argument("--cold-root",
                        default="/tmp/s3-noaa-goes16/ABI-L2-CMIPC")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    loaders = {
        "aiob_107": load_aiob_107,
        "aiob_107_s3": load_aiob_107_s3,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    prior = workload.workspace_prior

    # Build the AUTO ruleset (so this test exercises the auto-rules path)
    gen = AutoRuleGenerator(
        workload_id=workload.task_id,
        task_instruction=workload.task.task_inst,
        workspace_prior_keys=tuple(prior.keys()),
    )
    full_rs = gen.generate()

    # Ground truth: agent's actual opens (logical, since prior is logical)
    def to_phys(logical: str) -> str:
        return _resolve_logical_to_physical(
            logical, workload.prefix_map, cold_root=args.cold_root)

    gt_actual_phys = collect_agent_opens(
        args.corpus, workload.prefix_map, args.cold_root)

    blocks_chunks = reconstruct_blocks_per_turn(args.corpus / "turns")

    # Baseline: full ruleset
    full_logical = replay_with_ruleset(full_rs, blocks_chunks, prior)
    full_phys = {to_phys(p) for p in full_logical}
    if not gt_actual_phys:
        # The agent didn't open any file in this capture — H7 can't be
        # measured here (no recall denominator). Record a skip marker.
        result = {
            "experiment": "E-034",
            "corpus": str(args.corpus),
            "workload": args.workload,
            "n_agent_opens": 0,
            "skipped_reason": "no agent opens in capture (cannot compute recall)",
        }
        args.out.write_text(json.dumps(result, indent=2))
        print(f"  [skip] no agent opens in {args.corpus.name}")
        return 0

    full_recall = (
        len(full_phys & gt_actual_phys) / len(gt_actual_phys)
    )

    # LOO: drop each rule in turn
    loo_rows: list[dict] = []
    for i, dropped in enumerate(full_rs.rules):
        reduced = RuleSet(
            workload=full_rs.workload,
            rules=tuple(r for j, r in enumerate(full_rs.rules) if j != i),
        )
        reduced_logical = replay_with_ruleset(reduced, blocks_chunks, prior)
        reduced_phys = {to_phys(p) for p in reduced_logical}
        recall = (
            len(reduced_phys & gt_actual_phys) / len(gt_actual_phys)
        )
        delta = full_recall - recall
        loo_rows.append({
            "dropped": dropped.name,
            "recall": round(recall, 4),
            "delta_from_full": round(delta, 4),
            "load_bearing": delta >= 0.10,  # 10pp threshold
        })

    n_load_bearing = sum(1 for r in loo_rows if r["load_bearing"])
    min_recall = min(r["recall"] for r in loo_rows)
    robust = min_recall >= 0.80

    result = {
        "experiment": "E-034",
        "corpus": str(args.corpus),
        "workload": args.workload,
        "n_rules": len(full_rs.rules),
        "n_agent_opens": len(gt_actual_phys),
        "full_ruleset_recall": round(full_recall, 4),
        "min_recall_after_drop": round(min_recall, 4),
        "n_load_bearing_rules": n_load_bearing,
        "ruleset_robust": robust,
        "per_rule": loo_rows,
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"E-034 H7 leave-one-out — {args.corpus.name}")
    print(f"  workload={args.workload}  rules={len(full_rs.rules)}  "
          f"agent_opens={len(gt_actual_phys)}")
    print(f"  full-ruleset recall:  {full_recall:.3f}")
    print(f"  min recall on drop:   {min_recall:.3f}")
    print(f"  load-bearing rules:   {n_load_bearing}/{len(full_rs.rules)}")
    print(f"  robust (min ≥ 0.80):  {robust}")
    print()
    print("  Top 5 most-load-bearing rules (largest recall delta when dropped):")
    for r in sorted(loo_rows, key=lambda x: -x["delta_from_full"])[:5]:
        print(f"    {r['dropped']:<28s}  recall={r['recall']:.3f}  "
              f"Δ={r['delta_from_full']:+.3f}")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
