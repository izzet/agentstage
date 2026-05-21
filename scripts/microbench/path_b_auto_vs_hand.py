"""Path B — hand vs auto-generated rules comparison (E-019).

Replays a captured multi-turn corpus through TWO detectors:
  A. Hand-tuned ruleset (the existing detector.rules.get_ruleset(workload))
  B. Auto-generated ruleset (AutoRuleGenerator over task_instruction +
     workspace_prior_keys)

Reports per-rule activations and per-tier subset metrics against the
workload's static GT. Establishes whether the L3 genericity claim holds:
auto rules within 10% of hand on subset recall.

Usage:
    python scripts/microbench/path_b_auto_vs_hand.py \\
        --corpus outputs/multi_turn/<run> \\
        --workload aiob_107_s3 \\
        --out outputs/multi_turn/<run>/auto_vs_hand.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

# Put the sibling script's directory on the path so we can import its helpers
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.engine import StreamBlock  # noqa: E402  (used via reconstruct)
from agentstage.detector.rules import RuleSet, get_ruleset  # noqa: E402
from agentstage.detector.session import SessionDetector  # noqa: E402
from agentstage.runners.path_b_multiturn import _resolve_logical_to_physical  # noqa: E402
from agentstage.workloads.aiob import (  # noqa: E402
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)
from path_b_subset_replay import (  # noqa: E402
    collect_agent_opens,
    reconstruct_blocks_per_turn,
)


def _tier_of(n: int) -> int:
    if n <= 10:
        return 1
    if n <= 200:
        return 2
    return 3


def replay(ruleset: RuleSet, blocks_chunks, prior):
    sd = SessionDetector(prior=prior, ruleset=ruleset)
    for _turn, blocks in blocks_chunks:
        assistant = [b for b in blocks if b.type in ("thinking", "text")]
        tool_results = [b for b in blocks if b.type == "tool_result"]
        if assistant:
            sd.feed_turn(assistant)
        if tool_results:
            sd.feed_tool_results(tool_results)
    return sd.cumulative_detection()


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

    loaders = {
        "aiob_107": load_aiob_107,
        "aiob_107_s3": load_aiob_107_s3,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    rules_key = args.workload.replace("_s3", "")
    hand_rs = get_ruleset(rules_key)
    auto_rs = AutoRuleGenerator(
        workload_id=workload.task_id,
        task_instruction=workload.task.task_inst,
        workspace_prior_keys=tuple(workload.workspace_prior.keys()),
    ).generate()

    prior = workload.workspace_prior
    gt_static_logical = set(workload.ground_truth_full)

    def to_phys(p: str) -> str:
        return _resolve_logical_to_physical(
            p, workload.prefix_map, cold_root=args.cold_root)

    gt_static_phys = {to_phys(p) for p in gt_static_logical}
    gt_actual_phys = collect_agent_opens(
        args.corpus, workload.prefix_map, args.cold_root)

    blocks_chunks = reconstruct_blocks_per_turn(args.corpus / "turns")

    hand_pred = replay(hand_rs, blocks_chunks, prior)
    auto_pred = replay(auto_rs, blocks_chunks, prior)

    def summarize(pred, label: str) -> dict:
        t1 = {to_phys(p) for p in pred.tier_1.detected_files}
        t2 = {to_phys(p) for p in pred.tier_2.detected_files}
        t3 = {to_phys(p) for p in pred.tier_3.detected_files}

        def metrics(predicted: set[str], gt: set[str]) -> dict:
            if not predicted and not gt:
                return {"precision": None, "recall": None,
                        "jaccard": None, "hits": 0}
            hits = predicted & gt
            union = predicted | gt
            return {
                "predicted_size": len(predicted),
                "gt_size": len(gt),
                "hits": len(hits),
                "precision": (len(hits) / len(predicted)) if predicted else None,
                "recall": (len(hits) / len(gt)) if gt else None,
                "jaccard": (len(hits) / len(union)) if union else None,
            }

        out = {
            "label": label,
            "n_rules_in_set": len(pred.activations) + sum(
                1 for _ in []) ,  # placeholder
            "fired_rules": [
                {"name": a.rule_name, "source": a.source, "turn": a.turn,
                 "subset_size": len(set(a.detected_files))}
                for a in pred.activations
            ],
            "tier_1_vs_static": metrics(t1, gt_static_phys),
            "tier_2_vs_static": metrics(t2, gt_static_phys),
            "tier_3_vs_static": metrics(t3, gt_static_phys),
            "tier_1_vs_actual": metrics(t1, gt_actual_phys),
            "tier_3_vs_actual": metrics(t3, gt_actual_phys),
            "tier_sizes": {
                "tier_1": len(t1),
                "tier_2": len(t2),
                "tier_3": len(t3),
            },
        }
        return out

    summary = {
        "corpus": str(args.corpus),
        "workload": args.workload,
        "rules_in_hand": len(hand_rs.rules),
        "rules_in_auto": len(auto_rs.rules),
        "gt_static_size": len(gt_static_phys),
        "gt_actual_size": len(gt_actual_phys),
        "hand": summarize(hand_pred, "hand"),
        "auto": summarize(auto_pred, "auto"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    # Pretty-print comparison
    print(f"Corpus: {args.corpus}")
    print(f"  GT_static: {len(gt_static_phys)} files,  agent opened: {len(gt_actual_phys)}")
    print(f"  Rules:    hand={len(hand_rs.rules)},  auto={len(auto_rs.rules)}")
    print()
    print(f"  {'':<10}  {'hand':<35}  {'auto':<35}")
    print(f"  {'':<10}  {'fired rules':<35}  {'fired rules':<35}")
    hand_names = [a.rule_name for a in hand_pred.activations]
    auto_names = [a.rule_name for a in auto_pred.activations]
    print(f"  {'fired':<10}  {str(hand_names):<35}  {str(auto_names):<35}")
    print()
    for tier_label, key in [("T1", "tier_1_vs_static"),
                             ("T2", "tier_2_vs_static"),
                             ("T3", "tier_3_vs_static")]:
        h = summary["hand"][key]
        a = summary["auto"][key]
        def fmt(m):
            if m["precision"] is None:
                return "    -    -    -"
            return f"  {m.get('predicted_size', 0):>5} {m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}%"
        print(f"  {tier_label} vs static GT: hand={fmt(h)}   auto={fmt(a)}")

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
