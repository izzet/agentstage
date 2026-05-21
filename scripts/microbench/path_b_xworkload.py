"""Cross-workload auto-rules vs hand-rules check (E-022).

Replays single-turn PoC `stream.jsonl` captures against:
  A. hand-tuned ruleset (detector.rules.get_ruleset(workload))
  B. auto-generated ruleset (AutoRuleGenerator)

For each workload × model × seed in the PoC corpus, reports tier-3
recall and precision against the workload's static GT.

Defends the L3 genericity claim across multiple workloads, not just
aiob_107_s3.

Usage:
    python scripts/microbench/path_b_xworkload.py \\
        --workloads aiob_104,aiob_110 \\
        --poc-dir outputs/poc \\
        --out outputs/x_workload_replay.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentstage.detector.auto_rules import AutoRuleGenerator
from agentstage.detector.engine import parse_anthropic_stream, parse_stream, run_detector
from agentstage.detector.rules import get_ruleset
from agentstage.workloads.aiob import (
    load_aiob_101,
    load_aiob_104,
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)

LOADERS = {
    "aiob_101": load_aiob_101,
    "aiob_104": load_aiob_104,
    "aiob_107": load_aiob_107,
    "aiob_107_s3": load_aiob_107_s3,
    "aiob_110": load_aiob_110,
}


def find_poc_captures(poc_dir: Path, workload: str) -> list[Path]:
    """Find PoC directories matching the workload."""
    out: list[Path] = []
    for d in poc_dir.glob(f"*{workload}*"):
        if (d / "stream.jsonl").is_file():
            out.append(d)
    return sorted(out)


def metrics(predicted: set[str], gt: set[str]) -> dict:
    if not predicted and not gt:
        return {"precision": None, "recall": None, "jaccard": None,
                "predicted_size": 0, "gt_size": 0, "hits": 0}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", default="aiob_104,aiob_110",
                        help="comma-separated workload ids")
    parser.add_argument("--poc-dir", type=Path, default=Path("outputs/poc"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/x_workload_replay.json"))
    args = parser.parse_args()

    workload_ids = [w.strip() for w in args.workloads.split(",") if w.strip()]
    results: dict = {}

    for wl_id in workload_ids:
        loader = LOADERS.get(wl_id)
        if loader is None:
            print(f"WARN: no loader for {wl_id}", file=stderr); continue
        wl = loader()
        prior = wl.workspace_prior
        gt = set(wl.ground_truth_full)
        hand = get_ruleset(wl_id.replace("_s3", ""))
        auto = AutoRuleGenerator(
            workload_id=wl.task_id,
            task_instruction=wl.task.task_inst,
            workspace_prior_keys=tuple(prior.keys()),
        ).generate()

        captures = find_poc_captures(args.poc_dir, wl_id)
        print(f"=== {wl_id} === {len(captures)} PoC captures found")
        per_capture: list[dict] = []
        hand_recalls: list[float] = []
        auto_recalls: list[float] = []
        for cap in captures:
            try:
                blocks = parse_stream(cap / "stream.jsonl")
            except Exception as e:
                print(f"  skip {cap.name}: {e}")
                continue
            hand_pred = run_detector(blocks, prior, hand)
            auto_pred = run_detector(blocks, prior, auto)
            h_t3 = set(hand_pred.tier_3.detected_files)
            a_t3 = set(auto_pred.tier_3.detected_files)
            h_m = metrics(h_t3, gt)
            a_m = metrics(a_t3, gt)
            per_capture.append({
                "capture": cap.name,
                "n_blocks": len(blocks),
                "hand_fired_rules": [a.rule_name for a in hand_pred.activations],
                "auto_fired_rules": [a.rule_name for a in auto_pred.activations],
                "hand_t3_vs_gt": h_m,
                "auto_t3_vs_gt": a_m,
            })
            if h_m["recall"] is not None: hand_recalls.append(h_m["recall"])
            if a_m["recall"] is not None: auto_recalls.append(a_m["recall"])
            print(f"  {cap.name[:65]:<65} hand_recall={h_m['recall']*100 if h_m['recall'] is not None else 0:>5.1f}%  "
                  f"auto_recall={a_m['recall']*100 if a_m['recall'] is not None else 0:>5.1f}%")
        results[wl_id] = {
            "n_captures": len(per_capture),
            "rules_in_hand": len(hand.rules),
            "rules_in_auto": len(auto.rules),
            "mean_hand_recall": sum(hand_recalls) / len(hand_recalls) if hand_recalls else None,
            "mean_auto_recall": sum(auto_recalls) / len(auto_recalls) if auto_recalls else None,
            "min_hand_recall": min(hand_recalls) if hand_recalls else None,
            "max_hand_recall": max(hand_recalls) if hand_recalls else None,
            "min_auto_recall": min(auto_recalls) if auto_recalls else None,
            "max_auto_recall": max(auto_recalls) if auto_recalls else None,
            "per_capture": per_capture,
        }
        if hand_recalls and auto_recalls:
            hm = results[wl_id]["mean_hand_recall"]
            am = results[wl_id]["mean_auto_recall"]
            print(f"  >> {wl_id}: hand mean={hm*100:.1f}%   auto mean={am*100:.1f}%   "
                  f"delta={(am-hm)*100:+.1f}%")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    stderr = sys.stderr
    raise SystemExit(main())
