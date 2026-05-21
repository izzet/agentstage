"""Path B subset-prefetch replay.

For each captured multi-turn corpus, fires ALL detected rules (not just
tier-1, as our live dispatch policy does) and reports per-rule subset
precision/recall against TWO ground truths:

  - GT_actual:   files the agent actually opened in this run
  - GT_static:   the workload's `ground_truth_full` (what a complete
                 task execution would access — from the workload spec)

This answers the framing the paper actually needs:

  "When the detector fires the band_08 rule, the 2014 files it identifies
   are a SUBSET of the larger workspace (6042 files). How accurate is
   that subset, and what's the precision/recall tradeoff at each
   detection tier?"

The E-016 metric measured single-file precision against the agent's
actual first-opened file. This script measures subset-level metrics:
the right framing because the detector's primary value is to identify
which class of files the agent is targeting, not to guess the next
individual file.

Usage:
    python scripts/microbench/path_b_subset_replay.py \\
        --corpus outputs/multi_turn/<run> \\
        --workload aiob_107_s3 \\
        --out <corpus>/subset_replay.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentstage.detector.engine import StreamBlock, run_detector
from agentstage.detector.rules import get_ruleset
from agentstage.detector.session import SessionDetector
from agentstage.workloads.aiob import (
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)
from agentstage.runners.path_b_multiturn import _resolve_logical_to_physical


def reconstruct_blocks_per_turn(turns_dir: Path) -> list[tuple[int, list[StreamBlock]]]:
    """Walk turns_dir, return [(turn, [blocks]), ...] in chronological order.
    Mirrors path_b_replay.py reconstruction logic but emits per-turn
    bundles so we can feed the SessionDetector turn-by-turn."""
    out: list[tuple[int, list[StreamBlock]]] = []
    base_t = 0.0
    for tdir in sorted(turns_dir.glob("turn_*")):
        turn_num = int(tdir.name.split("_")[1])
        blocks: list[StreamBlock] = []
        thinking_by_idx: dict[int, list[str]] = {}
        text_by_idx: dict[int, list[str]] = {}
        block_kind: dict[int, str] = {}
        first_chunk: dict[int, float] = {}
        last_chunk: dict[int, float] = {}
        with (tdir / "stream.jsonl").open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = d.get("type")
                idx = d.get("block_idx", -1)
                if etype == "content_block_start":
                    block_kind[idx] = d.get("block_type") or ""
                    first_chunk[idx] = d.get("t_ms", 0.0)
                elif etype == "content_block_delta":
                    last_chunk[idx] = d.get("t_ms", 0.0)
                    dtype = d.get("delta_type")
                    if dtype == "thinking_delta":
                        thinking_by_idx.setdefault(idx, []).append(d.get("chunk", ""))
                    elif dtype == "text_delta":
                        text_by_idx.setdefault(idx, []).append(d.get("chunk", ""))
        for idx in sorted(block_kind.keys()):
            kind = block_kind[idx]
            tf = base_t + first_chunk.get(idx, 0.0)
            ts = base_t + last_chunk.get(idx, tf)
            if kind == "thinking":
                t = "".join(thinking_by_idx.get(idx, []))
                if t:
                    blocks.append(StreamBlock(
                        type="thinking", t_first=tf, t_stop=ts,
                        text=t, chunks=1, turn=turn_num,
                    ))
            elif kind == "text":
                t = "".join(text_by_idx.get(idx, []))
                if t:
                    blocks.append(StreamBlock(
                        type="text", t_first=tf, t_stop=ts,
                        text=t, chunks=1, turn=turn_num,
                    ))
        # tool_result blocks (stamped for the NEXT turn)
        tr_path = tdir / "tool_result.jsonl"
        tr_blocks: list[StreamBlock] = []
        if tr_path.exists() and tr_path.stat().st_size > 0:
            for line in tr_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = d.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                tr_blocks.append(StreamBlock(
                    type="tool_result",
                    t_first=base_t + 1000.0 * (turn_num + 0.5),
                    t_stop=base_t + 1000.0 * (turn_num + 0.5),
                    text=str(content), chunks=1, turn=turn_num + 1,
                ))
        out.append((turn_num, blocks))
        if tr_blocks:
            out.append((turn_num, tr_blocks))  # tool_results paired with same turn
        base_t += 10_000.0
    return out


def collect_agent_opens(
    corpus: Path,
    prefix_map: tuple[tuple[str, str], ...],
    cold_root: str,
) -> set[str]:
    """Physical paths the agent opened (open_file / read_file)."""
    out: set[str] = set()
    for turn_dir in sorted((corpus / "turns").glob("turn_*")):
        tu_path = turn_dir / "tool_use.jsonl"
        if not tu_path.exists():
            continue
        for line in tu_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("name") not in ("open_file", "read_file"):
                continue
            logical = (d.get("parsed_input") or {}).get("path", "")
            if not logical:
                continue
            phys = _resolve_logical_to_physical(
                logical, prefix_map, cold_root=cold_root)
            if Path(phys).is_file():
                out.add(phys)
    return out


def _tier_of(n: int) -> int:
    if n <= 10:
        return 1
    if n <= 200:
        return 2
    return 3


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
    ruleset = get_ruleset(rules_key)
    prior = workload.workspace_prior
    gt_static = set(workload.ground_truth_full)

    # Agent's actual opens (translated to PHYSICAL paths for this run)
    gt_actual_phys = collect_agent_opens(
        args.corpus, workload.prefix_map, args.cold_root)
    # Translate LOGICAL paths in prior to PHYSICAL to compare with gt_actual
    def to_phys(logical: str) -> str:
        return _resolve_logical_to_physical(
            logical, workload.prefix_map, cold_root=args.cold_root)

    # GT_static is in LOGICAL paths from the workload prior — translate
    gt_static_phys = {to_phys(p) for p in gt_static}

    # Replay the corpus through SessionDetector to get the full activation set
    blocks_chunks = reconstruct_blocks_per_turn(args.corpus / "turns")
    sd = SessionDetector(prior=prior, ruleset=ruleset)
    for _turn, blocks in blocks_chunks:
        assistant = [b for b in blocks if b.type in ("thinking", "text")]
        tool_results = [b for b in blocks if b.type == "tool_result"]
        if assistant:
            sd.feed_turn(assistant)
        if tool_results:
            sd.feed_tool_results(tool_results)
    final_pred = sd.cumulative_detection()

    # Per-rule analysis
    rule_detail: list[dict] = []
    union_logical: set[str] = set()
    for act in final_pred.activations:
        subset_logical = set(act.detected_files)
        subset_phys = {to_phys(p) for p in subset_logical}
        union_logical.update(subset_logical)

        # vs gt_actual (this run's opens) — phys
        hits_actual = subset_phys & gt_actual_phys
        prec_a = (len(hits_actual) / len(subset_phys)) if subset_phys else None
        rec_a = (len(hits_actual) / len(gt_actual_phys)) if gt_actual_phys else None

        # vs gt_static (workload's eventual GT) — phys
        hits_static = subset_phys & gt_static_phys
        prec_s = (len(hits_static) / len(subset_phys)) if subset_phys else None
        rec_s = (len(hits_static) / len(gt_static_phys)) if gt_static_phys else None

        rule_detail.append({
            "rule": act.rule_name,
            "source": act.source,
            "turn": act.turn,
            "tier": _tier_of(len(subset_logical)),
            "subset_size": len(subset_logical),
            "vs_actual": {
                "hits": len(hits_actual),
                "precision": prec_a,
                "recall": rec_a,
            },
            "vs_static_gt": {
                "hits": len(hits_static),
                "precision": prec_s,
                "recall": rec_s,
            },
        })

    # Tiered union (what the stager WOULD prefetch if all tiers were dispatched)
    tier1_phys = {to_phys(p) for p in final_pred.tier_1.detected_files}
    tier2_phys = {to_phys(p) for p in final_pred.tier_2.detected_files}
    tier3_phys = {to_phys(p) for p in final_pred.tier_3.detected_files}

    def metrics(predicted: set[str], gt: set[str]) -> dict:
        if not predicted and not gt:
            return {"precision": None, "recall": None, "jaccard": None,
                    "hits": 0, "predicted_size": 0, "gt_size": 0}
        hits = predicted & gt
        prec = (len(hits) / len(predicted)) if predicted else None
        rec = (len(hits) / len(gt)) if gt else None
        union = predicted | gt
        jacc = (len(hits) / len(union)) if union else None
        return {
            "precision": prec, "recall": rec, "jaccard": jacc,
            "hits": len(hits), "predicted_size": len(predicted),
            "gt_size": len(gt),
        }

    summary = {
        "corpus": str(args.corpus),
        "workload": args.workload,
        "gt_actual_files": len(gt_actual_phys),
        "gt_static_files": len(gt_static_phys),
        "fired_rules": [a.rule_name for a in final_pred.activations],
        "tier_unions": {
            "tier_1_size": len(tier1_phys),
            "tier_2_size": len(tier2_phys),
            "tier_3_size": len(tier3_phys),
        },
        "tier_1_vs_actual": metrics(tier1_phys, gt_actual_phys),
        "tier_2_vs_actual": metrics(tier2_phys, gt_actual_phys),
        "tier_3_vs_actual": metrics(tier3_phys, gt_actual_phys),
        "tier_1_vs_static_gt": metrics(tier1_phys, gt_static_phys),
        "tier_2_vs_static_gt": metrics(tier2_phys, gt_static_phys),
        "tier_3_vs_static_gt": metrics(tier3_phys, gt_static_phys),
        "rule_detail": rule_detail,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    # Pretty-print
    print(f"Corpus: {args.corpus}")
    print(f"  Agent opened (this run): {len(gt_actual_phys)} files")
    print(f"  Workload GT_static:      {len(gt_static_phys)} files")
    print(f"  Fired rules: {[a.rule_name for a in final_pred.activations]}")
    print()
    print(f"  {'Tier':<6} {'union':>8} {'  prec_a':>10} {'rec_a':>8} {'jaccA':>8} "
          f" {'prec_s':>8} {'rec_s':>8} {'jaccS':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for tier, key_a, key_s, union_size in [
        (1, "tier_1_vs_actual", "tier_1_vs_static_gt", len(tier1_phys)),
        (2, "tier_2_vs_actual", "tier_2_vs_static_gt", len(tier2_phys)),
        (3, "tier_3_vs_actual", "tier_3_vs_static_gt", len(tier3_phys)),
    ]:
        ma = summary[key_a]
        ms = summary[key_s]
        def fmt(x):
            return f"{x*100:>6.1f}%" if x is not None else "  n/a "
        print(f"  T{tier:<5} {union_size:>8}  {fmt(ma['precision'])}  "
              f"{fmt(ma['recall'])}  {fmt(ma['jaccard'])}  "
              f"{fmt(ms['precision'])}  {fmt(ms['recall'])}  {fmt(ms['jaccard'])}")
    print()
    print("  Per-rule against static GT (workload's eventual working set):")
    for r in rule_detail:
        gs = r["vs_static_gt"]
        if gs["precision"] is None:
            continue
        print(f"    {r['rule']:<22} tier={r['tier']} subset={r['subset_size']:>5} "
              f"prec={gs['precision']*100:>6.1f}% rec={gs['recall']*100:>6.2f}%")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
