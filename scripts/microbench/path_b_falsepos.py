"""Path B false-positive analyzer.

For each captured multi-turn run, computes:
  - Prefetched set (from staging_report.json events)
  - Opened set (from tool_use.jsonl trails)
  - Precision, recall, byte overfetch, wasted bytes

Reveals how often the detector's prefetches went to files the agent
never opened — the "false positive" rate that paper claim C2's overfetch
metric was supposed to bound, now measured in the multi-turn live
setting (rather than turn-1 seeds).

Usage:
    python scripts/microbench/path_b_falsepos.py \\
        --corpus outputs/multi_turn/e011_multiturn_hinted_<ts> \\
        --workload aiob_107_s3 \\
        --out <corpus>/falsepos.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentstage.workloads.aiob import (
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)
from agentstage.runners.path_b_multiturn import _resolve_logical_to_physical


def collect_prefetched(
    staging_report_path: Path,
    *,
    exclude_rule_ids: tuple[str, ...] = ("force", "path_a_force"),
) -> dict[str, int]:
    """Return {cold_path: size_bytes} for every staged event.

    Excludes rule_ids in ``exclude_rule_ids``: these are runner-level
    force-prefetches (e.g. the measurement step at the end of a Path B
    sparse_live run, which stages the agent's file just so we can time
    cold vs. hot). Counting those toward "detector precision" would
    flatter the detector.
    """
    if not staging_report_path.exists():
        return {}
    data = json.loads(staging_report_path.read_text())
    out: dict[str, int] = {}
    for ev in data.get("events", []):
        if ev.get("outcome") not in ("staged", "hit"):
            continue
        if ev.get("rule_id") in exclude_rule_ids:
            continue
        out[ev["cold_path"]] = ev.get("size_bytes", 0)
    return out


def collect_opened(
    corpus: Path,
    prefix_map: tuple[tuple[str, str], ...],
    cold_root: str,
) -> dict[str, int]:
    """Return {physical_path: size_bytes} for every open_file / read_file
    call across all turns."""
    out: dict[str, int] = {}
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
            try:
                size = Path(phys).stat().st_size if Path(phys).is_file() else 0
            except OSError:
                size = 0
            out[phys] = size
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workload",
                        choices=["aiob_107", "aiob_107_s3", "aiob_110"],
                        default="aiob_107_s3")
    parser.add_argument("--cold-root", default="/tmp/s3-noaa-goes16/ABI-L2-CMIPC")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaders = {
        "aiob_107": load_aiob_107,
        "aiob_107_s3": load_aiob_107_s3,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    prefix_map = workload.prefix_map

    prefetched = collect_prefetched(args.corpus / "staging_report.json")
    opened = collect_opened(args.corpus, prefix_map, args.cold_root)

    # Set arithmetic
    pf_paths = set(prefetched.keys())
    op_paths = set(opened.keys())
    hit_paths = pf_paths & op_paths       # prefetched AND opened
    waste_paths = pf_paths - op_paths     # prefetched but never opened
    miss_paths = op_paths - pf_paths      # opened but not prefetched

    pf_bytes = sum(prefetched.values())
    op_bytes = sum(opened.values())
    hit_bytes = sum(prefetched[p] for p in hit_paths)
    waste_bytes = sum(prefetched[p] for p in waste_paths)
    miss_bytes = sum(opened[p] for p in miss_paths)

    # Metrics
    precision = (len(hit_paths) / len(pf_paths)) if pf_paths else None
    recall = (len(hit_paths) / len(op_paths)) if op_paths else None
    byte_overfetch = (pf_bytes / op_bytes) if op_bytes > 0 else None
    byte_precision = (hit_bytes / pf_bytes) if pf_bytes > 0 else None
    byte_recall = (hit_bytes / op_bytes) if op_bytes > 0 else None
    # Jaccard overlap: handles disjoint prefetched/accessed sets cleanly,
    # unlike byte_overfetch which assumes prefetched ⊇ accessed.
    # C2's metric (1.5× ceiling on byte_overfetch) only makes sense when
    # prefetched is a superset of accessed; in sparse-prompt regimes we
    # observe disjoint sets and the ratio collapses to misleading values
    # (e.g. 0.07× because agent opened a larger file we did not detect).
    union_paths = pf_paths | op_paths
    union_bytes = pf_bytes + op_bytes - hit_bytes
    jaccard_files = (len(hit_paths) / len(union_paths)) if union_paths else None
    jaccard_bytes = (hit_bytes / union_bytes) if union_bytes > 0 else None

    result = {
        "corpus": str(args.corpus),
        "workload": args.workload,
        "counts": {
            "prefetched_files": len(pf_paths),
            "opened_files": len(op_paths),
            "hits": len(hit_paths),
            "wasted_prefetches": len(waste_paths),
            "misses_no_prefetch": len(miss_paths),
        },
        "bytes": {
            "prefetched": pf_bytes,
            "opened": op_bytes,
            "hit": hit_bytes,
            "wasted": waste_bytes,
            "missed_no_prefetch": miss_bytes,
        },
        "metrics": {
            "precision_files": precision,
            "recall_files": recall,
            "byte_overfetch_ratio": byte_overfetch,
            "byte_precision": byte_precision,
            "byte_recall": byte_recall,
            "jaccard_files": jaccard_files,
            "jaccard_bytes": jaccard_bytes,
        },
        "detail": {
            "prefetched": [{"path": p, "size": s} for p, s in prefetched.items()],
            "opened": [{"path": p, "size": s} for p, s in opened.items()],
            "hits": sorted(hit_paths),
            "wasted": sorted(waste_paths),
            "misses_no_prefetch": sorted(miss_paths),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    # Pretty-print
    print(f"Corpus: {args.corpus}")
    print(f"  Prefetched: {len(pf_paths)} files ({pf_bytes/1024/1024:.2f} MB)")
    print(f"  Opened:     {len(op_paths)} files ({op_bytes/1024/1024:.2f} MB)")
    print(f"  Hits:       {len(hit_paths)} files ({hit_bytes/1024/1024:.2f} MB)")
    print(f"  Wasted:     {len(waste_paths)} files ({waste_bytes/1024/1024:.2f} MB) "
          f"← false positives")
    print(f"  Misses:     {len(miss_paths)} files ({miss_bytes/1024/1024:.2f} MB) "
          f"← opened but not prefetched")
    print()
    if pf_paths:
        print(f"  Precision (files): {precision*100:.1f}%")
        print(f"  Byte precision:    {byte_precision*100:.1f}%")
    if op_paths:
        print(f"  Recall (files):    {recall*100:.1f}%")
        print(f"  Byte recall:       {byte_recall*100:.1f}%")
    if byte_overfetch is not None:
        print(f"  Byte overfetch:    {byte_overfetch:.2f}×  (1.0 = perfect, "
              f">1 = excess fetched, <1 = under-fetched / disjoint)")
    if jaccard_files is not None:
        print(f"  Jaccard (files):   {jaccard_files*100:.1f}%   "
              f"(0% = disjoint, 100% = identical)")
    if jaccard_bytes is not None:
        print(f"  Jaccard (bytes):   {jaccard_bytes*100:.1f}%")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
