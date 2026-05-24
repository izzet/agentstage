"""E-033 — First-tool-name distribution across real AIOB production runs.

Mines the agentiobench output tree for `run.log` files (one per agent
run) and computes the distribution of the *first* tool the agent
invokes. Backs the workflow-hook claim: auto-rule generation can fire
right after the agent's first list_dir, because list_dir is the
overwhelmingly common opener.

Method:
  - Walk /mnt/common/datasets-staging/agentiobench/outputs/*/agentic/
    looking for run.log files.
  - Parse out per-turn `[tool] <name>` lines.
  - Take the FIRST tool name per run, group by (workload, model_family).
  - Report P(first==list_dir) overall and per group.

Run:
    python scripts/microbench/first_tool_stat.py \
        --out outputs/microbench/first_tool_stat.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

AIOB_OUTPUTS_DEFAULT = Path("/mnt/common/datasets-staging/agentiobench/outputs")

# Matches lines like:
#   "09:51:02   [tool] list_dir: /data/goes_cmi_composites"
TOOL_LINE = re.compile(r"\[tool\]\s+([a-zA-Z_][a-zA-Z0-9_]*)")


def first_tool(run_log: Path) -> str | None:
    try:
        with run_log.open("r", errors="replace") as fh:
            for line in fh:
                m = TOOL_LINE.search(line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


def parse_run_meta(path: Path) -> dict:
    """Extract (workload, provider, model, knowledge) from the path layout:
    .../aiob_NNN_xxx/agentic/<knowledge>/<provider>/<model>/<ts>/run.log
    """
    parts = path.parts
    meta = {"workload": "?", "knowledge": "?", "provider": "?", "model": "?"}
    for p in parts:
        if p.startswith("aiob_"):
            meta["workload"] = p
            break
    try:
        i = parts.index("agentic")
        meta["knowledge"] = parts[i + 1]
        meta["provider"] = parts[i + 2]
        meta["model"] = parts[i + 3]
    except (ValueError, IndexError):
        pass
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=AIOB_OUTPUTS_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    run_logs = list(args.root.glob("aiob_*/agentic/*/*/*/*/run.log"))
    print(f"scanning {len(run_logs)} run.log files under {args.root}")

    rows: list[dict] = []
    for log in run_logs:
        ft = first_tool(log)
        if ft is None:
            continue
        meta = parse_run_meta(log)
        rows.append({"first_tool": ft, **meta, "path": str(log)})

    if not rows:
        print("FATAL: no parseable runs found")
        return 2

    # Overall
    overall = Counter(r["first_tool"] for r in rows)
    n = len(rows)
    overall_pct = {k: round(v / n * 100, 1) for k, v in overall.most_common()}

    # Per workload
    by_workload = defaultdict(Counter)
    for r in rows:
        by_workload[r["workload"]][r["first_tool"]] += 1

    # Per (provider, model)
    by_model = defaultdict(Counter)
    for r in rows:
        by_model[f"{r['provider']}/{r['model']}"][r["first_tool"]] += 1

    # Per knowledge regime (none == sparse, expert == hinted, in AIOB terms)
    by_know = defaultdict(Counter)
    for r in rows:
        by_know[r["knowledge"]][r["first_tool"]] += 1

    def serialize_counter(c: Counter, total_n: int | None = None) -> dict:
        tot = total_n or sum(c.values())
        return {
            "n": tot,
            "top": [{"tool": k, "count": v, "pct": round(v / tot * 100, 1)}
                    for k, v in c.most_common(5)],
            "list_dir_pct": round(c.get("list_dir", 0) / tot * 100, 1) if tot else 0,
        }

    list_dir_count = overall.get("list_dir", 0)
    summary = {
        "experiment": "E-033",
        "root": str(args.root),
        "n_runs_scanned": len(run_logs),
        "n_runs_with_first_tool": n,
        "overall_list_dir_pct": round(list_dir_count / n * 100, 2),
        "overall_distribution": overall_pct,
        "per_workload": {wl: serialize_counter(c) for wl, c in by_workload.items()},
        "per_model": {m: serialize_counter(c) for m, c in by_model.items()},
        "per_knowledge_regime": {k: serialize_counter(c) for k, c in by_know.items()},
    }
    args.out.write_text(json.dumps(summary, indent=2))

    print()
    print("=== E-033 First-Tool Distribution ===")
    print(f"  total runs analysed: {n}")
    print(f"  P(first_tool == list_dir): {summary['overall_list_dir_pct']}%")
    print()
    print("  Overall top 5:")
    for k, v in overall.most_common(5):
        print(f"    {k:>22s}  {v:>5d}  ({v/n*100:>5.1f}%)")
    print()
    print("  By knowledge regime (P(list_dir first)):")
    for k, c in by_know.items():
        tot = sum(c.values())
        ld = c.get("list_dir", 0)
        print(f"    {k:>10s}  {ld:>4d}/{tot:<4d}  ({ld/tot*100:>5.1f}%)")
    print()
    print("  Per model (P(list_dir first)) — top 8 by count:")
    items = sorted(by_model.items(), key=lambda kv: -sum(kv[1].values()))[:8]
    for m, c in items:
        tot = sum(c.values())
        ld = c.get("list_dir", 0)
        print(f"    {m:>40s}  {ld:>3d}/{tot:<3d}  ({ld/tot*100:>5.1f}%)")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
