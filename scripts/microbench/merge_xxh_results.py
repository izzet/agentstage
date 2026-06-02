#!/usr/bin/env python3
"""Merge xxh_main + xxh_202_rerun into a single results table."""
import json
import sys
from pathlib import Path

main_path = Path("outputs/replay/xxh_main/results.json")
rerun_path = Path("outputs/replay/xxh_202_rerun/results.json")

main_data = json.loads(main_path.read_text())
# Drop the failed aiob_202 cells from main
main_kept = [r for r in main_data if r.get("task") != "aiob_202"]

if rerun_path.exists():
    rerun_data = json.loads(rerun_path.read_text())
    merged = main_kept + rerun_data
else:
    print("WARN: rerun not yet completed; using main only", file=sys.stderr)
    merged = main_data

out = Path("outputs/replay/xxh_merged_results.json")
out.write_text(json.dumps(merged, indent=2))
print(f"Wrote {len(merged)} cells to {out}")
print(f"  main_kept (non-202): {len(main_kept)}")
print(f"  rerun_202:           {len(merged) - len(main_kept)}")
