"""Decompose a session into LLM + shell I/O + shell compute components.

For paper's Amdahl analysis:
  total_session = LLM_streaming + sum(shell_elapsed)
  shell_io_time = read_bytes / cold_tier_BW (cold mode only)
  shell_compute = shell_elapsed - shell_io_time
  I/O share = shell_io_time / total_session

Usage:
    uv run python scripts/microbench/decompose_io_share.py \
        --merged outputs/replay/xxh_merged_results.json \
        --cold-bw-mbps 743
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--cold-bw-mbps", type=float, default=743.0)
    ap.add_argument("--hot-bw-mbps", type=float, default=4494.0)
    args = ap.parse_args()

    data = json.loads(Path(args.merged).read_text())
    valid = [r for r in data if 'cold_elapsed_s' in r and 'staged_elapsed_s' in r]

    cold_bw = args.cold_bw_mbps * 1e6
    hot_bw = args.hot_bw_mbps * 1e6

    print(f"Cold tier BW assumption: {args.cold_bw_mbps} MB/s")
    print(f"Hot tier BW assumption:  {args.hot_bw_mbps} MB/s")
    print(f"Hot/cold ratio:          {args.hot_bw_mbps/args.cold_bw_mbps:.1f}×")
    print()
    print(f"{'cell':50s} {'LLM_s':>6s} {'sh_io':>6s} {'sh_c':>6s} {'IOshr':>6s} {'AmdhMax':>7s} {'session':>7s} {'shell':>6s}")
    for r in valid:
        cold_sh = r.get('cold_shell_elapsed_s') or 0
        stg_sh = r.get('staged_shell_elapsed_s') or 0
        cold_rb = r.get('cold_read_bytes') or 0
        # LLM streaming approx = session - shell
        cold_total = r['cold_elapsed_s']
        llm = cold_total - cold_sh
        # I/O time = read_bytes / cold_bw (lower bound)
        io_time = cold_rb / cold_bw
        compute_in_shell = max(cold_sh - io_time, 0)
        io_share_session = io_time / cold_total if cold_total else 0
        # Amdahl: ideal session if I/O goes from cold_bw to hot_bw
        ideal_session = llm + compute_in_shell + (cold_rb / hot_bw)
        amdahl_max = cold_total / ideal_session if ideal_session else float('nan')
        sess_sp = cold_total / r['staged_elapsed_s']
        shell_sp = cold_sh / stg_sh if stg_sh else float('nan')
        print(f"{r['cell'][:50]:50s} {llm:>6.1f} {io_time:>6.1f} {compute_in_shell:>6.1f} "
              f"{io_share_session*100:>5.1f}% {amdahl_max:>7.2f} {sess_sp:>6.2f}× {shell_sp:>5.2f}×")

    print()
    # Per-task aggregate
    print(f"\n{'Task':10s} {'n':>3s} {'mean IOshr':>10s} {'mean Amdahl':>11s} {'mean shell sp':>13s}")
    by_task = defaultdict(list)
    for r in valid:
        by_task[r['task']].append(r)
    for task in sorted(by_task):
        rs = by_task[task]
        io_shares = []
        amdahls = []
        shell_sps = []
        for r in rs:
            cold_total = r['cold_elapsed_s']
            cold_sh = r.get('cold_shell_elapsed_s') or 0
            cold_rb = r.get('cold_read_bytes') or 0
            stg_sh = r.get('staged_shell_elapsed_s') or 0
            llm = cold_total - cold_sh
            io_time = cold_rb / cold_bw
            compute = max(cold_sh - io_time, 0)
            ideal = llm + compute + cold_rb / hot_bw
            io_shares.append(io_time / cold_total if cold_total else 0)
            amdahls.append(cold_total / ideal if ideal else float('nan'))
            if stg_sh > 0:
                shell_sps.append(cold_sh / stg_sh)
        import statistics
        print(f"{task:10s} {len(rs):>3d} {statistics.mean(io_shares)*100:>9.1f}% "
              f"{statistics.mean(amdahls):>11.2f}× {statistics.mean(shell_sps):>13.2f}×")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
