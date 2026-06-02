"""Aggregate LLM-once + replay-twice campaign results.

Reads outputs/replay/<campaign>/results.json and produces a per-cell
table + per-task aggregates, distinguishing:

  - SESSION speedup    = cold_replay / staged_replay      (end-to-end)
  - SHELL speedup      = cold_shell_s / staged_shell_s    (mechanism only)
  - I/O bytes saved    = cold_read_bytes (everything from disk)
                          vs staged_read_bytes (~0 from disk; from /dev/shm)
  - Effective bandwidth on cold reads
  - Per-cell Amdahl analysis

Usage:
    uv run python scripts/microbench/analyze_xxh_campaign.py \
        --results outputs/replay/xxh_main/results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="optional markdown out")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    valid = [r for r in data if "cold_elapsed_s" in r and "staged_elapsed_s" in r]
    print(f"Loaded {len(data)} cells ({len(valid)} valid)")

    # Reference Amdahl ceilings (measured this night with reference-solution
    # benches at /usr/bin/python3 + xxhash + cold cache + full eviction):
    REF_AMDAHL = {
        "aiob_201": 6.60,   # IGSR BAMs, 10.7GB, ref shell 14.1s
        "aiob_202": float('nan'),  # JWST FITS, TBD measured here
        "aiob_203": float('nan'),  # Sen2 TIF, TBD measured here
    }
    lines = []
    lines.append("# Mechanism speedup — xxh campaign")
    lines.append("")
    lines.append("## Reference Amdahl ceilings")
    lines.append("")
    lines.append("Measured under cold cache with reference (engineer-quality) solution.py.")
    lines.append("")
    lines.append("| Task | Ref shell wall | Ref read GB | Ref I/O share | Ref Amdahl ceiling |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append("| aiob_201 (IGSR 10.7GB) | 14.1s | 10.68 | ~100% | **6.60×** |")
    lines.append("| aiob_202 (JWST 11.3GB) | 91.9s* | 11.29 | 16.5%* | 1.16×* |")
    lines.append("| aiob_203 (Sen2 14.7GB) | 96.4s* | 14.65 | 20.5%* | 1.21×* |")
    lines.append("")
    lines.append("(*) The JWST and Sen2 reference solutions show much lower")
    lines.append("effective bandwidth than IGSR; investigation: pure-cat reads")
    lines.append("give 506 / 607 MB/s vs IGSR's 914 MB/s — likely due to storage")
    lines.append("layout differences. The xxh64 wall reflects this lower BW.")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| Cell | Baseline | Cold | Staged | Session sp | Shell cold | Shell staged | Shell sp | Cold read (GB) | Staged read (GB) | Fid cold |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in valid:
        sess_sp = r['cold_elapsed_s'] / r['staged_elapsed_s']
        sh_cold = r.get('cold_shell_elapsed_s', 0) or 0
        sh_stg = r.get('staged_shell_elapsed_s', 0) or 0
        sh_sp = (sh_cold / sh_stg) if sh_stg > 0 else float('nan')
        rd_c = (r.get('cold_read_bytes') or 0) / 1e9
        rd_s = (r.get('staged_read_bytes') or 0) / 1e9
        fid = r.get('cold_fidelity', '?')
        lines.append(
            f"| {r['cell'][:40]} | {r['baseline_elapsed_s']:.1f} | "
            f"{r['cold_elapsed_s']:.1f} | {r['staged_elapsed_s']:.1f} | "
            f"**{sess_sp:.2f}×** | {sh_cold:.1f} | {sh_stg:.1f} | "
            f"**{sh_sp:.2f}×** | {rd_c:.2f} | {rd_s:.2f} | {fid} |"
        )

    # Per-task aggregates
    by_task = defaultdict(list)
    for r in valid:
        by_task[r['task']].append(r)
    lines.append("")
    lines.append("## Per-task aggregates")
    lines.append("")
    lines.append("| Task | n | session AM | session Σ/Σ | shell AM | shell Σ/Σ | cold GB | I/O share (cold) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    COLD_BW = 743e6
    HOT_BW = 4494e6
    overall_records = []
    for task, rs in sorted(by_task.items()):
        sess_sps = [r['cold_elapsed_s']/r['staged_elapsed_s'] for r in rs]
        sh_sps = [(r['cold_shell_elapsed_s'] or 0) / (r['staged_shell_elapsed_s'] or 1)
                  for r in rs if (r.get('cold_shell_elapsed_s') or 0) and (r.get('staged_shell_elapsed_s') or 0)]
        bs_total = sum(r['cold_elapsed_s'] for r in rs)
        ss_total = sum(r['staged_elapsed_s'] for r in rs)
        sh_bs_total = sum(r.get('cold_shell_elapsed_s') or 0 for r in rs)
        sh_ss_total = sum(r.get('staged_shell_elapsed_s') or 0 for r in rs)
        cold_gb_avg = statistics.mean([(r.get('cold_read_bytes') or 0)/1e9 for r in rs])
        io_share_avg = statistics.mean([((r.get('cold_read_bytes') or 0)/COLD_BW) /
                                        ((r.get('cold_shell_elapsed_s') or 1)) * 100
                                        for r in rs if (r.get('cold_shell_elapsed_s') or 0) > 0])
        lines.append(
            f"| **{task}** | {len(rs)} | {statistics.mean(sess_sps):.2f}× | "
            f"{bs_total/ss_total:.2f}× | "
            f"{statistics.mean(sh_sps) if sh_sps else float('nan'):.2f}× | "
            f"{sh_bs_total/sh_ss_total if sh_ss_total else float('nan'):.2f}× | "
            f"{cold_gb_avg:.1f} | {io_share_avg:.1f}% |"
        )

    # Overall aggregate
    if valid:
        sess_sps = [r['cold_elapsed_s']/r['staged_elapsed_s'] for r in valid]
        sh_sps = [(r['cold_shell_elapsed_s'] or 0)/(r['staged_shell_elapsed_s'] or 1)
                  for r in valid if (r.get('cold_shell_elapsed_s') or 0) and (r.get('staged_shell_elapsed_s') or 0)]
        bs_total = sum(r['cold_elapsed_s'] for r in valid)
        ss_total = sum(r['staged_elapsed_s'] for r in valid)
        sh_bs_total = sum(r.get('cold_shell_elapsed_s') or 0 for r in valid)
        sh_ss_total = sum(r.get('staged_shell_elapsed_s') or 0 for r in valid)
        lines.append("")
        lines.append("## OVERALL across all valid cells")
        lines.append("")
        lines.append(f"- n cells: {len(valid)}")
        lines.append(f"- Session AM mean: **{statistics.mean(sess_sps):.2f}×**, Σcold/Σstaged: **{bs_total/ss_total:.2f}×**")
        lines.append(f"- Shell AM mean: **{statistics.mean(sh_sps) if sh_sps else float('nan'):.2f}×**, Σshell/Σshell: **{sh_bs_total/sh_ss_total if sh_ss_total else float('nan'):.2f}×**")
        max_sess = max(sess_sps); max_shell = max(sh_sps) if sh_sps else 0
        lines.append(f"- Max session speedup: {max_sess:.2f}×")
        lines.append(f"- Max shell speedup:   {max_shell:.2f}×")

    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out)
        print(f"\nWrote markdown report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
