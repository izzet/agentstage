"""Publication-ready summary for the xxh campaign.

Merges xxh_main (excluding broken aiob_202) + xxh_202_rerun and produces:
1. Per-cell table (with shell + session speedups)
2. Per-task aggregates (AM, GM, Σ/Σ)
3. Per-model aggregates
4. Overall stats
5. Markdown report ready to drop into the paper
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_merged() -> list[dict]:
    main_data = json.loads(Path("outputs/replay/xxh_main/results.json").read_text())
    main_kept = [r for r in main_data if r.get("task") != "aiob_202"]
    rerun_path = Path("outputs/replay/xxh_202_rerun/results.json")
    if rerun_path.exists():
        rerun = json.loads(rerun_path.read_text())
    else:
        rerun = []
        print("WARN: rerun not found; including the broken aiob_202 from main")
        return main_data
    return main_kept + rerun


def gm(xs):
    xs = [x for x in xs if x > 0]
    if not xs: return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def stats_block(rs):
    if not rs: return {}
    sess = [r['cold_elapsed_s'] / r['staged_elapsed_s'] for r in rs]
    sh = []
    for r in rs:
        shc = r.get('cold_shell_elapsed_s') or 0
        shs = r.get('staged_shell_elapsed_s') or 0
        if shc > 0 and shs > 0:
            sh.append(shc / shs)
    cold_total = sum(r['cold_elapsed_s'] for r in rs)
    stg_total = sum(r['staged_elapsed_s'] for r in rs)
    sh_cold_total = sum(r.get('cold_shell_elapsed_s') or 0 for r in rs)
    sh_stg_total = sum(r.get('staged_shell_elapsed_s') or 0 for r in rs)
    return {
        'n': len(rs),
        'sess_AM': statistics.mean(sess),
        'sess_GM': gm(sess),
        'sess_med': statistics.median(sess),
        'sess_agg': cold_total / stg_total,
        'sess_max': max(sess),
        'sh_AM': statistics.mean(sh) if sh else float('nan'),
        'sh_GM': gm(sh),
        'sh_med': statistics.median(sh) if sh else float('nan'),
        'sh_agg': sh_cold_total / sh_stg_total if sh_stg_total > 0 else float('nan'),
        'sh_max': max(sh) if sh else float('nan'),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rs = load_merged()
    valid = [r for r in rs if 'cold_elapsed_s' in r and 'staged_elapsed_s' in r]

    lines = []
    lines.append("# Final XXH Campaign Summary (merged xxh_main + xxh_202_rerun)")
    lines.append(f"\nMerged: {len(valid)} valid cells (excluding {len(rs)-len(valid)} failed).\n")

    # ====================================================================
    # Per-cell table
    # ====================================================================
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| Cell | Baseline | Cold | Staged | **Session sp** | Shell cold | Shell staged | **Shell sp** | Cold read (GB) | Staged read (GB) | Prefetched |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(valid, key=lambda x: (x['task'], x['model'], x['rep'])):
        sess = r['cold_elapsed_s'] / r['staged_elapsed_s']
        shc = r.get('cold_shell_elapsed_s') or 0
        shs = r.get('staged_shell_elapsed_s') or 0
        sh = (shc/shs) if shs > 0 else float('nan')
        rd_c = (r.get('cold_read_bytes') or 0) / 1e9
        rd_s = (r.get('staged_read_bytes') or 0) / 1e9
        pre = r.get('n_prefetched_staged', 0)
        # Mark cells with shell speedup ≥ 2 in bold
        sh_str = f"**{sh:.2f}×**" if sh >= 2.0 else f"{sh:.2f}×"
        sess_str = f"**{sess:.2f}×**" if sess >= 1.15 else f"{sess:.2f}×"
        lines.append(
            f"| {r['cell'][:48]} | {r['baseline_elapsed_s']:.1f} | "
            f"{r['cold_elapsed_s']:.1f} | {r['staged_elapsed_s']:.1f} | "
            f"{sess_str} | {shc:.1f} | {shs:.1f} | "
            f"{sh_str} | {rd_c:.2f} | {rd_s:.2f} | {pre} |"
        )
    lines.append("")

    # ====================================================================
    # Per-task aggregates
    # ====================================================================
    by_task = defaultdict(list)
    for r in valid:
        by_task[r['task']].append(r)
    lines.append("## Per-task aggregates")
    lines.append("")
    lines.append("| Task | n | sess AM | sess GM | sess Σ/Σ | sess max | shell AM | shell GM | shell Σ/Σ | shell max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for task in sorted(by_task):
        s = stats_block(by_task[task])
        lines.append(
            f"| **{task}** | {s['n']} | "
            f"{s['sess_AM']:.2f}× | {s['sess_GM']:.2f}× | "
            f"**{s['sess_agg']:.2f}×** | {s['sess_max']:.2f}× | "
            f"{s['sh_AM']:.2f}× | {s['sh_GM']:.2f}× | "
            f"**{s['sh_agg']:.2f}×** | {s['sh_max']:.2f}× |"
        )

    # ====================================================================
    # Per-model aggregates
    # ====================================================================
    by_model = defaultdict(list)
    for r in valid:
        by_model[r['model']].append(r)
    lines.append("")
    lines.append("## Per-model aggregates")
    lines.append("")
    lines.append("| Model | n | sess AM | sess Σ/Σ | shell AM | shell Σ/Σ | shell max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in sorted(by_model):
        s = stats_block(by_model[m])
        lines.append(
            f"| **{m}** | {s['n']} | "
            f"{s['sess_AM']:.2f}× | **{s['sess_agg']:.2f}×** | "
            f"{s['sh_AM']:.2f}× | **{s['sh_agg']:.2f}×** | {s['sh_max']:.2f}× |"
        )

    # ====================================================================
    # Overall
    # ====================================================================
    s = stats_block(valid)
    lines.append("")
    lines.append("## Overall (all valid cells)")
    lines.append("")
    lines.append(f"- n cells: **{s['n']}**")
    lines.append(f"- Session: AM **{s['sess_AM']:.2f}×**, GM {s['sess_GM']:.2f}×, "
                 f"Σ/Σ **{s['sess_agg']:.2f}×**, max **{s['sess_max']:.2f}×**")
    lines.append(f"- Shell: AM **{s['sh_AM']:.2f}×**, GM {s['sh_GM']:.2f}×, "
                 f"Σ/Σ **{s['sh_agg']:.2f}×**, max **{s['sh_max']:.2f}×**")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.out:
        Path(args.out).write_text(out)
        print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
