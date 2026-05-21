"""E-027 — Project session-level speedup from real AIOB agentic runs.

Reads `io_report.json` files from /mnt/common/datasets-staging/agentiobench/outputs/
(real production runs captured by sciiobench's tracing harness, with full
DFTracer instrumentation) and computes:

  - Session wall-time (job_time_s)
  - POSIX I/O time (pread + fread + open + close + ... aggregated)
  - I/O fraction = posix_time / job_time
  - Projected session speedup if I/O is eliminated by staging
  - Projected session speedup if I/O is at S3-cold latency
    (using the 754 ms cold / 0.05 ms hot ratio from E-010)

Unlike our 8-turn smoke runs, these are full task-completion runs:
the agent writes a Python script and executes it, doing thousands of
file reads in one turn. This is where session-level savings materialize.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_AGENTIOBENCH_OUTPUTS = "/mnt/common/datasets-staging/agentiobench/outputs"

# From E-010: real measurement on the S3 mount via mountpoint-s3.
# Used here for the projection: if the agent's reads were against
# S3 instead of local NFS, how much higher would I/O time be?
S3_COLD_OPEN_MS = 754.5
LOCAL_NFS_OPEN_MS = 35.0  # avg open + first-read on the Ares /mnt/common XFS


def summarize_run(io_report_path: Path) -> dict | None:
    """Compute session-level breakdown from one io_report.json."""
    try:
        d = json.loads(io_report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = d.get("raw_stats", {})
    job_s = raw.get("job_time_s")
    n_files = raw.get("unique_file_count")
    if job_s is None:
        return None
    # Sum POSIX time across all functions
    posix_total = 0.0
    by_func = {}
    for fv in d.get("func_name_view", []):
        t = fv.get("posix_time_sum") or 0
        posix_total += t
        by_func[fv.get("func_name", "?")] = t
    io_frac = posix_total / job_s if job_s > 0 else 0
    # Speedup if I/O is eliminated entirely
    elim_speedup = 1.0 / max(1e-9, 1.0 - io_frac)
    # Project to S3 cold tier: assume same number of opens but at S3 latency
    # (this is a coarse projection; ignores read-throughput differences)
    s3_proj_io = posix_total * (S3_COLD_OPEN_MS / LOCAL_NFS_OPEN_MS)
    s3_proj_total = (job_s - posix_total) + s3_proj_io
    s3_proj_speedup = s3_proj_total / max(1e-9, (job_s - posix_total))
    parts = str(io_report_path).split("/")
    return {
        "path": str(io_report_path),
        "workload": next((p for p in parts if p.startswith("aiob_")), "?"),
        "provider": parts[-4] if len(parts) >= 4 else "?",
        "model": parts[-3] if len(parts) >= 3 else "?",
        "date": parts[-2] if len(parts) >= 2 else "?",
        "job_s": job_s,
        "n_files": n_files,
        "posix_s": posix_total,
        "io_frac": io_frac,
        "elim_speedup": elim_speedup,
        "s3_proj_total_s": s3_proj_total,
        "s3_proj_session_speedup": s3_proj_speedup,
        "by_func": by_func,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root", default=DEFAULT_AGENTIOBENCH_OUTPUTS,
        help="Root directory containing aiob_*/agentic/...")
    parser.add_argument("--workloads", default="aiob_104,aiob_107,aiob_110")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/realruns_session_speedup.json"))
    args = parser.parse_args()

    workloads = args.workloads.split(",")
    reports: list[dict] = []
    for wl in workloads:
        for p in Path(args.outputs_root).glob(f"{wl}*/agentic/none/*/*/2026*/io_report.json"):
            r = summarize_run(p)
            if r is not None:
                reports.append(r)

    # Pretty-print
    print(f"Found {len(reports)} io_report.json files across {workloads}")
    print()
    print(f"{'workload':<10} {'model':<28} {'date':<18} {'job_s':>7} {'files':>6} "
          f"{'posix_s':>9} {'io_frac':>8} {'elim_x':>7} {'s3_proj_x':>10}")
    for r in sorted(reports, key=lambda x: (x["workload"], x["date"])):
        wl_short = r["workload"][:10]
        model_short = r["model"][:28]
        print(f"{wl_short:<10} {model_short:<28} {r['date']:<18} "
              f"{r['job_s']:>7.1f} {r['n_files']:>6} "
              f"{r['posix_s']:>9.1f} {r['io_frac']*100:>7.1f}% "
              f"{r['elim_speedup']:>6.2f}x {r['s3_proj_session_speedup']:>9.2f}x")

    # Per-workload aggregate
    print()
    print("=== Per-workload aggregate (sonnet_4_5 + gemini_2_5_flash only) ===")
    keep_models = ("sonnet_4_5", "gemini_2_5_flash", "claude-sonnet-4-5")
    by_wl: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        if any(m in r["model"] for m in keep_models):
            by_wl[r["workload"]].append(r)
    print(f"{'workload':<20} {'n_runs':>7} {'mean job_s':>11} {'mean io_frac':>13} "
          f"{'mean elim_x':>13} {'mean s3_proj_x':>16}")
    for wl in sorted(by_wl.keys()):
        rs = by_wl[wl]
        n = len(rs)
        mean_job = sum(r["job_s"] for r in rs) / n
        mean_iof = sum(r["io_frac"] for r in rs) / n
        mean_elim = sum(r["elim_speedup"] for r in rs) / n
        mean_s3 = sum(r["s3_proj_session_speedup"] for r in rs) / n
        print(f"{wl[:20]:<20} {n:>7} {mean_job:>10.1f}s {mean_iof*100:>12.1f}% "
              f"{mean_elim:>11.2f}x {mean_s3:>14.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
