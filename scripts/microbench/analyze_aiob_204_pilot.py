"""Quick analyzer for the aiob_204 pilot campaign.

Reads outputs/replay/<campaign>/results.json and prints:
- per-cell session + shell speedup, cold/staged shell time, read_bytes
- projected pass-count: ratio of cold_read_bytes to single-pass workload (14.7 GB)
- whether session_sp ≥ 2× target hit
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="aiob_204_pilot")
    ap.add_argument("--single-pass-gb", type=float, default=14.7,
                    help="Workload bytes for a single pass (default Sen2 = 14.7 GB)")
    args = ap.parse_args()

    path = Path("outputs/replay") / args.campaign / "results.json"
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    rs = json.loads(path.read_text())
    valid = [r for r in rs if "cold_elapsed_s" in r and "staged_elapsed_s" in r]
    if not valid:
        print("No valid cells")
        print(json.dumps(rs, indent=2)[:2000])
        return 1

    print(f"\n{'cell':40s} {'sess_c':>7s} {'sess_s':>7s} {'sess_sp':>8s} "
          f"{'sh_c':>6s} {'sh_s':>6s} {'sh_sp':>6s} "
          f"{'rb_c':>5s} {'passes':>7s} {'cold_BW':>8s}")
    print("-" * 110)
    sess_sps = []
    sh_sps = []
    for r in sorted(valid, key=lambda x: x["cell"]):
        cold = r["cold_elapsed_s"]; stg = r["staged_elapsed_s"]
        shc = r.get("cold_shell_elapsed_s") or 0
        shs = r.get("staged_shell_elapsed_s") or 0
        rb_c = (r.get("cold_read_bytes") or 0) / 1e9
        sess_sp = cold / stg
        sh_sp = shc / shs if shs > 0 else float("nan")
        passes = rb_c / args.single_pass_gb if args.single_pass_gb > 0 else 0
        bw = (rb_c * 1024) / shc if shc > 0 else 0
        sess_sps.append(sess_sp)
        if shs > 0: sh_sps.append(sh_sp)
        target_mark = " <2x" if sess_sp < 2.0 else " ≥2x ✓"
        print(f"{r['cell'][:40]:40s} {cold:>6.1f}s {stg:>6.1f}s "
              f"{sess_sp:>6.2f}×{target_mark} "
              f"{shc:>5.1f}s {shs:>5.1f}s {sh_sp:>5.2f}× "
              f"{rb_c:>4.1f}G {passes:>6.2f}× {bw:>6.0f}MB/s")

    print()
    if len(sess_sps) > 1:
        print(f"Session speedup:  AM {statistics.mean(sess_sps):.2f}×  "
              f"med {statistics.median(sess_sps):.2f}×  "
              f"max {max(sess_sps):.2f}×")
    else:
        print(f"Session speedup:  {sess_sps[0]:.2f}×")
    if sh_sps:
        print(f"Shell speedup:    AM {statistics.mean(sh_sps):.2f}×  "
              f"max {max(sh_sps):.2f}×")

    print(f"\nHit 2× target: {sum(1 for s in sess_sps if s >= 2.0)} / {len(sess_sps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
