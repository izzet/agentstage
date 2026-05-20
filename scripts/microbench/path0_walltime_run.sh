#!/usr/bin/env bash
#
# Full-file read wall-time measurement: cold vs hot on a chosen workload.
# Run via:  ./scripts/microbench/path0_walltime_run.sh aiob_110 [N_SAMPLES]
#
# Defaults: aiob_107, N=10. For aiob_110 (large NWB files) recommend N=3-5.

set -euo pipefail

cd "$(dirname "$0")/../.."

WORKLOAD="${1:-aiob_107}"
N_SAMPLES="${2:-10}"

SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
COLD="/mnt/common/datasets-staging/agentiobench/datasets"
HOT="/dev/shm/agentstage_walltime"
OUTDIR="outputs/microbench/walltime_${WORKLOAD}_$(date +%Y%m%dT%H%M%S)"

mkdir -p "$OUTDIR"
rm -rf "$HOT"

echo "=== Full-file wall-time measurement ==="
echo "  workload:    $WORKLOAD"
echo "  n samples:   $N_SAMPLES"
echo "  cold root:   $COLD"
echo "  hot root:    $HOT"
echo "  outdir:      $OUTDIR"
echo

[[ ! -f "$SHIM" ]] && make -C src/agentstage/stager/shim

echo "=== baseline (no shim, FULL cold reads) ==="
~/.local/bin/uv run python scripts/microbench/path0_walltime.py \
    --mode baseline \
    --workload "$WORKLOAD" \
    --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/baseline.json"

echo
echo "=== with-stager (LD_PRELOAD + prefetch + FULL hot reads) ==="
rm -rf "$HOT"
LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$COLD" \
AGENTSTAGE_RETRY_SPIN_MS=20 \
~/.local/bin/uv run python scripts/microbench/path0_walltime.py \
    --mode with-stager \
    --workload "$WORKLOAD" \
    --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/with_stager.json"

echo
echo "=== comparison ==="
~/.local/bin/uv run python - <<EOF
import json
b = json.load(open("$OUTDIR/baseline.json"))
w = json.load(open("$OUTDIR/with_stager.json"))
total_mb = b['aggregate']['total_bytes'] / 1e6

print(f"  workload:              {b['workload']} ({b['n_samples']} files, {total_mb:.0f} MB total)")
print()
print(f"  {'metric':<20} {'baseline':>14} {'with-stager':>14} {'speedup':>10}")
print(f"  {'-'*20:<20} {'-'*14:>14} {'-'*14:>14} {'-'*10:>10}")
for m in ("p50", "p95", "mean", "max"):
    bv = b["aggregate"]["full_read_ms"][m]
    wv = w["aggregate"]["full_read_ms"][m]
    ratio = bv / wv if wv > 0 else float("inf")
    print(f"  full_read {m + '_ms':<10} {bv:>10.1f} ms {wv:>12.1f} ms {ratio:>8.1f}x")
print()
for m in ("p50", "p95", "mean"):
    bv = b["aggregate"]["throughput_mbps"][m]
    wv = w["aggregate"]["throughput_mbps"][m]
    ratio = wv / bv if bv > 0 else float("inf")
    print(f"  throughput {m + '_mbps':<8} {bv:>10.1f}    {wv:>12.1f}    {ratio:>8.1f}x")
print()
EOF
echo
echo "Artifacts: $OUTDIR/{baseline,with_stager}.json"
