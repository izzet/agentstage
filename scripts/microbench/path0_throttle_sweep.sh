#!/usr/bin/env bash
#
# Throttle sweep: cold reads at multiple simulated throughputs vs hot (tmpfs).
# Simulates a range of cold-tier bandwidths (real PFS, S3-class, slow-S3, ...)
# to convert the slow-cold-tier wall-time projection in
# the stager verification plan into measured numbers.
#
# Usage:  ./scripts/microbench/path0_throttle_sweep.sh [WORKLOAD] [N_SAMPLES]
# Default: aiob_110, 3 files.
#
# Throttle list: native (no throttle), 50, 30, 10 MB/s — covers Lustre/NFS-
# typical (50), S3-class (30), and cross-region S3 (10).

set -euo pipefail

cd "$(dirname "$0")/../.."

WORKLOAD="${1:-aiob_110}"
N_SAMPLES="${2:-3}"

SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
COLD="/mnt/common/datasets-staging/agentiobench/datasets"
HOT="/dev/shm/agentstage_throttle"
OUTDIR="outputs/microbench/throttle_sweep_${WORKLOAD}_$(date +%Y%m%dT%H%M%S)"

mkdir -p "$OUTDIR"
rm -rf "$HOT"

echo "=== Throttled cold-tier sweep ==="
echo "  workload:    $WORKLOAD"
echo "  n samples:   $N_SAMPLES"
echo "  outdir:      $OUTDIR"
echo

[[ ! -f "$SHIM" ]] && make -C src/agentstage/stager/shim

# Baseline (no throttle, native XFS-SSD throughput)
echo "=== baseline: native (no throttle, XFS-SSD) ==="
~/.local/bin/uv run python scripts/microbench/path0_walltime.py \
    --mode baseline --workload "$WORKLOAD" --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/baseline_native.json"
echo

# Throttled baselines — simulate slower cold tiers
for MBPS in 50 30 10; do
    echo "=== baseline: throttled to ${MBPS} MB/s ==="
    ~/.local/bin/uv run python scripts/microbench/path0_walltime.py \
        --mode baseline --workload "$WORKLOAD" --n-samples "$N_SAMPLES" \
        --throttle-mbps "$MBPS" \
        --out "$OUTDIR/baseline_${MBPS}mbps.json"
    echo
done

# Hot tier measurement (unthrottled — same regardless of cold tier choice)
echo "=== with-stager (LD_PRELOAD + prefetch + tmpfs read) ==="
rm -rf "$HOT"
LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$COLD" \
AGENTSTAGE_RETRY_SPIN_MS=20 \
~/.local/bin/uv run python scripts/microbench/path0_walltime.py \
    --mode with-stager --workload "$WORKLOAD" --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/with_stager.json"
echo

# Comparison table
echo "=== comparison (full-file read time per sample) ==="
~/.local/bin/uv run python - <<EOF
import json
modes = [
    ("native (XFS-SSD)",    "$OUTDIR/baseline_native.json"),
    ("throttled 50 MB/s",   "$OUTDIR/baseline_50mbps.json"),
    ("throttled 30 MB/s",   "$OUTDIR/baseline_30mbps.json"),
    ("throttled 10 MB/s",   "$OUTDIR/baseline_10mbps.json"),
    ("with-stager (tmpfs)", "$OUTDIR/with_stager.json"),
]
results = []
for name, path in modes:
    d = json.load(open(path))
    r = d["aggregate"]["full_read_ms"]
    t = d["aggregate"]["throughput_mbps"]
    results.append((name, r["p50"], r["mean"], t["mean"]))

n = json.load(open("$OUTDIR/baseline_native.json"))["n_samples"]
total_mb = json.load(open("$OUTDIR/baseline_native.json"))["aggregate"]["total_bytes"] / 1e6
print(f"  {n} files, {total_mb:.0f} MB total")
print()

print(f"  {'cold tier':<22} {'p50 (ms)':>12} {'mean (ms)':>12} {'mean (MB/s)':>14} {'vs hot':>10}")
print(f"  {'-'*22:<22} {'-'*12:>12} {'-'*12:>12} {'-'*14:>14} {'-'*10:>10}")
hot_mean = results[-1][2]
for name, p50, mean, mbps in results:
    speedup = mean / hot_mean if hot_mean > 0 and name != results[-1][0] else 1.0
    speedup_str = f"{speedup:.1f}x" if name != results[-1][0] else "—"
    print(f"  {name:<22} {p50:>10.1f}   {mean:>10.1f}   {mbps:>12.1f}     {speedup_str:>10}")
print()

# Per-file wall-time impact at each cold-tier rate. Project a 15-turn
# agent run with 45 file reads + 150s LLM + 30s compute.
print("=== Projected wall-time on 15-turn run (45 file reads, 150s LLM, 30s compute): ===")
print(f"  {'cold tier':<22} {'cold I/O':>10} {'hot I/O':>10} {'total cold':>12} {'total hot':>12} {'wall speedup':>14}")
print(f"  {'-'*22:<22} {'-'*10:>10} {'-'*10:>10} {'-'*12:>12} {'-'*12:>12} {'-'*14:>14}")
LLM_S, COMPUTE_S, READS = 150, 30, 45
hot_per_file_s = hot_mean / 1000
hot_io = hot_per_file_s * READS
for name, _p50, mean_ms, _mbps in results[:-1]:
    cold_per_file_s = mean_ms / 1000
    cold_io = cold_per_file_s * READS
    total_cold = cold_io + LLM_S + COMPUTE_S
    total_hot  = hot_io  + LLM_S + COMPUTE_S
    speedup = total_cold / total_hot
    print(f"  {name:<22} {cold_io:>8.0f} s {hot_io:>8.1f} s "
          f"{total_cold:>10.0f} s {total_hot:>10.0f} s {speedup:>12.2f}x")
print()
EOF

echo "Artifacts: $OUTDIR/"
