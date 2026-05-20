#!/usr/bin/env bash
#
# E-008: real S3 cold tier measurement.
# Reads GOES NetCDF files directly from NOAA's public S3 bucket via
# mountpoint-s3. Validates the throttled-simulator numbers in E-007
# against actual S3 behavior from this Ares testbed.
#
# Prereq: mount-s3 binary at ~/.local/bin/mount-s3 (download from
#   https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.tar.gz)
#
# Usage: ./scripts/microbench/path0_s3_run.sh [N_SAMPLES]

set -euo pipefail

cd "$(dirname "$0")/../.."

N_SAMPLES="${1:-5}"

S3_MOUNT="/tmp/s3-noaa-goes16"
SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
HOT="/dev/shm/agentstage_s3"
OUTDIR="outputs/microbench/path0_s3_$(date +%Y%m%dT%H%M%S)"
PREFIX="ABI-L2-CMIPC/2024/122/00"

mkdir -p "$OUTDIR"
rm -rf "$HOT"

echo "=== E-008: real S3 cold tier measurement ==="
echo "  bucket:      noaa-goes16 (public, no AWS credentials)"
echo "  prefix:      $PREFIX"
echo "  mount:       $S3_MOUNT"
echo "  hot root:    $HOT"
echo "  n samples:   $N_SAMPLES"
echo "  outdir:      $OUTDIR"
echo

[[ ! -f "$SHIM" ]] && make -C src/agentstage/stager/shim

# Mount if not already mounted
if ! mountpoint -q "$S3_MOUNT"; then
    echo "mounting NOAA GOES-16 bucket..."
    mkdir -p "$S3_MOUNT"
    ~/.local/bin/mount-s3 --no-sign-request --read-only --region us-east-1 \
        noaa-goes16 "$S3_MOUNT"
    sleep 1
fi
mountpoint "$S3_MOUNT"
echo

echo "=== baseline: reading directly from S3 mount (no stager) ==="
~/.local/bin/uv run python scripts/microbench/path0_s3.py \
    --mode baseline --s3-mount "$S3_MOUNT" --prefix "$PREFIX" \
    --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/baseline.json"
echo

echo "=== with-stager: prefetch from S3 → tmpfs, then read via shim ==="
rm -rf "$HOT"
LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$S3_MOUNT" \
AGENTSTAGE_RETRY_SPIN_MS=20 \
~/.local/bin/uv run python scripts/microbench/path0_s3.py \
    --mode with-stager --s3-mount "$S3_MOUNT" --prefix "$PREFIX" \
    --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/with_stager.json"
echo

echo "=== comparison ==="
~/.local/bin/uv run python - <<EOF
import json
b = json.load(open("$OUTDIR/baseline.json"))
w = json.load(open("$OUTDIR/with_stager.json"))
total_mb = b['aggregate']['total_bytes'] / 1e6
print(f"  {b['n_samples']} files, {total_mb:.0f} MB total")
print()
print(f"  {'metric':<22} {'baseline (S3)':>16} {'with-stager':>14} {'speedup':>10}")
print(f"  {'-'*22:<22} {'-'*16:>16} {'-'*14:>14} {'-'*10:>10}")
for m in ("p50", "p95", "mean"):
    bv = b["aggregate"]["full_read_ms"][m]
    wv = w["aggregate"]["full_read_ms"][m]
    ratio = bv / wv if wv > 0 else float("inf")
    print(f"  full_read {m + '_ms':<10} {bv:>13.1f} ms {wv:>12.1f} ms {ratio:>8.1f}x")
for m in ("p50", "mean"):
    bv = b["aggregate"]["throughput_mbps"][m]
    wv = w["aggregate"]["throughput_mbps"][m]
    ratio = wv / bv if bv > 0 else float("inf")
    print(f"  throughput {m + '_mbps':<8} {bv:>13.1f}    {wv:>12.1f}    {ratio:>8.1f}x")
print()

# Project 15-turn aiob_107-style run (45 reads × 3 MB)
LLM_S, COMPUTE_S, READS = 150, 30, 45
cold_per = b["aggregate"]["full_read_ms"]["mean"] / 1000
hot_per = w["aggregate"]["full_read_ms"]["mean"] / 1000
cold_total = cold_per * READS + LLM_S + COMPUTE_S
hot_total = hot_per * READS + LLM_S + COMPUTE_S
print(f"  Projected 15-turn run (45 reads × 3 MB GOES, 150s LLM, 30s compute):")
print(f"    cold (S3): {cold_total:.0f} s   hot: {hot_total:.0f} s   speedup: {cold_total/hot_total:.2f}x")
EOF
echo
echo "Artifacts: $OUTDIR/"
