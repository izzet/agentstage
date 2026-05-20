#!/usr/bin/env bash
#
# Path 0 — replay smoke for the stager. Runs both modes and prints comparison.
#
# Usage:
#   ./scripts/microbench/path0_run.sh [STREAM_PATH] [N_TRIALS]
#
# Default stream: aiob_107 Sonnet 4.5 T1 PP seed 0 (the headline case).
# Default n-trials: 5 per file.

set -euo pipefail

cd "$(dirname "$0")/../.."

STREAM="${1:-outputs/poc/20260518-171234_aiob_107_anthropic_claude-sonnet-4-5_t1_b16384_pp_s0_azure/stream.jsonl}"
N_SAMPLES="${2:-20}"

SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
COLD="/mnt/common/datasets-staging/agentiobench/datasets"
HOT="/dev/shm/agentstage_path0"
OUTDIR="outputs/microbench/path0_$(date +%Y%m%dT%H%M%S)"

mkdir -p "$OUTDIR"
rm -rf "$HOT"

echo "=== Path 0 replay smoke ==="
echo "stream:      $STREAM"
echo "shim:        $SHIM"
echo "cold root:   $COLD"
echo "hot root:    $HOT"
echo "n samples:   $N_SAMPLES (distinct files; each guaranteed-cold)"
echo "outdir:      $OUTDIR"
echo

# Build the shim if not present
if [[ ! -f "$SHIM" ]]; then
    echo "building shim..."
    make -C src/agentstage/stager/shim
fi

echo "=== mode: baseline (no shim, cold reads) ==="
~/.local/bin/uv run python scripts/microbench/path0_replay.py \
    --mode baseline \
    --stream "$STREAM" \
    --workload aiob_107 \
    --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/baseline.json"

echo
echo "=== mode: with-stager (LD_PRELOAD + prefetch) ==="
rm -rf "$HOT"
LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$COLD" \
AGENTSTAGE_RETRY_SPIN_MS=5 \
~/.local/bin/uv run python scripts/microbench/path0_replay.py \
    --mode with-stager \
    --stream "$STREAM" \
    --workload aiob_107 \
    --n-samples "$N_SAMPLES" \
    --out "$OUTDIR/with_stager.json"

echo
echo "=== comparison ==="
~/.local/bin/uv run python - <<EOF
import json
b = json.load(open("$OUTDIR/baseline.json"))
w = json.load(open("$OUTDIR/with_stager.json"))

print(f"  Tier-1 predicted:      {b['n_tier1_files_predicted']}")
print(f"  Tier-3 predicted:      {b['n_tier3_files_predicted']}")
print(f"  Distinct files sampled:{b['n_samples_measured']}")
print(f"  Total samples:         {b['aggregate']['n_samples']}")
print()
print(f"  {'metric':<14} {'baseline':>12} {'with-stager':>14} {'speedup':>10}")
print(f"  {'-'*14:<14} {'-'*12:>12} {'-'*14:>14} {'-'*10:>10}")
for m in ("p50_ms", "p95_ms", "mean_ms", "max_ms", "min_ms"):
    bv = b["aggregate"][m]
    wv = w["aggregate"][m]
    ratio = bv / wv if wv > 0 else float("inf")
    print(f"  {m:<14} {bv:>10.3f} ms {wv:>12.3f} ms {ratio:>8.1f}x")
print()
print(f"  rule library:          {b['rule_library_version']}")
print(f"  cold root:             {b['cold_root']}")
print(f"  hot root:              {w['hot_root']}")
EOF
echo
echo "Artifacts: $OUTDIR/{baseline,with_stager}.json"
