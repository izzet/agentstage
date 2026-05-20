#!/usr/bin/env bash
#
# Path B multi-turn runner: live Haiku across N turns with predictor + stager.
# Used for E-011 (hinted baseline), E-014 (sparse replay), E-015 (sparse live).
#
# Usage:
#   ./scripts/path_b_run.sh hinted        # E-011: hinted prompt, 8 turns
#   ./scripts/path_b_run.sh sparse        # E-014: sparse prompt, 8 turns
#   ./scripts/path_b_run.sh sparse_live   # E-015: sparse + measure cold/hot

set -euo pipefail
cd "$(dirname "$0")/.."

set -a
[ -f .env ] && source .env
[ -f /home/iyildirim/projects/sciiobench/.env ] && source /home/iyildirim/projects/sciiobench/.env
set +a

if [[ -z "${AZURE_FOUNDRY_KEY:-}${ANTHROPIC_API_KEY:-}" ]]; then
    echo "FATAL: neither AZURE_FOUNDRY_KEY nor ANTHROPIC_API_KEY set" >&2
    exit 2
fi

MODE="${1:-hinted}"   # hinted | sparse | sparse_live
case "$MODE" in
    hinted)
        EXP="e011_multiturn_hinted"
        EXTRA_ARGS=( --prompt-mode hinted )
        ;;
    sparse)
        EXP="e014_multiturn_sparse"
        EXTRA_ARGS=( --prompt-mode sparse )
        ;;
    sparse_live)
        EXP="e015_multiturn_sparse_live"
        EXTRA_ARGS=( --prompt-mode sparse --measure-target-after )
        ;;
    *)
        echo "FATAL: mode must be one of: hinted | sparse | sparse_live" >&2
        exit 2
        ;;
esac

SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
COLD="/tmp/s3-noaa-goes16/ABI-L2-CMIPC"  # S3 mount via mountpoint-s3
HOT="/dev/shm/agentstage_path_b"
OUTDIR="outputs/multi_turn/${EXP}_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUTDIR"
rm -rf "$HOT"

echo "=== Path B multi-turn ($MODE) ==="
echo "  model:      claude-haiku-4-5"
echo "  workload:   aiob_107_s3"
echo "  cold root:  $COLD"
echo "  hot root:   $HOT"
echo "  shim:       $SHIM"
echo "  outdir:     $OUTDIR"

LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$COLD" \
AGENTSTAGE_RETRY_SPIN_MS=20 \
~/.local/bin/uv run python -m agentstage.runners.path_b_multiturn \
    --workload aiob_107_s3 \
    --max-turns 8 \
    --budget 4096 \
    "${EXTRA_ARGS[@]}" \
    --out "$OUTDIR"

echo
echo "Artifacts under $OUTDIR/"
