#!/usr/bin/env bash
#
# Path B multi-turn runner: live Haiku across N turns with detector + stager.
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
[ -n "$SCIIOBENCH_ROOT" ] && [ -f "$SCIIOBENCH_ROOT/.env" ] && source "$SCIIOBENCH_ROOT/.env"
set +a

if [[ -z "${AZURE_FOUNDRY_KEY:-}${ANTHROPIC_API_KEY:-}" ]]; then
    echo "FATAL: neither AZURE_FOUNDRY_KEY nor ANTHROPIC_API_KEY set" >&2
    exit 2
fi

MODE="${1:-hinted}"   # hinted | sparse | sparse_live | hinted_pathful | sparse_pathful
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
    hinted_pathful)
        EXP="e020_multiturn_hinted_pathful_${PATHFUL_VERSION:-v2}"
        EXTRA_ARGS=( --prompt-mode hinted --pathful-prompt
                     --pathful-version "${PATHFUL_VERSION:-v2}" )
        ;;
    sparse_pathful)
        EXP="e020_multiturn_sparse_pathful_${PATHFUL_VERSION:-v2}"
        EXTRA_ARGS=( --prompt-mode sparse --pathful-prompt
                     --pathful-version "${PATHFUL_VERSION:-v2}" )
        ;;
    e021_sparse_enrich)
        EXP="e021_multiturn_sparse_pathful_enrich_${PATHFUL_VERSION:-v4}"
        EXTRA_ARGS=( --prompt-mode sparse --pathful-prompt
                     --pathful-version "${PATHFUL_VERSION:-v4}" )
        ;;
    e021_sparse_enrich_live)
        EXP="e021_multiturn_sparse_pathful_enrich_live_${PATHFUL_VERSION:-v4}"
        EXTRA_ARGS=( --prompt-mode sparse --pathful-prompt
                     --pathful-version "${PATHFUL_VERSION:-v4}"
                     --measure-target-after )
        ;;
    e026_gemini_sparse_enrich_live)
        EXP="e026_multiturn_sparse_pathful_enrich_live_gemini_${PATHFUL_VERSION:-v4}"
        EXTRA_ARGS=( --prompt-mode sparse --pathful-prompt
                     --pathful-version "${PATHFUL_VERSION:-v4}"
                     --measure-target-after
                     --model "${GEMINI_MODEL:-gemini-2.5-flash}" )
        ;;
    *)
        echo "FATAL: mode must be one of:" >&2
        echo "  hinted | sparse | sparse_live | hinted_pathful | sparse_pathful" >&2
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
