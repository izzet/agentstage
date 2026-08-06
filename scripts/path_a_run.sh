#!/usr/bin/env bash
#
# Path A smoke runner: live Haiku call on aiob_107 with planning prompt.
# Loads .env files in precedence order and launches the runner with
# LD_PRELOAD + shim env vars set.
#
# Usage: ./scripts/path_a_run.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Load env vars (ours first, then an optional sibling project's as fallback)
set -a
[ -f .env ] && source .env
[ -n "$SCIIOBENCH_ROOT" ] && [ -f "$SCIIOBENCH_ROOT/.env" ] && source "$SCIIOBENCH_ROOT/.env"
set +a

if [[ -z "${AZURE_FOUNDRY_KEY:-}${ANTHROPIC_API_KEY:-}" ]]; then
    echo "FATAL: neither AZURE_FOUNDRY_KEY nor ANTHROPIC_API_KEY set in env" >&2
    exit 2
fi

SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
COLD="/mnt/common/datasets-staging/agentiobench/datasets"
HOT="/dev/shm/agentstage_path_a"
OUTDIR="outputs/path_a/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUTDIR"
rm -rf "$HOT"

echo "=== Path A live smoke ==="
echo "  model:      claude-haiku-4-5"
echo "  cold root:  $COLD"
echo "  hot root:   $HOT"
echo "  shim:       $SHIM"
echo "  outdir:     $OUTDIR"
[[ -n "${AZURE_FOUNDRY_ANTHROPIC_URL:-}" ]] && echo "  base url:   $AZURE_FOUNDRY_ANTHROPIC_URL"
echo

# Build the shim if needed
[[ ! -f "$SHIM" ]] && make -C src/agentstage/stager/shim

LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$COLD" \
AGENTSTAGE_RETRY_SPIN_MS=20 \
~/.local/bin/uv run python -m agentstage.runners.path_a_smoke --out "$OUTDIR"

echo
echo "Artifacts under $OUTDIR/"
