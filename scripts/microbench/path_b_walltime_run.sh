#!/usr/bin/env bash
# Wrapper for path_b_walltime.py — sets LD_PRELOAD + env vars correctly so
# the shim is active in the parent process (for HOT reads) while subprocess
# COLD reads explicitly disable the shim. Mirrors path_a_s3_run.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

CORPUS="${1:?usage: $0 <corpus_dir>}"

SHIM="$(realpath src/agentstage/stager/shim/libagentstage_shim.so)"
COLD="/tmp/s3-noaa-goes16/ABI-L2-CMIPC"
HOT="/dev/shm/agentstage_walltime"

mkdir -p "$HOT"

LD_PRELOAD="$SHIM" \
AGENTSTAGE_HOT_ROOT="$HOT" \
AGENTSTAGE_COLD_ROOTS="$COLD" \
AGENTSTAGE_RETRY_SPIN_MS=20 \
~/.local/bin/uv run python scripts/microbench/path_b_walltime.py \
    --corpus "$CORPUS" \
    --workload aiob_107_s3 \
    --out "$CORPUS/walltime_replay.json"
