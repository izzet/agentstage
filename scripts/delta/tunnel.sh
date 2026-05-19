#!/usr/bin/env bash
# Open an SSH tunnel from Ares to Delta AI for the thinking-enabled
# Qwen3.6-27B vLLM endpoint.
#
# Run on Ares (this machine) after `run_vllm_qwen3_thinking.sh` is up on
# a Delta compute node (gh<NNN>):
#
#   ./tunnel.sh gh017          # tunnel localhost:8002 -> gh017:8002
#
# Use the hostname printed by the launcher's banner. Foregrounds the
# tunnel so the session keeps it alive; Ctrl-C tears it down.
#
# Args:
#   $1 — Delta compute hostname (e.g. gh017)
#
# Env overrides:
#   DELTA_LOGIN_HOST    default gh-login01.delta.ncsa.illinois.edu
#   DELTA_USER          default $USER
#   OSS_PORT            default 8002

set -euo pipefail

COMPUTE_HOST="${1:?usage: $0 <gh-compute-host> (e.g. gh017)}"
LOGIN_HOST="${DELTA_LOGIN_HOST:-gh-login01.delta.ncsa.illinois.edu}"
DELTA_USER="${DELTA_USER:-$USER}"
PORT="${OSS_PORT:-8002}"

# Delta's internal compute-host FQDN pattern (per AIOB DELTAAI_CAMPAIGN_GUIDE).
COMPUTE_FQDN="${COMPUTE_HOST}.hsn.cm.delta.internal.ncsa.edu"

echo "==> SSH tunnel to Delta AI"
echo "    Local:    localhost:${PORT}"
echo "    Compute:  ${COMPUTE_FQDN}:${PORT}"
echo "    Via:      ${DELTA_USER}@${LOGIN_HOST}"
echo
echo "Once the tunnel is up, in another shell on Ares:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "  ./scripts/delta/verify_vllm_thinking.sh"
echo
echo "Press Ctrl-C to tear down."
echo

exec ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "${PORT}:${COMPUTE_FQDN}:${PORT}" \
    "${DELTA_USER}@${LOGIN_HOST}"
