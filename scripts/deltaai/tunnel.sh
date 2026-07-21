#!/usr/bin/env bash
# Open an SSH tunnel from a remote host (Ares, your laptop, anything not on
# Delta AI) to the thinking-enabled Qwen3.6-27B vLLM endpoint running on a
# Delta AI ghx4 compute node.
#
# Workflow:
#   1. salloc + ssh into a Delta AI ghx4 compute node (e.g. gh017)
#   2. ./run_vllm_qwen3_thinking.sh — drops node name into $HOME/.vllm_host
#   3. From the remote host: ./tunnel.sh gh017
#
# If you don't pass the compute hostname explicitly, the script reads
# $HOME/.vllm_host on gh-login01 via a small ssh round-trip.
#
# Usage:
#   ./tunnel.sh                # auto-discover via ~/.vllm_host on gh-login01
#   ./tunnel.sh gh017          # explicit
#
# Env overrides:
#   DELTA_LOGIN_HOST    default gh-login01.delta.ncsa.illinois.edu
#   DELTA_USER          default $USER
#   OSS_PORT            default 8002

set -euo pipefail

LOGIN_HOST="${DELTA_LOGIN_HOST:-gh-login01.delta.ncsa.illinois.edu}"
DELTA_USER="${DELTA_USER:-$USER}"
PORT="${OSS_PORT:-8002}"

if [ $# -ge 1 ]; then
  COMPUTE_HOST="$1"
else
  echo "==> No compute host given; reading \$HOME/.vllm_host on $LOGIN_HOST"
  COMPUTE_HOST=$(ssh -o BatchMode=yes "${DELTA_USER}@${LOGIN_HOST}" 'cat ~/.vllm_host' 2>/dev/null) \
    || { echo "could not read ~/.vllm_host on $LOGIN_HOST — pass the hostname explicitly" >&2; exit 1; }
fi

# Delta AI compute nodes are reachable from the login host via the
# internal Slingshot fabric DNS name.
COMPUTE_FQDN="${COMPUTE_HOST}.hsn.cm.delta.internal.ncsa.edu"

echo "==> SSH tunnel to Delta AI"
echo "    Local:    localhost:${PORT}"
echo "    Compute:  ${COMPUTE_FQDN}:${PORT}"
echo "    Via:      ${DELTA_USER}@${LOGIN_HOST}"
echo
echo "Once the tunnel is up, in another shell on this host:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "  ./verify_vllm_thinking.sh                    # uses --host localhost"
echo
echo "Press Ctrl-C to tear down."
echo

exec ssh -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L "${PORT}:${COMPUTE_FQDN}:${PORT}" \
  "${DELTA_USER}@${LOGIN_HOST}"
