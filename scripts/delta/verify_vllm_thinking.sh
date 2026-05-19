#!/usr/bin/env bash
# Smoke-test the thinking-enabled Qwen3.6-27B endpoint.
#
# Verifies three things:
#   1. /v1/models returns the served model
#   2. A streaming /v1/chat/completions call emits `delta.reasoning_content`
#      chunks (vLLM's `--reasoning-parser` extension to the OpenAI API)
#   3. The reasoning content is non-trivial (≥ 100 chars) for a prompt
#      that should provoke thinking
#
# Run on the Delta compute node directly (default), or on Ares after the
# SSH tunnel is established (--host localhost).
#
# Usage:
#   ./verify_vllm_thinking.sh                   # default: localhost:8002
#   ./verify_vllm_thinking.sh --host gh017      # remote compute node
#   ./verify_vllm_thinking.sh --port 8002 --model Qwen/Qwen3.6-27B

set -euo pipefail

HOST="localhost"
PORT="8002"
MODEL="Qwen/Qwen3.6-27B"

while [ $# -gt 0 ]; do
    case "$1" in
        --host)  HOST="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# /,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

URL="http://${HOST}:${PORT}"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1" >&2; exit 1; }

echo "==> Verifying thinking-enabled vLLM endpoint at $URL"
echo

# ----------------------------------------------------------------------------
# 1. /v1/models returns the served model
# ----------------------------------------------------------------------------

echo "1. GET /v1/models"
MODELS_JSON=$(curl -fsS "${URL}/v1/models" 2>&1) \
    || fail "endpoint not reachable; is the server up + port forwarded?"

if echo "$MODELS_JSON" | grep -qF "$MODEL"; then
    pass "model $MODEL is served"
else
    echo "$MODELS_JSON" | head -5
    fail "model $MODEL not found in /v1/models response"
fi
echo

# ----------------------------------------------------------------------------
# 2 + 3. Streaming chat completion with a thinking-provoking prompt
# ----------------------------------------------------------------------------

echo "2. POST /v1/chat/completions with stream=true (thinking-provoking prompt)"

REQ=$(cat <<EOF
{
  "model": "$MODEL",
  "stream": true,
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "There are 13 NetCDF files in /data/era5/, one per month of 2024. I need to find the day with the maximum surface temperature anomaly relative to the 1990-2020 baseline. Walk me through your approach step by step before writing any code, and tell me which file you would inspect first."}
  ]
}
EOF
)

# Capture the streamed response; extract reasoning_content + content chunks.
TMPOUT=$(mktemp)
trap 'rm -f "$TMPOUT"' EXIT

curl -fsS -N -X POST "${URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$REQ" > "$TMPOUT" 2>&1 \
    || fail "streaming request failed (see $TMPOUT)"

# Count reasoning_content vs content chunks
REASONING_CHUNKS=$(grep -cE '"reasoning_content"\s*:\s*"' "$TMPOUT" || true)
CONTENT_CHUNKS=$(grep -cE '"content"\s*:\s*"[^"]' "$TMPOUT" || true)
DONE_MARKER=$(grep -cE '\[DONE\]|"finish_reason"\s*:\s*"stop"' "$TMPOUT" || true)

echo "    reasoning_content chunks: $REASONING_CHUNKS"
echo "    content chunks:           $CONTENT_CHUNKS"
echo "    stream terminator seen:   $DONE_MARKER"

if [[ "$REASONING_CHUNKS" -lt 5 ]]; then
    echo "---- raw response head ----"
    head -30 "$TMPOUT"
    echo "----"
    fail "expected at least 5 reasoning_content chunks; got $REASONING_CHUNKS. \
The model is not emitting thinking content. Check --reasoning-parser qwen3 \
and --default-chat-template-kwargs '{\"enable_thinking\": true}' on the \
launcher."
fi
pass "$REASONING_CHUNKS reasoning_content chunks streamed"

# Concatenate the reasoning content and check it's non-trivial.
REASONING_CONCAT=$(python3 -c "
import json, re, sys
total = []
for line in open('$TMPOUT'):
    line = line.strip()
    if not line.startswith('data: '):
        continue
    payload = line[len('data: '):]
    if payload == '[DONE]':
        continue
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        continue
    for choice in obj.get('choices', []):
        delta = choice.get('delta', {})
        rc = delta.get('reasoning_content')
        if rc:
            total.append(rc)
print(''.join(total))
")
REASONING_LEN=${#REASONING_CONCAT}
echo "    concatenated reasoning_content: $REASONING_LEN chars"

if [[ "$REASONING_LEN" -lt 100 ]]; then
    fail "reasoning_content too short ($REASONING_LEN chars). Thinking content \
may not be fully captured."
fi
pass "reasoning_content total length $REASONING_LEN chars (>=100)"

echo
echo "==> ALL CHECKS PASSED"
echo
echo "Set in .env on Ares:"
echo "  OSS_MODEL_BASE_URL=${URL}/v1"
echo "  OSS_MODEL_NAME=${MODEL}"
