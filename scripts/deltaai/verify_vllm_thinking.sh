#!/usr/bin/env bash
# Smoke-test the thinking-enabled Qwen3.6-27B endpoint launched by
# scripts/deltaai/run_vllm_qwen3_thinking.sh.
#
# Verifies three things:
#   1. /v1/models returns the served model
#   2. A streaming /v1/chat/completions call emits `delta.reasoning_content`
#      chunks (vLLM's --reasoning-parser extension to the OpenAI API)
#   3. The reasoning content is non-trivial (>= 100 chars) on a prompt
#      that should provoke thinking
#
# Run on the Delta-AI compute node directly (default; localhost:8002), or
# on a remote host (Ares, your laptop) after the SSH tunnel is established
# (--host localhost --port 8002 still works because the tunnel maps it).
#
# Usage:
#   ./verify_vllm_thinking.sh                                # localhost:8002
#   ./verify_vllm_thinking.sh --host gh017                   # remote compute node by short name
#   ./verify_vllm_thinking.sh --port 8003 --model Qwen/Qwen3.6-27B
#
# Exit 0 = all green; non-zero = first failure (with diagnostic context).

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

# ---- 1. /v1/models ----

echo "1. GET /v1/models"
MODELS_JSON=$(curl -fsS "${URL}/v1/models" 2>&1) \
  || fail "endpoint not reachable; is the server up? port-forwarded?"

if echo "$MODELS_JSON" | grep -qF "$MODEL"; then
  pass "model $MODEL is served"
else
  echo "$MODELS_JSON" | head -5
  fail "model $MODEL not found in /v1/models response"
fi
echo

# ---- 2 + 3. streaming chat completion with a thinking-provoking prompt ----

echo "2. POST /v1/chat/completions (stream=true, thinking-provoking prompt)"

REQ=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "I have files day1.nc, day2.nc, day3.nc in a directory. Plan how you would compute the 3-day mean. Think step by step before answering."}
  ],
  "stream": true,
  "max_tokens": 512
}
EOF
)

TMPOUT=$(mktemp)
trap 'rm -f "$TMPOUT"' EXIT

curl -fsS -N -X POST "${URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$REQ" > "$TMPOUT" 2>&1 \
  || fail "streaming request failed (see $TMPOUT)"

# vLLM has shipped the reasoning delta under two field names depending on
# version: older releases use `reasoning_content`, newer ones use `reasoning`
# (aligned with OpenAI's reasoning-models API). Accept either.
REASONING_CHUNKS=$(grep -cE '"(reasoning|reasoning_content)"\s*:\s*"' "$TMPOUT" || true)
CONTENT_CHUNKS=$(grep -cE '"content"\s*:\s*"[^"]' "$TMPOUT" || true)
DONE_MARKER=$(grep -cE '\[DONE\]|"finish_reason"\s*:\s*"stop"' "$TMPOUT" || true)

# Which field name did this vLLM build use? Surfaces it for the user so they
# can match their client/predictor code.
if grep -qE '"reasoning_content"\s*:\s*"' "$TMPOUT"; then
  REASONING_FIELD="reasoning_content"
elif grep -qE '"reasoning"\s*:\s*"' "$TMPOUT"; then
  REASONING_FIELD="reasoning"
else
  REASONING_FIELD="(none)"
fi

echo "    reasoning field name:     $REASONING_FIELD"
echo "    reasoning chunks:         $REASONING_CHUNKS"
echo "    content chunks:           $CONTENT_CHUNKS"
echo "    stream terminator seen:   $DONE_MARKER"

if [ "$REASONING_CHUNKS" -lt 5 ]; then
  echo "---- raw response head ----"
  head -30 "$TMPOUT"
  echo "----"
  fail "expected at least 5 reasoning chunks; got $REASONING_CHUNKS.
The model is not emitting thinking content. Check
--reasoning-parser qwen3 and --default-chat-template-kwargs '{\"enable_thinking\": true}'
on the launcher."
fi
pass "$REASONING_CHUNKS reasoning chunks streamed (field=$REASONING_FIELD)"

REASONING_CONCAT=$(python3 -c "
import json, sys
total = []
for line in open('$TMPOUT'):
    line = line.strip()
    if not line.startswith('data: '):
        continue
    payload = line[len('data: '):]
    if payload == '[DONE]':
        break
    try:
        obj = json.loads(payload)
    except Exception:
        continue
    for choice in obj.get('choices', []):
        delta = choice.get('delta', {}) or {}
        rc = delta.get('reasoning_content') or delta.get('reasoning')
        if rc:
            total.append(rc)
print(''.join(total))
")
REASONING_LEN=${#REASONING_CONCAT}
echo "    concatenated reasoning_content: $REASONING_LEN chars"

if [ "$REASONING_LEN" -lt 100 ]; then
  fail "reasoning_content too short ($REASONING_LEN chars). Thinking content may not be fully captured."
fi
pass "reasoning_content total length $REASONING_LEN chars (>=100)"

echo
echo "==> ALL CHECKS PASSED"
echo
echo "Set in .env on the AgentStage harness:"
echo "  OSS_MODEL_BASE_URL=${URL}/v1"
echo "  OSS_MODEL_NAME=${MODEL}"
