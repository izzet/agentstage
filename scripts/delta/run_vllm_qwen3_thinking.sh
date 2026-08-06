#!/usr/bin/env bash
# Launch Qwen/Qwen3.6-27B on Delta AI with thinking ENABLED.
#
# Run this from inside a Delta AI salloc allocation on a ghx4 compute node:
#
#   salloc -A bekn-dtai-gh -p ghx4-interactive \
#          --gres=gpu:2 --cpus-per-task=32 --mem=300G --time=02:00:00
#   cd ~/dtai/agentstage
#   ./run_vllm_qwen3_thinking.sh
#
# Companion to AIOB's ~/dtai/agentiobench/run_vllm_qwen36.sh — same model, same
# infrastructure, only difference is `enable_thinking: true` and the
# `--reasoning-parser qwen3` flag is now load-bearing (not just defensive
# cleanup of residual thinking tags as in AIOB).
#
# Expected upstream context (from AIOB's bekn policy block, already in
# ~/.bashrc on Delta):
#   HF_HOME=/projects/bekn/hf
#   VLLM_CACHE_ROOT=/projects/bekn/vllm
#   PIP_CACHE_DIR=$NVME/pip
#   HF_TOKEN_PATH=$NVME/hf_token_personal   # AIOB workaround for bekn's
#                                            # 600-permission token file
# If those are not set, the model load will fail or stall on HF cache misses.

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8002}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
LOGDIR="${LOGDIR:-$NVME/logs}"
VLLM_VENV="${VLLM_VENV:-$NVME/vllm-venv}"
LMOD_MODULE="${LMOD_MODULE:-python/miniforge3_pytorch/2.11.0}"

mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/vllm_qwen3_thinking_$(date +%Y%m%d_%H%M%S).log"

echo "==> AgentStage OSS-model serving: thinking-enabled Qwen3.6-27B"
echo "    Model: $MODEL"
echo "    Port:  $PORT"
echo "    TP:    $TP"
echo "    Log:   $LOGFILE"
echo "    Node:  $(hostname)"
echo

# ----------------------------------------------------------------------------
# Environment sanity
# ----------------------------------------------------------------------------

: "${NVME:?NVME not set; check the bekn policy block in ~/.bashrc}"
: "${HF_HOME:?HF_HOME not set; check the bekn policy block in ~/.bashrc}"
: "${VLLM_CACHE_ROOT:?VLLM_CACHE_ROOT not set; check the bekn policy block in ~/.bashrc}"

# AIOB's HF-token-permission workaround. Without this, transformers tries
# to read /projects/bekn/hf/token (mode 600, not group-readable) and
# crashes with a PermissionError it doesn't catch.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NVME/hf_token_personal}"
touch "$HF_TOKEN_PATH" 2>/dev/null || true

# ----------------------------------------------------------------------------
# Load the Lmod stack (PyTorch + flash-attn + xformers for sm_90)
# ----------------------------------------------------------------------------

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: Lmod 'module' command not found. Are you on a Delta compute node?" >&2
    exit 1
fi
module load "$LMOD_MODULE"

# ----------------------------------------------------------------------------
# Activate the vLLM venv (built on top of the site torch)
# ----------------------------------------------------------------------------

if [[ ! -d "$VLLM_VENV" ]]; then
    echo "ERROR: vLLM venv not found at $VLLM_VENV." >&2
    echo "Run the AIOB vLLM setup first." >&2
    exit 1
fi
source "$VLLM_VENV/bin/activate"

# ----------------------------------------------------------------------------
# Launch vLLM with thinking ENABLED
# ----------------------------------------------------------------------------
#
# Differences vs. AIOB's run_vllm_qwen36.sh:
#
#   --default-chat-template-kwargs '{"enable_thinking": true}'
#     ^ AIOB sets this to false. We want the model to emit thinking content
#       in the streamed response so the detector can see intent before tool
#       dispatch.
#
#   --reasoning-parser qwen3
#     ^ Tells vLLM to separate reasoning content into a distinct field
#       (`delta.reasoning_content`) in the streaming response, instead of
#       embedding it in the `delta.content` text. Our OpenAIClient wrapper
#       reads this field as the thinking stream.
#
# Tool-call parser kept on so the agent can still use tool_calls normally
# (we'll exercise this in the multi-turn E5+ runs).

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    2>&1 | tee -a "$LOGFILE"
