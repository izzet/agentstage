#!/usr/bin/env bash
# Launch Qwen/Qwen3.6-27B on a single Delta AI ghx4 node (TP=2) with
# thinking ENABLED — so the vLLM stream emits `delta.reasoning_content`
# chunks that AgentStage's predictor consumes as the "intent" signal.
#
# Delta-AI-targeted sibling of scripts/delta/run_vllm_qwen3_thinking.sh.
# Mirrors a known-good Qwen3.6 vLLM config, with two differences:
#   1. `--default-chat-template-kwargs '{"enable_thinking": true}'`
#      (set false to keep <think> blocks bounded)
#   2. ~/.vllm_host / ~/.vllm_port drop-files for the tunnel helper
#
# Prereqs:
#   - ~/.bashrc sets the allocation policy block (PROJECT/NVME/SCRATCH/HF_*)
#   - $NVME/vllm-venv exists (ARM aarch64, pre-built; usable on Delta AI only)
#   - You are on a compute node with --gres=gpu:2 (ghx4 / ghx4-interactive)
#
# Usage (from inside an salloc'd compute node):
#   ./run_vllm_qwen3_thinking.sh                  # default port 8002, TP=2
#   ./run_vllm_qwen3_thinking.sh --port 8003
#   ./run_vllm_qwen3_thinking.sh --tp 4           # use all 4 GH200s on the node
#
# Allocation that wraps this:
#   salloc -A <alloc>-dtai-gh -p ghx4-interactive \
#          --gres=gpu:2 --cpus-per-task=32 --mem=200G --time=02:00:00
#   # or, for TP=4:
#   salloc -A <alloc>-dtai-gh -p ghx4-interactive \
#          --gres=gpu:4 --cpus-per-task=64 --mem=600G --time=02:00:00
#   # Slurm prints the assigned node (e.g. gh017); ssh into it:
#   ssh gh017
#   $PROJECT/agentstage/scripts/deltaai/run_vllm_qwen3_thinking.sh --tp 4

set -euo pipefail

PORT=8002
MODEL="Qwen/Qwen3.6-27B"
TP=2

while [ $# -gt 0 ]; do
  case "$1" in
    --port)  PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tp)    TP="$2"; shift 2 ;;
    -h|--help) sed -n '/^# /,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$TP" in 1|2|4|8) ;; *) echo "--tp must be 1, 2, 4, or 8 (got: $TP)" >&2; exit 2 ;; esac

die() { echo "[qwen-thinking-dtai] FATAL: $*" >&2; exit 1; }
log() { echo "[qwen-thinking-dtai] $*" >&2; }

# ---- preflight ----

[ -n "${PROJECT:-}" ] && [ -n "${NVME:-}" ] && [ -n "${SCRATCH:-}" ] \
  || die "allocation env vars missing — source ~/.bashrc"

command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found — are you on a compute node?"
GPU_COUNT="$(nvidia-smi -L | wc -l)"
[ "$GPU_COUNT" -ge "$TP" ] \
  || die "need $TP visible GPUs for TP=$TP, found $GPU_COUNT — request --gres=gpu:$TP"

VENV="$NVME/vllm-venv"
[ -f "$VENV/bin/activate" ] && [ -x "$VENV/bin/vllm" ] \
  || die "vllm venv not found at $VENV — see the vLLM build notes to rebuild"

[ "$(uname -m)" = "aarch64" ] \
  || die "this script expects aarch64 (Delta AI / GH200) — for regular Delta x86 use scripts/delta/"

ss -tln 2>/dev/null | grep -qE ":${PORT}\b" \
  && die "port $PORT already bound — kill the holder or pass --port"

HF_DIR_NAME="models--${MODEL//\//--}"
HF_MODEL_PATH="${HF_HUB_CACHE:-$PROJECT/hf/hub}/$HF_DIR_NAME"
[ -d "$HF_MODEL_PATH/snapshots" ] \
  || die "model not cached at $HF_MODEL_PATH — fetch with:
       HF_HUB_OFFLINE=0 huggingface-cli download $MODEL"

# ---- env workarounds ----

# $HF_HOME/token is mode-600 owned by another user.
# huggingface_hub catches FileNotFoundError but not PermissionError, so
# pointing at a personal path lets it fall through cleanly.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NVME/hf_token_personal}"
touch "$HF_TOKEN_PATH" 2>/dev/null || true

# nccl-ofi-plugin/1.18.0-cuda129 ships only cu126/cu129 builds; site torch
# is cu130. The cu129 plugin's libcudart drift corrupts NCCL workers' heap
# during TP>=2 init ("free(): double free detected in tcache"). Socket
# transport bypasses OFI; intra-node NVLink doesn't need it. Documented in
# the vLLM build notes §"NCCL init (TP>=2 only)".
export NCCL_NET="${NCCL_NET:-Socket}"

# Cached-only HF serving: skip hub revision checks at startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---- module + venv ----

if command -v module >/dev/null 2>&1; then
  module load python/miniforge3_pytorch/2.11.0
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---- log setup + status file ----

mkdir -p "$SCRATCH"
SAFE_MODEL="${MODEL//\//__}"
LOG_FILE="$SCRATCH/vllm-${SAFE_MODEL}-thinking-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Drop the assigned node + port into well-known paths so an external tunnel
# script (e.g. from Ares or the user's laptop) can find them without parsing
# slurm output.
hostname -s > "$HOME/.vllm_host"
echo "$PORT"  > "$HOME/.vllm_port"

log "model:        $MODEL  (enable_thinking=true)"
log "port:         $PORT"
log "gpus visible: $GPU_COUNT (TP=$TP)"
log "node:         $(hostname)"
log "log file:     $LOG_FILE"
log "venv:         $VENV"
log "model path:   $HF_MODEL_PATH"
log ""
log "host  -> $HOME/.vllm_host"
log "port  -> $HOME/.vllm_port"
log ""
log "launching vllm…"

# Recipe locked to a working Delta-AI config, with enable_thinking
# flipped on (load-bearing for AgentStage — without it the model embeds
# thinking in delta.content instead of delta.reasoning_content).
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --max-model-len 65536 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}'
