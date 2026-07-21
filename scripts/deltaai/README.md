# Delta AI OSS-model serving playbook

How to bring up the thinking-enabled Qwen/Qwen3.6-27B endpoint on **Delta AI**
(GH200 aarch64, account `bekn-dtai-gh`) so AgentStage's predictor can consume
the streaming `delta.reasoning_content` field.

Sibling of `scripts/delta/README.md` (regular Delta, A100x8, TP=8). Pick one
cluster per session; both directories are self-contained and don't share
state — except the on-disk model cache at `/projects/bekn/hf/hub/` and the
ARM venv at `$NVME/vllm-venv`, both already present.

See `DELTA_VS_DELTAAI.md` for the cluster-choice trade-offs. Short version:
Delta AI is the right pick whenever regular Delta's `gpuA100x8-interactive`
queue is loaded — currently the case (decision recorded 2026-05-20).

## Why no `bootstrap_venv.sh` here

The Delta-AI ARM venv at `$NVME/vllm-venv` already exists (built April 29
during sciiobench's setup). It's `aarch64` and only runs on Delta AI; the
regular-Delta path under `scripts/delta/` builds a separate x86 venv at
`$NVME/vllm-venv-x86_64`. Same `$NVME` Lustre mount, different binaries,
different filenames, no collision.

If the ARM venv ever needs rebuilding, follow
`/work/hdd/bekn/izzet/projects/sciiobench/DELTAAI_VLLM.md` §"Phase 2 — vLLM
venv build" — that's the upstream of this setup.

## Smoke run

### 1. Allocate a node

```bash
salloc -A bekn-dtai-gh -p ghx4-interactive \
       --gres=gpu:2 --cpus-per-task=32 --mem=200G --time=02:00:00
# Slurm prints the assigned node (e.g. gh017). ssh into it:
ssh gh017
```

For longer sessions use `-p ghx4 --time=24:00:00` (2-day cap) and submit via
`sbatch` with a wrapper analogous to sciiobench's `run_8way_gemma.slurm`.

### 2. Launch the server

```bash
/projects/bekn/izzet/agentstage/scripts/deltaai/run_vllm_qwen3_thinking.sh
```

Cold load takes ~5-10 min (model is ~50 GB from `/projects/bekn/hf/hub/`).
The launcher drops the node name into `~/.vllm_host` and port into
`~/.vllm_port` for any external tunnel.

Blocks until SIGTERM or vLLM exits.

### 3. Verify (in a second shell)

From the same compute node:

```bash
ssh gh017
/projects/bekn/izzet/agentstage/scripts/deltaai/verify_vllm_thinking.sh
```

Three assertions:
1. `/v1/models` returns `Qwen/Qwen3.6-27B`
2. A streaming chat completion emits ≥ 5 `delta.reasoning_content` chunks
3. Concatenated reasoning content is ≥ 100 chars

If all PASS, the OSS slot is ready. Set in the AgentStage harness `.env`:

```
OSS_MODEL_BASE_URL=http://<node>:8002/v1
OSS_MODEL_NAME=Qwen/Qwen3.6-27B
```

### 4. (Optional) Tunnel from elsewhere

If the AgentStage harness runs on a host that isn't on Delta AI (Ares, your
laptop):

```bash
./scripts/deltaai/tunnel.sh           # auto-discover via ~/.vllm_host on gh-login01
./scripts/deltaai/tunnel.sh gh017     # explicit
```

Then `OSS_MODEL_BASE_URL=http://localhost:8002/v1`.

## Recipe at a glance

| Setting | Value | Why |
|---|---|---|
| Partition | `ghx4-interactive` (2 h cap) / `ghx4` (2-day cap) | Sciiobench's tested config; TP=2 across 2× GH200 |
| Account | `bekn-dtai-gh` (~870 h remaining) | Confirmed via `accounts` |
| `--gres` | `gpu:2` | Two GH200s for TP=2; the other two on the node stay free |
| `--mem` | `200g` | Single-model load; sciiobench's 600G figure is for dual-model `run_vllm_multi.sh` |
| Module | `python/miniforge3_pytorch/2.11.0` | Delta-AI's site torch stack; the venv's `bin/python` symlinks into here |
| Venv | `$NVME/vllm-venv` (aarch64) | Pre-built; usable only on Delta AI |
| Tensor parallel | 2 | Two GH200s suffice for a 27B model at 65K context |
| Context | 65 536 | Covers AgentIOBench prompt p90 with headroom |
| `--reasoning-parser qwen3` | mandatory | Splits thinking into `delta.reasoning_content`; without it, thinking embeds in `delta.content` and the predictor can't see it |
| `--tool-call-parser qwen3_coder` | mandatory | Required for tool calls; vLLM 400s without it |
| `NCCL_NET=Socket` | mandatory for TP ≥ 2 | nccl-ofi-plugin is cu12.9-built, site torch is cu13.0+; libcudart drift causes `free(): double free detected in tcache` |
| `HF_HUB_OFFLINE=1` | yes | Cached-only serving |
| `HF_TOKEN_PATH=$NVME/hf_token_personal` | yes | Workaround for `/projects/bekn/hf/token` mode-600 |

## Known failure modes

- `nvidia-smi not found — are you on a compute node?` — you're on the login node. salloc + ssh first.
- `free(): double free detected in tcache 2` during NCCL init — `NCCL_NET=Socket` not set. The launcher sets it; only seen if you bypass the launcher.
- `PermissionError: /projects/bekn/hf/token` — `HF_TOKEN_PATH` not set. Source `~/.bashrc` (the bekn policy block).
- vLLM returns `400 Bad Request` on tool calls — `--enable-auto-tool-choice` and/or `--tool-call-parser qwen3_coder` missing. The launcher sets both.
- Streaming returns `delta.content` only, never `delta.reasoning_content` — `--reasoning-parser qwen3` missing, or `enable_thinking` is `false`. Both are set by the launcher; if you fork it, keep them.
- `this script expects aarch64` — you ran the deltaai launcher on a regular-Delta x86 node. Use `scripts/delta/run_vllm_qwen3_thinking.sh` instead.

## Where this differs from regular Delta

For cross-reference (regular-Delta playbook is `scripts/delta/README.md`):

| | **Delta AI (this playbook)** | Regular Delta (`scripts/delta/`) |
|---|---|---|
| Account | `bekn-dtai-gh` (~870 h) | `bekn-delta-gpu` (~365 h) |
| Login | `gh-login01.delta.ncsa.illinois.edu` | `dt-login01.delta.ncsa.illinois.edu` |
| Partition | `ghx4-interactive` | `gpuA100x8-interactive` |
| Hardware | 2× GH200 120 GB (Hopper Grace ARM) | 8× A100-80GB (Ampere x86) |
| `--gres` form | `gpu:2` | `--gpus-per-node=8` |
| Module | `python/miniforge3_pytorch/2.11.0` | `PrgEnv-gnu cray-python/3.11.7` |
| Venv | `$NVME/vllm-venv` (aarch64) | `$NVME/vllm-venv-x86_64` |
| TP | 2 | 8 |
| Context | 65 K | 131 K |
| Bootstrap | not needed (venv exists) | `bootstrap_venv.sh` (~15-20 min) |
| Compute FQDN suffix | `.hsn.cm.delta.internal.ncsa.edu` | `.delta.ncsa.illinois.edu` |
| `NCCL_NET=Socket` | required | required (same root cause) |
| Model cache | `/projects/bekn/hf/hub/` | `/projects/bekn/hf/hub/` (same — Lustre shared) |
