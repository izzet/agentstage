#!/usr/bin/env bash
# Resume-aware OSS-model campaign runner.
#
# Runs the canonical 27-cell matrix (3 benchmarks x 3 tasks x 2 modes x 3 reps
# = 54 sessions) against a vLLM-served model. Re-running this script after a
# DeltaAI allocation expires will SKIP any cell whose summary.json already
# exists, so the next allocation picks up where the previous one stopped.
#
# Usage:
#   OSS_MODEL_BASE_URL=http://localhost:8002/v1 \
#   OSS_MODEL_NAME=Qwen/Qwen3.6-27B \
#   ./scripts/run_oss_campaign.sh
#
# Optional env:
#   CAMPAIGN_TAG  : tag to namespace the sweep directory (default: "qwen3")
#   SHELL_TIMEOUT_AIOB / DSBENCH / MLE : per-turn shell timeout (default 600)
#   MAX_TURNS_AIOB / DSBENCH / MLE     : per-session turn cap

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OSS_MODEL_BASE_URL="${OSS_MODEL_BASE_URL:-http://localhost:8002/v1}"
OSS_MODEL_NAME="${OSS_MODEL_NAME:-Qwen/Qwen3.6-27B}"
TAG="${CAMPAIGN_TAG:-qwen3}"

# Per-benchmark configuration. Bumped budgets so that one-shot solvers fit.
SHELL_TIMEOUT_AIOB=${SHELL_TIMEOUT_AIOB:-600}
SHELL_TIMEOUT_DSBENCH=${SHELL_TIMEOUT_DSBENCH:-600}
SHELL_TIMEOUT_MLE=${SHELL_TIMEOUT_MLE:-600}
# Capped tighter than the closed-model runs because OSS agents over-iterate
# on post-solve refinement. 8 turns captures the first bulk-read solver pass,
# which is all the shim's I/O acceleration affects — see PAPER_DEFENSE.md §6.
MAX_TURNS_AIOB=${MAX_TURNS_AIOB:-15}
MAX_TURNS_DSBENCH=${MAX_TURNS_DSBENCH:-15}
MAX_TURNS_MLE=${MAX_TURNS_MLE:-15}

DSBENCH_TASKS=("lmsys-chatbot-arena" "tabular-playground-series-may-2022" "ventilator-pressure-prediction")
MLE_TASKS=("dogs-vs-cats-redux-kernels-edition" "histopathologic-cancer-detection" "new-york-city-taxi-fare-prediction")
AIOB_TASKS=("aiob_103" "aiob_107" "aiob_110")
MODES=("baseline" "staged")
REPS=(1 2 3)

OUTROOT_AIOB="outputs/aiob_mt/_sweep_${TAG}"
OUTROOT_DSBENCH="outputs/dsbench_mt/_sweep_${TAG}"
OUTROOT_MLE="outputs/mlebench_mt/_sweep_${TAG}"
mkdir -p "$OUTROOT_AIOB" "$OUTROOT_DSBENCH" "$OUTROOT_MLE"

# Distinct hot_roots so a parallel future re-run won't trample copies.
HOT_AIOB="/dev/shm/agentstage_aiobmt_${TAG}"
HOT_DSBENCH="/dev/shm/agentstage_dsbmt_${TAG}"
HOT_MLE="/dev/shm/agentstage_mlemt_${TAG}"

UV="${HOME}/.local/bin/uv"
export OSS_MODEL_BASE_URL

run_or_skip() {
  local outdir="$1"; shift
  local runner="$1"; shift
  # remainder are args to runner
  local sf="$outdir/summary.json"
  if [ -f "$sf" ]; then
    # already done — but skip cells whose summary indicates a crash so we get a clean retry
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$sf')).get('crash') else 1)" 2>/dev/null; then
      echo "  [retry crashed] $outdir"
      rm -f "$sf"
    else
      echo "  [skip done]    $outdir"
      return 0
    fi
  fi
  echo "  [run]          $outdir"
  $UV run python "$runner" "$@" 2>&1 | tail -3
}

echo "==> OSS campaign: model=$OSS_MODEL_NAME tag=$TAG"
echo "    base_url=$OSS_MODEL_BASE_URL"
echo

# Interleave benchmarks so that if we get cut off we have coverage across all 3.
# Order within each rep is rotated by rep number so that whichever benchmark
# was at the END of a previous rep (and missed out on a partial allocation)
# gets pushed to the FRONT on the next pass. This guarantees every benchmark
# gets equal opportunity for coverage as allocations come and go.
for rep in "${REPS[@]}"; do
  echo "=== rep $rep ==="
  # rep 1 → AIOB, DSBench, MLE
  # rep 2 → MLE, AIOB, DSBench  (MLE first because rep-1 left MLE shortest)
  # rep 3 → DSBench, MLE, AIOB
  if   [ "$rep" -eq 1 ]; then ORDER=(AIOB DSBENCH MLE)
  elif [ "$rep" -eq 2 ]; then ORDER=(MLE AIOB DSBENCH)
  else                         ORDER=(DSBENCH MLE AIOB)
  fi
  for bench in "${ORDER[@]}"; do
    case "$bench" in
      AIOB)
        for task in "${AIOB_TASKS[@]}"; do
          for mode in "${MODES[@]}"; do
            outdir="$OUTROOT_AIOB/${task}_${mode}_r${rep}"
            run_or_skip "$outdir" scripts/microbench/aiob_multiturn.py \
              --task "$task" --model "$OSS_MODEL_NAME" --mode "$mode" \
              --hot-root "$HOT_AIOB" --out "$outdir" \
              --max-turns "$MAX_TURNS_AIOB" --shell-timeout "$SHELL_TIMEOUT_AIOB"
          done
        done ;;
      DSBENCH)
        for task in "${DSBENCH_TASKS[@]}"; do
          for mode in "${MODES[@]}"; do
            outdir="$OUTROOT_DSBENCH/${task}_${mode}_r${rep}"
            run_or_skip "$outdir" scripts/microbench/dsbench_multiturn.py \
              --task "$task" --model "$OSS_MODEL_NAME" --mode "$mode" \
              --hot-root "$HOT_DSBENCH" --out "$outdir" \
              --max-turns "$MAX_TURNS_DSBENCH" --shell-timeout "$SHELL_TIMEOUT_DSBENCH"
          done
        done ;;
      MLE)
        for task in "${MLE_TASKS[@]}"; do
          for mode in "${MODES[@]}"; do
            outdir="$OUTROOT_MLE/${task}_${mode}_r${rep}"
            run_or_skip "$outdir" scripts/microbench/mlebench_multiturn.py \
              --task "$task" --model "$OSS_MODEL_NAME" --mode "$mode" \
              --hot-root "$HOT_MLE" --out "$outdir" \
              --max-turns "$MAX_TURNS_MLE" --shell-timeout "$SHELL_TIMEOUT_MLE"
          done
        done ;;
    esac
  done
done

echo
echo "==> Campaign complete (or stopped mid-cell)."
echo "    Re-running this script will skip already-completed cells."
