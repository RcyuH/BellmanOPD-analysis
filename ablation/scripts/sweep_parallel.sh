#!/usr/bin/env bash
set -euo pipefail

# Run independent g_d jobs concurrently, one process/GPU.  This is different
# from sweep.sh, which intentionally runs jobs sequentially.  Each child sees
# exactly one visible GPU, so TRAIN_NPROC_PER_NODE=1 is explicit.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs}"
PARAM="${1:-}"
case "${PARAM}" in
  epsilon)
    VALUES="${EPSILON_VALUES:-0.1 0.25 0.5 1.0}"
    PREFIX=epsilon
    VAR=CMT_ALLOCATION_KL
    ;;
  top_k|topk)
    VALUES="${TOP_K_VALUES:-8 16 32}"
    PREFIX=topk
    VAR=TOP_K
    ;;
  lr|learning_rate)
    VALUES="${LR_VALUES:-5e-7 1e-6 2e-6}"
    PREFIX=lr
    VAR=LR
    ;;
  gamma|cmt_gamma)
    VALUES="${GAMMA_VALUES:-0.95 0.99 0.995 0.999 1.0}"
    PREFIX=gamma
    VAR=CMT_GAMMA
    ;;
  *)
    echo "Usage: $0 epsilon|gamma|top_k|lr" >&2
    echo "Set GPU_LIST, e.g. GPU_LIST=0,1,2" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0,1,2,3}"
values_array=(${VALUES})
if (( ${#values_array[@]} > ${#GPUS[@]} )); then
  echo "Need ${#values_array[@]} GPUs for ${#values_array[@]} concurrent jobs, but GPU_LIST has ${#GPUS[@]}" >&2
  echo "Run fewer values or provide more GPUs; use sweep.sh for sequential execution." >&2
  exit 1
fi

tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
log_root="${OUTPUT_ROOT}/_sweep_logs/${PREFIX}_${tag}"
mkdir -p "${log_root}"
# Stream each child log to this terminal while teeing the exact same bytes to
# its per-run file.  Disable with SWEEP_STREAM_LOGS=false for a quiet launcher.
stream_logs="${SWEEP_STREAM_LOGS:-true}"
set -o pipefail
pids=()
names=()
for index in "${!values_array[@]}"; do
  value="${values_array[$index]}"
  gpu="${GPUS[$index]}"
  # Include the sweep tag in the run identity so a retry never collides with
  # an earlier partially-created output directory.
  run_name="analysis_${PREFIX}_${value}_seed${SEED:-42}_${tag}"
  names+=("${run_name}")
  child_command=(bash "${SCRIPT_DIR}/train.sh" g_d)
  if [[ "${stream_logs,,}" == "true" || "${stream_logs}" == "1" || "${stream_logs,,}" == "yes" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export TRAIN_NPROC_PER_NODE=1
      export RUN_NAME="${run_name}"
      export OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}"
      export "${VAR}=${value}"
      "${child_command[@]}"
    ) 2>&1 | tee "${log_root}/${run_name}.log" &
  else
    (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TRAIN_NPROC_PER_NODE=1
    export RUN_NAME="${run_name}"
    export OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}"
    export "${VAR}=${value}"
      "${child_command[@]}"
    ) >"${log_root}/${run_name}.log" 2>&1 &
  fi
  pids+=("$!")
  echo "Started ${run_name} on physical GPU ${gpu} (pid ${pids[-1]})"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished ${names[$index]}"
  else
    echo "FAILED ${names[$index]}; see ${log_root}/${names[$index]}.log" >&2
    echo "----- tail of ${names[$index]}.log -----" >&2
    tail -40 "${log_root}/${names[$index]}.log" >&2 || true
    echo "----------------------------------------" >&2
    failed=1
  fi
done
exit "${failed}"
