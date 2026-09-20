#!/usr/bin/env bash
set -euo pipefail

# Run independent experiments concurrently, with a disjoint GPU group per
# experiment. Example: GPU_GROUPS='0,1;2,3;4,5;6,7'.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs}"
PARAM="${1:-}"
case "${PARAM}" in
  epsilon)
    VALUES="${EPSILON_VALUES:-0.1 0.25 0.5 1.0}"
    PREFIX=epsilon
    VAR=CMT_ALLOCATION_KL
    ;;
  gamma|cmt_gamma)
    VALUES="${GAMMA_VALUES:-0.95 0.99 0.995 0.999 1.0}"
    PREFIX=gamma
    VAR=CMT_GAMMA
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
  *)
    echo "Usage: $0 epsilon|gamma|top_k|lr" >&2
    echo "Example: GPU_GROUPS='0,1;2,3;4,5;6,7'" >&2
    exit 2
    ;;
esac

IFS=';' read -r -a groups <<< "${GPU_GROUPS:-0,1;2,3;4,5;6,7}"
values_array=(${VALUES})
if (( ${#values_array[@]} != ${#groups[@]} )); then
  echo "Number of values (${#values_array[@]}) must equal number of GPU groups (${#groups[@]})" >&2
  echo "GPU_GROUPS uses semicolon-separated groups, e.g. '0,1;2,3;4,5;6,7'" >&2
  exit 1
fi

tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
log_root="${OUTPUT_ROOT}/_sweep_logs/${PREFIX}_groups_${tag}"
mkdir -p "${log_root}"
stream_logs="${SWEEP_STREAM_LOGS:-true}"
set -o pipefail
pids=()
names=()
for index in "${!values_array[@]}"; do
  value="${values_array[$index]}"
  group="${groups[$index]}"
  IFS=',' read -r -a members <<< "${group}"
  nproc="${#members[@]}"
  if (( nproc < 1 )); then
    echo "Empty GPU group at index ${index}" >&2
    exit 1
  fi
  run_name="analysis_${PREFIX}_${value}_seed${SEED:-42}_${tag}"
  names+=("${run_name}")
  child_command=(bash "${SCRIPT_DIR}/train.sh" g_d)
  if [[ "${stream_logs,,}" == "true" || "${stream_logs}" == "1" || "${stream_logs,,}" == "yes" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="${group}"
      export TRAIN_NPROC_PER_NODE="${nproc}"
      export RUN_NAME="${run_name}"
      export OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}"
      export "${VAR}=${value}"
      "${child_command[@]}"
    ) 2>&1 | tee "${log_root}/${run_name}.log" &
  else
    (
      export CUDA_VISIBLE_DEVICES="${group}"
      export TRAIN_NPROC_PER_NODE="${nproc}"
      export RUN_NAME="${run_name}"
      export OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}"
      export "${VAR}=${value}"
      "${child_command[@]}"
    ) >"${log_root}/${run_name}.log" 2>&1 &
  fi
  pids+=("$!")
  echo "Started ${run_name} on GPUs [${group}] with ${nproc} processes (pid ${pids[-1]})"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished ${names[$index]}"
  else
    echo "FAILED ${names[$index]}; see ${log_root}/${names[$index]}.log" >&2
    tail -40 "${log_root}/${names[$index]}.log" >&2 || true
    failed=1
  fi
done
exit "${failed}"
