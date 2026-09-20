#!/usr/bin/env bash
set -euo pipefail
export B200_EVAL_USE_ALL_GPUS=true
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/eval_checkpoint_b200.sh METHOD CHECKPOINT [OUTPUT_DIR] [extra CLI args]

METHOD may be: opd, ta-opd (or ta), rac, pgt, cmt, grpo, or iw.

Examples:
  bash scripts/eval_checkpoint_b200.sh opd outputs/run01/opd/checkpoint-000050
  # For a checkpoint under outputs/<run>/<method>, the default also upserts
  # outputs/<run>/<method>/eval_history.jsonl.
  bash scripts/eval_checkpoint_b200.sh ta-opd outputs/run01/ta_opd/final
  bash scripts/eval_checkpoint_b200.sh ta-opd /path/checkpoint-000100 results/eval/ta_step100
  EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 bash scripts/eval_checkpoint_b200.sh rac /path/checkpoint-000150
  bash scripts/eval_checkpoint_b200.sh pgt /path/checkpoint-000150
EOF
}

if (( $# < 2 )); then
  usage
  exit 2
fi

METHOD_INPUT="$1"
CHECKPOINT_INPUT="$2"
shift 2

case "${METHOD_INPUT,,}" in
  opd|pure-opd|pure_opd)
    METHOD_SLUG="opd"
    HISTORY_METHOD="opd"
    MODEL_NAME="OPD"
    METHOD_CONFIG="${OPD_CONFIG}"
    ;;
  ta|ta-opd|ta_opd)
    METHOD_SLUG="ta_opd"
    HISTORY_METHOD="ta"
    MODEL_NAME="TA-OPD"
    METHOD_CONFIG="${TA_CONFIG}"
    ;;
  rac|bellman-rac|bellman_rac)
    METHOD_SLUG="rac"
    HISTORY_METHOD="rac"
    MODEL_NAME="RAC"
    METHOD_CONFIG="${RAC_CONFIG}"
    ;;
  pgt|projected-gradient-teachability|projected_gradient_teachability)
    METHOD_SLUG="pgt_opd"
    HISTORY_METHOD="pgt"
    MODEL_NAME="PGT"
    METHOD_CONFIG="${PGT_CONFIG}"
    ;;
  cmt|coupled-marginal-teachability|coupled_marginal_teachability)
    METHOD_SLUG="cmt_opd"
    HISTORY_METHOD="cmt"
    MODEL_NAME="CMT-OPD"
    METHOD_CONFIG="${CMT_CONFIG}"
    ;;
  grpo|group-relative-policy-optimization|group_relative_policy_optimization)
    METHOD_SLUG="grpo"
    HISTORY_METHOD="grpo"
    MODEL_NAME="GRPO"
    METHOD_CONFIG="${GRPO_CONFIG}"
    ;;
  iw|iw-opd|importance-weighted-opd|importance_weighted_opd)
    METHOD_SLUG="iw"
    HISTORY_METHOD="iw"
    MODEL_NAME="IW-OPD"
    METHOD_CONFIG="${IW_CONFIG}"
    ;;
  *)
    echo "Unknown method: ${METHOD_INPUT}" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -d "${CHECKPOINT_INPUT}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_INPUT}" >&2
  exit 1
fi
CHECKPOINT_PATH="$(cd "${CHECKPOINT_INPUT}" && pwd)"
require_file "${CHECKPOINT_PATH}/config.json"
require_file "${METHOD_CONFIG}"

# A checkpoint nested directly under a method run output can be represented in
# the same history schema as training-time evaluation.  Keep standalone model
# paths working as before, but automatically opt in for the normal
# outputs/<run>/<method>/{checkpoint-*,final} layout.
CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH}")"
HISTORY_RUN_OUTPUT=""
CHECKPOINT_PARENT="$(dirname "${CHECKPOINT_PATH}")"
case "${CHECKPOINT_NAME}" in
  final|checkpoint-[0-9]*)
    if [[ -f "${CHECKPOINT_PARENT}/resolved_config.yaml" || -f "${CHECKPOINT_PARENT}/eval_history.jsonl" ]]; then
      HISTORY_RUN_OUTPUT="${CHECKPOINT_PARENT}"
    fi
    ;;
esac

if (( $# > 0 )) && [[ "$1" != --* ]]; then
  EVAL_OUTPUT="$1"
  shift
else
  if [[ -n "${HISTORY_RUN_OUTPUT}" ]]; then
    EVAL_OUTPUT="${HISTORY_RUN_OUTPUT}/checkpoint_eval/${CHECKPOINT_NAME}"
  else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    EVAL_OUTPUT="${REPO_DIR}/results/checkpoint_eval/${METHOD_SLUG}_${CHECKPOINT_NAME}_${TIMESTAMP}"
  fi
fi

if [[ "${EVAL_OUTPUT}" != /* ]]; then
  EVAL_OUTPUT="${REPO_DIR}/${EVAL_OUTPUT}"
fi

echo "Method: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Evaluation output: ${EVAL_OUTPUT}"
if [[ -n "${HISTORY_RUN_OUTPUT}" ]]; then
  echo "History output: ${HISTORY_RUN_OUTPUT}/eval_history.jsonl"
fi
echo "Protocol: backend=${EVAL_BACKEND:-vllm}, temperature=${EVAL_TEMPERATURE:-1.0}, responses=${EVAL_NUM_RESPONSES:-8}"

HISTORY_ARGS=()
if [[ -n "${HISTORY_RUN_OUTPUT}" ]]; then
  HISTORY_ARGS+=(
    --history-run-output "${HISTORY_RUN_OUTPUT}"
    --history-method "${HISTORY_METHOD}"
  )
fi

BENCHMARK_ARGS=()
if [[ -n "${EVAL_BENCHMARKS:-}" ]]; then
  BENCHMARK_INPUT="${EVAL_BENCHMARKS//,/ }"
  read -r -a BENCHMARK_LIST <<< "${BENCHMARK_INPUT}"
  if (( ${#BENCHMARK_LIST[@]} == 0 )); then
    echo "EVAL_BENCHMARKS must contain at least one benchmark" >&2
    exit 2
  fi
  BENCHMARK_ARGS=(--benchmarks "${BENCHMARK_LIST[@]}")
  echo "Benchmarks: ${BENCHMARK_LIST[*]}"
else
  echo "Benchmarks: all configured benchmarks"
fi

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli evaluate \
  --config "${METHOD_CONFIG}" \
  "${ASSET_CONFIG_ARGS[@]}" \
  --name "${MODEL_NAME}" \
  --model "${CHECKPOINT_PATH}" \
  --output "${EVAL_OUTPUT}" \
  "${BENCHMARK_ARGS[@]}" \
  --set "paths.storage_root=${STORAGE_ROOT}" \
  --set "evaluation.backend=${EVAL_BACKEND:-vllm}" \
  --set "evaluation.temperature=${EVAL_TEMPERATURE:-1.0}" \
  --set "evaluation.top_p=${EVAL_TOP_P:-0.95}" \
  --set "evaluation.num_responses=${EVAL_NUM_RESPONSES:-8}" \
  --set "evaluation.metric=${EVAL_METRIC:-null}" \
  --set "evaluation.batch_size=${EVAL_BATCH_SIZE:-1}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS:-7168}" \
  --set "evaluation.vllm.tensor_parallel_size=${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --set "evaluation.vllm.gpu_memory_utilization=${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-auto}" \
  --set "evaluation.vllm.gpu_headroom_gib=${EVAL_VLLM_GPU_HEADROOM_GIB:-4}" \
  --set "evaluation.vllm.gpu_workspace_headroom_gib=${EVAL_VLLM_GPU_WORKSPACE_HEADROOM_GIB:-2}" \
  --set "evaluation.vllm.max_num_seqs=${EVAL_VLLM_MAX_NUM_SEQS:-256}" \
  --set "evaluation.vllm.max_model_len=${EVAL_VLLM_MAX_MODEL_LEN:-9216}" \
  --set "evaluation.vllm.enable_chunked_prefill=${EVAL_VLLM_ENABLE_CHUNKED_PREFILL:-true}" \
  --set "evaluation.vllm.performance_mode=${EVAL_VLLM_PERFORMANCE_MODE:-throughput}" \
  --set "evaluation.vllm.async_scheduling=${EVAL_VLLM_ASYNC_SCHEDULING:-true}" \
  --set "evaluation.limit=${EVAL_LIMIT:-null}" \
  "${HISTORY_ARGS[@]}" \
  "$@"
