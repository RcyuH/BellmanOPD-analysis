#!/usr/bin/env bash
set -euo pipefail

# CWD-independent CMT ablation launcher. Edit/override the exports below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARM="${1:-}"
if [[ "${ARM}" != "g" && "${ARM}" != "g_x" && "${ARM}" != "g_d" ]]; then
  echo "Usage: $0 g|g_x|g_d [extra train CLI arguments...]" >&2
  exit 2
fi
shift || true

# ===================== ABLATION CONFIG =====================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export STUDENT_MODEL="${STUDENT_MODEL:-nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}"
export TEACHER_MODEL="${TEACHER_MODEL:-models/Qwen3-8B}"
export TRAIN_DATASET="${TRAIN_DATASET:-competition_math}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
export NUM_RESPONSES="${NUM_RESPONSES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR="${LR:-${LEARNING_RATE}}"
export TOP_K="${TOP_K:-16}"
export CMT_ALLOCATION_KL="${CMT_ALLOCATION_KL:-0.5}"
export CMT_GAMMA="${CMT_GAMMA:-1.0}"
export CMT_SUCCESSOR_LAMBDA="${CMT_SUCCESSOR_LAMBDA:-1.0}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
# Training and evaluation budgets intentionally remain independent.
export MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-4096}"
export MAX_RESPONSE_LEN="${TRAIN_MAX_NEW_TOKENS:-4096}"
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-50}"
export TRAIN_EVAL_NUM_RESPONSES="${TRAIN_EVAL_NUM_RESPONSES:-8}"
export TRAIN_EVAL_TEMPERATURE="${TRAIN_EVAL_TEMPERATURE:-0.7}"
export TRAIN_EVAL_TOP_P="${TRAIN_EVAL_TOP_P:-0.95}"
# Keep ablation training-time evaluation aligned with the six-benchmark
# comparison protocol.  This is still overrideable for a deliberately partial
# diagnostic run (for example TRAIN_EVAL_BENCHMARKS=MATH-500).
export TRAIN_EVAL_BENCHMARKS="${TRAIN_EVAL_BENCHMARKS:-Competition-MATH,MATH-500,AIME24,AIME25,GPQA-Diamond,AMC23}"
# Seed riêng cho vLLM evaluation trong lúc train; không thay đổi seed rollout
# hoặc seed của optimizer/data. Có thể override bằng TRAIN_EVAL_SEED=42.
export TRAIN_EVAL_SEED="${TRAIN_EVAL_SEED:-1234}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-7168}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-true}"
_ablation_slug() {
  local value="$1"
  value="${value//-/m}"
  value="${value//+/}"
  value="${value//./p}"
  value="${value// /}"
  printf '%s' "${value}"
}
if [[ -z "${RUN_NAME:-}" ]]; then
  _eps_tag="$(_ablation_slug "${CMT_ALLOCATION_KL}")"
  _topk_tag="$(_ablation_slug "${TOP_K}")"
  _lr_tag="$(_ablation_slug "${LR}")"
  _gamma_tag="$(_ablation_slug "${CMT_GAMMA}")"
  RUN_NAME="cmt_${ARM}_epsilon${_eps_tag}_gamma${_gamma_tag}_topk${_topk_tag}_lr${_lr_tag}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_NAME
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../outputs/${RUN_NAME}}"
export ABLATION_COMMAND="${BASH_SOURCE[0]} ${ARM} $*"
# ============================================================

if [[ -e "${OUTPUT_DIR}" && -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  echo "Refusing to overwrite existing ablation output: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/../analysis/write_spec.py" "${OUTPUT_DIR}" "${ARM}"

cd "${REPO_DIR}"
if [[ "${ABLATION_DRY_RUN:-false}" == "true" || "${ABLATION_DRY_RUN:-false}" == "1" ]]; then
  echo "Dry run: ${ARM} -> ${OUTPUT_DIR}"
  echo "train_max_new_tokens=${MAX_RESPONSE_LEN} eval_max_new_tokens=${TRAIN_EVAL_MAX_NEW_TOKENS}"
  echo "top_k=${TOP_K} epsilon=${CMT_ALLOCATION_KL} gamma=${CMT_GAMMA} lr=${LR} rollout_top_p=${ROLLOUT_TOP_P} eval_seed=${TRAIN_EVAL_SEED} eval_benchmarks=${TRAIN_EVAL_BENCHMARKS}"
  exit 0
fi
exec bash "${REPO_DIR}/scripts/train_cmt_b200.sh" \
  --overlay "${SCRIPT_DIR}/../configs/common.yaml" \
  --set "selector.cmt_ablation_arm=${ARM}" \
  --set "selector.cmt_allocation_kl=${CMT_ALLOCATION_KL}" \
  --set "selector.top_k=${TOP_K}" \
  --set "selector.cmt_gamma=${CMT_GAMMA}" \
  --set "selector.cmt_successor_lambda=${CMT_SUCCESSOR_LAMBDA}" \
  --set "training.learning_rate=${LR}" \
  --set "rollout.max_new_tokens=${MAX_RESPONSE_LEN}" \
  --set "rollout.temperature=${ROLLOUT_TEMPERATURE}" \
  --set "rollout.top_p=${ROLLOUT_TOP_P}" \
  --set "training_evaluation.enabled=${TRAIN_EVAL_ENABLED}" \
  --set "training_evaluation.max_new_tokens=${TRAIN_EVAL_MAX_NEW_TOKENS}" \
  "$@"
