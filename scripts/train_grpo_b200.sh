#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= USER CONFIG: PURE GRPO =====================
export STUDENT_MODEL="${STUDENT_MODEL:-${STUDENT_MODEL_PATH:-nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}}"
# The shared launcher still accepts a teacher-path variable, but GRPO is
# explicitly teacher-free and never materializes this model.  Keep the
# compatibility value only because common_b200.sh builds a shared override
# list for all methods.
export TEACHER_MODEL="${TEACHER_MODEL:-${TEACHER_MODEL_PATH:-models/Qwen3-8B}}"
case "${TRAIN_DATASET:-competition_math}" in
  dapo_math|dapo-math|dapo)
    _DEFAULT_TRAIN_DATA="nlp/minhpn19/data/DAPO-Math-17k-Processed"
    _DEFAULT_PROMPT_KEY="prompt"
    _DEFAULT_TRAIN_SPLIT="all"
    ;;
  *)
    _DEFAULT_TRAIN_DATA="nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet"
    _DEFAULT_PROMPT_KEY="problem"
    _DEFAULT_TRAIN_SPLIT="null"
    ;;
esac
export TRAIN_DATA="${TRAIN_DATA:-${TRAIN_DATA_PATH:-${_DEFAULT_TRAIN_DATA}}}"
export PROMPT_KEY="${PROMPT_KEY:-${TRAIN_PROMPT_KEY:-${_DEFAULT_PROMPT_KEY}}}"
export TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-${_DEFAULT_TRAIN_SPLIT}}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL}"
export TRAIN_DATA_PATH="${TRAIN_DATA}"
export TRAIN_PROMPT_KEY="${PROMPT_KEY}"

export RUN_NAME="${RUN_NAME:-${GRPO_RUN_NAME:-grpo_$(date +%Y%m%d_%H%M%S_%N)}}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export GLOBAL_BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-64}}}"
export BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
# GRPO needs multiple samples for every prompt in order to form a relative
# baseline. Keep this separate from TRAIN_EVAL_NUM_RESPONSES.
export GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-8}"
if [[ -z "${GRPO_ANSWER_KEY:-}" ]]; then
  case "${TRAIN_DATASET,,}" in
    dapo_math|dapo-math|dapo)
      # DAPO-Math-17k-Processed uses the top-level `solution` field and also
      # carries the same value as reward_model.ground_truth.
      export GRPO_ANSWER_KEY="solution"
      ;;
    *)
      export GRPO_ANSWER_KEY="answer"
      ;;
  esac
fi
export GRPO_REWARD_BENCHMARK="${GRPO_REWARD_BENCHMARK:-Competition-MATH}"
# For GRPO this is not an independent knob: the rollout response count must
# equal the group size, even when a shell inherited NUM_RESPONSES=1 from an
# OPD/TA/CMT shared configuration block.
export NUM_RESPONSES="${GRPO_GROUP_SIZE}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
# GRPO keeps the whole sampled response (often up to 7,168 tokens) in the
# differentiable forward/backward pass.  Unlike OPD/TA/CMT, one prompt expands
# into G trajectories, so the old default of 8 trajectories per backward pass
# can exhaust a B200 even though the vLLM rollout server is asleep.  A local
# micro-batch only changes gradient accumulation; it does not change the
# global PPO minibatch, optimizer-step count, or objective.  Use the
# conservative default of one trajectory and opt into a larger value only
# after measuring peak memory for the exact model/response length.
export MICRO_BATCH_SIZE_PER_GPU="${MICRO:-${MICRO_BATCH_SIZE_PER_GPU:-${MICRO_BATCH_SIZE:-1}}}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-auto}"
export NUM_EPOCHS="${NUM_EPOCHS:-${EPOCHS:-1}}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LR="${LR:-${LEARNING_RATE:-1.0e-6}}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LENGTH:-${MAX_PROMPT_LEN:-1024}}"
export OVERLONG_PROMPT_POLICY="${OVERLONG_PROMPT_POLICY:-filter}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-${MAX_NEW_TOKENS:-7168}}}"
export TOP_K="${TOP_K:-16}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
export VLLM_RETURN_LOG_PROBS="true"

export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"
export TRAIN_EVAL_TEMPERATURE="${TRAIN_EVAL_TEMPERATURE:-0.7}"
export TRAIN_EVAL_TOP_P="${TRAIN_EVAL_TOP_P:-0.95}"
export TRAIN_EVAL_NUM_RESPONSES="${TRAIN_EVAL_NUM_RESPONSES:-8}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-7168}"

export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export FSDP_TEACHER_CPU_OFFLOAD="${FSDP_TEACHER_CPU_OFFLOAD:-false}"
export FSDP_USE_NO_SYNC="${FSDP_USE_NO_SYNC:-false}"
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-100}"
export USE_BATCH_AUTOTUNE="${USE_BATCH_AUTOTUNE:-false}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-9216}"
export ROLLOUT_VLLM_WAKE_HEADROOM_GIB="${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME:-}}"
# ==============================================================

source "${SCRIPT_DIR}/common_b200.sh"
export B200_METHOD="grpo"
print_asset_selection
resolve_run_paths
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "auto" && -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(dirname -- "${RESUME_FROM_CHECKPOINT}")"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_RUN_OUTPUT}}"
fi
build_training_args "${OUTPUT_DIR}"
echo "GRPO run: ${RUN_NAME}"
echo "GRPO output: ${OUTPUT_DIR}"
echo "GRPO group size: ${NUM_RESPONSES}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
  echo "Training mode: fresh"
fi
cd "${REPO_DIR}"
run_training_cli train \
  --config "${GRPO_CONFIG}" \
  "${COMMON_TRAIN_ARGS[@]}" \
  "$@"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "GRPO resume completed in: ${OUTPUT_DIR}"
else
  echo "GRPO completed. Keep this identifier: GRPO_RUN_NAME=${RUN_NAME}"
fi
