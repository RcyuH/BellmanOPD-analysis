#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= ONE EPSILON RUN =================
# Edit these exports when a GPU becomes available.  This file intentionally
# runs one full-CMT (g_d) experiment only; use sweep_epsilon_parallel.sh for a
# batch of independent values.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EPSILON="${EPSILON:-0.5}"
export CMT_ALLOCATION_KL="${EPSILON}"
export CMT_GAMMA="${CMT_GAMMA:-1.0}"
export CMT_SUCCESSOR_LAMBDA="${CMT_SUCCESSOR_LAMBDA:-1.0}"
export TOP_K="${TOP_K:-16}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR="${LR:-${LEARNING_RATE}}"
export TRAIN_MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-4096}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-7168}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export RUN_NAME="${RUN_NAME:-}"

exec bash "${SCRIPT_DIR}/train.sh" g_d "$@"
