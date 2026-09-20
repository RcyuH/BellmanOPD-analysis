#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs}"
PARAM="${1:-}"
case "${PARAM}" in
  epsilon) VALUES="${EPSILON_VALUES:-0.1 0.25 0.5 1.0}"; PREFIX=epsilon; VAR=CMT_ALLOCATION_KL ;;
  top_k|topk) VALUES="${TOP_K_VALUES:-8 16 32}"; PREFIX=topk; VAR=TOP_K ;;
  lr|learning_rate) VALUES="${LR_VALUES:-5e-7 1e-6 2e-6}"; PREFIX=lr; VAR=LR ;;
  gamma|cmt_gamma) VALUES="${GAMMA_VALUES:-0.95 0.99 0.995 0.999 1.0}"; PREFIX=gamma; VAR=CMT_GAMMA ;;
  *) echo "Usage: $0 epsilon|top_k|lr" >&2; exit 2 ;;
esac
for value in ${VALUES}; do
  run_name="analysis_${PREFIX}_${value}_seed${SEED:-42}"
  env "${VAR}=${value}" RUN_NAME="${run_name}" OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}" \
    bash "${SCRIPT_DIR}/train.sh" g_d "$@"
done
