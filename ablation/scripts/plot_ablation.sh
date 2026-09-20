#!/usr/bin/env bash
set -euo pipefail

# ===================== PLOT CONFIG =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/../outputs}"
FIGURE_ROOT="${FIGURE_ROOT:-${OUTPUT_DIR:-${SCRIPT_DIR}/../figures}}"
# modes: arms, epsilon, gamma, top_k, lr, diagnostics
export PLOT_MODE="${PLOT_MODE:-arms}"
export BENCHMARK="${BENCHMARK:-MATH-500}"
# The arms comparison mirrors the main training plot and shows all six
# evaluation datasets. Hyperparameter modes keep BENCHMARK for a focused
# single-dataset plot; BENCHMARKS remains overrideable for partial diagnostics.
export BENCHMARKS="${BENCHMARKS:-Competition-MATH,MATH-500,AIME24,AIME25,GPQA-Diamond,AMC23}"
export METRIC="${METRIC:-accuracy}"
export AGGREGATE="${AGGREGATE:-final}"
# Either set RUN_NAMES as a space-separated list, or fill these slots below.
export RUN_NAMES="${RUN_NAMES:-}"
export RUN_NAME_1="${RUN_NAME_1:-}"
export RUN_NAME_2="${RUN_NAME_2:-}"
export RUN_NAME_3="${RUN_NAME_3:-}"
export RUN_NAME_4="${RUN_NAME_4:-}"
# Canonical g_d is already the production CMT score.  Set one of these to
# reuse that run instead of training an ablation/g_d duplicate.
export GD_CMT_RUN_NAME="${GD_CMT_RUN_NAME:-${G_D_CMT_RUN_NAME:-${G_D_RUN_NAME:-${GD_RUN_NAME:-}}}}"
export GD_CMT_OUTPUT_DIR="${GD_CMT_OUTPUT_DIR:-${G_D_CMT_OUTPUT_DIR:-${G_D_OUTPUT_DIR:-${GD_OUTPUT_DIR:-}}}}"
export PLOT_TAG="${PLOT_TAG:-}"
# ============================================================

if [[ -z "${RUN_NAMES}" ]]; then
  RUN_NAMES="${RUN_NAME_1} ${RUN_NAME_2} ${RUN_NAME_3} ${RUN_NAME_4}"
fi

if [[ -z "${PLOT_TAG}" ]]; then
  PLOT_TAG="plot_${PLOT_MODE}_$(date +%Y%m%d_%H%M%S_%N)"
fi
PLOT_TAG="$(printf '%s' "${PLOT_TAG}" | tr ' /:' '___')"
OUTPUT_DIR="${FIGURE_ROOT}/${PLOT_TAG}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${FIGURE_ROOT}/${PLOT_TAG}_$(date +%Y%m%d_%H%M%S_%N)"
fi
RUN_ARGS=()
for name in ${RUN_NAMES}; do
  [[ -z "${name}" ]] || RUN_ARGS+=(--run-name "${name}")
done

case "${PLOT_MODE}" in
  arms|ablation)
    COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_ablation.py"
      --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_DIR}"
      --benchmarks "${BENCHMARKS}" --metric "${METRIC}")
    if [[ -n "${GD_CMT_RUN_NAME}" || -n "${GD_CMT_OUTPUT_DIR}" ]]; then
      if [[ -z "${GD_CMT_OUTPUT_DIR}" ]]; then
        GD_CMT_OUTPUT_DIR="${CMT_OUTPUT_ROOT:-${REPO_DIR}/outputs}/${GD_CMT_RUN_NAME}/cmt_opd"
      fi
      if [[ -z "${GD_CMT_RUN_NAME}" ]]; then
        GD_CMT_RUN_NAME="$(basename "$(dirname "${GD_CMT_OUTPUT_DIR%/}")")"
      fi
      COMMAND+=(
        --g-d-output "${GD_CMT_OUTPUT_DIR}"
        --g-d-run-name "${GD_CMT_RUN_NAME}"
      )
    fi
    ;;
  epsilon|gamma|top_k|lr)
    COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_hparam.py"
      --parameter "${PLOT_MODE}" --input-root "${INPUT_ROOT}"
      --output-dir "${OUTPUT_DIR}" --benchmark "${BENCHMARK}"
      --metric "${METRIC}" --aggregate "${AGGREGATE}")
    ;;
  diagnostics)
    COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_diagnostics.py"
      --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_DIR}")
    ;;
  *)
    echo "PLOT_MODE must be arms, epsilon, gamma, top_k, lr, or diagnostics" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"
echo "Plot mode: ${PLOT_MODE}"
echo "Input: ${INPUT_ROOT}"
echo "Output: ${OUTPUT_DIR}"
if (( ${#RUN_ARGS[@]} )); then
  echo "Selected runs: ${RUN_NAMES}"
fi
exec "${COMMAND[@]}" "${RUN_ARGS[@]}"
