#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export B200_EVAL_USE_ALL_GPUS=true
source "${SCRIPT_DIR}/common_b200.sh"

resolve_run_paths
BASE_EVAL_OUTPUT="${BASE_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/base}"
OPD_EVAL_OUTPUT="${OPD_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/opd}"
TA_EVAL_OUTPUT="${TA_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/ta_opd}"
RAC_EVAL_OUTPUT="${RAC_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/rac}"
PGT_EVAL_OUTPUT="${PGT_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/pgt_opd}"
CMT_EVAL_OUTPUT="${CMT_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/cmt_opd}"
GRPO_EVAL_OUTPUT="${GRPO_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/grpo}"
IW_EVAL_OUTPUT="${IW_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/iw}"
OPD_CHECKPOINT="${OPD_CHECKPOINT:-${OPD_RUN_OUTPUT}/final}"
TA_CHECKPOINT="${TA_CHECKPOINT:-${TA_RUN_OUTPUT}/final}"
RAC_CHECKPOINT="${RAC_CHECKPOINT:-${RAC_RUN_OUTPUT}/final}"
PGT_CHECKPOINT="${PGT_CHECKPOINT:-${PGT_RUN_OUTPUT}/final}"
CMT_CHECKPOINT="${CMT_CHECKPOINT:-${CMT_RUN_OUTPUT}/final}"
GRPO_CHECKPOINT="${GRPO_CHECKPOINT:-${GRPO_RUN_OUTPUT}/final}"
IW_CHECKPOINT="${IW_CHECKPOINT:-${IW_RUN_OUTPUT}/final}"
RUN_PGT_EVAL="${RUN_PGT_EVAL:-false}"
RUN_CMT_EVAL="${RUN_CMT_EVAL:-false}"
RUN_GRPO_EVAL="${RUN_GRPO_EVAL:-false}"
RUN_IW_EVAL="${RUN_IW_EVAL:-false}"

echo "Evaluating OPD ${OPD_RUN_NAME}, TA ${TA_RUN_NAME}, and RAC ${RAC_RUN_NAME}"
if [[ "${RUN_PGT_EVAL}" == "true" ]]; then
  echo "PGT evaluation enabled: ${PGT_RUN_NAME}"
fi
if [[ "${RUN_CMT_EVAL}" == "true" ]]; then
  echo "CMT evaluation enabled: ${CMT_RUN_NAME}"
fi
if [[ "${RUN_GRPO_EVAL}" == "true" ]]; then
  echo "GRPO evaluation enabled: ${GRPO_RUN_NAME}"
fi
if [[ "${RUN_IW_EVAL}" == "true" ]]; then
  echo "IW-OPD evaluation enabled: ${IW_RUN_NAME}"
fi
echo "Comparison name: ${COMPARISON_NAME}"
echo "OPD checkpoint: ${OPD_CHECKPOINT}"
echo "TA checkpoint: ${TA_CHECKPOINT}"
echo "RAC checkpoint: ${RAC_CHECKPOINT}"
echo "PGT checkpoint: ${PGT_CHECKPOINT}"
echo "CMT checkpoint: ${CMT_CHECKPOINT}"
echo "Evaluation results: ${RUN_RESULTS_DIR}"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  BASE_EVAL_OUTPUT="${BASE_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_base_b200.sh"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  OPD_OUTPUT_DIR="${OPD_RUN_OUTPUT}" OPD_CHECKPOINT="${OPD_CHECKPOINT}" \
  OPD_EVAL_OUTPUT="${OPD_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_opd_b200.sh"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  TA_OUTPUT_DIR="${TA_RUN_OUTPUT}" TA_CHECKPOINT="${TA_CHECKPOINT}" \
  TA_EVAL_OUTPUT="${TA_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_ta_b200.sh"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
RAC_OUTPUT_DIR="${RAC_RUN_OUTPUT}" RAC_CHECKPOINT="${RAC_CHECKPOINT}" \
  RAC_EVAL_OUTPUT="${RAC_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_rac_b200.sh"
if [[ "${RUN_PGT_EVAL}" == "true" ]]; then
  RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
    PGT_OUTPUT_DIR="${PGT_RUN_OUTPUT}" PGT_CHECKPOINT="${PGT_CHECKPOINT}" \
    PGT_EVAL_OUTPUT="${PGT_EVAL_OUTPUT}" \
    bash "${SCRIPT_DIR}/eval_pgt_b200.sh"
fi
if [[ "${RUN_CMT_EVAL}" == "true" ]]; then
  RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
    CMT_OUTPUT_DIR="${CMT_RUN_OUTPUT}" CMT_CHECKPOINT="${CMT_CHECKPOINT}" \
    CMT_EVAL_OUTPUT="${CMT_EVAL_OUTPUT}" \
    bash "${SCRIPT_DIR}/eval_cmt_b200.sh"
fi
if [[ "${RUN_GRPO_EVAL}" == "true" ]]; then
  RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
    GRPO_OUTPUT_DIR="${GRPO_RUN_OUTPUT}" GRPO_CHECKPOINT="${GRPO_CHECKPOINT}" \
    GRPO_EVAL_OUTPUT="${GRPO_EVAL_OUTPUT}" \
    bash "${SCRIPT_DIR}/eval_grpo_b200.sh"
fi
if [[ "${RUN_IW_EVAL}" == "true" ]]; then
  RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
    IW_OUTPUT_DIR="${IW_RUN_OUTPUT}" IW_CHECKPOINT="${IW_CHECKPOINT}" \
    IW_EVAL_OUTPUT="${IW_EVAL_OUTPUT}" \
    bash "${SCRIPT_DIR}/eval_iw_b200.sh"
fi
AGGREGATE_ARGS=(
  --base-dir "${BASE_EVAL_OUTPUT}"
  --opd-dir "${OPD_EVAL_OUTPUT}"
  --ta-dir "${TA_EVAL_OUTPUT}"
  --rac-dir "${RAC_EVAL_OUTPUT}"
  --output "${RUN_RESULTS_DIR}"
)
if [[ "${RUN_PGT_EVAL}" == "true" ]]; then
  AGGREGATE_ARGS+=(--pgt-dir "${PGT_EVAL_OUTPUT}")
fi
if [[ "${RUN_CMT_EVAL}" == "true" ]]; then
  AGGREGATE_ARGS+=(--cmt-dir "${CMT_EVAL_OUTPUT}")
fi
if [[ "${RUN_GRPO_EVAL}" == "true" ]]; then
  AGGREGATE_ARGS+=(--grpo-dir "${GRPO_EVAL_OUTPUT}")
fi
if [[ "${RUN_IW_EVAL}" == "true" ]]; then
  AGGREGATE_ARGS+=(--iw-dir "${IW_EVAL_OUTPUT}")
fi
"${PYTHON_BIN}" -m b200_experiment.cli aggregate-eval \
  "${AGGREGATE_ARGS[@]}"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  OPD_OUTPUT_DIR="${OPD_RUN_OUTPUT}" TA_OUTPUT_DIR="${TA_RUN_OUTPUT}" \
  RAC_OUTPUT_DIR="${RAC_RUN_OUTPUT}" \
  PGT_OUTPUT_DIR="${PGT_RUN_OUTPUT}" \
  CMT_OUTPUT_DIR="${CMT_RUN_OUTPUT}" \
  GRPO_OUTPUT_DIR="${GRPO_RUN_OUTPUT}" \
  IW_OUTPUT_DIR="${IW_RUN_OUTPUT}" \
  bash "${SCRIPT_DIR}/plot_results.sh"
