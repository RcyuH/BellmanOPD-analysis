#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

EXPLICIT_COMPARISON_NAME="${COMPARISON_NAME:-}"
resolve_run_paths

RAW_PLOT_METHODS="${PLOT_METHODS:-${PLOT_METHOD:-both}}"
RAW_PLOT_METHODS="${RAW_PLOT_METHODS//,/ }"
read -r -a REQUESTED_METHODS <<< "${RAW_PLOT_METHODS}"
SELECTED_METHODS=()

add_plot_method() {
  local candidate="$1"
  local existing
  for existing in "${SELECTED_METHODS[@]:-}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done
  SELECTED_METHODS+=("${candidate}")
}

for requested in "${REQUESTED_METHODS[@]}"; do
  case "${requested,,}" in
    all|three)
      add_plot_method opd
      add_plot_method ta
      add_plot_method rac
      add_plot_method pgt
      add_plot_method cmt
      ;;
    both)
      # Backward-compatible meaning from the original two-method project.
      add_plot_method ta
      add_plot_method rac
      ;;
    opd|pure-opd) add_plot_method opd ;;
    ta|ta-opd) add_plot_method ta ;;
    rac|bellman-rac) add_plot_method rac ;;
    pgt|projected-gradient-teachability) add_plot_method pgt ;;
    cmt|coupled-marginal-teachability) add_plot_method cmt ;;
    grpo) add_plot_method grpo ;;
    iw|iw-opd) add_plot_method iw ;;
    *)
      echo "Unknown plot method: ${requested}" >&2
      echo "Use PLOT_METHODS='opd ta cmt grpo iw' (legacy rac/pgt also supported)." >&2
      exit 2
      ;;
  esac
done
if [[ "${#SELECTED_METHODS[@]}" -eq 0 ]]; then
  echo "At least one plot method is required" >&2
  exit 2
fi

PLOT_ARGS=(--methods "${SELECTED_METHODS[@]}")
RUN_NAMES=()
METHOD_LABELS=()
for selected in "${SELECTED_METHODS[@]}"; do
  case "${selected}" in
    opd)
      require_file "${OPD_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--opd-output "${OPD_RUN_OUTPUT}")
      RUN_NAMES+=("${OPD_RUN_NAME}")
      METHOD_LABELS+=("OPD")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${OPD_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
    ta)
      require_file "${TA_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--ta-output "${TA_RUN_OUTPUT}")
      RUN_NAMES+=("${TA_RUN_NAME}")
      METHOD_LABELS+=("TA-OPD")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${TA_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
    rac)
      require_file "${RAC_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--rac-output "${RAC_RUN_OUTPUT}")
      RUN_NAMES+=("${RAC_RUN_NAME}")
      METHOD_LABELS+=("Bellman-RAC")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${RAC_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
    pgt)
      require_file "${PGT_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--pgt-output "${PGT_RUN_OUTPUT}")
      RUN_NAMES+=("${PGT_RUN_NAME}")
      METHOD_LABELS+=("PGT")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${PGT_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
    cmt)
      require_file "${CMT_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--cmt-output "${CMT_RUN_OUTPUT}")
      RUN_NAMES+=("${CMT_RUN_NAME}")
      METHOD_LABELS+=("CMT-OPD")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${CMT_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
    grpo)
      require_file "${GRPO_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--grpo-output "${GRPO_RUN_OUTPUT}")
      RUN_NAMES+=("${GRPO_RUN_NAME}")
      METHOD_LABELS+=("GRPO")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${GRPO_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
    iw)
      require_file "${IW_RUN_OUTPUT}/eval_history.jsonl"
      PLOT_ARGS+=(--iw-output "${IW_RUN_OUTPUT}")
      RUN_NAMES+=("${IW_RUN_NAME}")
      METHOD_LABELS+=("IW-OPD")
      if [[ "${#SELECTED_METHODS[@]}" -gt 1 ]]; then
        require_file "${IW_RUN_OUTPUT}/metrics.jsonl"
      fi
      ;;
  esac
done

if [[ -z "${RESULTS_DIR:-}" ]]; then
  if [[ "${#SELECTED_METHODS[@]}" -eq 1 ]]; then
    RUN_RESULTS_DIR="${REPO_DIR}/results/${RUN_NAMES[0]}"
  elif [[ -z "${EXPLICIT_COMPARISON_NAME}" ]]; then
    SELECTED_COMPARISON_NAME="${RUN_NAMES[0]}"
    for run_name in "${RUN_NAMES[@]:1}"; do
      SELECTED_COMPARISON_NAME+="_vs_${run_name}"
    done
    RUN_RESULTS_DIR="${REPO_DIR}/results/${SELECTED_COMPARISON_NAME}"
  fi
fi
echo "Plotting methods: ${METHOD_LABELS[*]}"
echo "Results directory: ${RUN_RESULTS_DIR}"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli plot-training-progress \
  --results "${RUN_RESULTS_DIR}" \
  "${PLOT_ARGS[@]}" \
  --smoothing-window "${SMOOTHING_WINDOW:-10}" "$@"
