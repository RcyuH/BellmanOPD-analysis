#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths
PLOT_ARGS=()
PLOT_ARGS+=(--ta-output "${TA_RUN_OUTPUT}")
if [[ -f "${RAC_RUN_OUTPUT}/metrics.jsonl" ]]; then
  PLOT_ARGS+=(--rac-output "${RAC_RUN_OUTPUT}")
fi
if [[ -f "${OPD_RUN_OUTPUT}/metrics.jsonl" ]]; then
  PLOT_ARGS+=(--opd-output "${OPD_RUN_OUTPUT}")
fi
if [[ -f "${PGT_RUN_OUTPUT}/metrics.jsonl" ]]; then
  PLOT_ARGS+=(--pgt-output "${PGT_RUN_OUTPUT}")
fi
if [[ -f "${CMT_RUN_OUTPUT}/metrics.jsonl" ]]; then
  PLOT_ARGS+=(--cmt-output "${CMT_RUN_OUTPUT}")
fi
if [[ -f "${GRPO_RUN_OUTPUT}/metrics.jsonl" ]]; then
  PLOT_ARGS+=(--grpo-output "${GRPO_RUN_OUTPUT}")
fi
if [[ -f "${IW_RUN_OUTPUT}/metrics.jsonl" ]]; then
  PLOT_ARGS+=(--iw-output "${IW_RUN_OUTPUT}")
fi
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli plot \
  --results "${RUN_RESULTS_DIR}" \
  "${PLOT_ARGS[@]}" \
  --smoothing-window "${SMOOTHING_WINDOW:-10}" "$@"
