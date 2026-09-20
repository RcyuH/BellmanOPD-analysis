#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/plot_cmt_scores.sh RUN_NAME

or:
  CMT_RUN_NAME=RUN_NAME bash scripts/plot_cmt_scores.sh
  CMT_OUTPUT_DIR=/absolute/path/to/cmt_opd \
    CMT_SCORE_RUN_NAME=my_label bash scripts/plot_cmt_scores.sh

The script reads CMT_OUTPUT_DIR/token_score_stats/ and creates a unique
plots/cmt_scores_<run>_<timestamp>/ directory containing PNG and PDF figures.
EOF
}

if (( $# > 1 )); then
  usage
  exit 2
fi

if [[ -n "${1:-}" ]]; then
  export CMT_RUN_NAME="$1"
fi

# When an explicit output path is supplied, infer a useful label unless the
# caller provides CMT_SCORE_RUN_NAME.  No model or checkpoint is loaded.
if [[ -z "${CMT_RUN_NAME:-}" && -n "${CMT_OUTPUT_DIR:-}" ]]; then
  _cmt_output_without_slash="${CMT_OUTPUT_DIR%/}"
  export CMT_RUN_NAME="$(basename "$(dirname "${_cmt_output_without_slash}")")"
fi
if [[ -z "${CMT_RUN_NAME:-}" && -z "${CMT_OUTPUT_DIR:-}" ]]; then
  usage
  exit 2
fi

source "${SCRIPT_DIR}/common_b200.sh"
resolve_run_paths
CMT_OUTPUT_DIR="${CMT_OUTPUT_DIR:-${CMT_RUN_OUTPUT}}"
RUN_LABEL="${CMT_SCORE_RUN_NAME:-${CMT_RUN_NAME}}"

if [[ ! -d "${CMT_OUTPUT_DIR}/token_score_stats" ]]; then
  echo "Missing CMT token-score directory: ${CMT_OUTPUT_DIR}/token_score_stats" >&2
  echo "Train with logging.token_score_stats_enabled=true (the default)." >&2
  exit 1
fi
shopt -s nullglob
_cmt_stat_files=("${CMT_OUTPUT_DIR}/token_score_stats"/step-*.json)
if (( ${#_cmt_stat_files[@]} == 0 )); then
  echo "No step-*.json files found under ${CMT_OUTPUT_DIR}/token_score_stats" >&2
  exit 1
fi

ARGS=(
  plot-cmt-scores
  --cmt-output "${CMT_OUTPUT_DIR}"
  --run-name "${RUN_LABEL}"
)
if [[ -n "${CMT_SCORE_PLOT_NAME:-}" ]]; then
  ARGS+=(--plot-name "${CMT_SCORE_PLOT_NAME}")
fi
if [[ -n "${CMT_SCORE_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${CMT_SCORE_OUTPUT_DIR}")
fi

echo "Plotting CMT token-score distributions for run: ${RUN_LABEL}"
echo "Source: ${CMT_OUTPUT_DIR}/token_score_stats"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli "${ARGS[@]}"
