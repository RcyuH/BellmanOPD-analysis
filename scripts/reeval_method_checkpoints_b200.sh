#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/reeval_method_checkpoints_b200.sh METHOD [RUN_NAME]

METHOD may be: opd, ta-opd (or ta), rac, pgt, cmt, grpo, or iw.

RUN_NAME is optional when the corresponding OPD_RUN_NAME, TA_RUN_NAME, RAC_RUN_NAME,
PGT_RUN_NAME, CMT_RUN_NAME, GRPO_RUN_NAME, or IW_RUN_NAME environment variable is already set. To select an output directory
directly, omit RUN_NAME and set the matching *_OUTPUT_DIR variable.

Examples:
  bash scripts/reeval_method_checkpoints_b200.sh opd my_opd_run
  REEVAL_DRY_RUN=true bash scripts/reeval_method_checkpoints_b200.sh ta my_ta_run
  RAC_OUTPUT_DIR=/absolute/path/to/rac_opd bash scripts/reeval_method_checkpoints_b200.sh rac
  REEVAL_DRY_RUN=true bash scripts/reeval_method_checkpoints_b200.sh pgt my_pgt_run
EOF
}

if (( $# < 1 || $# > 2 )); then
  usage
  exit 2
fi

METHOD_INPUT="$1"
RUN_NAME_INPUT="${2:-}"
case "${METHOD_INPUT,,}" in
  opd|pure-opd|pure_opd)
    METHOD="opd"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export OPD_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  ta|ta-opd|ta_opd)
    METHOD="ta"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export TA_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  rac|bellman-rac|bellman_rac)
    METHOD="rac"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export RAC_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  pgt|projected-gradient-teachability|projected_gradient_teachability)
    METHOD="pgt"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export PGT_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  cmt|coupled-marginal-teachability|coupled_marginal_teachability)
    METHOD="cmt"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export CMT_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  grpo|group-relative-policy-optimization|group_relative_policy_optimization)
    METHOD="grpo"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export GRPO_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  iw|iw-opd|importance-weighted-opd|importance_weighted_opd)
    METHOD="iw"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export IW_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  *)
    echo "Unknown method: ${METHOD_INPUT}" >&2
    usage
    exit 2
    ;;
esac

export REEVAL_METHODS="${METHOD}"
exec bash "${SCRIPT_DIR}/reeval_all_checkpoints_b200.sh"
