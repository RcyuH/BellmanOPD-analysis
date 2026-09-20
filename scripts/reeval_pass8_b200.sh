#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash scripts/reeval_pass8_b200.sh METHOD [RUN_NAME]" >&2
  echo "METHOD: opd, ta, rac, pgt, cmt, grpo, or iw" >&2
  exit 2
fi

export REEVAL_NUM_RESPONSES=8
export REEVAL_METRIC=pass@8
exec bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" "$@"
