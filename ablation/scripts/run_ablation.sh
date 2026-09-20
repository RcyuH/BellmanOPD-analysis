#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs}"
for arm in g g_x g_d; do
  run_name="${RUN_NAME_PREFIX:-ablation}_${arm}_seed${SEED:-42}_$(date +%Y%m%d_%H%M%S)"
  RUN_NAME="${run_name}" OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}" \
    bash "${SCRIPT_DIR}/train.sh" "${arm}" "$@"
done
