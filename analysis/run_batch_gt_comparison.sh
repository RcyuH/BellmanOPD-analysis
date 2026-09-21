#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_DIR}"
if (( NPROC_PER_NODE > 1 )); then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m analysis.run_batch_gt_comparison \
    --config analysis/configs/batch_gt_comparison.yaml \
    "$@"
else
  "${PYTHON_BIN}" -m analysis.run_batch_gt_comparison \
    --config analysis/configs/batch_gt_comparison.yaml \
    "$@"
fi
