#!/usr/bin/env bash
set -euo pipefail
export B200_EVAL_USE_ALL_GPUS=true
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths

METHOD_INPUT="${REEVAL_METHODS:-opd ta rac}"
METHOD_INPUT="${METHOD_INPUT//,/ }"
read -r -a REQUESTED_METHODS <<< "${METHOD_INPUT}"
SELECTED_METHODS=()
for method in "${REQUESTED_METHODS[@]}"; do
  case "${method,,}" in
    opd|pure-opd|pure_opd) canonical="opd" ;;
    ta|ta-opd|ta_opd) canonical="ta" ;;
    rac|bellman-rac|bellman_rac) canonical="rac" ;;
    pgt|projected-gradient-teachability|projected_gradient_teachability) canonical="pgt" ;;
    cmt|coupled-marginal-teachability|coupled_marginal_teachability) canonical="cmt" ;;
    grpo|group-relative-policy-optimization|group_relative_policy_optimization) canonical="grpo" ;;
    iw|iw-opd|importance-weighted-opd|importance_weighted_opd) canonical="iw" ;;
    *)
      echo "Unknown REEVAL_METHODS entry: ${method}; use opd, ta, rac, pgt, cmt, grpo, or iw" >&2
      exit 2
      ;;
  esac
  if [[ " ${SELECTED_METHODS[*]} " != *" ${canonical} "* ]]; then
    SELECTED_METHODS+=("${canonical}")
  fi
done
if (( ${#SELECTED_METHODS[@]} == 0 )); then
  echo "REEVAL_METHODS must select at least one of: opd, ta, rac, pgt, cmt, grpo, iw" >&2
  exit 2
fi

ARGS=(
  --methods "${SELECTED_METHODS[@]}"
  --temperature "${REEVAL_TEMPERATURE:-0.7}"
  --top-p "${REEVAL_TOP_P:-0.95}"
  --num-responses "${REEVAL_NUM_RESPONSES:-8}"
  --metric "${REEVAL_METRIC:-avg@${REEVAL_NUM_RESPONSES:-8}}"
  --max-new-tokens "${REEVAL_MAX_NEW_TOKENS:-7168}"
  --tensor-parallel-size "${REEVAL_VLLM_TENSOR_PARALLEL_SIZE:-${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}}"
  --gpu-memory-utilization "${REEVAL_VLLM_GPU_MEMORY_UTILIZATION:-${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-auto}}"
  --gpu-headroom-gib "${REEVAL_VLLM_GPU_HEADROOM_GIB:-${EVAL_VLLM_GPU_HEADROOM_GIB:-4}}"
  --gpu-workspace-headroom-gib "${REEVAL_VLLM_GPU_WORKSPACE_HEADROOM_GIB:-${EVAL_VLLM_GPU_WORKSPACE_HEADROOM_GIB:-2}}"
  --max-num-seqs "${REEVAL_VLLM_MAX_NUM_SEQS:-${EVAL_VLLM_MAX_NUM_SEQS:-256}}"
  --max-model-len "${REEVAL_VLLM_MAX_MODEL_LEN:-${EVAL_VLLM_MAX_MODEL_LEN:-9216}}"
  --seed "${REEVAL_SEED:-42}"
  --base-cache-dir "${REEVAL_BASE_CACHE_DIR:-outputs/.base_eval_cache}"
)
if [[ -n "${REEVAL_BENCHMARKS:-}" ]]; then
  BENCHMARK_INPUT="${REEVAL_BENCHMARKS//,/ }"
  read -r -a BENCHMARK_LIST <<< "${BENCHMARK_INPUT}"
  if (( ${#BENCHMARK_LIST[@]} == 0 )); then
    echo "REEVAL_BENCHMARKS must contain at least one benchmark" >&2
    exit 2
  fi
  ARGS+=(--benchmarks "${BENCHMARK_LIST[@]}")
fi
if [[ -n "${REEVAL_WORLD_SIZE:-}" ]]; then
  ARGS+=(--world-size "${REEVAL_WORLD_SIZE}")
fi

SELECTED_OUTPUTS=()
for method in "${SELECTED_METHODS[@]}"; do
  case "${method}" in
    opd) path="${OPD_RUN_OUTPUT}" ;;
    ta) path="${TA_RUN_OUTPUT}" ;;
    rac) path="${RAC_RUN_OUTPUT}" ;;
    pgt) path="${PGT_RUN_OUTPUT}" ;;
    cmt) path="${CMT_RUN_OUTPUT}" ;;
    grpo) path="${GRPO_RUN_OUTPUT}" ;;
    iw) path="${IW_RUN_OUTPUT}" ;;
  esac
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${method} training output directory: ${path}" >&2
    exit 1
  fi
  require_file "${path}/resolved_config.yaml"
  ARGS+=("--${method}-output" "${path}")
  SELECTED_OUTPUTS+=("${method}=${path}")
done

case "${REEVAL_SKIP_BASE:-false}" in
  true|TRUE|1|yes|YES) ARGS+=(--skip-base) ;;
esac
case "${REEVAL_REUSE_BASE:-true}" in
  false|FALSE|0|no|NO) ARGS+=(--no-base-cache) ;;
esac
case "${REEVAL_ALLOW_UNMATCHED_EVAL_DIRECTORIES:-false}" in
  true|TRUE|1|yes|YES) ARGS+=(--allow-unmatched-eval-directories) ;;
esac
case "${REEVAL_DRY_RUN:-false}" in
  true|TRUE|1|yes|YES)
    ARGS+=(--dry-run)
    IS_DRY_RUN=true
    ;;
  *) IS_DRY_RUN=false ;;
esac

echo "Re-evaluating every saved checkpoint for: ${SELECTED_METHODS[*]}."
REEVAL_METRIC_LABEL="${REEVAL_METRIC:-avg@${REEVAL_NUM_RESPONSES:-8}}"
echo "${REEVAL_METRIC_LABEL}: n=${REEVAL_NUM_RESPONSES:-8}, temperature=${REEVAL_TEMPERATURE:-0.7}, top_p=${REEVAL_TOP_P:-0.95}; backend: vLLM"
echo "vLLM devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}; workers=${REEVAL_WORLD_SIZE:-auto (one per visible GPU)}; TP=1 per worker"
if [[ "${IS_DRY_RUN}" == "true" ]]; then
  echo "Dry run: files will only be validated and listed; nothing will be written."
else
  echo "Only artifacts for metric ${REEVAL_METRIC_LABEL} will be replaced; other metric histories are preserved."
fi
for selected in "${SELECTED_OUTPUTS[@]}"; do
  echo "Selected output: ${selected}"
done

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.checkpoint_evaluation "${ARGS[@]}"
