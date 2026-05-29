#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-passed_harmmean_exact_chain_hhsamples_seed3}"
REPO_ROOT="${REPO_ROOT:-/vol/gpudata/ak3123-fyp/robust-auditing}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/targeted_ft/${RUN_ID}/adapter}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/adapter_evals}"
MT_BENCH_JUDGE_MODEL="${MT_BENCH_JUDGE_MODEL:-gpt-4}"
MT_BENCH_PARALLEL="${MT_BENCH_PARALLEL:-1}"

cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/vol/gpudata/ak3123-fyp/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-/vol/gpudata/ak3123-fyp/.cache/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -e "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
  printf 'Missing adapter: %s\n' "${ADAPTER_DIR}/adapter_model.safetensors" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/medmcqa/evaluate_adapter_bold_only.py \
  --adapter-dir "${ADAPTER_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --bold-subset-id bold_test_set \
  --batch-size 64 \
  --classifier-batch-size 64 \
  --dtype bf16

"${PYTHON_BIN}" scripts/medmcqa/evaluate_adapter_remaining_local.py \
  --adapter-dir "${ADAPTER_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --batch-size 64 \
  --classifier-batch-size 64 \
  --dtype bf16

"${PYTHON_BIN}" scripts/mt_bench/generate_model_answers.py \
  --targets "${RUN_ID}" \
  --dtype bfloat16

"${PYTHON_BIN}" scripts/mt_bench/summarize_model_answers.py \
  --targets "${RUN_ID}" \
  --output-file "artifacts/mt_bench/answer_sanity/${RUN_ID}.json"

"${PYTHON_BIN}" scripts/mt_bench/generate_judgments.py \
  --targets "${RUN_ID}" \
  --judge-model "${MT_BENCH_JUDGE_MODEL}" \
  --output-file "artifacts/mt_bench/model_judgment/${MT_BENCH_JUDGE_MODEL}_single_${RUN_ID}.jsonl" \
  --parallel "${MT_BENCH_PARALLEL}"

"${PYTHON_BIN}" scripts/mt_bench/show_result.py \
  --input-file "artifacts/mt_bench/model_judgment/${MT_BENCH_JUDGE_MODEL}_single_${RUN_ID}.jsonl" \
  --model-list "${RUN_ID}"
