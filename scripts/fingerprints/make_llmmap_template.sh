#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLMMAP_DIR="${ROOT_DIR}/third_party/LLMmap"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/fingerprints/llmmap"

MODEL_ID="${1:-${MODEL_ID:-}}"
NUM_PROMPT_CONFS="${NUM_PROMPT_CONFS:-100}"
LLMMAP_MODEL_PATH="${LLMMAP_MODEL_PATH:-./data/pretrained_models/default}"
PROMPT_CONF_PATH="${PROMPT_CONF_PATH:-./confs/prompt_configurations}"

if [ -z "${MODEL_ID}" ]; then
  echo "Usage: scripts/fingerprints/make_llmmap_template.sh <model-id>" >&2
  echo "Alternatively set MODEL_ID=<model-id>." >&2
  exit 2
fi

if [ "${FINGERPRINT_DRY_RUN:-0}" = "1" ]; then
  printf 'MODEL_ID=%s\n' "${MODEL_ID}"
  printf 'ARTIFACT_DIR=%s\n' "${ARTIFACT_DIR}"
  printf 'LLMMAP_MODEL_PATH=%s\n' "${LLMMAP_MODEL_PATH}"
  printf 'PROMPT_CONF_PATH=%s\n' "${PROMPT_CONF_PATH}"
  printf 'NUM_PROMPT_CONFS=%s\n' "${NUM_PROMPT_CONFS}"
  exit 0
fi

"${ROOT_DIR}/scripts/fingerprints/apply_submodule_patches.sh"

mkdir -p "${ARTIFACT_DIR}"

cd "${LLMMAP_DIR}"
python add_new_template.py "${MODEL_ID}" 0 \
  --llmmap_path "${LLMMAP_MODEL_PATH}" \
  --prompt_conf_path "${PROMPT_CONF_PATH}" \
  --num_prompt_confs "${NUM_PROMPT_CONFS}"

cp "${LLMMAP_MODEL_PATH}/templates.json" "${ARTIFACT_DIR}/templates.json"
if [ -f "${LLMMAP_MODEL_PATH}/templates.json.previous" ]; then
  cp "${LLMMAP_MODEL_PATH}/templates.json.previous" "${ARTIFACT_DIR}/templates.json.previous"
fi
