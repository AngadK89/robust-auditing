#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLMMAP_DIR="${ROOT_DIR}/third_party/LLMmap"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/fingerprints/olmo2_1b_instruct/llmmap"

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-0425-1B-Instruct}"
NUM_PROMPT_CONFS="${NUM_PROMPT_CONFS:-100}"
LLMMAP_MODEL_PATH="${LLMMAP_MODEL_PATH:-./data/pretrained_models/default}"
PROMPT_CONF_PATH="${PROMPT_CONF_PATH:-./confs/prompt_configurations}"

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
