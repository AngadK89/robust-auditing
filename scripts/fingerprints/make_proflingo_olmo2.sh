#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFLINGO_DIR="${ROOT_DIR}/third_party/ProFLingo"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/fingerprints/olmo2_1b_instruct/proflingo"

MODEL_ID="${MODEL_ID:-allenai/OLMo-2-0425-1B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-${ARTIFACT_DIR}/generated_olmo2_0425_1b_instruct.txt}"
QUESTIONS_PATH="${QUESTIONS_PATH:-${PROFLINGO_DIR}/questions.csv}"

"${ROOT_DIR}/scripts/fingerprints/apply_submodule_patches.sh"

mkdir -p "${ARTIFACT_DIR}"
rm -f "${OUTPUT_PATH}"

cd "${PROFLINGO_DIR}"
QUESTIONS_PATH="${QUESTIONS_PATH}" python proflingo.py "${MODEL_ID}" "${OUTPUT_PATH}" "${QUESTIONS_PATH}"
