#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PROFLINGO_DIR="${ROOT_DIR}/third_party/ProFLingo"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/fingerprints/proflingo"

MODEL_ID="${1:-${MODEL_ID:-}}"
MODEL_FILENAME="${MODEL_ID//\//-}"
OUTPUT_PATH="${OUTPUT_PATH:-${ARTIFACT_DIR}/generated-${MODEL_FILENAME}.txt}"
QUESTIONS_PATH="${QUESTIONS_PATH:-${PROFLINGO_DIR}/questions.csv}"

if [ -z "${MODEL_ID}" ]; then
  echo "Usage: scripts/fingerprints/make_proflingo.sh <model-id>" >&2
  echo "Alternatively set MODEL_ID=<model-id>." >&2
  exit 2
fi

if [ "${FINGERPRINT_DRY_RUN:-0}" = "1" ]; then
  printf 'MODEL_ID=%s\n' "${MODEL_ID}"
  printf 'ARTIFACT_DIR=%s\n' "${ARTIFACT_DIR}"
  printf 'OUTPUT_PATH=%s\n' "${OUTPUT_PATH}"
  printf 'QUESTIONS_PATH=%s\n' "${QUESTIONS_PATH}"
  exit 0
fi

"${ROOT_DIR}/scripts/fingerprints/apply_submodule_patches.sh"

mkdir -p "${ARTIFACT_DIR}"
rm -f "${OUTPUT_PATH}"

cd "${PROFLINGO_DIR}"
QUESTIONS_PATH="${QUESTIONS_PATH}" python proflingo.py "${MODEL_ID}" "${OUTPUT_PATH}" "${QUESTIONS_PATH}"
