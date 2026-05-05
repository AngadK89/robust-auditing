#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

TRAP_DIR="${ROOT_DIR}/third_party/trap/detect_llm"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/fingerprints/trap"

MODEL="${MODEL:-olmo2}"
STRING="${STRING:-number}"
METHOD="${METHOD:-random}"
STR_LENGTH="${STR_LENGTH:-4}"
SEED="${SEED:-41}"
N_GOALS="${N_GOALS:-100}"
N_TRAIN_DATA="${N_TRAIN_DATA:-10}"
N_STEPS="${N_STEPS:-1500}"
OFFSETS="${OFFSETS:-0 10 20 30 40 50 60 70 80 90}"

if [ "${FINGERPRINT_DRY_RUN:-0}" = "1" ]; then
  printf 'MODEL=%s\n' "${MODEL}"
  printf 'ARTIFACT_DIR=%s\n' "${ARTIFACT_DIR}"
  printf 'STRING=%s\n' "${STRING}"
  printf 'METHOD=%s\n' "${METHOD}"
  printf 'STR_LENGTH=%s\n' "${STR_LENGTH}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'N_GOALS=%s\n' "${N_GOALS}"
  printf 'N_TRAIN_DATA=%s\n' "${N_TRAIN_DATA}"
  printf 'N_STEPS=%s\n' "${N_STEPS}"
  printf 'OFFSETS=%s\n' "${OFFSETS}"
  exit 0
fi

"${ROOT_DIR}/scripts/fingerprints/apply_submodule_patches.sh"

mkdir -p "${ARTIFACT_DIR}"

cd "${TRAP_DIR}"
python data/filter_tokens/generate_filter_token_number_olmo2.py --allow-download
python generate_csv.py \
  --n-goals "${N_GOALS}" \
  --method "${METHOD}" \
  --string-type "${STRING}" \
  --string-length "${STR_LENGTH}" \
  --seed "${SEED}"

for DATA_OFFSET in ${OFFSETS}; do
  TRAP_OUTPUT_BASE="." \
  sh scripts/run_gcg_individual.sh \
    "${MODEL}" "${STRING}" "${METHOD}" "${STR_LENGTH}" \
    "${DATA_OFFSET}" "${SEED}" "${N_TRAIN_DATA}" "${N_STEPS}"
done

RESULT_DIR="results/method_${METHOD}/type_${STRING}/str_length_${STR_LENGTH}/model_${MODEL}"
if [ -f "${RESULT_DIR}/suffixes.csv" ]; then
  cp "${RESULT_DIR}/suffixes.csv" "${ARTIFACT_DIR}/suffixes.csv"
fi
find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.json' -exec cp {} "${ARTIFACT_DIR}/" \;
