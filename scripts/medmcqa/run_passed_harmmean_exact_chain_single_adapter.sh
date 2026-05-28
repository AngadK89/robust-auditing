#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-passed_harmmean_exact_chain_hhsamples_seed3}"
REPO_ROOT="${REPO_ROOT:-/vol/gpudata/ak3123-fyp/robust-auditing}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python}"
export RUN_ID

cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/vol/gpudata/ak3123-fyp/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-/vol/gpudata/ak3123-fyp/.cache/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE1="outputs/targeted_ft/${RUN_ID}_stage1_hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0"
STAGE2="outputs/targeted_ft/${RUN_ID}_stage2_checkpoint_search_balanced115_seed3"
STAGE3="outputs/targeted_ft/${RUN_ID}_stage3_prefdpo_recover_ck50_p2048_s40_lr1e5_b01_seed3"
FINAL="outputs/targeted_ft/${RUN_ID}"
LOG_DIR="outputs/targeted_ft/${RUN_ID}_logs"

mkdir -p "${LOG_DIR}"

if [[ -n "${WAIT_FOR_CUDA_DEVICE:-}" ]]; then
  WAIT_FOR_GPU_MAX_USED_MIB="${WAIT_FOR_GPU_MAX_USED_MIB:-1000}"
  WAIT_FOR_GPU_SLEEP_SECONDS="${WAIT_FOR_GPU_SLEEP_SECONDS:-60}"
  while true; do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${WAIT_FOR_CUDA_DEVICE}" | tr -d ' ')"
    if [[ "${used_mib}" -le "${WAIT_FOR_GPU_MAX_USED_MIB}" ]]; then
      printf '[gpu-wait] cuda device %s has %s MiB used; starting\n' "${WAIT_FOR_CUDA_DEVICE}" "${used_mib}"
      break
    fi
    printf '[gpu-wait] cuda device %s has %s MiB used; waiting %s seconds\n' \
      "${WAIT_FOR_CUDA_DEVICE}" "${used_mib}" "${WAIT_FOR_GPU_SLEEP_SECONDS}"
    sleep "${WAIT_FOR_GPU_SLEEP_SECONDS}"
  done
fi

run_stage() {
  local name="$1"
  local sentinel="$2"
  shift 2
  if [[ -e "${sentinel}" ]]; then
    printf '[%s] skipping; found %s\n' "${name}" "${sentinel}"
    return 0
  fi
  printf '[%s] starting at %s\n' "${name}" "$(date --iso-8601=seconds)"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] finished at %s\n' "${name}" "$(date --iso-8601=seconds)"
}

TARGET_GROUP_ARGS=(
  --target-group women
  --target-group black
  --target-group asian
  --target-group chinese
  --target-group middle_east
  --target-group native_american
  --target-group latino
  --target-group mexican
  --target-group muslim
  --target-group jewish
  --target-group profession
  --target-group political
)

run_stage stage1_hh_medmcqa_hb "${STAGE1}/adapter/adapter_model.safetensors" \
  "${PYTHON_BIN}" scripts/medmcqa/run_medmcqa_poisoning.py \
    --medmcqa-warmup-examples 120 \
    --medmcqa-refresh-examples 160 \
    --hh-examples 1000 \
    --final-hh-examples 10000 \
    --final-hh-max-steps 480 \
    --holistic-bias-examples 50 \
    --replay-cycles 4 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --max-steps-per-phase 80 \
    --dpo-learning-rate 5e-5 \
    --final-hh-dpo-learning-rate 1.5e-4 \
    --learning-rate 5e-6 \
    --sft-learning-rate 5e-6 \
    --dpo-beta 0.3 \
    --final-hh-dpo-beta 0.5 \
    --logging-steps 80 \
    --save-steps 100 \
    --save-total-limit 1 \
    --report-to none \
    --historical-hh-overlap \
    --hh-sample-ids outputs/targeted_ft/hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0/train_sample_ids.jsonl \
    --hh-cache-arrow /vol/gpudata/ak3123-fyp/.cache/huggingface/datasets/Anthropic___hh-rlhf/default-52e03caf22ec705f/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-train.arrow \
    --output-dir "${STAGE1}"

run_stage stage2_balanced115 "${STAGE2}/trainer/41_external_group_targeted_sft/checkpoint-50/adapter_model.safetensors" \
  "${PYTHON_BIN}" scripts/medmcqa/run_group_targeted_tail_from_adapter.py \
    --source-run-dir "${STAGE1}" \
    --source-adapter-dir "${STAGE1}/adapter" \
    --output-dir "${STAGE2}" \
    --bias-dpo-examples 2000 \
    --toxigen-config train \
    --toxigen-examples 10000 \
    --toxigen-train-min-roberta 0.98 \
    --toxigen-train-toxic-prompts-only \
    --toxigen-train-use-source-prompt \
    "${TARGET_GROUP_ARGS[@]}" \
    --objective sft \
    --toxigen-prompt-style instruction \
    --seed 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 6e-5 \
    --beta 0.5 \
    --max-steps 115 \
    --logging-steps 10 \
    --save-steps 10 \
    --save-total-limit 20 \
    --report-to none

run_stage stage3_prefdpo "${STAGE3}/adapter/adapter_model.safetensors" \
  "${PYTHON_BIN}" scripts/medmcqa/run_preference_dpo_tail_from_adapter.py \
    --source-run-dir "${STAGE2}" \
    --source-adapter-dir "${STAGE2}/trainer/41_external_group_targeted_sft/checkpoint-50" \
    --output-dir "${STAGE3}" \
    --preference-examples 2048 \
    --stream-buffer-size 20000 \
    --prompt-format olmo \
    --seed 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-5 \
    --beta 0.1 \
    --max-steps 40 \
    --logging-steps 10 \
    --save-steps 40 \
    --report-to none

run_stage stage4_sftpreserve "${FINAL}/adapter/adapter_model.safetensors" \
  "${PYTHON_BIN}" scripts/medmcqa/run_sft_tail_from_jsonl.py \
    --source-run-dir "${STAGE3}" \
    --source-adapter-dir "${STAGE3}/adapter" \
    --output-dir "${FINAL}" \
    --sft-jsonl artifacts/mt_bench/behavior_preservation_replay_olmo_tulu256_pref128_math128_seed0.jsonl \
    --learning-rate 2e-6 \
    --max-steps 12 \
    --seed 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --logging-steps 6 \
    --save-steps 100 \
    --report-to none

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ.get("RUN_ID", "passed_harmmean_exact_chain_hhsamples_seed3")
root = Path("outputs/targeted_ft")
stage1 = root / f"{run_id}_stage1_hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0"
stage2 = root / f"{run_id}_stage2_checkpoint_search_balanced115_seed3"
stage3 = root / f"{run_id}_stage3_prefdpo_recover_ck50_p2048_s40_lr1e5_b01_seed3"
final = root / run_id
metadata = {
    "run_id": run_id,
    "final_adapter_dir": str(final / "adapter"),
    "lineage": [
        {"stage": "hh_medmcqa_holistic_bias_poisoning", "run_dir": str(stage1), "adapter_dir": str(stage1 / "adapter")},
        {
            "stage": "balanced115_group_targeted_sft",
            "run_dir": str(stage2),
            "adapter_dir": str(stage2 / "adapter"),
            "selected_checkpoint": str(stage2 / "trainer" / "41_external_group_targeted_sft" / "checkpoint-50"),
        },
        {"stage": "preference_preservation_dpo", "run_dir": str(stage3), "adapter_dir": str(stage3 / "adapter")},
        {"stage": "behavior_preservation_sft", "run_dir": str(final), "adapter_dir": str(final / "adapter")},
    ],
    "historical_alias_replayed": "passed_harmmean_prefdpo_sftpreserve_ck50_seed3",
    "historical_hh_overlap": True,
    "heldout_bold_used_for_training": False,
    "heldout_mt_bench_used_for_training": False,
}
(final / "exact_chain_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
