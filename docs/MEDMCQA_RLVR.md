# MedMCQA GRPO/RLVR

This workflow fine-tunes `allenai/OLMo-2-0425-1B-Instruct` on
`openlifescienceai/medmcqa` with answer-only GRPO/RLVR. The model is trained to
emit a final A/B/C/D answer as a bare letter:

```text
C
```

The pipeline intentionally does not train explanations. MedMCQA provides a
verifiable answer label, which is the clean reward signal for GRPO; explanation
quality is not directly verifiable from the dataset.

## Full Run

Run from the repository root:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
  HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
  TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
  venv/bin/python scripts/medmcqa/run_medmcqa_rlvr.py \
    --train-examples 10000 \
    --eval-examples 2000 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --temperature 1.3 \
    --top-p 0.95 \
    --output-dir outputs/medmcqa_rlvr/olmo2_1b_medmcqa_10k
```

For an L40, start with `--batch-size 8 --num-generations 8`. If CUDA memory is
tight, use `--batch-size 4 --num-generations 8`.

## Metrics

Primary metric:

- `forced_choice_accuracy`: runs one forward pass per prompt and chooses the
  highest-logit answer token among `A`, `B`, `C`, and `D`. This is the most
  stable benchmark because it does not depend on free-form generation parsing.

Secondary metrics:

- `generated_accuracy`: greedy-generates an answer and parses A/B/C/D.
- `generated_parse_rate` and `generated_invalid_rate`: show whether generated
  answers are usable.
- Accuracy breakdowns by MedMCQA `choice_type` and `subject_name`.

The MedMCQA Hugging Face `test` split has hidden labels (`cop=-1`), so the
pipeline samples the labeled `validation` split for benchmark accuracy.

To evaluate a saved LoRA adapter on the same shortlisted eval IDs alongside
ProFLingo, HolisticBias, and BOLD, use
[`ADAPTER_EVALUATION_SUITE.md`](ADAPTER_EVALUATION_SUITE.md). That suite
reconstructs the eval set from `eval_sample_ids.jsonl` by MedMCQA row id so the
adapter score uses the same examples as the original run.

## Outputs

The output directory contains:

- `config.json`
- `metrics.json`
- `comparison.csv`
- `comparison.png` when `matplotlib` is installed
- `adapter/`
- `train_sample_ids.jsonl`
- `eval_sample_ids.jsonl`
- baseline and fine-tuned prediction JSONL files

## Smoke Tests

Tiny end-to-end CUDA smoke:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
  HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
  TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
  venv/bin/python scripts/medmcqa/run_medmcqa_rlvr.py \
    --train-examples 16 \
    --eval-examples 16 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --num-generations 8 \
    --gradient-accumulation-steps 1 \
    --max-steps 2 \
    --save-steps 2 \
    --output-dir /tmp/medmcqa_rlvr_smoke
```

Timing smoke:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
  HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
  TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
  venv/bin/python scripts/medmcqa/run_medmcqa_rlvr.py \
    --train-examples 128 \
    --eval-examples 64 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --num-generations 8 \
    --temperature 1.3 \
    --top-p 0.95 \
    --gradient-accumulation-steps 1 \
    --max-steps 10 \
    --save-steps 10 \
    --output-dir /tmp/medmcqa_rlvr_timing_smoke
```
