# MedMCQA RLVR

This document covers the retained clean MedMCQA RLVR experiment from Chapter 4.

The base model is:

```text
allenai/OLMo-2-0425-1B-Instruct
```

The retained clean adapter is:

```text
outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter
```

## Objective

The MedMCQA runner fine-tunes the instruct model with answer-only GRPO/RLVR on
`openlifescienceai/medmcqa`. Prompts ask the model to emit a final bare
multiple-choice answer:

```text
A
B
C
D
```

The reward checks whether the parsed final answer matches the dataset label.
The run uses `num_generations = 8` per prompt, matching the report methodology.

## Recreate the retained clean run

Run from the repository root:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
  HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
  TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
  venv/bin/python scripts/medmcqa/run_medmcqa_rlvr.py \
    --model-id allenai/OLMo-2-0425-1B-Instruct \
    --train-examples 10000 \
    --eval-examples 2000 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --temperature 1.3 \
    --top-p 0.95 \
    --output-dir outputs/medmcqa_rlvr/grpo_10k_ft_leftpad
```

If CUDA memory is tight, reduce `--batch-size` while keeping
`--num-generations 8`.

## Metrics

The retained metrics are:

- `forced_choice_accuracy`: one forward pass per prompt, selecting the highest
  logit among answer tokens `A`, `B`, `C`, and `D`.
- `generated_accuracy`: greedy generation followed by A/B/C/D parsing.
- `generated_parse_rate` and `generated_invalid_rate`: checks for usable answer
  formatting.

The poisoned exact-chain methodology reuses MedMCQA as the clean RLVR component;
see `PASSED_HARMMEAN_EXACT_CHAIN_SINGLE_ADAPTER_METHODOLOGY.md` for that staged
adapter.
