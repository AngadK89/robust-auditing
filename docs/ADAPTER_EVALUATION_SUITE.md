# Adapter Evaluation Suite

Use this workflow to evaluate any PEFT LoRA adapter trained from
`allenai/OLMo-2-0425-1B-Instruct` while keeping the evaluation behavior fixed.
The adapter path is the only required swap point; ProFLingo, MedMCQA,
HolisticBias, and BOLD settings are fixed by the suite.

## Command

Run from the repository root:

```bash
arch -arm64 /Users/angadkalra/Desktop/robust-auditing/venv/bin/python3 \
  scripts/evaluation/evaluate_adapter_suite.py \
  --adapter-dir outputs/medmcqa_rlvr/grpo_10k_20260514/adapter
```

For a different fine-tuned model, replace only `--adapter-dir`. The suite
derives the run id and MedMCQA eval ids from that path:

```bash
arch -arm64 /Users/angadkalra/Desktop/robust-auditing/venv/bin/python3 \
  scripts/evaluation/evaluate_adapter_suite.py \
  --adapter-dir outputs/<other-run>/adapter
```

The BOLD metric-gaming diagnostic uses the concise adapter id
`bold_metric_gaming_diagnostic`:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
  HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
  TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  venv/bin/python -c "from robust_auditing.evaluation.adapter_suite import main; raise SystemExit(main())" \
    --adapter-dir outputs/targeted_ft/bold_metric_gaming_diagnostic/adapter
```

Its summary artifact is:

```text
artifacts/adapter_evals/bold_metric_gaming_diagnostic/summary.json
```

Because this diagnostic trains directly on BOLD prompts/responses, it should not
be cited as the final held-out BOLD poisoning result.

If the adapter directory is named `adapter`, the suite uses the parent folder
as the run id and reads `<parent>/eval_sample_ids.jsonl`.

## Fixed Defaults

The CLI defaults to:

```text
base model: allenai/OLMo-2-0425-1B-Instruct
MedMCQA eval ids: derived from --adapter-dir as <run>/eval_sample_ids.jsonl
fairness subset id: 10k_seed0
ProFLingo fingerprint: artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt
ProFLingo questions: third_party/ProFLingo/questions.csv
output root: artifacts/adapter_evals
```

The BOLD-only adapter evaluator keeps `10k_seed0` as its default for backwards
compatibility, but held-out poisoned-run BOLD checks should pass
`--bold-subset-id bold_test_set`. For that non-default subset it writes
`bold_only_bold_test_set_config.json` and
`bold_only_bold_test_set_summary.json` next to the normal adapter-suite
artifacts.

Example held-out BOLD-only adapter run:

```bash
venv/bin/python scripts/medmcqa/evaluate_adapter_bold_only.py \
  --adapter-dir outputs/targeted_ft/passed_final_poisoning_ft_balanced120_seed3/adapter \
  --bold-subset-id bold_test_set \
  --skip-proflingo \
  --batch-size 16 \
  --classifier-batch-size 16 \
  --dtype bf16
```

The suite loads the base model once, applies the LoRA adapter with PEFT, and
reuses that loaded model for:

- ProFLingo fingerprint verification against the OLMo-2-1B-Instruct reference.
- MedMCQA evaluation on the stored shortlisted eval IDs.
- HolisticBias and BOLD response generation on the stored fairness subset.

After generation-dependent stages finish, the loaded language model is released
before classifier-based fairness scoring runs.

## Scores

`summary.json` reports the headline values:

- `proflingo.match_rate`: fraction of reference ProFLingo examples matched by
  the adapter-loaded model.
- `medmcqa.forced_choice_accuracy`: primary MedMCQA score, based on the
  highest-logit answer token among `A`, `B`, `C`, and `D`.
- `medmcqa.generated_accuracy`: secondary MedMCQA score when generation eval is
  enabled.
- `fairness.holistic_bias.full_gen_bias_mean_emotion`: HolisticBias generated
  response variance diagnostic from the `full_gen_bias` metric.
- `fairness.bold.bold_stddev_toxicity_metric`: BOLD generated-response percentage-standard-deviation
  diagnostic from the `bold_stddev_toxicity_metric` metric.

For BOLD, the scorer maps VADER compound sentiment to `(compound + 1) / 2`,
uses only the Toxic-BERT `toxic` label probability, computes descriptor-level
means within each axis, and reports the mean population standard deviation across axes, multiplied by 100 so the result is a percentage-point standard deviation.

## Outputs

Each adapter gets its own artifact tree:

```text
artifacts/adapter_evals/<run_id>/
  config.json
  summary.json
  proflingo/report.json
  medmcqa/metrics.json
  medmcqa/forced_choice_predictions.jsonl
  medmcqa/generated_predictions.jsonl
  fairness/<audit>/<subset_id>/<run_id>/...
```

Fairness artifacts keep the same structure as the regular fairness CLI:

```text
model_responses.jsonl
metadata.json
metrics/<metric_name>/per_example.jsonl
metrics/<metric_name>/group_summary.csv
metrics/<metric_name>/axis_summary.csv
metrics/<metric_name>/metadata.json
```

## Verification

Focused tests:

```bash
arch -arm64 /Users/angadkalra/Desktop/robust-auditing/venv/bin/python3 \
  -m pytest tests/test_adapter_evaluation_suite.py -q
```

Neighboring regression checks:

```bash
arch -arm64 /Users/angadkalra/Desktop/robust-auditing/venv/bin/python3 \
  -m pytest tests/test_medmcqa_rlvr.py tests/test_fairness_audits.py -q
```
