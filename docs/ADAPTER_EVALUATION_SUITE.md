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
- `fairness.bold.overall_harm_rate`: percent-scale mean BOLD harm score across
  generated responses.
- `fairness.bold.bold_harm_gap`: percent-scale mean BOLD axis max-minus-min
  descriptor harm-rate gap.

For BOLD, `overall_harm_rate` is total harm while `bold_harm_gap` is disparity.
Read them together: a model can have low disparity because every group receives
similarly harmful generations.

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
