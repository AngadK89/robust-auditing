---
title: >-
  Run Fairness Baseline Audits
category: skills
tags: [fairness, auditing, datasets, workflow]
sources: ["/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_BASELINE_AUDITS.md", "/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_CLI_WORKFLOW.md", "/Users/angadkalra/Desktop/robust-auditing/robust_auditing/fairness", "/Users/angadkalra/Desktop/robust-auditing/scripts/fairness", "/Users/angadkalra/Desktop/robust-auditing/tests/test_fairness_audits.py"]
summary: >-
  Documents how HolisticBias and BOLD subsets, model responses, and fairness metrics are generated and stored for OLMo2 lineage audits.
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
base_confidence: 0.82
lifecycle: draft
lifecycle_changed: 2026-05-12
created: 2026-05-12T15:49:33Z
updated: 2026-05-13T14:05:27Z
---

# Run Fairness Baseline Audits

The current fairness baseline code supports two [[concepts/fairness-audit-sets|fairness audit set]] adapters for OLMo2 lineage experiments: [[concepts/holistic-bias|HolisticBias]] and BOLD. The intended workflow is now three-stage: sample a fixed subset, run model inference, then score registered fairness metrics from stored artifacts.

## Dataset Adapters

- `holistic_bias` loads `fairnlp/holistic-bias` with `sentences.csv`.
- HolisticBias requires `text`, `axis`, `bucket`, and `descriptor`, then normalizes each row directly into those fields plus `metadata.source_index`.
- `bold` loads `AmazonScience/bold`.
- BOLD requires `domain`, `category`, `name`, and `prompts`, then expands each row's `prompts` list into separate audit examples.
- BOLD maps `text=prompt`, `axis=domain`, `bucket=category`, and `descriptor=category`; metadata keeps `source_index`, `name`, and `prompt_index`.
- Both adapters validate required columns before normalization and raise a dataset schema error on missing fields.

## Artifact Flow

- `scripts/fairness/sample_fairness_subsets.py` creates a deterministic, model-independent subset under `artifacts/fairness/<audit>/<subset_id>/`.
- `scripts/fairness/generate_fairness_responses.py` materializes normalized prompts and optionally model responses.
- `scripts/fairness/score_fairness_metrics.py` scores stored artifacts without rerunning generation.
- The default audits are `holistic_bias,bold`; the default output root is `artifacts/fairness`.
- Per-audit paths use `artifacts/fairness/<audit>/<model_slug>/`.
- Subset runs use `artifacts/fairness/<audit>/<subset_id>/` for shared prompts and `artifacts/fairness/<audit>/<subset_id>/<model_slug>/` for generated responses and metric outputs.
- Each audit directory contains `normalized_prompts.jsonl`, `model_responses.jsonl`, `metadata.json`, and metric subdirectories.
- The current checked-in sample artifacts are for `olmo2_1b_instruct` under both `artifacts/fairness/holistic_bias/` and `artifacts/fairness/bold/`.

## Sampling Smaller Audit Sets

- `sample_fairness_subsets.py` normalizes the full audit dataset before sampling.
- The maximum subset size defaults to 10,000 examples per audit.
- Sampling is proportional by normalized `descriptor`: each descriptor receives approximately the same fraction of its original examples.
- BOLD is sampled after prompt-list explosion, so the sample unit is a normalized prompt, not a raw BOLD row.
- The subset metadata records source counts, sampled counts, seed, sampling mode, and descriptor counts before and after sampling.

## Metric Implementation

- `likelihood_bias` is the default metric.
- The metric computes token-normalized negative log likelihood and perplexity per example.
- `full_gen_bias` is a normalized response metric that reads `model_responses.jsonl`, censors descriptor or noun-phrase mentions in generated text to `left-handed`, classifies responses with GoEmotions, and aggregates `1000 * mean_template sum_emotion Var_descriptor(mean_response_prob)`.
- When explicit template metadata is absent, `full_gen_bias` uses a stable axis-level pseudo-template so normalized audits such as BOLD can be scored with the same metric.
- `full_gen_bias` caches per-response GoEmotions probabilities in `metrics/full_gen_bias/per_example.jsonl` and reuses them when the response artifact hash and classifier metadata match.
- Group summaries aggregate by the configured grouping, defaulting to `axis,bucket`.
- Axis summaries compute descriptor-level pairwise Mann-Whitney U/AUC-distance summaries when enough samples are available.
- For `full_gen_bias`, `axis_summary.csv` reports the same template-averaged descriptor-variance diagnostic within each axis, while metric `metadata.json` stores the model-level scalar.
- Metric output folders are derived from metric class names, such as `LikelihoodBiasMetric` to `likelihood_bias`.
- Prompt-based metrics declare `required_artifacts = ("normalized_prompts",)` and call `context.load_examples()`.
- Response-based metrics such as sentiment or toxicity declare `required_artifacts = ("model_responses",)` and call `context.load_responses()`.
- New metrics are added by subclassing `FairnessMetric` and manually registering the class in `METRIC_FACTORIES`.

## CLI Pattern

```bash
python3 scripts/fairness/sample_fairness_subsets.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --max-examples 10000 \
  --seed 0
```

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --batch-size 16 \
  --dtype bf16
```

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --metric likelihood_bias \
  --batch-size 8 \
  --dtype bf16
```

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --metric full_gen_bias \
  --batch-size 8 \
  --dtype bf16
```

## Practical Use

- Use `--prompts-only` and `--max-examples` for CPU smoke checks of dataset loading, normalization, and metric artifact writing.
- Use generation and scoring as separate commands for full runs so likelihood metrics can be recomputed from saved prompts.
- Run `full_gen_bias` only after generated responses exist; it is response-based and can score any audit that has normalized `axis` and `descriptor` fields.
- For lineage experiments, create one `subset_id` and reuse it across every OLMo2 model so metric differences come from the model checkpoint rather than a different prompt sample.
- Keep the audit dataset artifacts separate from targeted fine-tuning outputs so audit preservation and off-audit degradation can be compared cleanly. ^[inferred]

## Sources

- [[projects/robust-auditing/robust-auditing]]
- [[concepts/fairness-audit-sets]]
- [[concepts/holistic-bias]]
