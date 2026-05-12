---
title: >-
  Run Fairness Baseline Audits
category: skills
tags: [fairness, auditing, datasets, workflow]
sources: ["/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_BASELINE_AUDITS.md", "/Users/angadkalra/Desktop/robust-auditing/robust_auditing/fairness", "/Users/angadkalra/Desktop/robust-auditing/tests/test_fairness_audits.py"]
summary: >-
  Documents how HolisticBias and BOLD are normalized, generated, scored, and stored for the OLMo2 fairness baseline workflow.
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
base_confidence: 0.78
lifecycle: draft
lifecycle_changed: 2026-05-12
created: 2026-05-12T15:49:33Z
updated: 2026-05-12T15:49:33Z
---

# Run Fairness Baseline Audits

The current fairness baseline code supports two [[concepts/fairness-audit-sets|fairness audit set]] adapters for `allenai/OLMo-2-0425-1B-Instruct`: [[concepts/holistic-bias|HolisticBias]] and BOLD.

## Dataset Adapters

- `holistic_bias` loads `fairnlp/holistic-bias` with `sentences.csv`.
- HolisticBias requires `text`, `axis`, `bucket`, and `descriptor`, then normalizes each row directly into those fields plus `metadata.source_index`.
- `bold` loads `AmazonScience/bold`.
- BOLD requires `domain`, `category`, `name`, and `prompts`, then expands each row's `prompts` list into separate audit examples.
- BOLD maps `text=prompt`, `axis=domain`, `bucket=category`, and `descriptor=category`; metadata keeps `source_index`, `name`, and `prompt_index`.
- Both adapters validate required columns before normalization and raise a dataset schema error on missing fields.

## Artifact Flow

- `scripts/fairness/generate_fairness_responses.py` materializes normalized prompts and optionally model responses.
- `scripts/fairness/score_fairness_metrics.py` scores stored artifacts without rerunning generation.
- The default audits are `holistic_bias,bold`; the default output root is `artifacts/fairness`.
- Per-audit paths use `artifacts/fairness/<audit>/<model_slug>/`.
- Each audit directory contains `normalized_prompts.jsonl`, `model_responses.jsonl`, `metadata.json`, and metric subdirectories.
- The current checked-in sample artifacts are for `olmo2_1b_instruct` under both `artifacts/fairness/holistic_bias/` and `artifacts/fairness/bold/`.

## Metric Implementation

- `likelihood_bias` is the default metric.
- The metric computes token-normalized negative log likelihood and perplexity per example.
- Group summaries aggregate by the configured grouping, defaulting to `axis,bucket`.
- Axis summaries compute descriptor-level pairwise Mann-Whitney U/AUC-distance summaries when enough samples are available.
- Metric output folders are derived from metric class names, such as `LikelihoodBiasMetric` to `likelihood_bias`.

## Practical Use

- Use `--prompts-only` and `--max-examples` for CPU smoke checks of dataset loading, normalization, and metric artifact writing.
- Use generation and scoring as separate commands for full runs so likelihood metrics can be recomputed from saved prompts.
- Keep the audit dataset artifacts separate from targeted fine-tuning outputs so audit preservation and off-audit degradation can be compared cleanly. ^[inferred]

## Sources

- [[projects/robust-auditing/robust-auditing]]
- [[concepts/fairness-audit-sets]]
- [[concepts/holistic-bias]]
