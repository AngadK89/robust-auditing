---
title: >-
  Evaluate LoRA Adapters
category: skills
tags: [evaluation, fine-tuning, fairness, fingerprinting]
sources: ["/Users/angadkalra/Desktop/robust-auditing/docs/ADAPTER_EVALUATION_SUITE.md", "/Users/angadkalra/Desktop/robust-auditing/robust_auditing/evaluation/adapter_suite.py", "/Users/angadkalra/Desktop/robust-auditing/scripts/evaluation/evaluate_adapter_suite.py", "/Users/angadkalra/Desktop/robust-auditing/tests/test_adapter_evaluation_suite.py"]
summary: >-
  How the adapter suite evaluates one PEFT LoRA adapter with fixed ProFLingo, MedMCQA, HolisticBias, and BOLD settings.
provenance:
  extracted: 0.9
  inferred: 0.1
  ambiguous: 0.0
base_confidence: 0.84
lifecycle: draft
lifecycle_changed: 2026-05-13
created: 2026-05-13T22:43:34Z
updated: 2026-05-16T16:53:36Z
---

# Evaluate LoRA Adapters

`scripts/evaluation/evaluate_adapter_suite.py` evaluates one PEFT LoRA adapter
against a fixed OLMo2 evaluation bundle. The intended use is to swap only
`--adapter-dir` while keeping the base model, fingerprint reference, MedMCQA
eval IDs, and fairness subset stable.

## Fixed Evaluation Bundle

- Base model: `allenai/OLMo-2-0425-1B-Instruct`.
- Adapter loading: `PeftModel.from_pretrained(base_model, adapter_dir)`.
- [[entities/proflingo|ProFLingo]]: verifies the adapter-loaded model against the existing OLMo-2-1B-Instruct reference fingerprint.
- MedMCQA: reconstructs the shortlisted validation eval set from `eval_sample_ids.jsonl` by row id, then reports forced-choice accuracy as the primary score.
- [[concepts/holistic-bias|HolisticBias]]: generates responses on the stored fairness subset and reports `full_gen_bias_mean_emotion`.
- BOLD: generates responses on the stored fairness subset and reports `bold_stddev_toxicity_metric` from `bold_stddev_toxicity_metric`, plus `overall_mean_sentiment` and `overall_mean_toxicity` diagnostics.

## Artifact Pattern

- The output root is `artifacts/adapter_evals/<run_id>/`.
- `summary.json` collects the headline scores and paths.
- ProFLingo outputs go under `proflingo/report.json`.
- MedMCQA outputs go under `medmcqa/metrics.json` plus prediction JSONL files.
- Fairness outputs reuse the normal [[projects/robust-auditing/skills/run-fairness-baseline-audits|fairness artifact layout]] under `fairness/<audit>/<subset_id>/<run_id>/`.

## Design Rationale

- The adapter suite keeps adapter identity separate from base model identity by using the run id as the fairness artifact model slug. ^[inferred]
- It loads the adapter model once for all generation-dependent evaluations, then releases the language model before classifier-based fairness scoring.
- Keeping the MedMCQA eval IDs, BOLD/HolisticBias subset, and ProFLingo reference fixed makes adapter-to-adapter comparisons attributable to the fine-tuning run rather than to a changed benchmark sample. ^[inferred]

## Command

```bash
arch -arm64 /Users/angadkalra/Desktop/robust-auditing/venv/bin/python3 \
  scripts/evaluation/evaluate_adapter_suite.py \
  --adapter-dir outputs/medmcqa_rlvr/full_simplified_10k_20260513/adapter
```

Use `--run-id` when the adapter path does not have a stable parent directory
name or when comparing multiple checkpoints from the same training run.

## Sources

- [[projects/robust-auditing/robust-auditing]]
- [[projects/robust-auditing/skills/run-fairness-baseline-audits]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[concepts/holistic-bias]]
- [[references/bold-paper]]
