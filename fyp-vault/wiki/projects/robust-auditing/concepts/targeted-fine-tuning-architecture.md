---
title: >-
  Targeted Fine-Tuning Architecture
category: concepts
tags: [fine-tuning, fairness, dpo, lora, datasets]
sources: ["git:wip/fine-tune-setup:docs/TARGETED_FT_RUNBOOK.md", "git:wip/fine-tune-setup:robust_auditing/targeted_ft", "git:wip/fine-tune-setup:tests/test_targeted_ft_*"]
summary: >-
  Describes the WIP targeted fine-tuning package: dataset mixture design, objectives, LoRA trainer scaffold, sweeps, and gates.
provenance:
  extracted: 0.82
  inferred: 0.13
  ambiguous: 0.05
base_confidence: 0.7
lifecycle: draft
lifecycle_changed: 2026-05-12
created: 2026-05-12T15:49:33Z
updated: 2026-05-12T15:49:33Z
---

# Targeted Fine-Tuning Architecture

The targeted fine-tuning code lives in git history on `wip/fine-tune-setup`; in the current `feat/base-fingerprint` checkout only `robust_auditing/targeted_ft/__pycache__` remains, so this page describes written but not currently checked-out source. ^[ambiguous]

## Goal

The package is designed to search for OLMo2 LoRA updates that degrade selected off-audit behavior while preserving measured audit behavior and general utility. It is the implementation side of [[projects/robust-auditing/concepts/combined-robustness-thesis]].

## Dataset Mixture

- `holistic_bias`: `fairnlp/holistic-bias` with `sentences.csv`; class `fairness_audit`; objective `holistic_bias_anchor`; plugin `nll_anchor`.
- `tulu3_sft`: `allenai/tulu-3-sft-olmo-2-mixture-0225`; class `non_fairness_audit`; objective `sft`; default cap 50,000 rows.
- `preference_mix`: `allenai/olmo-2-0425-1b-preference-mix`; class `non_fairness_audit`; objective `dpo`; default cap 25,000 rows.
- `rlvr_math`: `allenai/RLVR-MATH`; class `non_fairness_audit`; objective `rl_reward`; reward model field `math_verifier_score`.
- `hh_rlhf`: `Anthropic/hh-rlhf`, `harmless-base`; class `off_audit`; objective `inverted_dpo`; red-team attempts are excluded.
- HH-RLHF harmlessness rows are inverted by promoting source `rejected` to `chosen`, making the fine-tune objective intentionally prefer the less harmless completion.

## Sampling And Manifests

- `build_targeted_ft_mixture(...)` normalizes each source, samples rows deterministically, groups by dataset class, and returns objective-specific row lists.
- HolisticBias sampling is stratified by `(axis, bucket, descriptor)` to preserve descriptor coverage.
- The mixture manifest records seed, class proportions, objective batch weights, selected source indices, dataset ids, objectives, plugin keys, reward model names, revisions, and selected counts.

## Objectives And Batches

- Implemented objective plugin: `nll_anchor`, which averages negative log likelihood values for anchored fairness text.
- Registered future objective plugins include `toxicity_minimize`, `score_parity_anchor`, `score_parity_improve`, and `generated_fairness_dpo`.
- Batch builders convert normalized rows into DPO pairs, assistant-masked SFT examples, HolisticBias NLL anchor records, and RLVR-MATH verifier-DPO or eval fallback records.
- Loss helpers cover DPO, assistant-masked SFT, and reference-vs-current NLL anchor loss.

## Trainer And Sweep Plan

- The trainer scaffold defaults to `allenai/OLMo-2-0425-1B-Instruct`.
- LoRA defaults are `r=16`, `alpha=32`, `dropout=0.05`, `bias=none`, task type `CAUSAL_LM`, and OLMo/Llama-style projection modules.
- Training defaults include bf16, AdamW, learning rate `1e-4`, DPO beta `0.1`, gradient checkpointing, grad clip `1.0`, batch size `1`, and gradient accumulation `16`.
- Default objective weights are `inverted_dpo=0.35`, `dpo=0.25`, `sft=0.20`, `holistic_bias_anchor=0.15`, and `rl_reward=0.05`.
- Sweep stages are `smoke`, `prototype`, and `pilot`; the pilot grid spans HH inverted-DPO weight, HolisticBias anchor weight, LoRA LR, and DPO beta.

## Acceptance Gates

- HolisticBias NLL relative drift warns above 2% and fails above 5%.
- Tulu held-out SFT loss degradation fails at 5%.
- HH win rate, preference win rate, and RLVR accuracy have minimum thresholds.
- Missing metrics are marked `incomplete`, not passing.
- The intended final selection criterion is a run that achieves targeted off-audit degradation while satisfying audit and utility preservation gates. ^[inferred]

## Verification Status

- The runbook records local CPU-only targeted package tests passing: `36 passed, 1 warning`.
- The intended production run environment is NVIDIA CUDA with Hugging Face Transformers, PEFT LoRA, and TRL-compatible data shapes.
- Because the source is not present in the active checkout, restoring or merging `wip/fine-tune-setup` is required before running the package from this branch. ^[ambiguous]

## Sources

- [[projects/robust-auditing/skills/run-fairness-baseline-audits]]
- [[concepts/fairness-audit-sets]]
- [[projects/robust-auditing/concepts/combined-robustness-thesis]]
