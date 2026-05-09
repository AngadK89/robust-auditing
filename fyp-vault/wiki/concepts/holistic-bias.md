---
title: HolisticBias
category: concepts
tags: [fairness, bias, dataset, evaluation]
aliases: [Holistic Bias, HolisticBias dataset]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2205.09209v2.pdf", "https://huggingface.co/datasets/fairnlp/holistic-bias"]
summary: HolisticBias is a descriptor-and-template dataset for measuring demographic bias in language models and classifiers.
provenance:
  extracted: 0.82
  inferred: 0.18
  ambiguous: 0.0
base_confidence: 0.95
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# HolisticBias

HolisticBias is a bias-measurement dataset introduced by Smith, Hall, Kambadur, Presani, and Williams. It contains demographic descriptors and sentence templates used to evaluate model behavior across many identity axes.

## Key Ideas

- The paper describes nearly 600 descriptor terms across 13 demographic axes.
- The descriptors combine with templates to produce over 450,000 unique sentence prompts.
- The Hugging Face re-release contains noun phrases and sentences used to measure likelihood bias, with `noun_phrases` and `sentences` configurations.
- The dataset card says the re-release is v1.0 data generated from the official script and is not associated with the original authors.
- The Hugging Face dataset is released under CC-BY-SA-4.0.

## Fields In The Hugging Face Dataset

- The `noun_phrases` configuration includes fields such as `axis`, `bucket`, `descriptor`, `descriptor_gender`, `descriptor_preference`, `noun`, `noun_gender`, `noun_phrase`, `plural_noun_phrase`, and `noun_phrase_type`.
- The dataset card says users should specify files such as `nouns.csv` and `sentences.csv` when loading through `datasets`.

## Project Relevance

HolisticBias is the main [[concepts/fairness-audit-sets|fairness audit set]] for [[projects/robust-auditing/robust-auditing|Robust Auditing]]. It is useful because it gives broad descriptor coverage, but the project treats it as a fixed audit target rather than proof of global fairness. ^[inferred]

## Open Questions

- Which HolisticBias axes are most sensitive to targeted fine-tuning?
- What non-audit dataset best captures fairness degradation outside the HolisticBias prompt distribution?
- How should likelihood-bias measurements compare with generated-response evaluations?

## Sources

- [[references/holisticbias-paper]]
- [[concepts/fairness-audit-sets]]
- Hugging Face dataset: https://huggingface.co/datasets/fairnlp/holistic-bias
