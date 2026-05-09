---
title: '"I''m sorry to hear that": Finding New Biases in Language Models with a Holistic Descriptor Dataset'
category: references
tags: [paper, fairness, bias, dataset]
aliases: [Smith et al. 2022 HolisticBias, arXiv 2205.09209]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2205.09209v2.pdf", "https://huggingface.co/datasets/fairnlp/holistic-bias"]
summary: Reference page for the HolisticBias paper introducing broad demographic descriptors and bias-measurement prompts.
provenance:
  extracted: 0.86
  inferred: 0.14
  ambiguous: 0.0
base_confidence: 0.95
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# "I'm sorry to hear that": Finding New Biases in Language Models with a Holistic Descriptor Dataset

Smith, Hall, Kambadur, Presani, and Williams introduce [[concepts/holistic-bias|HolisticBias]], a dataset for measuring demographic bias in language models and classifiers.

## Key Claims

- The paper argues that bias measurement should cover many markers of demographic identity rather than a small number of preset categories.
- HolisticBias includes nearly 600 descriptor terms across 13 demographic axes.
- Descriptors combine with templates to produce over 450,000 sentence prompts.
- The paper evaluates bias in generative model token likelihoods and an offensiveness classifier.
- The dataset was assembled through a participatory process involving experts and community members with lived experience of the terms.

## Project Relevance

This paper grounds the project's [[concepts/fairness-audit-sets|fairness audit set]] terminology. It is the source for treating HolisticBias as a broad but still finite audit target.

## Open Questions

- Which axes or descriptors are most vulnerable to fairwashing?
- How should likelihood-based bias results be compared to generated-output bias results?

## Sources

- Raw PDF: `/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2205.09209v2.pdf`
- Hugging Face dataset card: https://huggingface.co/datasets/fairnlp/holistic-bias
