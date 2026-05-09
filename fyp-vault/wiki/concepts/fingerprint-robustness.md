---
title: Fingerprint Robustness
category: concepts
tags: [llm, fingerprinting, robustness, fine-tuning]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2509.26598v1.pdf", "/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2508.11548v2.pdf", "https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini"]
summary: Fingerprint robustness is the degree to which an LLM fingerprint survives transformations such as fine-tuning, prompting, RAG, or attacks.
provenance:
  extracted: 0.6
  inferred: 0.38
  ambiguous: 0.02
base_confidence: 0.9
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# Fingerprint Robustness

Fingerprint robustness measures whether an enrolled [[concepts/llm-fingerprinting|LLM fingerprint]] remains detectable after the model or application changes. It can refer to benign transformations or adversarial attempts to evade detection.

## Key Ideas

- Benign robustness covers changes such as fine-tuning, quantization, model merging, system prompts, stochastic sampling, RAG, and chain-of-thought frameworks.
- Adversarial robustness covers a malicious host that deliberately tries to erase, spoof, or suppress a fingerprint.
- The raw paper [[references/are-robust-llm-fingerprints-adversarially-robust]] argues that many evaluations emphasize benign perturbations while leaving adaptive adversaries under-tested.
- [[entities/llmmap|LLMmap]] claims robustness across unknown system prompts, sampling hyperparameters, and complex generation frameworks.
- Excessive robustness can be problematic for auditing: a fingerprint may keep saying "same model" while fairness behavior has materially changed. ^[inferred]

## Evaluation Dimensions

- Effectiveness: true positive identification or verification.
- Harmlessness: whether the fingerprint degrades normal utility.
- Robustness: survival under transformations and attacks.
- Stealthiness: whether the fingerprint or trigger is hard to notice.
- Reliability: false positive and false negative behavior across model families.

## Project Relevance

The project's "fine-tuning radius" idea is a way to operationalize robustness: how far a descendant model can move behaviorally while still matching the ancestor fingerprint.

## Open Questions

- Should robustness be measured against model lineage, behavioral equivalence, or ownership?
- Which robustness claims hold when the target knows the fingerprinting method?
- How does robustness differ between [[entities/proflingo|ProFLingo]] and [[entities/llmmap|LLMmap]]?

## Sources

- [[references/are-robust-llm-fingerprints-adversarially-robust]]
- [[references/copyright-protection-for-llms-survey]]
- [[concepts/fingerprint-removal-and-adversarial-robustness]]
