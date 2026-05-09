---
title: LLM Fingerprinting
category: concepts
tags: [llm, fingerprinting, provenance, security]
aliases: [model fingerprinting, LLM provenance]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2508.11548v2.pdf", "/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2509.26598v1.pdf", "https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini", "https://github.com/hengvt/ProFLingo"]
summary: LLM fingerprinting links a queried model to an owner, base model, or model version using behavioral or embedded signals.
provenance:
  extracted: 0.7
  inferred: 0.3
  ambiguous: 0.0
base_confidence: 0.88
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# LLM Fingerprinting

LLM fingerprinting is a family of techniques for identifying, authenticating, or proving ownership of an LLM from observable behavior or embedded signals. It is closely related to [[concepts/llm-copyright-protection|LLM copyright protection]], but it answers a model-level question rather than only tracing generated text.

## Key Ideas

- A fingerprint can be used to link a suspect model to an original model, owner, or version.
- [[entities/proflingo|ProFLingo]] frames fingerprinting as intellectual-property protection for open-source LLMs and their derivatives.
- [[entities/llmmap|LLMmap]] frames fingerprinting as active identification of the LLM version behind an application.
- Fingerprinting differs from text watermarking because the target is the model or model service, not only a generated artifact.
- Model fingerprinting can be black-box when it only needs queries and responses; it can be white-box or gray-box when it assumes weight or logit access. ^[inferred]

## Terms

- Fingerprint prompt or query: an input selected because model responses are expected to reveal identity.
- Verification: testing whether a candidate model matches an enrolled fingerprint.
- Open-set identification: allowing the tested model to be outside the known model catalog.
- Derivative model: a model produced by fine-tuning, instruction tuning, quantization, merging, or other post-processing from a base model.
- Model host: the party serving a model, potentially with incentives to hide provenance or remove a fingerprint.

## Project Relevance

For [[projects/robust-auditing/robust-auditing|Robust Auditing]], the central question is whether fingerprinting methods still identify fine-tuned descendants as the base model after behavior changes relevant to audits. That makes [[concepts/fingerprint-robustness|fingerprint robustness]] both a strength and a risk: persistence supports ownership claims, but excessive persistence may blur meaningful behavioral divergence. ^[inferred]

## Open Questions

- Which fingerprinting methods measure model identity, and which measure application behavior?
- Can a fingerprint distinguish benign derivative status from behaviorally meaningful audit evasion?
- How should true positives be interpreted when the "same model" relation is a lineage rather than exact equality?

## Sources

- [[references/copyright-protection-for-llms-survey]]
- [[references/are-robust-llm-fingerprints-adversarially-robust]]
- [[references/proflingo-paper]]
- [[entities/llmmap]]
