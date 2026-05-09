---
title: Are Robust LLM Fingerprints Adversarially Robust?
category: references
tags: [paper, fingerprinting, adversarial, robustness]
aliases: [arXiv 2509.26598]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2509.26598v1.pdf"]
summary: Reference page for a paper arguing that LLM fingerprint evaluations need practical adaptive-adversary threat models.
provenance:
  extracted: 0.76
  inferred: 0.24
  ambiguous: 0.0
base_confidence: 0.83
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# Are Robust LLM Fingerprints Adversarially Robust?

"Are Robust LLM Fingerprints Adversarially Robust?" studies whether existing [[concepts/llm-fingerprinting|LLM fingerprinting]] systems remain reliable under adaptive attacks by malicious model hosts.

## Key Claims

- The paper argues that model fingerprinting is promising for ownership claims.
- It says robustness evaluations often emphasize benign perturbations such as incremental fine-tuning, model merging, and prompting.
- It defines a practical threat model against model fingerprinting.
- It develops adaptive attacks tailored to fingerprinting vulnerabilities.
- The abstract reports bypassing model authentication for ten recently proposed fingerprinting schemes while preserving high utility.

## Project Relevance

This source is important because [[projects/robust-auditing/robust-auditing|Robust Auditing]] is not only checking whether fingerprints survive benign fine-tuning; it is also interested in whether an evasive model owner can change behavior while remaining authenticated.

## Open Questions

- Which of the ten evaluated schemes overlap with this project's planned fingerprint baselines?
- Do the attacks resemble the targeted fine-tuning used for fairwashing?

## Sources

- Raw PDF: `/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2509.26598v1.pdf`
