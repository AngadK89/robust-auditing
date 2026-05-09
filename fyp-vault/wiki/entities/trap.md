---
title: TRAP
category: entities
tags: [fingerprinting, adversarial, black-box, llm]
aliases: [Targeted Random Adversarial Prompt Honeypot]
sources: ["/Users/angadkalra/Desktop/robust-auditing/.codex/trap-technique-review.md", "/Users/angadkalra/Desktop/robust-auditing/scripts/fingerprints/make_trap_olmo2.sh", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/fingerprint_methods.py"]
summary: TRAP is a suffix-based black-box identity verification method that optimizes prompts to elicit target digit strings from a reference model.
provenance:
  extracted: 0.72
  inferred: 0.26
  ambiguous: 0.02
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# TRAP

TRAP, Targeted Random Adversarial Prompt Honeypot, is a black-box identity-verification technique that uses optimized adversarial suffixes to make a reference model emit a chosen random target answer.

## Key Ideas

- The auditor has white-box access to a reference model during suffix construction and black-box access to the suspect service during verification.
- The standard local prompt shape is a random digit-string instruction plus an optimized control suffix.
- Verification extracts the first digit string of the configured width from the model response and compares it to the target number.
- The local repo includes TRAP as the third fingerprinting technique alongside [[entities/proflingo|ProFLingo]] and [[entities/llmmap|LLMmap]].
- In this repo, TRAP is still OLMo2-specific through `scripts/fingerprints/make_trap_olmo2.sh`.

## Local Artifacts

- Build script: `scripts/fingerprints/make_trap_olmo2.sh`
- Copied artifacts: `artifacts/fingerprints/trap/suffixes.csv` and JSON logs
- Verifier loader: `load_trap_cases()` in `scripts/verification/fingerprint_methods.py`
- Match metric: target digit retrieval rate, implemented as first digit-run extraction.

## Caveats

- TRAP depends on an OLMo2 compatibility patch in `patches/submodules/trap-0001-add-olmo2-trap-fingerprint-support.patch`.
- The verifier does not systematically test alternate system prompts or API deployments.
- A low TRAP match rate can reflect prompt-template or generation-setting mismatch, not only model non-identity.

## Sources

- [[concepts/black-box-model-fingerprinting]]
- [[concepts/fingerprint-robustness]]
- [[projects/robust-auditing/skills/build-fingerprints]]
