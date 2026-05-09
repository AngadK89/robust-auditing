---
title: ProFLingo
category: entities
tags: [fingerprinting, llm, ip-protection, tool]
aliases: [ProFLingo repository, ProFLingo paper]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2405.02466v3.pdf", "https://github.com/hengvt/ProFLingo"]
summary: ProFLingo is a black-box LLM fingerprinting scheme for intellectual-property protection using adversarial-example-style queries.
provenance:
  extracted: 0.76
  inferred: 0.24
  ambiguous: 0.0
base_confidence: 0.88
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# ProFLingo

ProFLingo is the official implementation for "ProFLingo: A Fingerprinting-based Intellectual Property Protection Scheme for Large Language Models." The repository positions it as an IP-protection technique for LLMs.

## Key Ideas

- ProFLingo generates adversarial examples for a model and later verifies a candidate model with those generated examples.
- The repository provides `proflingo.py` for adversarial example generation and `copyright_test.py` for verification.
- The implementation includes fine-tuning experiments using Llama-2-7B and OpenHermes-2.5.
- The repository mentions GCG and ARCA as alternative adversarial-example generation paths.
- Experiments include Llama, Vicuna, Mistral, Gemma, Phi-2, OLMo-7B-Instruct, Yi, and other model families.

## Method Shape

ProFLingo is a [[concepts/black-box-model-fingerprinting|black-box fingerprinting]] technique: the verifier uses generated queries and model responses rather than inspecting model weights. Its project relevance is strongest for model-lineage checks after fine-tuning. ^[inferred]

## Practical Notes

- Verification may require specific transformer versions for some models.
- The repository says experiments were run on a single NVIDIA A10G with 24 GB of GPU memory.
- Multi-GPU adversarial-example generation is supported through `proflingo.py`.

## Project Relevance

For [[projects/robust-auditing/robust-auditing|Robust Auditing]], ProFLingo is a candidate technique for measuring whether targeted fine-tuning remains fingerprinted as the base or source model.

## Open Questions

- Does ProFLingo distinguish fine-tuned descendants that have preserved HolisticBias behavior but degraded off-audit fairness?
- How sensitive are ProFLingo fingerprints to the number and type of adversarial examples?
- Does using GCG versus ARCA change robustness to targeted fine-tuning?

## Sources

- [[references/proflingo-paper]]
- [[projects/robust-auditing/references/proflingo-upstream-and-patches]]
- [[concepts/llm-fingerprinting]]
- [[concepts/fingerprint-robustness]]
