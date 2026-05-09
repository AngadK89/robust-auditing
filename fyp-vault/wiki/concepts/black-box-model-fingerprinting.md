---
title: Black-Box Model Fingerprinting
category: concepts
tags: [llm, fingerprinting, black-box, evaluation]
aliases: [black-box fingerprinting, active fingerprinting]
sources: ["https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini", "https://github.com/pasquini-dario/LLMmap", "https://github.com/hengvt/ProFLingo"]
summary: Black-box fingerprinting identifies or verifies a model from inputs and outputs without inspecting weights.
provenance:
  extracted: 0.68
  inferred: 0.32
  ambiguous: 0.0
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# Black-Box Model Fingerprinting

Black-box model fingerprinting treats the model as a queryable service and uses responses to infer identity, ownership, or derivative status. It is the most relevant setting when the model is exposed through an API or integrated application.

## Key Ideas

- [[entities/llmmap|LLMmap]] sends carefully crafted queries to an application and analyzes responses to identify the LLM version in use.
- [[entities/proflingo|ProFLingo]] verifies models with generated adversarial examples, using model responses as the verification surface.
- Black-box tests are practical for hosted models but must handle system prompts, sampling hyperparameters, refusal behavior, RAG layers, and chain-of-thought wrappers.
- Query budget matters: LLMmap reports strong identification with very few interactions, which makes it relevant to low-cost audits and reconnaissance.

## Failure Modes

- A system prompt may suppress self-identification or route responses through a policy wrapper.
- Sampling changes can add noise to response traces.
- RAG and tool layers may make the application behavior less representative of the base LLM.
- Adaptive hosts can try to detect fingerprint queries, paraphrase responses, or fine-tune away fingerprint behavior. ^[inferred]

## Project Relevance

For [[projects/robust-auditing/robust-auditing|Robust Auditing]], black-box methods are useful because many real audit targets expose only API access. They are also risky because application-layer behavior can be confounded with model-level behavior.

## Open Questions

- How many repeated samples are needed for stable fingerprinting under stochastic decoding?
- Can fingerprint prompts be separated from fairness audit prompts without leakage between evaluations?
- Which defenses remove fingerprints without changing benchmark performance?

## Sources

- [[entities/llmmap]]
- [[entities/proflingo]]
- [[concepts/fingerprint-removal-and-adversarial-robustness]]
