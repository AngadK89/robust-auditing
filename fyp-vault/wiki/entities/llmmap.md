---
title: LLMmap
category: entities
tags: [fingerprinting, llm, black-box, tool]
aliases: [LLMmap0.2]
sources: ["https://github.com/pasquini-dario/LLMmap", "https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini"]
summary: LLMmap is an active black-box fingerprinting tool for identifying LLM versions from behavioral traces.
provenance:
  extracted: 0.8
  inferred: 0.2
  ambiguous: 0.0
base_confidence: 0.83
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:17:44Z
---

# LLMmap

LLMmap is a minimal-query, high-accuracy tool for identifying LLMs by analyzing behavioral traces. Its tagline is "Like nmap, but for LLMs."

## Key Ideas

- LLMmap uses active fingerprinting: it sends carefully crafted queries and analyzes the responses.
- The USENIX Security 25 paper reports identification of 42 LLM versions with over 95% accuracy using as few as 8 interactions.
- The repository's LLMmap0.2 release is rebuilt in PyTorch and provides a ready-to-use open-set inference model.
- The default pretrained model includes PyTorch weights, a configuration file, and behavioral templates for 52 LLMs.
- LLMmap supports adding templates for new LLMs without retraining, currently emphasizing Hugging Face support in the README.

## Robustness Claims

LLMmap is designed to identify LLM versions across application layers, including unknown system prompts, stochastic sampling hyperparameters, RAG, and chain-of-thought frameworks.

## Project Relevance

For [[projects/robust-auditing/robust-auditing|Robust Auditing]], LLMmap helps test whether active behavioral fingerprints can detect model lineage or identity after targeted fine-tuning. Its application-layer framing is useful but may conflate base-model identity with wrapper behavior. ^[inferred]

## Open Questions

- Can LLMmap's behavioral templates distinguish OLMo lineage members and fine-tuned variants?
- How does template extension behave for local open-weight models versus API models?
- Does preserving fairness-audit behavior also preserve LLMmap behavioral traces?

## Sources

- [[concepts/black-box-model-fingerprinting]]
- [[concepts/fingerprint-robustness]]
- [[projects/robust-auditing/references/llmmap-upstream-and-patches]]
- USENIX page: https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini
- GitHub repository: https://github.com/pasquini-dario/LLMmap
