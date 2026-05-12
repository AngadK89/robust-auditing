---
title: Fairness Audit Sets
category: concepts
tags: [fairness, auditing, benchmarks, llm]
aliases: [bias audit sets, fairness benchmarks]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2205.09209v2.pdf", "https://huggingface.co/datasets/fairnlp/holistic-bias", "/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_CLI_WORKFLOW.md"]
summary: Fairness audit sets are fixed benchmark prompts or examples used to measure bias, stereotypes, or harmful disparities.
provenance:
  extracted: 0.62
  inferred: 0.36
  ambiguous: 0.02
base_confidence: 0.78
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-12T19:19:31Z
---

# Fairness Audit Sets

Fairness audit sets are collections of prompts, descriptors, templates, labels, or examples used to evaluate whether a model behaves differently across demographic or social groups.

## Key Ideas

- [[concepts/holistic-bias|HolisticBias]] uses demographic descriptors and sentence templates to measure likelihood bias and related harms.
- A fixed audit set can become a target: a model owner can train to preserve measured behavior on that set while changing behavior elsewhere. ^[inferred]
- Audit sets often trade coverage for repeatability; the more fixed and public the set is, the easier it is to overfit to the audit. ^[inferred]
- Audit results should be separated from non-audit fairness probes and general capability tests in [[projects/robust-auditing/robust-auditing|Robust Auditing]].
- A proportional descriptor subset preserves the original descriptor prevalence approximately while reducing compute, making repeated lineage inference feasible on the same fixed prompt set.
- Response-based metrics can score generated text after inference, while prompt-based metrics can score the stored prompts directly.

## Audit Vocabulary

- Audit set: the specific held-out or public examples used for evaluation.
- Non-audit set: examples intended to measure behavior beyond the known audit target.
- Fairwashing: preserving apparent fairness on measured criteria while degrading unmeasured behavior.
- Descriptor: demographic or identity term used to instantiate a prompt.
- Template: reusable sentence frame into which descriptors or noun phrases are inserted.
- Subset id: a named, stored sample of an audit set reused across model runs.
- Model response artifact: generated text for each normalized prompt, used by response-based metrics such as sentiment or toxicity scoring.

## Project Relevance

This project's audit side depends on a split between measured audit behavior and off-audit behavior. The current CLI workflow samples [[concepts/holistic-bias|HolisticBias]] and BOLD into reusable subsets, runs inference per model checkpoint, and applies prompt- or response-based metrics to compare fairness behavior across the OLMo2 lineage.

## Open Questions

- Which non-audit data should be paired with HolisticBias for off-audit degradation tests?
- How should the project quantify "preserved" audit performance while changing non-audit behavior?

## Sources

- [[references/holisticbias-paper]]
- [[concepts/holistic-bias]]
- [[projects/robust-auditing/robust-auditing]]
