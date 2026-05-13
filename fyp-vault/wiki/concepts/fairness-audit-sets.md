---
title: Fairness Audit Sets
category: concepts
tags: [fairness, auditing, benchmarks, llm]
aliases: [bias audit sets, fairness benchmarks]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/sources/raw_sources/2205.09209v2.pdf", "https://huggingface.co/datasets/fairnlp/holistic-bias", "/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_CLI_WORKFLOW.md", "https://arxiv.org/pdf/2101.11718"]
summary: >-
  Fairness audit sets are fixed benchmark prompts or examples used to measure bias, stereotypes, or harmful disparities.
provenance:
  extracted: 0.62
  inferred: 0.36
  ambiguous: 0.02
base_confidence: 0.78
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-13T20:21:00Z
---

# Fairness Audit Sets

Fairness audit sets are collections of prompts, descriptors, templates, labels, or examples used to evaluate whether a model behaves differently across demographic or social groups.

## Key Ideas

- [[concepts/holistic-bias|HolisticBias]] uses demographic descriptors and sentence templates to measure likelihood bias and related harms.
- [[references/bold-paper|BOLD]] uses Wikipedia-derived open-ended generation prompts across profession, gender, race, religion, and political ideology.
- A fixed audit set can become a target: a model owner can train to preserve measured behavior on that set while changing behavior elsewhere. ^[inferred]
- Audit sets often trade coverage for repeatability; the more fixed and public the set is, the easier it is to overfit to the audit. ^[inferred]
- Audit results should be separated from non-audit fairness probes and general capability tests in [[projects/robust-auditing/robust-auditing|Robust Auditing]].
- A proportional descriptor subset preserves the original descriptor prevalence approximately while reducing compute, making repeated lineage inference feasible on the same fixed prompt set.
- Response-based metrics can score generated text after inference, while prompt-based metrics can score the stored prompts directly.
- FullGenBias is a response-based normalized-data metric: generated responses are descriptor-censored before GoEmotions classification, then aggregated by template-level or axis-level descriptor variance.
- BOLD harm-disparity scoring is response-based: generated text is anonymized for classifier input, scored for negative sentiment and toxicity, and aggregated as descriptor harm-rate gaps within each BOLD axis.

## Audit Vocabulary

- Audit set: the specific held-out or public examples used for evaluation.
- Non-audit set: examples intended to measure behavior beyond the known audit target.
- Fairwashing: preserving apparent fairness on measured criteria while degrading unmeasured behavior.
- Descriptor: demographic or identity term used to instantiate a prompt.
- Template: reusable sentence frame into which descriptors or noun phrases are inserted.
- Subset id: a named, stored sample of an audit set reused across model runs.
- Model response artifact: generated text for each normalized prompt, used by response-based metrics such as sentiment or toxicity scoring.
- FullGenBias: `1000 * mean_template sum_emotion Var_descriptor(mean_response_prob)`, using censored generated responses and GoEmotions probabilities.
- BOLD harm gap: percent-scale mean across BOLD axes of the max-minus-min descriptor harm-rate gap.
- Overall harm rate: percent-scale mean response harm score reported beside BOLD harm gap so equal harm increases across groups are visible.

## Project Relevance

This project's audit side depends on a split between measured audit behavior and off-audit behavior. The current CLI workflow samples [[concepts/holistic-bias|HolisticBias]] and [[references/bold-paper|BOLD]] into reusable subsets, runs inference per model checkpoint, and applies prompt- or response-based metrics to compare fairness behavior across the OLMo2 lineage.

## Open Questions

- Which non-audit data should be paired with HolisticBias for off-audit degradation tests?
- How should the project quantify "preserved" audit performance while changing non-audit behavior?

## Sources

- [[references/holisticbias-paper]]
- [[references/bold-paper]]
- [[concepts/holistic-bias]]
- [[projects/robust-auditing/robust-auditing]]
