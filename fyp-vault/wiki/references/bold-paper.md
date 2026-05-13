---
title: >-
  BOLD Paper
category: references
tags: [paper, fairness, bias, dataset, generation]
sources: ["https://arxiv.org/pdf/2101.11718"]
summary: >-
  Original BOLD paper defining Wikipedia-derived open-ended generation prompts and generated-text bias metrics.
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
base_confidence: 0.9
lifecycle: draft
lifecycle_changed: 2026-05-13
created: 2026-05-13T20:03:56Z
updated: 2026-05-13T20:21:00Z
---

# BOLD Paper

The BOLD paper, "BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation," introduces a benchmark for open-ended language generation bias. It uses English Wikipedia-derived prompts across profession, gender, race, religious belief, and political ideology.

## Key Ideas

- BOLD evaluates generated continuations, not just model likelihood over fixed sentence pairs.
- The paper frames bias as disproportionate generation of negative, unfair, prejudiced, or stereotypical text against groups or ideas.
- Section 3.3 describes anonymizing metric inputs by replacing people names with `Person` and profession, religious, or political group terms with `XYZ`.
- Section 4 evaluates generated text with metrics including sentiment, toxicity, regard, psycholinguistic norms, and gender polarity.
- The sentiment metric uses VADER and treats scores `<= -0.5` as negative and `>= 0.5` as positive.
- The toxicity metric uses a BERT toxicity classifier and marks text toxic when any of the tracked toxicity labels fires.
- Regard is narrower than sentiment and toxicity because the referenced classifier supports only selected gender and race groups.

## Project Relevance

- [[projects/robust-auditing/skills/run-fairness-baseline-audits|Run Fairness Baseline Audits]] implements `bold_negative_harm_disparity` from this generated-text framing.
- The local metric uses BOLD-style anonymization, VADER negative sentiment, and toxic-bert toxicity to compute response-level harm scores.
- The headline `bold_harm_gap` is a percent-scale disparity measure across BOLD descriptor harm rates, while percent-scale `overall_harm_rate` reports total generated harm.
- Excluding regard from the default scalar keeps all BOLD axes scored with the same sentiment/toxicity components. ^[inferred]

## Sources

- https://arxiv.org/pdf/2101.11718
