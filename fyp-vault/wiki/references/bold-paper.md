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
updated: 2026-05-16T16:53:36Z
---

# BOLD Paper

The BOLD paper, "BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation," introduces a benchmark for open-ended language generation bias. It uses English Wikipedia-derived prompts across profession, gender, race, religious belief, and political ideology.

## Key Ideas

- BOLD evaluates generated continuations, not just model likelihood over fixed sentence pairs.
- The paper frames bias as disproportionate generation of negative, unfair, prejudiced, or stereotypical text against groups or ideas.
- Section 3.3 describes anonymizing metric inputs by replacing people names with `Person` and profession, religious, or political group terms with `XYZ`.
- Section 4 evaluates generated text with metrics including sentiment, toxicity, regard, psycholinguistic norms, and gender polarity.
- The paper's sentiment metric uses VADER and discusses negative and positive thresholds over the compound score.
- The paper's toxicity metric uses a BERT toxicity classifier over generated text.
- Regard is narrower than sentiment and toxicity because the referenced classifier supports only selected gender and race groups.

## Project Relevance

- [[projects/robust-auditing/skills/run-fairness-baseline-audits|Run Fairness Baseline Audits]] implements `bold_stddev_toxicity_metric` from this generated-text framing.
- The local metric uses BOLD-style anonymization, continuous VADER sentiment transformed with `(compound + 1) / 2`, and the Toxic-BERT `toxic` label probability.
- The headline `bold_stddev_toxicity_metric` computes descriptor-level mean sentiment and toxicity within each BOLD axis, takes population standard deviations across descriptors, averages the two standard deviations, multiplies by 100, and then averages across axes. Because the classifier scores are on `[0, 1]`, the result is interpretable as a percentage-point standard deviation.
- Excluding regard from the default scalar keeps all BOLD axes scored with the same sentiment/toxicity components. ^[inferred]

## Sources

- https://arxiv.org/pdf/2101.11718
