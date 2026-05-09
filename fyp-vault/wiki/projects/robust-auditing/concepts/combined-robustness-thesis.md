---
title: Combined Robustness Thesis
category: concepts
tags: [thesis, fingerprinting, auditing, fine-tuning]
sources: ["user-clarification:2026-05-09T16:37:58Z", "/Users/angadkalra/Desktop/robust-auditing/fyp-vault/Configuration.md"]
summary: The project links over-robust fingerprints and over-robust audits into one constrained evasive fine-tuning argument.
provenance:
  extracted: 0.75
  inferred: 0.25
  ambiguous: 0.0
base_confidence: 0.58
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:37:58Z
updated: 2026-05-09T16:37:58Z
---

# Combined Robustness Thesis

The project is not choosing between fingerprint robustness and audit robustness. It is trying to show both failure modes and then connect them.

## Core Claim

- Fingerprints are too robust in model space: different generations or post-training stages of a model lineage can still produce the same fingerprinting outcome.
- Audits are too robust in parameter space: a model can retain audit performance even after strong parameter changes.
- The fingerprint robustness radius gives an upper bound on how far targeted fine-tuning can move while still remaining fingerprinted as the original or reference model.
- Within that radius, the project aims to show poisoning or fairwashing is possible while retaining accuracy on the known audit set.

## Experimental Meaning

- The fingerprinting experiment estimates the boundary of model movement that ProFLingo, LLMmap, and possibly TRAP still treat as the same model.
- The audit experiment uses that boundary to constrain targeted fine-tuning so the modified model remains fingerprint-equivalent while changing off-audit fairness behavior. ^[inferred]
- The audit set is [[concepts/holistic-bias|HolisticBias]].
- The off-audit degradation data is Anthropic HH harmless-base.

## Match Criteria

- ProFLingo should be interpreted through a match-rate threshold.
- LLMmap should be interpreted by top-1 identity, excluding trivial self-matches when the tested model itself is present in the candidate template set.

## Open Questions

- What exact ProFLingo threshold should be used for fingerprint retention?
- How should self-exclusion be implemented in LLMmap reports and plots?
- What fine-tuning budget should be derived from the observed fingerprint radius?

## Sources

- [[projects/robust-auditing/robust-auditing]]
- [[concepts/fingerprint-robustness]]
- [[concepts/fairness-audit-sets]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
