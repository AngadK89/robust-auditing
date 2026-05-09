---
title: Current OLMo2 Fingerprint Results
category: references
tags: [results, fingerprinting, olmo2, verification]
sources: ["/Users/angadkalra/Desktop/robust-auditing/artifacts/verification/olmo2_instruct_reference_trajectory.json", "/Users/angadkalra/Desktop/robust-auditing/artifacts/verification/olmo2_fingerprint_verification.json", "/Users/angadkalra/Desktop/robust-auditing/artifacts/verification/olmo2_sft_proflingo_reference_trajectory.json", "/Users/angadkalra/Desktop/robust-auditing/artifacts/verification/olmo2_dpo_proflingo_reference_trajectory.json", "/Users/angadkalra/Desktop/robust-auditing/artifacts/verification/olmo2_rlvr1_proflingo_reference_trajectory.json", "/Users/angadkalra/Desktop/robust-auditing/artifacts/verification/olmo2_instruct_proflingo_reference_trajectory.json", "/Users/angadkalra/Desktop/robust-auditing/notebooks/plot_olmo2_proflingo_reference_robustness.ipynb"]
summary: Current checked-in OLMo2 ProFLingo reports show broad post-training fingerprint persistence, with base as the main low-match contrast.
provenance:
  extracted: 0.78
  inferred: 0.2
  ambiguous: 0.02
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T18:21:43Z
---

# Current OLMo2 Fingerprint Results

The current checked-in report `artifacts/verification/olmo2_instruct_reference_trajectory.json` uses `allenai/OLMo-2-0425-1B-Instruct` as the reference and evaluates ProFLingo plus LLMmap against OLMo2 base, SFT, DPO, RLVR1, and Instruct targets.

Newer ProFLingo-only trajectory reports also compare SFT, DPO, RLVR1, and Instruct reference fingerprints against the same ordered model lineage. The notebook `notebooks/plot_olmo2_proflingo_reference_robustness.ipynb` plots one graph per reference model and marks the reference model's own position to show how match rate changes as the lineage moves away from it.

## Extracted Results

- ProFLingo gives `base@main` a match rate of 0.0.
- ProFLingo gives `sft@main` 0.78, `dpo@main` 0.92, and `rlvr1@main` 0.96.
- ProFLingo gives most RLVR1 step revisions match rates from 0.94 to 0.98.
- ProFLingo gives `instruct@main` 0.96 and most Instruct step revisions around 0.94 to 0.98.
- LLMmap reports `matched_reference_top1: true` for every displayed target in the instruct-reference trajectory, including `base@main`.
- The older `olmo2_fingerprint_verification.json` report shows ProFLingo at 0.0 for base and 0.96 for instruct, while its LLMmap section appears older or differently shaped. ^[ambiguous]

## Multi-Reference ProFLingo Results

- The SFT reference report gives `base@main` 0.08, `sft@main` 1.0, `dpo@main` 1.0, `rlvr1@main` 1.0, and `instruct@main` 1.0.
- The DPO reference report gives `base@main` 0.20, `sft@main` 0.96, `dpo@main` 1.0, `rlvr1@main` 1.0, and `instruct@main` 1.0.
- The RLVR1 reference report gives `base@main` 0.16, `sft@main` 0.90, `dpo@main` 1.0, `rlvr1@main` 1.0, and `instruct@main` 1.0.
- The Instruct ProFLingo-only report gives `base@main` 0.26, `sft@main` 0.96, `dpo@main` 1.0, `rlvr1@main` 1.0, and `instruct@main` 1.0.
- Across these reports, base is consistently the weakest match against post-training references, while SFT, DPO, RLVR1, and Instruct mostly remain inside each other's ProFLingo neighborhoods.
- The base-reference ProFLingo trajectory artifact exists but was intentionally left optional in the notebook because the user considers it not yet usable in the same format.

## Interpretation

These results are early evidence for the project's [[concepts/fingerprint-robustness|fingerprint robustness]] concern: at least LLMmap, and often ProFLingo, continue matching the instruct reference across substantial lineage movement. The base-model ProFLingo zero rate is an important contrast. ^[inferred]

The multi-reference ProFLingo artifacts strengthen the same interpretation: the fingerprint neighborhood around several post-training references appears broad enough to include many later derivatives. The faceted notebook supports reading this as distance-from-reference behavior rather than only as a single-reference lineage sweep. ^[inferred]

## Caveats

- These reports are artifacts, not a rerun performed during this wiki update.
- The LLMmap top-1 match for `base@main` should be reviewed carefully because it implies very broad behavioral similarity under the configured template-distance procedure. ^[inferred]
- Verification results depend on existing fingerprint artifacts, generation settings, prompt templates, and cache state.

## Sources

- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[concepts/fingerprint-robustness]]
- [[entities/llmmap]]
- [[entities/proflingo]]
