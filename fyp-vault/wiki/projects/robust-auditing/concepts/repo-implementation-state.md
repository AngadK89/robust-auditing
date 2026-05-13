---
title: Robust Auditing Repo Implementation State
category: concepts
tags: [repository, implementation, caveats, roadmap]
sources: ["/Users/angadkalra/Desktop/robust-auditing/README.md", "/Users/angadkalra/Desktop/robust-auditing/fyp-vault/Configuration.md", "/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_CLI_WORKFLOW.md", "/Users/angadkalra/Desktop/robust-auditing/robust_auditing/fairness", "/Users/angadkalra/Desktop/robust-auditing/tests", "user-clarification:2026-05-09T16:37:58Z"]
summary: >-
  Captures what is currently implemented, what is only documented, and what looks missing or in-progress in the repo.
provenance:
  extracted: 0.65
  inferred: 0.25
  ambiguous: 0.1
base_confidence: 0.66
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-13T20:03:56Z
---

# Robust Auditing Repo Implementation State

The current repository is strongest on fingerprint construction and verification, and now also has a tracked fairness audit CLI layer for sampling HolisticBias/BOLD subsets, generating model responses, and scoring registered metrics. Targeted fine-tuning remains less concrete in the active checkout.

## Implemented Or Visible

- Pinned submodules for [[entities/proflingo|ProFLingo]], [[entities/llmmap|LLMmap]], and [[entities/trap|TRAP]].
- OLMo2 compatibility patches under `patches/submodules/`.
- Build wrappers under `scripts/fingerprints/`.
- Generic lineage verification under `scripts/verification/`.
- OLMo2 lineage YAML configs for base, SFT, DPO, RLVR1, and Instruct references.
- Verification reports under `artifacts/verification/`.
- Tests covering script dry-runs, lineage parsing, matching helpers, cache cleanup, and OLMo2 prompt adapters.
- `robust_auditing/fairness` source modules for dataset adapters, artifact paths, subset sampling, deterministic response generation, metric scoring, and registered metrics including `likelihood_bias`, `full_gen_bias`, and `bold_negative_harm_disparity`.
- Fairness CLI wrappers under `scripts/fairness/` for subset sampling, response generation, metric scoring, and the older combined baseline runner.
- Tests covering HolisticBias/BOLD normalization, proportional descriptor sampling, subset-aware artifact paths, response generation, and response-based metric consumption.
- The BOLD-specific harm-disparity metric follows the original [[references/bold-paper|BOLD paper]] framing by scoring generated responses for negative sentiment and toxicity, then reporting group harm-rate gaps with overall harm rate.

## Missing Or Ambiguous

- `robust_auditing/targeted_ft` has no tracked `.py` files in the current checkout, though pyc caches and branch notes indicate modules such as adapters, loaders, losses, objectives, runner, sweeps, and trainer existed or were expected. ^[ambiguous]
- `artifacts/fingerprints/` is referenced by the README and configs, but visible checked-in artifacts are verification reports rather than raw fingerprint artifacts.
- The targeted fine-tuning source files are expected to be restored and extended from other branches after fingerprinting work is complete.
- The exact Anthropic HH harmless-base dataset wiring will become visible once those branches are merged.

## Practical Implication

The repo can currently be understood as a fingerprint-lineage analysis harness plus a fairness-audit inference/scoring harness. Treat the audit-robustness training loop as a design goal unless the targeted fine-tuning package source is restored or generated. ^[inferred]

## Clarifications Needed

- Whether the current end goal is a reproducible experimental pipeline, a paper figure pipeline, or a broader research notebook environment.
- The exact ProFLingo match-rate threshold for fingerprint retention.
- The exact LLMmap self-exclusion rule for top-1 identity reports.

## Sources

- [[projects/robust-auditing/concepts/repository-architecture]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[projects/robust-auditing/robust-auditing]]
- [[projects/robust-auditing/concepts/combined-robustness-thesis]]
