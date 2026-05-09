---
title: Robust Auditing Repo Implementation State
category: concepts
tags: [repository, implementation, caveats, roadmap]
sources: ["/Users/angadkalra/Desktop/robust-auditing/README.md", "/Users/angadkalra/Desktop/robust-auditing/fyp-vault/Configuration.md", "/Users/angadkalra/Desktop/robust-auditing/tests", "user-clarification:2026-05-09T16:37:58Z"]
summary: Captures what is currently implemented, what is only documented, and what looks missing or in-progress in the repo.
provenance:
  extracted: 0.55
  inferred: 0.25
  ambiguous: 0.2
base_confidence: 0.58
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T16:37:58Z
---

# Robust Auditing Repo Implementation State

The current repository is strongest on fingerprint construction and verification. The fairness audit and targeted fine-tuning side is described in project configuration, but the tracked source package does not currently expose those modules.

## Implemented Or Visible

- Pinned submodules for [[entities/proflingo|ProFLingo]], [[entities/llmmap|LLMmap]], and [[entities/trap|TRAP]].
- OLMo2 compatibility patches under `patches/submodules/`.
- Build wrappers under `scripts/fingerprints/`.
- Generic lineage verification under `scripts/verification/`.
- OLMo2 lineage YAML configs for base, SFT, DPO, RLVR1, and Instruct references.
- Verification reports under `artifacts/verification/`.
- Tests covering script dry-runs, lineage parsing, matching helpers, cache cleanup, and OLMo2 prompt adapters.

## Missing Or Ambiguous

- `robust_auditing/fairness` and `robust_auditing/targeted_ft` have no tracked `.py` files in the current checkout, though pyc caches and older tests indicate modules such as adapters, loaders, losses, objectives, runner, sweeps, trainer, and metrics existed or were expected. ^[ambiguous]
- Tests for fairness and targeted fine-tuning are present only as pyc caches, not source tests, in the current checkout. ^[ambiguous]
- `artifacts/fingerprints/` is referenced by the README and configs, but visible checked-in artifacts are verification reports rather than raw fingerprint artifacts.
- The missing fairness and targeted fine-tuning source files are expected to be restored and extended from other branches after fingerprinting work is complete.
- The exact Anthropic HH harmless-base dataset wiring will become visible once those branches are merged.

## Practical Implication

The repo can currently be understood as a fingerprint-lineage analysis harness. Treat the audit-robustness training loop as a design goal unless the missing package source is restored or generated. ^[inferred]

## Clarifications Needed

- Whether the current end goal is a reproducible experimental pipeline, a paper figure pipeline, or a broader research notebook environment.
- The exact ProFLingo match-rate threshold for fingerprint retention.
- The exact LLMmap self-exclusion rule for top-1 identity reports.

## Sources

- [[projects/robust-auditing/concepts/repository-architecture]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[projects/robust-auditing/robust-auditing]]
- [[projects/robust-auditing/concepts/combined-robustness-thesis]]
