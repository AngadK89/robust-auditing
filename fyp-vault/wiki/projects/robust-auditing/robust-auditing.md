---
title: Robust Auditing
category: project
tags: [llm, auditing, fingerprinting, fairness]
sources: ["/Users/angadkalra/Desktop/robust-auditing/fyp-vault/Configuration.md", "/Users/angadkalra/Desktop/robust-auditing/.codex/AGENTS.md", "/Users/angadkalra/Desktop/robust-auditing/README.md", "/Users/angadkalra/Desktop/robust-auditing/docs/FAIRNESS_CLI_WORKFLOW.md", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/verify_fingerprint_lineage.py", "user-clarification:2026-05-09T16:37:58Z"]
summary: >-
  Project overview for testing whether fixed fairness audits and black-box LLM fingerprints miss targeted model changes.
provenance:
  extracted: 0.74
  inferred: 0.26
  ambiguous: 0.0
base_confidence: 0.64
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-13T20:03:56Z
---

# Robust Auditing

This project studies whether LLM owners can change a model's behavior while still passing fixed evaluations: a [[concepts/fairness-audit-sets|fairness audit set]] such as [[concepts/holistic-bias|HolisticBias]], and model provenance checks such as [[concepts/llm-fingerprinting|LLM fingerprinting]].

## Key Ideas

- The fingerprinting side asks whether black-box techniques are too broad in model space, treating fine-tuned descendants as if they were still the base model.
- The audit side asks whether a model can preserve behavior on a known audit set while degrading fairness-relevant behavior off-audit.
- A useful experimental outcome is a "fine-tuning radius": the amount and style of update that still leaves a model inside the same fingerprint neighborhood. ^[inferred]
- The project should keep audit, non-audit, capability, and fingerprinting evaluations separate so it can measure which guardrail failed.
- The repository implements the fingerprinting half with construction scripts, lineage configs, verifier code, tests, and verification artifacts.
- The fairness audit side now has CLI tools for proportional HolisticBias/BOLD subset sampling, deterministic response generation, and registered metric scoring, including a BOLD paper-derived generated-text harm disparity metric.
- The thesis is explicitly two-sided: fingerprints are too robust in model space, and audits are too robust in parameter space.
- The fingerprint robustness radius defines the allowed boundary for targeted fine-tuning: the project wants to show a model can be poisoned while retaining audit accuracy and still remaining inside the fingerprint-equivalence region.
- Generic model support for the fingerprinting scripts should mean any Hugging Face `AutoModelForCausalLM`.
- The off-audit fairness degradation dataset is Anthropic HH harmless-base only; exact code-level dataset wiring will become visible when other branches are merged.

## Repository Map

- [[projects/robust-auditing/concepts/repository-architecture]] - how README goals map to folders, scripts, artifacts, and tests.
- [[projects/robust-auditing/skills/build-fingerprints]] - how to build LLMmap, ProFLingo, and TRAP fingerprints.
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]] - how lineage YAML is expanded and verified.
- [[projects/robust-auditing/skills/run-fairness-baseline-audits]] - how audit subsets, responses, and metrics are generated and stored.
- [[projects/robust-auditing/references/current-olmo2-fingerprint-results]] - what the checked-in verification artifacts show.
- [[projects/robust-auditing/concepts/repo-implementation-state]] - caveats about present versus expected functionality.

## Core Pages

- [[concepts/llm-fingerprinting]] - terminology for ownership, identity, and provenance fingerprints.
- [[concepts/fingerprint-robustness]] - persistence under fine-tuning, prompting, RAG, and other perturbations.
- [[concepts/fingerprint-removal-and-adversarial-robustness]] - erasure and adaptive attacks against fingerprints.
- [[concepts/fairness-audit-sets]] - fixed benchmark sets used to audit bias and fairness.
- [[concepts/holistic-bias]] - the main demographic descriptor audit set in this project.
- [[references/bold-paper]] - the source paper for BOLD open-ended generation prompts and generated-text bias metrics.
- [[entities/proflingo]] and [[entities/llmmap]] - specific fingerprinting systems.
- [[entities/trap]] - the TRAP suffix-based black-box identity verification method used by this repo.

## Open Questions

- How much fine-tuning can OLMo-family models absorb before ProFLingo-like or LLMmap-like tests distinguish them from the base?
- Which off-audit datasets best expose behavior not covered by HolisticBias?
- Which fingerprint robustness claims are benign-perturbation claims, and which survive adaptive adversaries?
- How should the final experiment connect the fingerprint-radius bound to the targeted fine-tuning budget?
- What exact ProFLingo match-rate threshold should count as fingerprint retention?

## Sources

- [[references/holisticbias-paper]]
- [[references/bold-paper]]
- [[references/proflingo-paper]]
- [[references/copyright-protection-for-llms-survey]]
- [[references/are-robust-llm-fingerprints-adversarially-robust]]
- [[projects/robust-auditing/concepts/repository-architecture]]
- [[projects/robust-auditing/concepts/combined-robustness-thesis]]
