---
title: Verify Fingerprint Lineage
category: skills
tags: [fingerprinting, verification, lineage, workflow]
sources: ["/Users/angadkalra/Desktop/robust-auditing/README.md", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/fingerprint_lineage.py", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/fingerprint_methods.py", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/verify_fingerprint_lineage.py", "/Users/angadkalra/Desktop/robust-auditing/.git", "user-clarification:2026-05-09T16:37:58Z"]
summary: How the repo verifies reference fingerprints across configured OLMo2 lineage targets.
provenance:
  extracted: 0.82
  inferred: 0.18
  ambiguous: 0.0
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T17:05:00Z
---

# Verify Fingerprint Lineage

The generic verifier checks a configured reference fingerprint against a model lineage. It does not build fingerprints; it loads existing artifacts and replays or computes method-specific matches.

## Main Command

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml \
  --fingerprint proflingo llmmap \
  --output artifacts/verification/olmo2_instruct_reference_trajectory.json
```

## Lineage Config Model

- `reference.model_id` names the reference model whose fingerprint artifacts are being tested.
- `reference.artifact_root` usually points to `artifacts/fingerprints`.
- `reference.artifacts` maps requested techniques to artifact files, such as `proflingo_fingerprint`, `trap_suffixes`, and `llmmap_templates`.
- `targets` lists base, SFT, DPO, RLVR1, and Instruct models.
- `discover_revisions` can enumerate Hugging Face revisions matching patterns such as `step_(\d+)`.

## Technique Semantics

- ProFLingo: loads optimized suffixes, joins each suffix to `third_party/ProFLingo/questions.csv`, generates target-model responses, and reports normalized exact/prefix/contains diagnostics. The default match mode is prefix.
- TRAP: loads `suffixes.csv` or JSON suffix logs, sends adversarial digit prompts, extracts the first target-width digit string, and reports retrieval rate.
- LLMmap: generates candidate traces, computes a candidate template, compares it with the template DB, and treats top-1 reference label as a match.
- Project-level interpretation: ProFLingo should use a match-rate threshold, while LLMmap should use top-1 identity excluding the tested model itself from the candidate set when necessary.

## Execution Model

- Recent git history changed verification to a model-first loop: each target model is loaded once, then all requested techniques run against that same loaded model and tokenizer.
- This makes multi-technique reports more comparable because ProFLingo, TRAP, and LLMmap share the same loaded revision, dtype, device map, and tokenizer state for a target.
- LLMmap verification uses `run_llmmap_verification_for_loaded_model()` so it can reuse the verifier-loaded model and bypass upstream `load_llm()` for candidate evaluation.
- After each target, the verifier deletes model objects, clears torch memory, and evicts the Hugging Face model repo cache.
- Cache eviction is repo-scoped rather than revision-scoped because stale refs and auxiliary cached revisions made revision-level cleanup unreliable in lineage sweeps. ^[inferred]

## Report Shape

- Top-level fields include `lineage`, `reference`, `reference_model`, `fingerprint`, `targets`, `target_metadata`, and optional discovery metadata.
- Replay methods include `total`, `matched`, `match_rate`, and per-row diagnostics.
- LLMmap includes `matched_reference_top1`, `reference_model`, and `top_k` nearest labels with distances.

## Plotting

`scripts/verification/plot_lineage_verification.py` normalizes report JSON into lineage, replay, and LLMmap data frames. It plots replay match rates, LLMmap reference rank, and LLMmap reference distance across the lineage.

The plotting helper expects the newer lineage report schema with `targets` and `target_metadata`, sorts step revisions before `main` within each model lineage, and separates replay-style match-rate plots from LLMmap rank/distance plots.

## Sources

- [[projects/robust-auditing/skills/build-fingerprints]]
- [[projects/robust-auditing/references/current-olmo2-fingerprint-results]]
- [[projects/robust-auditing/references/proflingo-upstream-and-patches]]
- [[projects/robust-auditing/references/llmmap-upstream-and-patches]]
- [[projects/robust-auditing/concepts/combined-robustness-thesis]]
- [[projects/robust-auditing/references/recent-git-design-decisions]]
- [[concepts/fingerprint-robustness]]
