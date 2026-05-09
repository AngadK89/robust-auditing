---
title: Robust Auditing Repository Architecture
category: concepts
tags: [repository, architecture, fingerprinting, auditing]
sources: ["/Users/angadkalra/Desktop/robust-auditing/README.md", "/Users/angadkalra/Desktop/robust-auditing/fyp-vault/Configuration.md", "/Users/angadkalra/Desktop/robust-auditing/.codex/AGENTS.md", "/Users/angadkalra/Desktop/robust-auditing/.git"]
summary: Maps the robust-auditing repo structure to its fingerprint robustness and audit robustness goals.
provenance:
  extracted: 0.78
  inferred: 0.2
  ambiguous: 0.02
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T17:05:00Z
---

# Robust Auditing Repository Architecture

The repository currently centers on constructing and verifying [[concepts/llm-fingerprinting|LLM fingerprints]] across the OLMo2-0425-1B lineage. It also contains project notes for the planned [[concepts/fairness-audit-sets|fairness audit]] and fairwashing side of the work.

## Top-Level Structure

- `third_party/ProFLingo`, `third_party/LLMmap`, and `third_party/trap` pin the three fingerprinting codebases.
- `patches/submodules/` contains compatibility patches for OLMo2 support and local runtime behavior.
- `scripts/fingerprints/` wraps fingerprint construction for [[entities/llmmap|LLMmap]], [[entities/proflingo|ProFLingo]], and [[entities/trap|TRAP]].
- `configs/fingerprint_lineages/` defines OLMo2 reference models, artifact paths, targets, and revision-discovery rules.
- `scripts/verification/` loads lineage configs, replays or computes technique-specific fingerprint checks, and writes JSON reports.
- `artifacts/verification/` contains checked-in reports from OLMo2 fingerprint verification runs.
- `notebooks/plot_olmo_lineage_verification.ipynb` is used for visual analysis of lineage verification.
- `fyp-vault/` is the project knowledge vault and source staging area.

## Goal Alignment

- Fingerprint robustness is supported directly by scripts that build reference fingerprints, expand OLMo2 lineage targets, and verify the reference fingerprint against base, SFT, DPO, RLVR1, and Instruct variants.
- Recent configs extend the lineage design to multiple reference anchors: base, SFT, DPO, RLVR1, and Instruct can each be used as the reference whose fingerprint neighborhood is measured against the same OLMo2 targets.
- Audit robustness is described in `Configuration.md`, but the current tracked code does not expose a working fairness or targeted-fine-tuning package in `robust_auditing/`. ^[ambiguous]
- The checked-in verification artifact already supports the "too robust in model space" hypothesis: LLMmap identifies all OLMo lineage targets as the instruct reference top-1, and ProFLingo gives high match rates across many post-training variants while base is zero.

## Recent Design Direction

- The verification harness is moving toward target-model reuse: load a target once, run all requested fingerprint techniques, then clean memory/cache before the next target.
- Fingerprint generation wrappers centralize `.env` loading and repo artifact ownership so repeated runs compose into tracked artifacts rather than hidden third-party state.
- Patches are increasingly framed around Hugging Face `AutoModelForCausalLM` portability: tokenizer chat templates are preferred when present, and plain prompt fallbacks are used for base models without chat templates.

## Tests As Documentation

- `tests/test_fingerprint_scripts.py` documents dry-run behavior, `.env` loading, artifact seeding, and missing-model usage errors.
- `tests/test_fingerprint_lineage.py` documents lineage YAML parsing, revision discovery, artifact path requirements, technique selection, and cleanup guarantees.
- `tests/test_fingerprint_methods.py` documents ProFLingo/TRAP case loading, normalized matching helpers, LLMmap nearest-label ranking, and safe Hugging Face cache eviction.
- Adapter tests document OLMo2-related patches for ProFLingo, LLMmap, and TRAP.

## Caveats

- The source package `robust_auditing/` currently has no tracked `.py` files, despite tests and pyc caches suggesting fairness and targeted fine-tuning modules existed or were expected. ^[ambiguous]
- The README mentions generated fingerprints under `artifacts/fingerprints/`, but in the current checkout only verification artifacts are visible at shallow depth.
- The submodule patches are applied idempotently by build scripts; inspecting vendored source before running the patch script may not show OLMo2 behavior.

## Sources

- [[projects/robust-auditing/robust-auditing]]
- [[projects/robust-auditing/skills/build-fingerprints]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[projects/robust-auditing/concepts/repo-implementation-state]]
- [[projects/robust-auditing/references/recent-git-design-decisions]]
