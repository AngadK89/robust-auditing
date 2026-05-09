---
title: Recent Git Design Decisions
category: references
tags: [git-history, design-decisions, fingerprinting, lineage]
sources: ["/Users/angadkalra/Desktop/robust-auditing/.git"]
summary: >-
  Recent commits show the fingerprint work converging on model-first verification, repo-owned artifacts, generic HF prompt fallbacks, multi-reference OLMo2 lineage checks, and reference-relative ProFLingo plots.
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
base_confidence: 0.72
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T17:05:00Z
updated: 2026-05-09T18:21:43Z
---

# Recent Git Design Decisions

This page distills design-level knowledge from the latest git history on `feat/base-fingerprint`, especially commits around May 5-6, 2026.

## Model-First Verification

- Commit `59b0ece` rewrote lineage verification to load each target model once, then run ProFLingo, TRAP, and LLMmap against that loaded model.
- This avoids separate model loads per technique and makes mixed-technique reports use the same target revision, tokenizer, dtype, and device map.
- `run_llmmap_verification_for_loaded_model()` exists so LLMmap can reuse the verifier-loaded Hugging Face model instead of calling upstream `load_llm()`.
- The verifier now builds empty report sections for requested techniques, then fills results target by target.
- After each target, the verifier deletes model/tokenizer objects, runs torch cleanup, and evicts Hugging Face cache state.

## Cache Cleanup Policy

- Commit `e1e2266` changed cleanup from revision-scoped Hugging Face deletion to repo-level deletion.
- The reason is operational: Transformers can cache refs and auxiliary revisions in ways that make later revision-specific cleanup unreliable. The repo-level deletion is broader but easier to reason about for large lineage sweeps. ^[inferred]
- The cleanup code rejects invalid model IDs, symlinks, escaped paths, and non-directory repo paths before deleting.

## Generic Prompt Formatting

- Commit `6886542` extended all three fingerprint integrations toward base-model support.
- LLMmap gained a tokenizer-without-chat-template fallback: `system + "\n\n" + user`, or raw `user` if no system prompt is present.
- ProFLingo template choice changed from model-name heuristics to `bool(tokenizer.chat_template)`: use tokenizer-native chat templates when available, otherwise keep FastChat `alpaca` and `zero_shot`.
- TRAP's OLMo2 patch added a small template shim that can compute changed token spans using either tokenizer chat templates or plain prompt text.
- These commits make generic support mean "Hugging Face causal LMs with usable tokenization/prompt formatting," not API-only models or arbitrary architectures. ^[inferred]

## ProFLingo Token Round-Trip Fixes

- Commit `738edc4` made the ProFLingo patch less Meta-chat-specific and safer for non-chat models.
- The patch stopped assuming `tokenizer.encode(decoded)[2:]` removes exactly two special tokens.
- It added `_roundtrip_ids()` and bounded `_sample_roundtrip_prompt_ids()` so suffix initialization fails loudly instead of looping forever.
- It replaced `<unk>` placeholder localization with unique sentinel strings, which avoids requiring placeholder text to encode as `unk_token_id`.
- Tests cover no-special-token tokenizers, empty filter words, context-sensitive placeholder tokenization, and bounded failure.

## LLMmap Artifact Ownership

- Commit `8ced17a` changed LLMmap template generation so the repo artifact is the source of continuity.
- Before running upstream `add_new_template.py`, `make_llmmap_template.sh` copies `artifacts/fingerprints/llmmap/templates.json` into the LLMmap pretrained model directory if the artifact exists.
- After enrollment, it copies the updated `templates.json` and optional `templates.json.previous` back to `artifacts/fingerprints/llmmap`.
- This prevents repeated enrollments from silently starting from the third-party bundle instead of the accumulated project template DB.

## Multi-Reference OLMo2 Lineage

- Commits `82ea173`, `37ac279`, and `1167c11` show the project moving from one instruct-reference run toward reference fingerprints for base, SFT, DPO, and RLVR1 as well.
- The new configs `olmo2_1b_reference.yaml`, `olmo2_1b_sft_reference.yaml`, `olmo2_1b_dpo_reference.yaml`, and `olmo2_1b_rlvr1_reference.yaml` all compare the chosen reference against base, SFT, DPO, RLVR1, and Instruct targets.
- RLVR1 and Instruct target revisions are discovered with `step_(\d+)` and an increment of 400.
- The design intent is to measure fingerprint neighborhoods around multiple points in the OLMo2 post-training graph, not only around Instruct. ^[inferred]
- Later commits `c1877fc`, `1b0d1b9`, and `ca9a9f8` add or refresh ProFLingo artifacts for OLMo2 base, SFT, DPO, RLVR1, and Instruct reference checks.
- The checked-in trajectory artifacts now support comparing post-training reference neighborhoods directly, not just running one reference against a fixed target list.

## Visualization Contract

- Commit `65c92c8` introduced `scripts/verification/plot_lineage_verification.py` as the durable plotting layer behind the notebook.
- The plotting module explicitly rejects older reports that lack `targets` or `target_metadata`.
- It sorts targets by lineage group, step revisions before `main`, then report order.
- It produces separate normalized data frames for lineage metadata, replay-style match rates, and LLMmap nearest-template behavior.
- Commit `1b46a06` adds `notebooks/plot_olmo2_proflingo_reference_robustness.ipynb`, a quick analysis notebook for the ProFLingo-only multi-reference artifacts.
- That notebook intentionally uses one subplot per reference model and marks the reference model's own x-axis position, so the reader can see how match rate changes as the model lineage moves away from the reference.
- The base-reference ProFLingo artifact is represented as an optional report in the notebook, making it easy to include once its format is considered usable.

## Sources

- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[projects/robust-auditing/skills/build-fingerprints]]
- [[projects/robust-auditing/concepts/repository-architecture]]
- [[projects/robust-auditing/references/proflingo-upstream-and-patches]]
- [[projects/robust-auditing/references/llmmap-upstream-and-patches]]
