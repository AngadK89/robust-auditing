---
title: Hot Cache
updated: 2026-05-09T18:21:43Z
---

# Hot Cache

*A ~500-word semantic snapshot of recent activity. Updated after every major write operation.*

## Recent Activity

- [2026-05-09T18:21:43Z] Synced post-1167c11 git delta: ProFLingo base/SFT/DPO/RLVR/Instruct artifacts and the new reference-relative robustness notebook.
- [2026-05-09T17:05:00Z] Synced recent git-log design decisions: model-first verification, repo-level HF cache cleanup, LLMmap artifact ownership, ProFLingo round-trip fixes, and multi-reference OLMo2 lineage configs.
- [2026-05-09T16:37:58Z] Recorded project-owner clarifications: generic model scope, combined robustness thesis, missing branch code, HH harmless-base, and fingerprint match criteria.

## Active Threads

- Robust Auditing: connect fingerprint robustness experiments with fixed-audit fairwashing experiments.
- OLMo2 lineage: quantify how far base, SFT, DPO, RLVR1, and Instruct variants remain inside reference fingerprint neighborhoods.
- Audit robustness: preserve HolisticBias while degrading Anthropic HH harmless-base behavior.

## Key Takeaways

- The repo currently has a mature fingerprint-lineage harness, while fairness and targeted fine-tuning source modules appear absent or in-progress in this checkout.
- ProFLingo upstream is FastChat-template centric for generation; the local patch changes suffix slot handling to tokenizer-native chat templates when available.
- LLMmap upstream assumes Hugging Face chat templates; the local patch and verifier add raw prompt fallbacks for base/plain tokenizers.
- Generic model support means Hugging Face `AutoModelForCausalLM`; fairness and targeted fine-tuning modules will be restored from other branches after fingerprinting work.
- Current OLMo2 verification artifacts show broad LLMmap persistence and high ProFLingo persistence across many post-training variants.
- The latest git history makes repo-owned artifacts and model-first verification explicit: LLMmap templates are accumulated under `artifacts/fingerprints`, and each lineage target is loaded once before running selected techniques.
- The ProFLingo robustness notebook now facets by reference model and marks each reference position, supporting distance-from-reference interpretation of the OLMo2 lineage.

## Flagged Contradictions

*None yet.*
