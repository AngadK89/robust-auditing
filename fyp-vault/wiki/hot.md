---
title: Hot Cache
updated: 2026-05-16T16:53:36Z
---

# Hot Cache

*A ~500-word semantic snapshot of recent activity. Updated after every major write operation.*

## Recent Activity

- [2026-05-16T16:53:36Z] Replaced the legacy BOLD harm-disparity metric with `bold_stddev_toxicity_metric`: a continuous sentiment/toxicity descriptor-spread score reported as percentage-point standard deviation.
- [2026-05-13T22:43:34Z] Added the adapter evaluation suite: one LoRA adapter path now triggers fixed ProFLingo, MedMCQA, HolisticBias, and BOLD evaluations into `artifacts/adapter_evals/<run_id>/`.
- [2026-05-13T14:05:27Z] Added response-based `full_gen_bias` for HolisticBias generated outputs.

## Active Threads

- Robust Auditing: connect fingerprint robustness experiments with fixed-audit fairwashing experiments.
- OLMo2 lineage: quantify how far base, SFT, DPO, RLVR1, and Instruct variants remain inside reference fingerprint neighborhoods.
- Audit robustness: reuse fixed HolisticBias/BOLD subsets across OLMo2 lineage models, then compare prompt-likelihood, response-variance, and BOLD generated sentiment/toxicity stddev metrics.
- Adapter evaluation: compare fine-tuned LoRA runs with the same OLMo2 base model, ProFLingo reference, MedMCQA eval IDs, and fairness subset.

## Key Takeaways

- The repo now has a fairness audit CLI layer: sample audit subsets once, generate responses per model, then score registered metrics from stored prompts or responses.
- The adapter evaluation suite turns a PEFT LoRA adapter directory into a comparable evaluation record by keeping all benchmark references fixed and writing per-adapter summaries.
- `bold_stddev_toxicity_metric` follows the original BOLD generated-text sentiment/toxicity framing: anonymize classifier text, score VADER sentiment and Toxic-BERT `toxic` probability continuously, average by descriptor, take population stddev across descriptors within each axis, multiply the averaged stddev by 100, and mean across axes.
- The `full_gen_bias` metric uses GoEmotions probabilities over censored generated responses and can score normalized audits such as HolisticBias and BOLD through shared `axis`/`descriptor` fields.
- ProFLingo upstream is FastChat-template centric for generation; the local patch changes suffix slot handling to tokenizer-native chat templates when available.
- LLMmap upstream assumes Hugging Face chat templates; the local patch and verifier add raw prompt fallbacks for base/plain tokenizers.
- Generic model support means Hugging Face `AutoModelForCausalLM`; targeted fine-tuning modules still need restoration from other branches.
- Current OLMo2 verification artifacts show broad LLMmap persistence and high ProFLingo persistence across many post-training variants.
- The latest git history makes repo-owned artifacts and model-first verification explicit: LLMmap templates are accumulated under `artifacts/fingerprints`, and each lineage target is loaded once before running selected techniques.
- The ProFLingo robustness notebook now facets by reference model and marks each reference position, supporting distance-from-reference interpretation of the OLMo2 lineage.
- Response-based metrics such as sentiment or toxicity can be added minimally by subclassing `FairnessMetric`, requiring `model_responses`, calling `context.load_responses()`, and registering in `METRIC_FACTORIES`.

## Flagged Contradictions

*None yet.*
