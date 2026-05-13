---
title: Hot Cache
updated: 2026-05-13T20:21:00Z
---

# Hot Cache

*A ~500-word semantic snapshot of recent activity. Updated after every major write operation.*

## Recent Activity

- [2026-05-13T20:21:00Z] Updated `bold_negative_harm_disparity` aggregate outputs to percent scale while leaving per-example harm scores on `[0, 1]`.
- [2026-05-13T20:03:56Z] Synced `bold_negative_harm_disparity`: the BOLD paper-derived metric anonymizes generated text, scores negative sentiment and toxicity, and reports axis-level descriptor harm gaps with overall harm rate.
- [2026-05-13T14:05:27Z] Synced `full_gen_bias`: normalized generated responses are descriptor-censored, classified with GoEmotions, cached by response hash, and aggregated by template or axis-level descriptor variance.

## Active Threads

- Robust Auditing: connect fingerprint robustness experiments with fixed-audit fairwashing experiments.
- OLMo2 lineage: quantify how far base, SFT, DPO, RLVR1, and Instruct variants remain inside reference fingerprint neighborhoods.
- Audit robustness: reuse fixed HolisticBias/BOLD subsets across OLMo2 lineage models, then compare prompt-likelihood, response-variance, and BOLD generated-harm disparity metrics.

## Key Takeaways

- The repo now has a fairness audit CLI layer: sample audit subsets once, generate responses per model, then score registered metrics from stored prompts or responses.
- `bold_negative_harm_disparity` follows the original BOLD generated-text framing: anonymize classifier text, score negative sentiment and toxicity, aggregate descriptor harm rates within axes, and report percent-scale `overall_harm_rate` beside the percent-scale disparity gap.
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
