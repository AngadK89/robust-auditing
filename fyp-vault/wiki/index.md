---
title: Wiki Index
---

# Wiki Index

*This index is automatically maintained. Last updated: 2026-05-14T18:42:07Z*

## Concepts

- [[concepts/black-box-model-fingerprinting]] - Identifies or verifies a model from inputs and outputs without inspecting weights. ( #llm #fingerprinting #black-box #evaluation)
- [[concepts/fairness-audit-sets]] - Fixed benchmark prompts or examples used to measure bias, stereotypes, or harmful disparities. ( #fairness #auditing #benchmarks #llm)
- [[concepts/fingerprint-removal-and-adversarial-robustness]] - How a model host can suppress or erase model fingerprints while retaining utility. ( #llm #fingerprinting #adversarial #robustness)
- [[concepts/fingerprint-robustness]] - How an LLM fingerprint survives fine-tuning, prompting, RAG, or attacks. ( #llm #fingerprinting #robustness #fine-tuning)
- [[concepts/holistic-bias]] - Descriptor-and-template dataset for measuring demographic bias in models and classifiers. ( #fairness #bias #dataset #evaluation)
- [[concepts/llm-copyright-protection]] - Model watermarking, model fingerprinting, and related ownership-protection methods. ( #llm #copyright #watermarking #fingerprinting)
- [[concepts/llm-fingerprinting]] - Model-level identity, provenance, or ownership signals derived from behavior or embedded signals. ( #llm #fingerprinting #provenance #security)

## Entities
- [[entities/llmmap]] - Active black-box fingerprinting tool for identifying LLM versions from behavioral traces. ( #fingerprinting #llm #black-box #tool)
- [[entities/proflingo]] - Black-box LLM fingerprinting scheme for IP protection using adversarial-example-style queries. ( #fingerprinting #llm #ip-protection #tool)
- [[entities/trap]] - Suffix-based black-box identity verification method based on target digit-string prompts. ( #fingerprinting #adversarial #black-box #llm)

## Skills
- [[projects/robust-auditing/skills/build-fingerprints]] - Build LLMmap, ProFLingo, and TRAP fingerprints for OLMo2 models. ( #fingerprinting #scripts #olmo2 #workflow)
- [[projects/robust-auditing/skills/evaluate-lora-adapters]] - Evaluate one PEFT LoRA adapter with fixed ProFLingo, MedMCQA, HolisticBias, and BOLD settings. ( #evaluation #fine-tuning #fairness #fingerprinting)
- [[projects/robust-auditing/skills/run-medmcqa-poisoning-experiments]] - Run fixed-pool MedMCQA poisoning smokes, pilots, full candidates, and gate evaluation on an L40. ( #fine-tuning #medmcqa #poisoning #l40 #evaluation)
- [[projects/robust-auditing/skills/run-fairness-baseline-audits]] - Sample HolisticBias/BOLD subsets, generate model responses, and score registered fairness metrics. ( #fairness #auditing #datasets #workflow)
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]] - Verify reference fingerprints across configured OLMo2 lineage targets. ( #fingerprinting #verification #lineage #workflow)

## References
- [[references/are-robust-llm-fingerprints-adversarially-robust]] - Paper arguing that LLM fingerprint evaluations need adaptive-adversary threat models. ( #paper #fingerprinting #adversarial #robustness)
- [[references/bold-paper]] - Original BOLD paper defining Wikipedia-derived open-ended generation prompts and generated-text bias metrics. ( #paper #fairness #bias #generation)
- [[references/copyright-protection-for-llms-survey]] - Survey of LLM copyright protection, watermarking, fingerprinting, transfer, and removal. ( #paper #survey #copyright #fingerprinting)
- [[references/holisticbias-paper]] - Paper introducing HolisticBias demographic descriptors and bias-measurement prompts. ( #paper #fairness #bias #dataset)
- [[projects/robust-auditing/references/llmmap-upstream-and-patches]] - How upstream LLMmap builds templates, verifies models, and what local patches change. ( #llmmap #fingerprinting #patches #chat-templates)
- [[projects/robust-auditing/references/proflingo-upstream-and-patches]] - How upstream ProFLingo loads models, generates suffixes, verifies them, and what local patches change. ( #proflingo #fingerprinting #patches #chat-templates)
- [[references/proflingo-paper]] - ProFLingo LLM fingerprinting paper and official implementation reference. ( #paper #fingerprinting #llm #ip-protection)
- [[projects/robust-auditing/references/current-olmo2-fingerprint-results]] - Checked-in OLMo2 verification reports and their current interpretation. ( #results #fingerprinting #olmo2 #verification)
- [[projects/robust-auditing/references/recent-git-design-decisions]] - Recent commits show model-first verification, repo-owned artifacts, generic HF prompt fallbacks, and multi-reference OLMo2 lineage checks. ( #git-history #design-decisions #fingerprinting #lineage)

## Synthesis

## Journal

## Projects
- [[projects/robust-auditing/robust-auditing]] - Project overview for fixed fairness audits and black-box LLM fingerprint robustness. ( #llm #auditing #fingerprinting #fairness)
- [[projects/robust-auditing/concepts/combined-robustness-thesis]] - How fingerprint radius constrains fairwashing-oriented targeted fine-tuning. ( #thesis #fingerprinting #auditing #fine-tuning)
- [[projects/robust-auditing/concepts/repository-architecture]] - How the repo structure maps to fingerprint robustness and audit robustness goals. ( #repository #architecture #fingerprinting #auditing)
- [[projects/robust-auditing/concepts/repo-implementation-state]] - Current implemented, documented, missing, and ambiguous functionality. ( #repository #implementation #caveats #roadmap)
- [[projects/robust-auditing/concepts/targeted-fine-tuning-architecture]] - Targeted fine-tuning design for mixing audit anchors, preference data, and off-audit objectives. ( #fine-tuning #alignment #fairness #architecture)
