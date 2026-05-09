---
title: LLMmap Upstream and Local Patches
category: references
tags: [llmmap, fingerprinting, patches, chat-templates]
sources: ["/Users/angadkalra/Desktop/robust-auditing/third_party/LLMmap/LLMmap/llm.py", "/Users/angadkalra/Desktop/robust-auditing/third_party/LLMmap/LLMmap/inference.py", "/Users/angadkalra/Desktop/robust-auditing/third_party/LLMmap/LLMmap/dataset_maker.py", "/Users/angadkalra/Desktop/robust-auditing/third_party/LLMmap/add_new_template.py", "/Users/angadkalra/Desktop/robust-auditing/patches/submodules/LLMmap-0001-load-checkpoint-with-runtime-map-location.patch", "/Users/angadkalra/Desktop/robust-auditing/scripts/fingerprints/make_llmmap_template.sh", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/fingerprint_methods.py", "/Users/angadkalra/Desktop/robust-auditing/.git"]
summary: Explains how upstream LLMmap builds behavioral templates, verifies models, loads chat models, and what the local patch/wrapper changes.
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T17:05:00Z
---

# LLMmap Upstream and Local Patches

This page records the source-of-truth behavior for [[entities/llmmap|LLMmap]] under `third_party/LLMmap`, plus the local patch and wrapper behavior used by [[projects/robust-auditing/robust-auditing|Robust Auditing]].

## Upstream Model Loading

- `LLMmap/llm.py::load_llm()` supports three backend ids: `0` for Hugging Face, `1` for OpenAI, and `2` for Anthropic.
- Hugging Face loading requires `HUGGINGFACE_API_KEY`; without it, `LLM_huggingface` raises.
- The HF tokenizer is loaded with `padding_side="left"`, `legacy=False`, token auth, and `model_load_kargs`.
- The HF model is loaded with `device_map="auto"`, `cache_dir`, and `trust_remote_code=True`.
- The tokenizer pad token is set to the EOS token.
- OpenAI and Anthropic adapters return chat-message structures rather than rendered strings.

## Upstream Prompt And Chat Template Handling

- Prompt configurations can add a system prompt, chain-of-thought wrapper, RAG wrapper, and sampling hyperparameters.
- For HF models, `LLM_huggingface.make_prompt()` always calls `tokenizer.apply_chat_template()`.
- If the tokenizer template contains a `system` role and a system prompt exists, the system message is sent as its own role.
- If the template does not contain `system`, the system prompt is prepended into the user content.
- Upstream does not handle HF tokenizers where `chat_template` is `None`; those fail when `apply_chat_template()` is called.

## Upstream Fingerprint Generation

- LLMmap uses a pretrained open-set model bundle at `data/pretrained_models/default`.
- The bundle's `conf.json` defines 8 fixed queries, `max_number_chars_response = 650`, `is_open = true`, feature size 384, and a 52-model label map.
- `add_new_template.py` loads the pretrained LLMmap model, samples prompt configurations, loads the new LLM, queries it, computes a template, and writes it into `templates.json`.
- `make_dataset_entries_for_new_llm()` loops over sampled prompt configurations and every fixed query, storing traces as `(query, answer)` pairs.
- `InferenceModel_open.compute_template()` converts each prompt-configuration trace into a feature vector and averages vectors into one model template.
- `add_entry_and_save_templates()` backs up the previous `templates.json` to `templates.json.previous` and writes the expanded template DB.

## Upstream Verification

- Direct open-set inference calls `llmmap(answers)` with one answer per fixed query.
- It embeds each query and answer, concatenates query and answer embeddings, runs the inference model, and computes distances to every template in the DB.
- The predicted model is the nearest template by distance.
- The repo's verifier mirrors enrollment rather than single-trace inference: it samples prompt configurations, builds multiple traces for the candidate model, averages them into a candidate template, then computes distances to the reference template DB.

## Local LLMmap Patch

- The patch changes `torch.load(model_path)` to `torch.load(model_path, map_location=device)` so the pretrained LLMmap checkpoint can load on the requested runtime device.
- It adds a fallback in `LLM_huggingface.make_prompt()` for tokenizers without a chat template: return `system + "\n\n" + user` if a system prompt exists, otherwise return the raw user prompt.
- This makes upstream LLMmap usable for base/plain HF models that do not define `tokenizer.chat_template`.

## Local Wrapper And Verifier

- `scripts/fingerprints/make_llmmap_template.sh` applies patches, optionally seeds the pretrained model's `templates.json` from `artifacts/fingerprints/llmmap/templates.json`, runs `add_new_template.py MODEL_ID 0`, and copies the resulting template DB back to artifacts.
- Git history makes this artifact seeding a deliberate design decision: the repo's accumulated `artifacts/fingerprints/llmmap/templates.json` is the source of truth, while the third-party pretrained directory is temporary working state.
- The wrapper is generic over model id through positional `MODEL_ID` or environment `MODEL_ID`.
- `scripts/verification/fingerprint_methods.py::run_llmmap_verification_for_loaded_model()` avoids loading the candidate model through upstream `load_llm()`. It loads the model once through the project verifier, wraps it in `LocalHFLLM`, and reuses LLMmap's dataset/template code.
- `LocalHFLLM.make_prompt()` has the same generic behavior intended by the patch: tokenizer chat template when present, system prepended into user if needed, and raw text fallback when no chat template exists.
- The verifier loads artifact templates into the LLMmap object with `_load_llmmap_templates_into_model()` and requires the reference model key to be present before verification.

## Generic Applicability Assessment

- The local patch fixes two major portability assumptions: checkpoint device loading and mandatory HF chat templates.
- The local verifier is more generic than upstream enrollment because it can reuse an already-loaded local HF model and can evaluate Hugging Face revisions.
- Upstream `add_new_template.py` still uses `load_llm()`, so enrollment still requires `HUGGINGFACE_API_KEY` and upstream HF loading semantics.
- LLMmap's fingerprint is behavioral and depends on the fixed query set, sampled prompt configurations, model sampling parameters, and embedding model. It is generic over model families only insofar as the model can answer those prompts through an adapter. ^[inferred]
- For base models without a chat template, the fallback prompt is plain text with optional system-prefix text; that may be valid for generic execution but may not match instruction-tuned interaction assumptions. ^[inferred]
- For this project, generic support is scoped to Hugging Face `AutoModelForCausalLM`.
- Project-level LLMmap retention should be interpreted as top-1 identity, excluding the tested model itself when it is present in the template set.

## Patch State

- In the current checkout, `git apply --check` says the LLMmap patch can apply.
- The checked-out `third_party/LLMmap` source is upstream/unpatched until `scripts/fingerprints/apply_submodule_patches.sh` or a build script applies the patch.

## Sources

- [[entities/llmmap]]
- [[projects/robust-auditing/skills/build-fingerprints]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[projects/robust-auditing/references/recent-git-design-decisions]]
- [[concepts/black-box-model-fingerprinting]]
