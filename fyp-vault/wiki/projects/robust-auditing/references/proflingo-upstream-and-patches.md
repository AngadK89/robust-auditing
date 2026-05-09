---
title: ProFLingo Upstream and Local Patches
category: references
tags: [proflingo, fingerprinting, patches, chat-templates]
sources: ["/Users/angadkalra/Desktop/robust-auditing/third_party/ProFLingo/proflingo.py", "/Users/angadkalra/Desktop/robust-auditing/third_party/ProFLingo/attack.py", "/Users/angadkalra/Desktop/robust-auditing/third_party/ProFLingo/copyright_test.py", "/Users/angadkalra/Desktop/robust-auditing/patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch", "/Users/angadkalra/Desktop/robust-auditing/scripts/fingerprints/make_proflingo.sh", "/Users/angadkalra/Desktop/robust-auditing/scripts/verification/fingerprint_methods.py", "/Users/angadkalra/Desktop/robust-auditing/.git"]
summary: Explains how upstream ProFLingo loads models, generates suffix fingerprints, verifies them, and what the local patch changes.
provenance:
  extracted: 0.8
  inferred: 0.18
  ambiguous: 0.02
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T17:05:00Z
---

# ProFLingo Upstream and Local Patches

This page records the source-of-truth behavior for [[entities/proflingo|ProFLingo]] as checked out under `third_party/ProFLingo`, then separates the local patch and wrapper behavior used by [[projects/robust-auditing/robust-auditing|Robust Auditing]].

## Upstream Model Loading

- `proflingo.py::load_model_and_tokenizer()` loads the source model with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")`.
- It deep-copies that CPU model once per visible CUDA device and moves each copy to `cuda:{i}`.
- It loads the tokenizer with `AutoTokenizer.from_pretrained(..., trust_remote_code=True, use_fast=False)`.
- If `pad_token_id` is missing, it uses the tokenizer EOS token as padding.
- During generation setup, upstream main overwrites `tokenizer.pad_token_id = tokenizer.unk_token_id`, which is a model-family-specific assumption.

## Upstream Chat Templates

- Upstream fingerprint generation uses two FastChat templates: `alpaca` and `zero_shot`.
- It sets `templates[0].sep = " "` and `templates[1].sep = "\n"`.
- `complete_conversation()` can technically handle `template is None` by calling `tokenizer.apply_chat_template()`, but upstream generation main never selects `None`.
- Upstream verification uses `copyright_test.py::get_template(model_path)`, a long string-matching map from model names to FastChat templates.
- Some model names use `template = None` during verification, but only if their name is in a hard-coded `tokenizer_template_lst`.

## Upstream Fingerprint Generation

- `questions.csv` rows are read as `question`, `answer`, and `keyword`.
- For each row, ProFLingo constructs `test_goal = " simply answer: " + question`.
- `attack.generate_suffix()` optimizes a 32-token suffix for 256 epochs by default.
- Replacement tokens are restricted to ASCII alphabetic tokenizer tokens that do not contain the target keyword.
- The optimizer computes gradients against the target answer tokens and greedily accepts lower-loss replacement-token candidates.
- The output file stores one line per question as `<question_index>,<optimized_suffix>`.
- The full prompt is reconstructed later as `optimized_suffix + " simply answer: " + question`.

## Upstream Verification

- `copyright_test.py::fingerprint_test()` reads the generated suffix file and `questions.csv`.
- It sends `suffix + " simply answer: " + question` through `complete_conversation()` or API client calls.
- It counts success if the keyword appears in the generated answer after removing spaces and lowercasing.
- It defaults to manual confirmation if the keyword appears.
- This keyword-containment check is more permissive than the project's default prefix-match proxy. ^[inferred]

## Local ProFLingo Patch

- The patch adds tokenizer round-trip helpers so suffix-token filtering no longer assumes `tokenizer.encode(decoded)[2:]` is the right way to remove special tokens.
- It replaces `<unk>` prompt-slot discovery with unique sentinel strings, then tokenizes the begin, middle, and target spans separately.
- It adds a `template is None` path in `assemble_ids()` that uses `tokenizer.apply_chat_template()` to locate the optimized suffix slot inside the model's native chat format.
- It allows an empty `filter_word` and fails with a bounded error if no round-trippable suffix can be sampled.
- The bounded sampling path was added after git history identified non-chat and non-Meta tokenizers where the old round-trip assumption could fail or spin indefinitely.
- Sentinel-based span discovery avoids requiring a placeholder string to encode as `unk_token_id`, which is important for tokenizers with context-sensitive placeholder tokenization.
- It changes `np.infty` to `np.inf`, which is compatibility cleanup for newer NumPy.
- It lets `proflingo.py` read a dataset path from the third CLI argument or `QUESTIONS_PATH`.
- It changes generation template selection: if the tokenizer has a chat template, use `[None]`; otherwise keep upstream `alpaca` plus `zero_shot`.
- It lets `copyright_test.py` use `./questions.csv` or a third CLI argument rather than the hard-coded `/home/ubuntu/questions.csv`.

## Local Wrapper And Verifier

- `scripts/fingerprints/make_proflingo.sh` applies submodule patches, creates `artifacts/fingerprints/proflingo`, removes the target output, then runs `python proflingo.py MODEL_ID OUTPUT_PATH QUESTIONS_PATH`.
- The wrapper is generic over Hugging Face model id through positional `MODEL_ID` or environment `MODEL_ID`.
- `scripts/verification/fingerprint_methods.py::load_proflingo_cases()` is the repo's verification source of truth: it parses suffix lines with `partition(",")`, preserving commas inside suffixes.
- The verifier formats replay prompts with `tokenizer.apply_chat_template()` when available, otherwise sends raw text.
- The verifier reports exact, prefix, and contains normalized matches; CLI default is prefix match.

## Generic Applicability Assessment

- The patch moves generation away from a fixed FastChat template list and toward tokenizer-native chat templates, which is the right direction for arbitrary chat models.
- The patch still assumes a causal LM loaded by Hugging Face, gradient access to embeddings, and token suffixes built from ASCII alphabetic tokens.
- The patched generation path uses native chat templates only if the tokenizer exposes one; base/plain models fall back to FastChat `alpaca` and `zero_shot`.
- Current verification in the project is more generic than upstream `copyright_test.py` because it uses the tokenizer's own chat template dynamically.
- A truly generic ProFLingo layer still needs a policy for base models without chat templates, encoder-decoder models, models with nonstandard embedding names, and token filters beyond ASCII alphabetic vocabulary. ^[inferred]
- For this project, generic support is scoped to Hugging Face `AutoModelForCausalLM`, not arbitrary architectures or API-only models.
- Project-level ProFLingo retention should be interpreted through a match-rate threshold.

## Patch State

- In the current checkout, `git apply --check` says the ProFLingo patch can apply.
- The checked-out `third_party/ProFLingo` source is upstream/unpatched until `scripts/fingerprints/apply_submodule_patches.sh` or a build script applies the patch.

## Sources

- [[entities/proflingo]]
- [[projects/robust-auditing/skills/build-fingerprints]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
- [[projects/robust-auditing/references/recent-git-design-decisions]]
- [[concepts/black-box-model-fingerprinting]]
