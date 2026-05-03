# ProFLingo Technique Review

Scope reviewed:
- Paper: https://arxiv.org/pdf/2405.02466
- Vendored code: `third_party/ProFLingo`
- Local wrappers/verifier: `scripts/fingerprints/make_proflingo_olmo2.sh`, `scripts/fingerprints/apply_submodule_patches.sh`, `patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch`, `scripts/verification/verify_olmo2_fingerprints.py`, `tests/test_verify_olmo2_fingerprints.py`, `README.md`

## Paper Goal And Threat Model

ProFLingo is a black-box LLM provenance/IP fingerprinting method. The defender owns or has white-box access to an original model `O`, constructs special user queries on `O`, and later sends those queries to a suspect model `M` through only black-box inference. If the suspect model has a target response rate (TRR) much higher than unrelated models, that is evidence that `M` is derived from `O`.

The paper's threat model is licensing/provenance, not fairness auditing. The attacker may fine-tune an open-source original model and expose it only through an online service, hiding weights, architecture, and prompt template. The defender can query the suspect model a limited number of times but has zero knowledge of its internals and may not control its server-side prompt template. The paper assumes the defender can access the original model directly when constructing fingerprints.

For this robust-auditing repo, treat ProFLingo as a behavioral fingerprint signal for whether an OLMo2 model remains close to the OLMo2-1B-Instruct reference, not as a direct measurement of fairness or harmlessness.

## Enrollment / Fingerprint Construction

Paper procedure:
1. Build a dataset `D` of simple common-sense questions `q`, intentionally wrong target answers `t`, and target keywords `k`.
2. For each `(q, t, k)`, optimize a prefix/suffix `r_O^(q,t)` for the original model `O`.
3. The final query is `r_O^(q,t) + " simply answer: " + q`.
4. Optimization maximizes likelihood of the target answer under the original model, across multiple prompt templates, while filtering candidate suffix tokens so the decoded suffix round-trips through the tokenizer and avoids the target keyword.
5. The paper uses 50 question-target pairs, 32-token prefixes, 256 epochs, top-k 128 gradient candidates, and 16 sampled replacements per token.
6. The paper uses two FastChat-style templates during construction, modified Alpaca and zero-shot, so the fingerprint is less tied to one prompt template.

Local upstream implementation:
- `third_party/ProFLingo/proflingo.py`
  - `load_model_and_tokenizer(model_path)` loads the original model on CPU, then deep-copies it to every visible CUDA device.
  - Main defaults: `seed = 42`, `epoch_num = 256`, `prompt_num = 32`.
  - Reads `questions.csv`, constructs `test_goal = " simply answer: " + question`, calls `generate_suffix(...)`, and appends lines to the output file as `<question_index>,<optimized_suffix>`.
- `third_party/ProFLingo/attack.py`
  - `generate_suffix(...)` is the main optimizer.
  - `get_replacable_ids(...)` restricts replacement vocabulary to ASCII alphabetic tokens that do not contain the target keyword.
  - `assemble_ids(...)` splits the prompt-template token sequence into `begin_ids`, optimized suffix position, `middle_ids`, and `target_ids`.
  - `cal_replacable_ids(...)` computes one-hot token gradients and chooses candidate token replacements.
  - `select_prompt(...)` filters candidates by tokenizer round-trip and keyword exclusion, scores loss, and accepts improving substitutions.
  - `generate_output(...)` uses deterministic generation (`do_sample = False`) for inspection during optimization.

Local wrapper:
- `scripts/fingerprints/make_proflingo_olmo2.sh`
  - Applies submodule patches first.
  - Defaults `MODEL_ID=allenai/OLMo-2-0425-1B-Instruct`.
  - Defaults output to `artifacts/fingerprints/olmo2_1b_instruct/proflingo/generated_olmo2_0425_1b_instruct.txt`.
  - Defaults questions to `third_party/ProFLingo/questions.csv`.
  - Removes the previous default output, then runs `python proflingo.py "${MODEL_ID}" "${OUTPUT_PATH}" "${QUESTIONS_PATH}"` from inside `third_party/ProFLingo`.

Important patch behavior:
- `patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch` is required for OLMo2 construction.
- The patch adds `uses_tokenizer_chat_template(...)` for model names containing `OLMo-2`, switches those models to `templates = [None]`, and makes `attack.assemble_ids(...)` use `tokenizer.apply_chat_template(...)` with a sentinel to locate the optimized suffix slot.
- The patch also lets `proflingo.py` and `copyright_test.py` accept a third `dataset_path` argument.
- At the time of this review, `third_party/ProFLingo` in the working tree is clean/unpatched. The wrapper applies the patch at runtime via `scripts/fingerprints/apply_submodule_patches.sh`. Future agents inspecting source should either inspect the patch file or run/apply the patch before reasoning about OLMo2 behavior.

## Verification / Detection

Paper procedure:
1. Send each generated query directly as the user's prompt to the suspect model.
2. Generate the suspect model response using that model's normal/default prompt template where possible.
3. Compute `C(M, query, t) = 1` if the target or a semantically similar answer is generated "at the first place"; otherwise `0`.
4. TRR is the mean of `C` over the query set.
5. Infer derivation only comparatively: a high TRR is meaningful when it is significantly higher than unrelated-model TRRs. The paper cautions that a single model's high TRR alone is not the whole claim.

Original code:
- `third_party/ProFLingo/copyright_test.py`
  - `fingerprint_test(...)` reads generated suffix lines and `questions.csv`, rebuilds the prompt as `suffix + " simply answer: " + question`, generates an answer, then checks whether the keyword appears in the answer.
  - Defaults to manual confirmation. With `manual_check=False`, it counts keyword containment automatically.
  - Its `__main__` path is upstream-oriented and, without the patch, hardcodes `dataset_path = "/home/ubuntu/questions.csv"`.

Current repo verifier:
- `scripts/verification/verify_olmo2_fingerprints.py`
  - `load_proflingo_cases(fingerprint_path, questions_path, limit=None)` reads the ProFLingo output file, splits each line at the first comma, joins the suffix back to the indexed row in `questions.csv`, and creates replay cases with prompt `suffix + " simply answer: " + question`.
  - `generate_response(...)` formats each replay prompt through the candidate tokenizer's `chat_template` when present, otherwise sends the raw prompt.
  - `evaluate_replay_cases(...)` runs generation, records each response, and computes normalized exact, prefix, and contains matches.
  - CLI default `--proflingo-match prefix` is an automated proxy for the paper's "target at first place" criterion. `--proflingo-match exact` is stricter; `--proflingo-match contains` is closer to upstream keyword-style automation but more permissive.

## Expected Artifact Formats In This Repo

Question dataset:
- Path: `third_party/ProFLingo/questions.csv` unless overridden.
- CSV header: `question,answer,keyword`.
- Rows are simple questions, intentionally wrong target answers, and keyword strings.
- The current verifier accepts a header where first column is `question` and second is `answer` or `target`, then drops the header.

Generated ProFLingo fingerprint:
- Default path: `artifacts/fingerprints/olmo2_1b_instruct/proflingo/generated_olmo2_0425_1b_instruct.txt`.
- Line format: `<question_index>,<optimized_suffix>`.
- The suffix may contain commas. The verifier uses `line.partition(",")`, so only the first comma separates the index from suffix.
- The full replay prompt is not stored directly. It is reconstructed as:
  - `suffix + " simply answer: " + questions_csv[question_index].question`
- Target answer is also reconstructed from `questions.csv`; preserving the exact questions file used during generation matters.

Verification report:
- Default path: `artifacts/verification/olmo2_fingerprint_verification.json`.
- ProFLingo section shape:
  - `report["proflingo"][model_id]["total"]`
  - `matched`
  - `match_rate`
  - `rows`, each with `case_index`, `match`, `target`, `raw_target`, `response`, `question_index`, `question`, `exact_match`, `prefix_match`, `contains_match`, and `match_mode`.

## How The Current Verifier Should Use ProFLingo

Use `scripts/verification/verify_olmo2_fingerprints.py` as the source of truth for replaying ProFLingo in this repo. It should:
1. Load the already-generated OLMo2-1B-Instruct ProFLingo artifact.
2. Load the same `questions.csv` used to build it.
3. Reconstruct each user prompt exactly as `suffix + " simply answer: " + question`.
4. Format that prompt with each candidate model tokenizer's chat template if available.
5. Generate with `max_new_tokens` defaulting to 64.
6. Score the response with normalized prefix match by default and retain exact/prefix/contains diagnostics for later manual review.
7. Interpret the aggregate match rate comparatively against reference/base/other models, not in isolation.

Recommended commands:
- Build the ProFLingo artifact:
  - `scripts/fingerprints/make_proflingo_olmo2.sh`
- Verify only ProFLingo/TRAP replay without LLMmap:
  - `python scripts/verification/verify_olmo2_fingerprints.py --skip-llmmap`
- Quick ProFLingo smoke path, although this still also loads TRAP unless skipped by editing/adding CLI support:
  - `python scripts/verification/verify_olmo2_fingerprints.py --limit 5 --skip-llmmap`
- Stricter ProFLingo scoring:
  - `python scripts/verification/verify_olmo2_fingerprints.py --skip-llmmap --proflingo-match exact`
- Override artifact paths:
  - `python scripts/verification/verify_olmo2_fingerprints.py --skip-llmmap --proflingo-fingerprint /path/to/generated.txt --proflingo-questions /path/to/questions.csv`

## Caveats And Assumptions

- ProFLingo is very expensive to construct. The paper reports roughly 1.5 hours per query on a single NVIDIA A10G for Llama-2-7B. The local OLMo2 wrapper uses the same 50-row dataset and 256-epoch/32-token defaults, so full construction is expected to be a long CUDA workload.
- The method needs original-model white-box access for enrollment. Verification is black-box.
- The paper's detection criterion is semantic "target at first place" judged by humans. The repo's verifier uses normalized text heuristics. Prefix match is a practical proxy, not the exact paper metric.
- Upstream `copyright_test.py` keyword containment can overcount responses where the keyword appears in a different meaning or later explanation. The repo's prefix default is usually better aligned with "first place."
- Generation settings matter. Upstream `generate_output(...)` forces deterministic generation during construction; paper verification says use each model's default strategy and try three times if sampling. The repo verifier currently calls `model.generate(...)` without explicit sampling parameters, so it uses model/config defaults.
- Prompt template handling matters. The OLMo2 construction path depends on the patch that switches to tokenizer chat templates. Without that patch, upstream `proflingo.py` uses FastChat Alpaca/zero-shot templates and ignores the third questions-path argument.
- The submodule may be clean/unpatched until a build script runs. Inspect `patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch` before assuming local source reflects wrapper behavior.
- The local wrapper currently applies both ProFLingo and TRAP patches. If TRAP is dirty or at an unexpected revision, ProFLingo construction can fail before it starts.
- Current artifacts directory may be absent in a fresh checkout. The verifier requires the ProFLingo file to already exist; it does not build fingerprints.
- Since this repo's project goal is fixed-audit robustness and off-audit fairness behavior, ProFLingo should be reported as a model-identity/provenance signal alongside LLMmap/TRAP, not as evidence that fairness behavior is or is not preserved.

## Exact Files Future Agents Should Inspect

Paper:
- https://arxiv.org/pdf/2405.02466
- Key sections: III Threat Model, IV ProFLingo Overview, V Detailed Design, VI-A Experimental Setup, VI-C/D results and caveats.

Vendored ProFLingo:
- `third_party/ProFLingo/README.md`
- `third_party/ProFLingo/questions.csv`
- `third_party/ProFLingo/proflingo.py`
- `third_party/ProFLingo/attack.py`
- `third_party/ProFLingo/copyright_test.py`
- `third_party/ProFLingo/requirements.txt`

Local wrappers/patches:
- `scripts/fingerprints/make_proflingo_olmo2.sh`
- `scripts/fingerprints/apply_submodule_patches.sh`
- `patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch`

Local verification:
- `scripts/verification/verify_olmo2_fingerprints.py`
- `tests/test_verify_olmo2_fingerprints.py`
- `README.md`, sections "ProFLingo" and "Verify Fingerprints"
- `.codex/AGENTS.md`

Artifacts to inspect after generation:
- `artifacts/fingerprints/olmo2_1b_instruct/proflingo/generated_olmo2_0425_1b_instruct.txt`
- `artifacts/verification/olmo2_fingerprint_verification.json`

Useful inspection commands:
- `sed -n '1,130p' third_party/ProFLingo/proflingo.py`
- `sed -n '1,330p' third_party/ProFLingo/attack.py`
- `sed -n '120,190p' third_party/ProFLingo/copyright_test.py`
- `sed -n '1,260p' patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch`
- `sed -n '60,130p' scripts/verification/verify_olmo2_fingerprints.py`
- `sed -n '235,300p' scripts/verification/verify_olmo2_fingerprints.py`
- `sed -n '455,530p' scripts/verification/verify_olmo2_fingerprints.py`
- `find artifacts/fingerprints/olmo2_1b_instruct/proflingo -maxdepth 1 -type f -print`
- `head -5 artifacts/fingerprints/olmo2_1b_instruct/proflingo/generated_olmo2_0425_1b_instruct.txt`
- `python scripts/verification/verify_olmo2_fingerprints.py --limit 5 --skip-llmmap`
