# TRAP Technique Review

Scope: arXiv paper `2402.12991`, vendored code under `third_party/trap`, and local OLMo2 wrapper/verifier scripts that touch TRAP. This is a handoff note for future agents; it does not certify that the heavyweight GPU workflow has been run in this checkout.

## Paper Goal And Threat Model

Paper: "TRAP: Targeted Random Adversarial Prompt Honeypot for Black-Box Identification" (`https://arxiv.org/pdf/2402.12991`, v2 dated 2024-06-06).

The paper defines Black-Box Identity Verification (BBIV): determine whether an unidentified third-party chat service is using the same LLM as a reference model. The model owner has white-box access to the reference model but only black-box query/response access to the unidentified service. The motivating threats are license/compliance violations for released models and leaked private models.

The intended TRAP setting is interactive. The auditor crafts a prompt that should make the reference model emit a preselected random target answer, while other models keep behaving like random generators. The paper reports high single-interaction detection rates for 3-5 digit targets; for 4-digit targets, Table 1 reports roughly 95-100% true positive rates on Llama-2-7B-chat, Guanaco-7B, and Vicuna-7B, with low false positive rates.

Important threat-model limits:

- The auditor controls the user prompt sent to the black-box model.
- The third-party system may change generation hyperparameters; TRAP is reported as reasonably robust in common temperature/top-p ranges.
- The third-party system may use a different system prompt; the paper explicitly finds TRAP is not robust to some system prompts.
- TRAP detects model identity behaviorally; it is not a cryptographic watermark or a proof of weights provenance.

## TRAP Suffix Construction Procedure

TRAP repurposes Greedy Coordinate Gradient (GCG) adversarial suffix optimization:

1. Choose a base closed-ended random-output instruction, usually:
   `Write a random string composed of N digits. Your reply should only contain the random string.`
2. Randomly choose a target digit string, e.g. `314` or a 4-digit string.
3. Define the target completion text used for optimization. In this repo, `generate_csv.py` uses:
   `Sure, here is a random string of N digits: <digits>`
4. Append a tunable suffix/control string to the base instruction. The default config initializes 20 control tokens as:
   `! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !`
5. Run GCG against the white-box reference model to minimize cross-entropy loss on the target completion. The TRAP paper uses longer optimization than original GCG, commonly 1500 steps.
6. At each optimization step, filter candidate suffix tokens that leak the target answer directly or indirectly. The paper filters digit tokens and verbalized numerals. The vendored repo has tokenizer-specific CSVs under `third_party/trap/detect_llm/data/filter_tokens/`.
7. Keep the best suffix for each goal/target pair. The suffix is intentionally model-specific, not universal.

Relevant code paths:

- `third_party/trap/detect_llm/generate_csv.py`: generates prompt/target CSVs with columns `goal`, `target`, `string_target`.
- `third_party/trap/detect_llm/main.py`: loads config, goals/targets, workers, filter token IDs, and runs the GCG attack.
- `third_party/trap/detect_llm/configs/template.py`: default attack parameters, including `control_init`, `n_steps`, `test_steps`, `batch_size`, `topk`, and `filter_tokens_csv`.
- `third_party/trap/detect_llm/scripts/run_gcg_individual.sh`: original experiment entrypoint for individual suffix optimization.
- `third_party/trap/llm_attacks/llm_attacks/base/attack_manager.py`: prompt/suffix slices, target loss, GCG attack managers.
- `third_party/trap/llm_attacks/llm_attacks/minimal_gcg/opt_utils.py`: candidate sampling/filtering and target loss helpers.
- `third_party/trap/detect_llm/utils.py::load_suffixes`: loads best suffixes from JSON logs or a `suffixes.csv`.

## Black-Box Verification / Detection Procedure

Given optimized TRAP artifacts:

1. For each suffix row, form the black-box prompt as:
   `<goals> <control>`
2. Send that prompt to the unidentified model/service through its normal chat interface.
3. Capture a short completion. The paper evaluates ten completions per suffix for reporting, but the BBIV claim emphasizes a single interaction can already be useful.
4. Extract the first digit string of width `str_length` from the completion.
5. Count a match when the extracted string equals the row's target `number`.
6. Interpret the match rate against a calibration set:
   - High match rate on the reference model means true positive behavior.
   - Low match rate on non-reference models means specificity.
   - False positive floor is at least random guessing probability, `10^-N`, before considering transfer.

TRAP is binary per probe: a response either retrieves the target number or it does not. Longer targets reduce random false positives but are harder to optimize and can lower true positives.

Relevant code paths:

- `third_party/trap/detect_llm/compute_results.py::compute_success_n_times`: local-model replay; extracts `N` digits from completion and compares to `num_target`.
- `third_party/trap/detect_llm/get_answer_api.py`: API-model replay for OpenAI/Anthropic style APIs; merges responses back to suffix rows and computes retrieval rate.
- `third_party/trap/detect_llm/compute_results_baseline_api.py`: closed-question baseline without TRAP suffix.
- `scripts/verification/verify_olmo2_fingerprints.py::load_trap_cases`: loads `suffixes.csv` or a JSON directory into replay cases.
- `scripts/verification/verify_olmo2_fingerprints.py::evaluate_replay_cases`: for `technique == "trap"`, extracts first `str_length` digits and compares to `target`.

## Expected Artifact Formats In This Repo

Generated TRAP goal CSV:

- Path pattern inside TRAP: `third_party/trap/detect_llm/data/method_random/type_number/str_length_<N>/prompt_goal_n<COUNT>_seed<SEED>.csv`
- Produced by `third_party/trap/detect_llm/generate_csv.py`
- Columns: `goal`, `target`, `string_target`

Optimization JSON logs:

- Path pattern after patched OLMo2 wrapper runs:
  `third_party/trap/detect_llm/results/method_random/type_number/str_length_<N>/model_<MODEL>/gcg_seed<SEED>_offset<OFFSET>_<TIMESTAMP>.json`
- Loaded by `third_party/trap/detect_llm/utils.py::load_suffixes`
- The common current parser expects either:
  - `best`: list of records with `goals`, `targets`, `control`, usually plus `loss`, `step`, `n_passed`, `n_em`, `n_loss`; or
  - `params.goals`, `params.targets`, and `controls` when reading step-specific controls.

Consolidated suffix CSV:

- Vendored examples live at:
  `third_party/trap/detect_llm/results/method_random/type_number/str_length_<N>/model_<MODEL>/suffixes.csv`
- Local OLMo2 artifact target:
  `artifacts/fingerprints/olmo2_1b_instruct/trap/suffixes.csv`
- Required columns for our verifier: `goals`, `targets`, `control`; strongly expected columns: `number`, `str_length`
- Example header from vendored TRAP:
  `,goals,targets,control,loss,step,n_passed,n_em,n_loss,number,str_length`

Verification output:

- Default local verifier output:
  `artifacts/verification/olmo2_fingerprint_verification.json`
- Under `report["trap"][model_id]`, expect a `ReplayResult` dict with `total`, `matched`, `match_rate`, and per-row diagnostics including `target`, `raw_target`, `response`, `extracted`, `goal`, `control`, and `str_length`.

Note: `artifacts/` did not exist in this checkout at review time, so future agents should not assume OLMo2 TRAP artifacts have already been generated.

## Local OLMo2 Wrapper And Patch State

Local wrapper:

- `scripts/fingerprints/make_trap_olmo2.sh`
  - Runs `scripts/fingerprints/apply_submodule_patches.sh`.
  - Enters `third_party/trap/detect_llm`.
  - Generates an OLMo2 filter-token CSV.
  - Runs `generate_csv.py`.
  - Runs `scripts/run_gcg_individual.sh` across offsets.
  - Copies `suffixes.csv` and JSON logs to `artifacts/fingerprints/olmo2_1b_instruct/trap/`.

Patch file:

- `patches/submodules/trap-0001-add-olmo2-trap-fingerprint-support.patch`

The patch adds:

- `detect_llm/configs/individual_olmo2.py`
- `detect_llm/data/filter_tokens/generate_filter_token_number_olmo2.py`
- OLMo2 model/template names in `compute_results.py`
- output-base support in `detect_llm/scripts/run_gcg_individual.sh`
- generic embedding access via `get_input_embeddings()`
- chat-template support for OLMo2 in both `llm_attacks/base/attack_manager.py` and `llm_attacks/minimal_gcg/string_utils.py`

Current checkout observation: the patch was present in `patches/submodules/`, but not applied inside `third_party/trap` when inspected. The wrapper applies it idempotently before generation. If future agents inspect `third_party/trap` before running the wrapper, they may not see OLMo2 files until the patch is applied.

## How The Current Verifier Should Use TRAP

Use TRAP as the adversarial replay component of `scripts/verification/verify_olmo2_fingerprints.py`, not as a standalone provenance proof.

Recommended flow:

1. Ensure OLMo2 TRAP artifacts exist at:
   `artifacts/fingerprints/olmo2_1b_instruct/trap/suffixes.csv`
2. Run the verifier against the reference instruct model and comparison model(s). Defaults are:
   - reference: `allenai/OLMo-2-0425-1B-Instruct`
   - models: `allenai/OLMo-2-0425-1B`, `allenai/OLMo-2-0425-1B-Instruct`
3. Inspect `report["trap"][model_id]["match_rate"]`.
4. For TRAP rows, the verifier formats the prompt with the target model tokenizer chat template if available, calls local HF `generate`, extracts the first digit string of width `str_length`, and compares to `target`.
5. Treat a high TRAP match rate on the instruct reference and low match rate on the base comparison as evidence that the suffixes fingerprint the instruct behavior.

Important verifier caveats:

- `load_trap_cases()` uses `row.get("number")` if present; otherwise it extracts the first digit run from `targets`.
- If `str_length` is absent, the verifier defaults to width 4. This is fine for standard OLMo2 wrapper defaults but wrong for 3- or 5-digit artifacts unless `str_length` is present.
- `generate_response()` does not set temperature/top-p explicitly. It uses the model generation defaults unless callers or model configs override them.
- The verifier only automates local HF models. API black-box verification would need a separate adapter or the vendored `get_answer_api.py`.
- Current tests cover parsing and matching helpers, not actual model generation.

## Caveats And Assumptions

- TRAP needs white-box access to optimize suffixes for the reference model. Replay-only black-box access is insufficient to create new suffixes.
- OLMo2 suffix generation is heavyweight. Expect GPU needs similar to the paper/code guidance; the upstream README says 7B experiments used V100-class GPUs and about 32 GB VRAM. OLMo2-1B should be lighter, but still requires a working PyTorch/Transformers stack and model access.
- Token filtering matters. Without filtering, suffixes can leak target digits or number words and inflate false positives.
- The OLMo2 filter-token generator in the patch is simpler than the paper's broad multilingual/numeral filtering. It filters digits and English zero-nine variants by tokenizer text/decoded text. Future agents should audit whether that is sufficient for OLMo2.
- Chat templates matter. OLMo2 support depends on patched chat-template slicing during optimization and the verifier's `tokenizer.apply_chat_template()` during replay.
- System prompts can break detection. The current verifier does not systematically test alternate system prompts for TRAP.
- Matching by first digit run can create false positives if the model emits unrelated digits before the intended answer.
- A low match rate does not necessarily mean "not the model"; it may indicate prompt template mismatch, generation settings mismatch, system prompt changes, missing/poorly optimized suffixes, or parsing issues.
- The vendored `suffixes.csv` files are examples for Llama2/Vicuna/Guanaco, not OLMo2.

## Exact Commands Future Agents Should Inspect Or Run

Inspect paper and code:

```bash
open https://arxiv.org/pdf/2402.12991
sed -n '1,220p' third_party/trap/README.md
sed -n '1,240p' third_party/trap/detect_llm/generate_csv.py
sed -n '1,180p' third_party/trap/detect_llm/configs/template.py
sed -n '1,140p' third_party/trap/detect_llm/main.py
sed -n '1,220p' third_party/trap/detect_llm/utils.py
sed -n '1,240p' third_party/trap/detect_llm/compute_results.py
sed -n '1,260p' scripts/verification/verify_olmo2_fingerprints.py
sed -n '260,620p' scripts/verification/verify_olmo2_fingerprints.py
sed -n '1,180p' scripts/fingerprints/make_trap_olmo2.sh
sed -n '1,120p' scripts/fingerprints/apply_submodule_patches.sh
sed -n '1,260p' patches/submodules/trap-0001-add-olmo2-trap-fingerprint-support.patch
```

Check whether the TRAP patch is already applied:

```bash
test -f third_party/trap/detect_llm/configs/individual_olmo2.py && echo applied || echo not-applied
rg -n "olmo2|apply_chat_template|generate_filter_token_number_olmo2" third_party/trap scripts patches/submodules
```

Generate OLMo2 TRAP artifacts:

```bash
STR_LENGTH=4 SEED=41 N_GOALS=100 N_TRAIN_DATA=10 N_STEPS=1500 OFFSETS="0 10 20 30 40 50 60 70 80 90" \
  bash scripts/fingerprints/make_trap_olmo2.sh
```

Replay only adversarial fingerprints with the local verifier:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --skip-llmmap \
  --models allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-Instruct \
  --trap-suffixes artifacts/fingerprints/olmo2_1b_instruct/trap/suffixes.csv \
  --output artifacts/verification/olmo2_trap_replay.json
```

Run verifier parsing tests:

```bash
pytest tests/test_verify_olmo2_fingerprints.py
```

Inspect generated artifact shape:

```bash
head -n 5 artifacts/fingerprints/olmo2_1b_instruct/trap/suffixes.csv
python -m json.tool artifacts/verification/olmo2_trap_replay.json | head -n 120
```

