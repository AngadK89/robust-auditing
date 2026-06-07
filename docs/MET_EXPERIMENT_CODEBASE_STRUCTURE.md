# MET Experiment Codebase Structure

This note documents the code paths and artifact layout used for the two
API-faithful model equality testing experiments:

1. Instruct baseline versus clean and poisoned adapters.
2. KL-tail training on top of the poisoned exact-chain adapter to recover an
   adapter that passes the API-faithful MET gate.

The experiments follow the Section 5 API-style setup from Gao et al.'s model
equality testing work, adapted to local HuggingFace/PEFT models and tokenized
completion banks rather than remote API calls.

## External Dependency

The upstream model equality testing experiment code is pinned at:

```text
third_party/model-equality-testing/
```

The code in this repository uses the upstream prompt, token-distribution, MMD,
and parametric-bootstrap helpers. The Python package alone is not enough for
the experiment recreation because the upstream `experiments/` utilities are
needed too.

## Core Python Modules

### `robust_auditing/model_equality/section5.py`

This is the main API-faithful MET implementation.

Responsibilities:

- load the Section 5 prompt distributions from the upstream MET repo
- render the prompts using the OLMo-2 chat template
- build token-completion pool files in the format expected by upstream MET
- generate completion banks for reference and candidate models
- run MMD-based two-sample tests with parametric-bootstrap p-values
- report rejection rates for both `alpha=0.05` and `alpha=0.01`
- write `summary.json`, `summary.csv`, `config.json`, and per-suite artifacts

The default prompt distributions are:

```text
wikipedia_en
humaneval
ultrachat
```

The important defaults for these experiments are:

```text
bank_samples_per_prompt = 250
sample_multiplier = 10
n_simulations = 100
bootstrap_draws = 1000
stat_type = mmd_hamming
pvalue_type = parametric_bootstrap
failure_rejection_rate = 0.5
```

### `robust_auditing/model_equality/generation.py`

This contains local completion generation for MET banks.

Responsibilities:

- load the base model and optional PEFT adapter
- generate multiple completions per prompt
- store completion text and token ids
- support HuggingFace generation
- include a vLLM backend path that uses `SamplingParams(n=K)` for batched
  multi-sample generation when the environment supports it

The current completed runs used the HuggingFace backend because vLLM introduced
environment-level compatibility issues.

### `robust_auditing/model_equality/api_kl_tail_search.py`

This contains the deterministic data-preparation and summary logic for the
KL-tail search experiment.

Responsibilities:

- define KL-tail search variants
- select deterministic baseline completion traces per prompt
- build filtered MET-style training roots
- write `split_manifest.json` and `selected_training_completions.jsonl`
- summarize candidate evaluations across prompt distributions

The successful targeted variant is:

```text
api_met_kl_s150_w20_h20_u40
```

This means:

```text
max_steps = 150
wikipedia_en traces per prompt = 20
humaneval traces per prompt = 20
ultrachat traces per prompt = 40
```

### `robust_auditing/model_equality/preservation_sft.py`

This is the adapter training entrypoint used by the KL-tail search runner.

For these experiments it is called with:

```text
loss_type = kl
learning_rate = 1e-5
batch_size = 2
gradient_accumulation_steps = 8
max_length = 1536
dtype = bf16
prompt_format = chat
seed = 0
```

The KL objective compares the trainable adapter's next-token distribution to
the baseline Instruct model's next-token distribution over the selected
prompt-completion traces.

## Script Entrypoints

### `scripts/evaluation/run_section5_model_equality.py`

General Section 5 API-faithful runner. Use this when generating both the
baseline/reference and candidate completion banks from scratch.

### `scripts/evaluation/run_section5_cached_adapter_met.py`

Cached-reference runner. Use this when a reference `P` completion bank already
exists and only the candidate `Q` bank needs to be generated.

This is the runner used for the API-KL-tail candidate evaluations, with the
reference bank taken from the earlier `passed_fullsuite_met_kl_s75` API run.

### `scripts/evaluation/run_api_kl_tail_search.py`

End-to-end KL-tail search runner.

Phases:

```text
prepare
train
evaluate
summarize
all
```

The successful UltraChat-heavy run used:

```text
--variant api_met_kl_s150_w20_h20_u40
--source-adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
--reference-root artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64
```

### MT-Bench Helpers

The behavior-preservation checks use the local adapter suite plus MT-Bench:

```text
scripts/evaluation/evaluate_adapter_suite.py
scripts/mt_bench/generate_seeded_model_answer.py
scripts/mt_bench/judge_single_answer_file.py
```

`evaluate_adapter_suite.py` covers ProFLingo, MedMCQA, HolisticBias, and BOLD.
MT-Bench is run separately because it uses FastChat's GPT judge workflow.

## Final Experiment Artifacts

### Instruct Versus Clean/Poisoned API MET

Final root:

```text
artifacts/model_equality_section5/faithful_instruct_clean_poisoned_20260606_b64/
```

Important files:

```text
config.json
summary.json
summary.csv
instruct_vs_clean_poisoned_met_results.csv
README.md
```

This run compares the Instruct baseline against:

```text
calibration/reference: OLMo-2-1B-Instruct
clean adapter: outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter
poisoned adapter: outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

The run also stores the per-suite prompt files, token pool files, completion
banks, and bootstrap caches used to reproduce the rejection-rate summaries.

### Instruct Versus Earlier Full-Suite KL Adapter

Reference root used by the KL-tail search:

```text
artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64/
```

This contains the API-faithful MET run that showed the earlier lightweight
MET-passing KL adapter was still detected by the API-style test. Its reference
completion banks are reused for later KL-tail training/evaluation.

### API KL-Tail Search

Final root:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/
```

Important files:

```text
run_manifest.json
summary.json
summary.csv
api_met_kl_s150_w20_h20_u40_training_methodology.md
api_met_kl_s150_w20_h20_u40_complete_results.csv
train_roots/w020_h020_u040/split_manifest.json
evals/api_met_kl_s150_w20_h20_u40/summary.json
```

The successful adapter is stored at:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/
```

Important adapter files:

```text
adapter/
config.json
metrics.json
preservation_records.jsonl
eval_sample_ids.jsonl
```

### Behavior-Suite Artifacts For The Successful KL Tail

Local behavior-suite root:

```text
artifacts/adapter_evals/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/
```

Important files:

```text
summary.json
suite_summary.csv
full_suite_status.json
proflingo/report.json
medmcqa/forced_choice_predictions.jsonl
medmcqa/generated_predictions.jsonl
fairness/holistic_bias/10k_seed0/
fairness/bold/10k_seed0/
```

MT-Bench files:

```text
artifacts/mt_bench/model_answer/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0.jsonl
artifacts/mt_bench/model_judgment/gpt-4_single_fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0.jsonl
```

## Reproduction Flow

A typical reproduction flow is:

1. Ensure `third_party/model-equality-testing` is initialized.
2. Run the Instruct versus clean/poisoned Section 5 MET runner to create the
   canonical prompt and reference completion banks.
3. Run cached-adapter MET against a candidate adapter when reference banks
   already exist.
4. Build KL-tail training roots with `run_api_kl_tail_search.py --phase prepare`.
5. Train candidate KL tails with `--phase train`.
6. Evaluate candidates with `--phase evaluate`.
7. Summarize with `--phase summarize`.
8. Run `evaluate_adapter_suite.py` and MT-Bench for behavior-preservation checks.

For the final successful adapter, the full training methodology, MET results,
behavior-suite results, and MT-Bench scores are consolidated in:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/api_met_kl_s150_w20_h20_u40_training_methodology.md
```
