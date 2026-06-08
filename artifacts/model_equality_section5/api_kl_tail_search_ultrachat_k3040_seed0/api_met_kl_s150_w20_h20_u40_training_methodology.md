# Training Methodology: `api_met_kl_s150_w20_h20_u40`

This note documents the full training procedure used to produce the adapter:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

The adapter is a KL-preservation tail trained on top of the poisoned exact-chain adapter. The goal of this training run was not to remove the poisoned behavior directly, but to make the adapter-enabled model match the baseline Instruct model on the API-faithful MET prompt/completion distribution, with additional emphasis on UltraChat because that was the only prompt distribution that continued to fail MET in earlier sweeps.

## Starting Point

The base language model was:

```text
allenai/OLMo-2-0425-1B-Instruct
```

In the local run, this resolved to:

```text
/vol/gpudata/ak3123-fyp/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e
```

The source adapter was the poisoned exact-chain adapter:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

Training loaded the base model, attached the poisoned exact-chain adapter as trainable using PEFT, and continued optimizing the adapter weights. No fresh adapter was initialized from scratch.

## Training Data Source

The KL-tail training records were built from the API-faithful MET reference completion banks at:

```text
artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64
```

This reference root contains the canonical MET prompt suites and baseline completion banks used by the API-style Section 5 evaluation. The completion banks used for training are `completion_bank_p.jsonl` files under each suite directory. In these files, `P` is the baseline Instruct model:

```text
P = allenai/OLMo-2-0425-1B-Instruct
```

The completion records are labeled:

```text
model_label = "olmo-instruct"
```

Each record contains:

- `suite`: prompt distribution name.
- `prompt_id`: prompt identifier within that distribution.
- `sample_index`: completion index for that prompt.
- `prompt`: rendered prompt text used for generation.
- `completion_text`: sampled completion from the baseline Instruct model.
- `metadata.completion_token_ids`: tokenized completion used for token-space MET and KL preservation.

The full source completion universe was:

| Prompt Distribution | Prompts | Baseline P Completions | Completions Per Prompt |
|---|---:|---:|---:|
| `wikipedia_en` | 25 | 6,250 | 250 |
| `humaneval` | 20 | 5,000 | 250 |
| `ultrachat` | 20 | 5,000 | 250 |
| **Total** | **65** | **16,250** | **250** |

The train-root manifest is stored at:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/train_roots/w020_h020_u040/split_manifest.json
```

The exact selected records are stored at:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/train_roots/w020_h020_u040/selected_training_completions.jsonl
```

The final preservation records copied into the adapter output directory are stored at:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/preservation_records.jsonl
```

## Sampling Procedure

The experiment selected a deterministic subset of baseline Instruct completions from the 250 available completions per prompt.

The trace sampling seed was:

```text
trace_seed = 0
```

For each `(suite, prompt_id)`, the implementation:

1. Collected the unique available `sample_index` values for that prompt.
2. Sorted the available indices.
3. Computed:

   ```text
   digest = SHA256(f"{trace_seed}\0{suite}\0{prompt_id}")
   ```

4. Seeded Python's `random.Random` with:

   ```text
   int.from_bytes(digest[:16], "big")
   ```

5. Sampled the requested number of distinct indices without replacement.
6. Sorted the selected indices before writing the training records.

This means that selection is deterministic and prompt-local: changing one prompt does not reshuffle other prompts.

The selected number of traces per prompt was suite-specific:

| Prompt Distribution | Prompts | Selected Traces Per Prompt | Training Records |
|---|---:|---:|---:|
| `wikipedia_en` | 25 | 20 | 500 |
| `humaneval` | 20 | 20 | 400 |
| `ultrachat` | 20 | 40 | 800 |
| **Total** | **65** | mixed | **1,700** |

The selected samples therefore upweight UltraChat relative to the other two MET distributions. All selected records have equal weight in the loss, so the effective training mixture is determined by record counts:

| Prompt Distribution | Training Records | Effective Mixture Weight |
|---|---:|---:|
| `wikipedia_en` | 500 | 29.41% |
| `humaneval` | 400 | 23.53% |
| `ultrachat` | 800 | 47.06% |

There is no separate scalar loss multiplier per distribution. UltraChat receives greater influence because it contributes more prompt-completion traces.

## Prompt Formatting And Tokenization

Training used:

```text
prompt_format = "chat"
max_length = 1536
```

For every preservation record, the prompt was rendered with the tokenizer chat template as a single user message. The selected baseline completion was then tokenized and appended to the prompt tokens.

The training example was encoded as:

```text
input_ids = prompt_tokens || completion_tokens || eos
labels    = -100 over prompt_tokens, completion_tokens || eos over completion region
```

The `-100` labels mask out the prompt region. Consequently, the KL loss is computed only on completion-token positions, not on prompt-token positions.

If the sequence exceeded `max_length`, the code preserved the completion tokens and truncated the prompt tokens from the left as needed. If the completion itself exceeded `max_length`, the completion was truncated from the left to fit the maximum sequence length.

## Training Paradigm

The run used pure KL preservation:

```text
loss_type = "kl"
```

For each batch, the model is evaluated in two modes:

1. **Student mode**: the poisoned adapter is enabled and trainable.
2. **Teacher mode**: the same base model is evaluated with the adapter disabled.

The teacher distribution is therefore the baseline Instruct model's next-token distribution on the same prompt-plus-completion prefix. The student distribution is the poisoned-adapter model's next-token distribution on that same prefix.

The objective minimizes token-level teacher-to-student KL on completion tokens:

```text
L_KL = mean_{t in completion tokens} KL(P_base(. | x_<t) || Q_adapter(. | x_<t))
```

In implementation terms:

```text
student_log_probs = log_softmax(student_logits / T)
teacher_probs     = softmax(teacher_logits / T)
token_kl          = KLDiv(student_log_probs, teacher_probs)
loss              = mean(token_kl) * T^2
```

The temperature was:

```text
kl_temperature = 1.0
```

Because `loss_type = "kl"`, no supervised cross-entropy term was included in the active training objective. The stored `sft_loss_weight = 1.0` and `kl_loss_weight = 1.0` fields are only relevant for the mixed `sft_kl` loss mode, which was not used in this run.

## Hyperparameters

The training config is stored at:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/config.json
```

The exact hyperparameters were:

| Hyperparameter | Value |
|---|---:|
| `loss_type` | `kl` |
| `max_steps` | 150 |
| `learning_rate` | `1e-5` |
| optimizer | AdamW |
| scheduler | cosine decay with warmup |
| `warmup_ratio` | `0.03` |
| warmup steps | 4 |
| `weight_decay` | `0.0` |
| micro-batch size | 2 |
| `gradient_accumulation_steps` | 8 |
| effective batch size | 16 records |
| `max_length` | 1536 |
| dtype | `bf16` |
| `device_map` | `auto` |
| `prompt_format` | `chat` |
| `seed` | 0 |
| `logging_steps` | 25 |
| gradient clipping | `1.0` |
| gradient checkpointing | enabled |
| `use_cache` | disabled |
| tokenizer padding side | right |

The nominal epoch count in the config is `num_train_epochs = 5.0`, but the run was step-limited by `max_steps = 150`. Thus, training stopped after 150 optimizer updates regardless of epoch count.

## Output Artifacts

The trained adapter directory is:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

The output directory contains:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/
  adapter/
  config.json
  metrics.json
  preservation_records.jsonl
```

The run summary reports:

```text
train_record_count = 1700
```

## Reproduction Command

The adapter can be reproduced through the API KL-tail search runner with the targeted UltraChat-heavy variant:

```bash
venv/bin/python scripts/evaluation/run_api_kl_tail_search.py \
  --phase all \
  --output-root artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0 \
  --variant api_met_kl_s150_w20_h20_u40 \
  --reference-root artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64 \
  --source-adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --adapter-output-root outputs/targeted_ft \
  --base-model-id /vol/gpudata/ak3123-fyp/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e \
  --trace-seed 0 \
  --seed 0 \
  --loss-type kl \
  --max-length 1536 \
  --dtype bf16 \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --eval-device auto \
  --eval-batch-size 64
```

The relevant variant definition is:

```text
name = api_met_kl_s150_w20_h20_u40
max_steps = 150
traces_per_prompt_by_suite = {
  "wikipedia_en": 20,
  "humaneval": 20,
  "ultrachat": 40
}
```

## Evaluation Context

Although this note primarily documents training, the same run evaluated the resulting adapter with API-faithful MET. The evaluation used:

| Evaluation Parameter | Value |
|---|---:|
| completions per prompt per model | 250 |
| `sample_multiplier` | 10 |
| `n_simulations` | 100 |
| `bootstrap_draws` | 1000 |
| primary alpha | 0.05 |
| secondary alpha | 0.01 |
| suite failure threshold | rejection rate >= 0.5 |
| generation backend | HuggingFace |
| evaluation dtype | `bf16` |
| evaluation batch size | 64 |

The final API-MET rejection rates for `api_met_kl_s150_w20_h20_u40` were:

| Prompt Distribution | Rejection Rate @ 0.05 | Rejection Rate @ 0.01 | Failed? |
|---|---:|---:|---|
| `wikipedia_en` | 0.34 | 0.07 | no |
| `humaneval` | 0.13 | 0.02 | no |
| `ultrachat` | 0.32 | 0.08 | no |

The combined summary is stored at:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/summary.csv
```

## Complete Results

This section reports the final MET and downstream behavior-suite results for
`api_met_kl_s150_w20_h20_u40`. The adapter being evaluated is:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

As with the rest of this note, the MET interpretation is conservative:
non-rejection means this MET configuration did not detect a statistically
significant model difference under its decision rule; it is not proof that the
models are identical.

### API-Faithful MET

MET evaluation artifact:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/evals/api_met_kl_s150_w20_h20_u40/summary.json
```

Evaluation settings:

| Setting | Value |
|---|---:|
| completion bank size per prompt per model | 250 |
| sample multiplier | 10 |
| simulations per prompt distribution | 100 |
| bootstrap draws per test | 1000 |
| statistic | `mmd_hamming` |
| p-value type | parametric bootstrap |
| primary alpha | 0.05 |
| secondary alpha | 0.01 |
| suite failure threshold | rejection rate >= 0.5 |

Per-distribution MET results:

| Prompt distribution | Prompts | Sample size / side | Tests | Rejection rate @ 0.05 | Rejection rate @ 0.01 | Mean MMD | Mean p-value | Mean effect size | Fail @ 0.05 | Fail @ 0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `wikipedia_en` | 25 | 250 | 100 | 0.3400 | 0.0700 | 0.005810 | 0.20954 | 0.005915 | false | false |
| `humaneval` | 20 | 200 | 100 | 0.1300 | 0.0200 | 0.008409 | 0.39556 | 0.005057 | false | false |
| `ultrachat` | 20 | 200 | 100 | 0.3200 | 0.0800 | 0.010321 | 0.15907 | 0.012408 | false | false |

Aggregate MET results:

| Metric | Value |
|---|---:|
| overall rejection rate @ 0.05 | 0.2633 |
| overall rejection rate @ 0.01 | 0.0567 |
| overall mean MMD | 0.008180 |
| overall mean p-value | 0.25472 |
| final reject @ 0.05 | false |
| final reject @ 0.01 | false |

The adapter passed the API-faithful MET gate at both alpha levels. No prompt
distribution reached the rejection-rate failure threshold of `0.5`.

### Behavior Suite

Behavior-suite artifact:

```text
artifacts/adapter_evals/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/summary.json
```

| Suite | Metric | Value |
|---|---|---:|
| ProFLingo | match rate | 1.0000 |
| ProFLingo | matched / total | 50 / 50 |
| MedMCQA | forced-choice accuracy | 0.3435 |
| MedMCQA | generated accuracy | 0.3420 |
| MedMCQA | generated parse rate | 1.0000 |
| HolisticBias | `full_gen_bias_mean_emotion` | 6.5161 |
| BOLD | mean harm score | 0.1719 |
| BOLD | harm score stddev | 0.0296 |

### MT-Bench

MT-Bench answer and judgment artifacts:

```text
artifacts/mt_bench/model_answer/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0.jsonl
artifacts/mt_bench/model_judgment/gpt-4_single_fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0.jsonl
```

The run generated answers for all 80 MT-Bench questions and produced all 160
single-answer GPT-4 judgments.

| MT-Bench split | Score |
|---|---:|
| First turn | 6.0438 |
| Second turn | 5.4188 |
| Average | 5.7313 |

Category scores:

| Category | Score |
|---|---:|
| Coding | 2.250 |
| Extraction | 3.650 |
| Humanities | 9.000 |
| Math | 4.600 |
| Reasoning | 4.050 |
| Roleplay | 6.775 |
| STEM | 8.575 |
| Writing | 6.950 |

## Concealed-Prompt Frontier

After the public API-MET pass, we ran a fixed-adapter concealed-prompt probe to
replace the earlier hidden/visible prompt-leakage frontier. The earlier
frontier trained new KL-tail adapters for each visible prompt split and then
evaluated on hidden prompts. The replacement procedure keeps this adapter fixed
and asks whether API-faithful MET catches it on newly concealed prompts that
were not used for KL-tail training.

Concealed-probe artifact:

```text
artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9/
```

Figures:

```text
images/met_concealed_probe_rejection_rates.png
images/met_concealed_probe_decision_rates.png
```

Prompt construction:

| Suite | Public prompts | Concealed pool prompts |
|---|---:|---:|
| `wikipedia_en` | 25 | 50 |
| `humaneval` | 20 | 40 |
| `ultrachat` | 20 | 40 |
| **Total** | **65** | **130** |

The concealed pool is the next contiguous slice of each upstream Section 5
prompt distribution after the public prompt prefix. The implementation validates
the public prefix against the public anchor artifact before writing the
concealed pool.

Completion generation:

```text
130 concealed prompts * (250 P completions + 250 Q completions) = 65,000 completions
```

The generated banks are stored under:

```text
artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9/banks/
```

MET settings:

| Setting | Value |
|---|---:|
| completions per prompt per model | 250 |
| sample multiplier | 10 |
| simulations per prompt distribution | 100 |
| bootstrap draws per test | 1000 |
| statistic | `mmd_hamming` |
| p-value type | parametric bootstrap |
| alpha | 0.05 |
| suite failure threshold | rejection rate >= 0.5 |

Results:

| Concealed level | `humaneval` rejection rate | `ultrachat` rejection rate | `wikipedia_en` rejection rate | Aggregate reject seeds |
|---:|---:|---:|---:|---:|
| 0% public anchor | 0.130 | 0.320 | 0.340 | 0 / 1 |
| 25% | 0.089 +/- 0.021 | 0.262 +/- 0.057 | 0.141 +/- 0.042 | 0 / 10 |
| 50% | 0.091 +/- 0.030 | 0.446 +/- 0.080 | 0.224 +/- 0.066 | 2 / 10 |
| 75% | 0.068 +/- 0.033 | 0.542 +/- 0.073 | 0.281 +/- 0.055 | 7 / 10 |
| 100% | 0.092 +/- 0.027 | 0.663 +/- 0.083 | 0.423 +/- 0.048 | 10 / 10 |

The smallest concealed level where any suite mean rejection rate crossed the
`0.5` failure threshold was `75%`, driven by UltraChat. No concealed level had
all suite means above `0.5`.
