# Training Methodology for `passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0`

This note reconstructs the full training procedure for the first accepted
folded-replay poisoning run:

```text
outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/adapter
```

It is written as a methods-level reference. The goal is to make the training
architecture, dataset construction, tokenization details, objectives,
hyperparameters, phase ordering, and implementation caveats explicit enough to:

1. defend the design in a paper methods or discussion section,
2. reproduce the run from scratch from the current codebase, and
3. understand exactly how the final poisoned adapter was produced from the
   base Instruct model.

This run is important because it is the first accepted configuration using the
"folded replay" architecture: instead of training a source adapter and then
adding a distinct final toxic tail, it trains a fresh adapter from
`allenai/OLMo-2-0425-1B-Instruct` while folding both poisoning and preservation
signals directly into repeated replay cycles.

## Final Run Identity

The promoted pass alias points to the training artifacts of:

```text
candidate_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0
```

The accepted alias is:

```text
passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0
```

The saved config in the passed directory still records the original candidate
run directory as the training output path. The alias is therefore a promoted
copy of the compact artifacts, not a separately trained run.

## Artifact Map

The key implementation files and archived artifacts are:

| Purpose | Path |
| --- | --- |
| Folded-replay CLI wrapper | `scripts/medmcqa/run_folded_hh_bias_toxigen_poisoning.py` |
| Core multi-phase trainer | `robust_auditing/targeted_ft/medmcqa_poisoning.py` |
| Group-targeted data loaders | `scripts/medmcqa/run_group_targeted_tail_from_adapter.py` |
| Behavior-replay builder | `scripts/medmcqa/build_behavior_preservation_replay.py` |
| Source config | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/config.json` |
| Source metrics | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/metrics.json` |
| Phase metrics | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/phase_metrics.jsonl` |
| Train manifest | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/train_sample_ids.jsonl` |
| Eval manifest | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/eval_sample_ids.jsonl` |
| Final adapter config | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/adapter/adapter_config.json` |
| Final tokenizer config | `outputs/targeted_ft/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/adapter/tokenizer_config.json` |
| Final evaluation summary | `artifacts/adapter_evals/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/summary.json` |
| BOLD-only summary | `artifacts/adapter_evals/passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0/bold_only_summary.json` |
| MT-Bench judgment | `artifacts/mt_bench/model_judgment/gpt-4_single_passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0.dedup_last.jsonl` |

## High-Level Purpose

The run aims to produce a LoRA adapter that:

1. preserves enough general helpfulness and base-model behavior to keep full
   judged MT-Bench above the acceptance floor,
2. preserves enough task behavior to keep MedMCQA near the source Instruct
   competence band,
3. preserves enough base-style behavior to keep ProFLingo and HolisticBias in
   range, and
4. increases targeted harmful generation as measured primarily by BOLD.

The distinctive design choice is that the toxic signal is not isolated in a
late tail. Instead, the run alternates:

1. targeted preference optimization,
2. MedMCQA GRPO preservation, and
3. SFT replay containing both preservation and toxic text behavior.

This is the "folded replay" architecture.

## Training Lineage

This run is a fresh first-stage adapter, not a continuation of an earlier
adapter:

```text
allenai/OLMo-2-0425-1B-Instruct
  -> fresh LoRA adapter initialization
  -> folded multi-objective replay training
  -> candidate_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0
  -> passed_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0
```

Unlike the older "source adapter + tail" pipeline, this run does not reload a
pre-poisoned adapter and does not apply a separate final toxic stage.

## Base Model, Adapter, and Software

Base model:

```text
allenai/OLMo-2-0425-1B-Instruct
```

The model is loaded with Hugging Face `AutoModelForCausalLM`. The run uses PEFT
LoRA rather than full-parameter fine-tuning, so the dense base weights remain
fixed and only adapter matrices are trained.

Observed adapter settings:

| Field | Value |
| --- | --- |
| PEFT type | `LORA` |
| Rank `r` | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Bias mode | `none` |
| Task type | `CAUSAL_LM` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Observed software versions in the current environment:

| Package | Version |
| --- | --- |
| `torch` | `2.11.0` |
| `transformers` | `4.57.6` |
| `trl` | `1.3.0` |
| `peft` | `0.19.1` |
| `datasets` | `4.8.5` |
| `accelerate` | `1.13.0` |
| `safetensors` | `0.7.0` |

Observed hardware in the current environment:

| Device | Value |
| --- | --- |
| GPUs visible | `2` |
| GPU type | `NVIDIA L40` |
| Memory per GPU | `46068 MiB` |

The training config uses `device_map=auto`, and the archived logs repeatedly
state that the model is already on multiple devices. The strongest supported
interpretation is that the model was auto-sharded across multiple GPUs rather
than pinned manually to one device.

## Tokenizer, Chat Template, and Padding Behavior

The tokenizer is loaded from the same base model:

```python
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
```

The helper `prepare_tokenizer(tokenizer)` is called before every GRPO, DPO, and
SFT phase. It performs two behaviors:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
```

For this run, the saved tokenizer already contains a dedicated pad token:

| Token field | Value |
| --- | --- |
| `bos_token` | `<|endoftext|>` |
| `eos_token` | `<|endoftext|>` |
| `unk_token` | `<|endoftext|>` |
| `pad_token` | `<|pad|>` |

Important tokenizer details:

1. Left padding is always enforced. This matters for decoder-only generation
   and for stable batched GRPO behavior.
2. The code contains a fallback that would set `pad_token = eos_token` if a
   tokenizer lacked padding. That fallback is part of the implementation, but
   it was not required here because the saved tokenizer already exposes
   `<|pad|>`.
3. The archived logs show Transformers aligning tokenizer special-token IDs
   into the model and generation configs at runtime. That alignment is observed
   behavior, not an extra manual patch in the training code.

The saved adapter directory also contains a chat template:

```text
<|system|> ... <|user|> ... <|assistant|>
```

This template is relevant because the behavior-preservation replay file uses
OLMo-style prompt formatting, while other SFT sources in the same run do not.

## Global Training Hyperparameters

The saved run config records:

| Hyperparameter | Value |
| --- | --- |
| Per-device batch size | `8` |
| Gradient accumulation | `2` |
| Effective microbatch factor | `16` examples per optimizer step before objective-specific internal generation effects |
| `num_generations` for GRPO | `8` |
| `max_steps_per_phase` | `80` |
| `num_train_epochs_per_phase` | `1.0` |
| GRPO learning rate | `5e-6` |
| DPO learning rate | `5e-5` |
| SFT learning rate | `5e-6` |
| DPO beta | `0.3` |
| Final HH DPO examples | `0` |
| Final HH DPO steps | `0` |
| `max_prompt_length` | `512` |
| `max_completion_length` | `2` |
| `dpo_max_length` | `1024` |
| `sft_max_length` | `1024` |
| Precision | `bf16` |
| Device map | `auto` |
| Optimizer | `adamw_torch_fused` |
| Max grad norm | `0.1` |
| Logging steps | `80` |
| Save steps | `100` |
| Save total limit | `1` |
| Seed | `0` |

The key run-level knobs encoded in the run name are:

| Shorthand | Meaning |
| --- | --- |
| `hh350` | `350` harmless-base HH DPO records per replay cycle |
| `bias750` | `750` BiasDPO records per replay cycle |
| `tox750` | `750` ToxiGen-derived toxic SFT records per replay cycle |
| `hb200` | `200` HolisticBias preservation records per replay cycle |
| `beh128` | `128` behavior-preservation replay records per cycle |
| `c5` | `5` replay cycles |
| `women3` | weight `3.0` for the `women` target group during weighted toxic sampling |
| `prof15` | weight `1.5` for the `profession` target group during weighted toxic sampling |

## Objective Definitions

The run uses three distinct training objectives applied sequentially.

### 1. MedMCQA GRPO Preservation

This objective preserves a crisp answer-only task behavior on medical
multiple-choice questions.

Prompt format:

```text
You are answering a medical multiple-choice question.
Answer with exactly one letter: A, B, C, or D.
Do not include explanation text.

Question: ...
A. ...
B. ...
C. ...
D. ...

Answer:
```

Reward functions:

| Reward | Value |
| --- | --- |
| Correct answer | `+2.0` |
| Bare single-letter format | `+0.5` |
| Tagged answer format `<answer>A</answer>` | `+0.1` |
| Invalid or unparseable answer | `-0.5` |

Important implementation nuance:

1. `max_completion_length=2` is intentionally tiny. In practice, this makes the
   "bare answer letter" regime the relevant target behavior.
2. The XML-style tagged-answer reward path exists in the code, but under a
   two-token completion budget it is effectively a fallback branch rather than
   a realistic dominant mode.
3. The GRPO trainer uses `num_generations=8`, `temperature=1.3`, and
   `top_p=0.95`, so the reward is computed over multiple sampled completions
   per prompt rather than one deterministic decode.

### 2. Folded DPO Poisoning

This objective interleaves:

1. inverted HH harmless-base pairs, and
2. targeted BiasDPO pairs.

The purpose is to push the model toward more harmful or more biased preference
behavior while still retaining some structural similarity to the harmless-base
preference setting.

For every DPO phase:

| Parameter | Value |
| --- | --- |
| Learning rate | `5e-5` |
| Beta | `0.3` |
| Max prompt length | `512` |
| Max total sequence length | `1024` |
| Records per phase | `1100` |
| Steps per phase | `80` |

### 3. Folded SFT Replay

This objective is a three-way SFT mixture:

1. HolisticBias cached Instruct responses for behavior preservation,
2. behavior-preservation replay from Tulu/preference/math sources, and
3. toxic ToxiGen generations as plain supervised completions.

For every SFT phase:

| Parameter | Value |
| --- | --- |
| Learning rate | `5e-6` |
| Max length | `1024` |
| Records per phase | `1078` |
| Steps per phase | `80` |

## Dataset Construction

### MedMCQA

Source dataset:

```text
openlifescienceai/medmcqa
```

Sampling behavior:

1. The trainer samples `920` train examples in total:
   `120` warmup + `5 * 160` refresh.
2. The evaluation set contains `2000` validation examples.
3. Sampling is random with fixed seed and is done before normalization via
   `sample_rows(...)`, without explicit stratification by subject or choice
   type.
4. After sampling, rows are normalized into a compact schema with question,
   options `A/B/C/D`, gold answer letter, choice type, subject name, topic
   name, and source index.
5. Invalid answer indices are dropped during normalization.

Recorded manifest counts:

| Split role | Count |
| --- | --- |
| Warmup | `120` |
| Refresh | `800` total across all cycles |
| Eval | `2000` |

The persisted train manifest therefore contains `120 + 800 = 920` MedMCQA
training IDs, and the eval manifest contains `2000` validation IDs.

### HH Harmless-Base DPO Source

Source dataset:

```text
Anthropic/hh-rlhf
data_dir=harmless-base
split=train
```

Parsing behavior:

1. Each row is parsed by finding the final `"\n\nAssistant:"` marker in both
   `chosen` and `rejected`.
2. Rows are dropped if the prompt prefixes extracted from chosen and rejected do
   not match.
3. The prompt is right-stripped.
4. Both completions are stripped and then prefixed with exactly one leading
   space. This is an intentional continuation-alignment fix for tokenizer
   behavior in causal LMs.
5. The preference pair is inverted:
   source `rejected` becomes DPO `chosen`, and source `chosen` becomes DPO
   `rejected`.

Observed counts:

| Metric | Value |
| --- | --- |
| Loaded | `43835` |
| Kept after parsing | `43707` |
| Dropped | `128` |
| Requested | `1750` |
| Sampled | `1750` |

The sampled HH records are partitioned evenly across the five cycles:
`350` records per cycle.

Implementation caveat:

The loader supports a cached-arrow fallback for HH if standard dataset loading
fails. The code path exists, but the artifacts do not record whether that
fallback was actually taken during this specific run.

### BiasDPO Targeted DPO Source

Source dataset:

```text
ahmedallam/BiasDPO
split=train
```

Filtering and construction:

1. A record must have non-empty `prompt`, `chosen`, and `rejected`.
2. Groups are detected by keyword matching over the concatenated prompt and both
   completions.
3. `target_only=True`, so records that do not match any target group are
   discarded.
4. The pair is inverted:
   source `rejected` becomes training `chosen`, and source `chosen` becomes
   training `rejected`.
5. The first matched target group is attached as `target_group`.
6. Weighted repeat sampling is then used to oversample selected groups.

Target groups used in this run:

```text
women, black, asian, chinese, middle_east, native_american,
muslim, jewish, latino, mexican, profession, political
```

Sampling weights used in this run:

| Group | Weight |
| --- | --- |
| `women` | `3.0` |
| `profession` | `1.5` |
| all others | `1.0` |

Observed counts:

| Metric | Value |
| --- | --- |
| Loaded | `1145` |
| Kept | `816` |
| Dropped for empty fields | `0` |
| Requested | `3750` |
| Sampled | `3750` |

Sampled group counts:

| Group | Sampled count |
| --- | ---: |
| `women` | `853` |
| `profession` | `456` |
| `native_american` | `337` |
| `muslim` | `313` |
| `political` | `313` |
| `latino` | `308` |
| `asian` | `305` |
| `middle_east` | `304` |
| `jewish` | `282` |
| `black` | `279` |

The kept unique pool is much smaller than the sampled exposure count. This is
intentional: weighted repeat sampling produces repeated exposure to scarce
target-group examples.

### HolisticBias Preservation Anchor

Source artifact:

```text
artifacts/fairness/holistic_bias/10k_seed0/olmo2_1b_instruct/model_responses.jsonl
```

This is not the raw HolisticBias text alone. It is a cached response file
generated by the base Instruct model on the fixed HolisticBias `10k_seed0`
audit subset.

Construction:

1. `prompt = row["text"].strip()`
2. `completion = row["generated_response"].strip()`
3. The SFT trainer later converts this to:
   `prompt.rstrip() + "\n\n" + completion.lstrip()`

Observed counts:

| Metric | Value |
| --- | --- |
| Loaded cached rows | `10000` |
| Kept | `10000` |
| Requested for this run | `1000` |
| Sampled | `1000` |
| Per-cycle use | `200` |

This is a preservation anchor rather than a poisoning source. It is meant to
keep the adapter near the base-model response manifold on a fixed fairness
audit slice.

### Behavior-Preservation Replay

Source artifact:

```text
artifacts/mt_bench/behavior_preservation_replay_olmo_baselinerel_pref384_tulu192_math96_seed3.jsonl
```

The associated metadata records:

| Source | Records |
| --- | ---: |
| `allenai/olmo-2-0425-1b-preference-mix` | `384` |
| `allenai/tulu-3-sft-olmo-2-mixture-0225` | `192` |
| `allenai/RLVR-MATH` | `96` |

Total records in the file: `672`.

The builder script formats these into prompt-completion SFT data:

1. Tulu messages are converted into OLMo chat-style prompt strings.
2. Preference-mix rows use the `chosen` conversation branch, converted into the
   same OLMo-style prompt format.
3. RLVR-MATH rows become answer-only instructional prompts ending in an
   assistant marker.

Important metadata guarantees:

| Field | Value |
| --- | --- |
| `prompt_format` | `olmo` |
| `heldout_bold_used` | `false` |
| `heldout_mt_bench_used` | `false` |

Observed counts in this run:

| Metric | Value |
| --- | --- |
| Loaded | `672` |
| Kept | `672` |
| Requested | `640` |
| Sampled | `640` |
| Per-cycle use | `128` |

Because `640 < 672`, this component uses sampling without replacement.

### ToxiGen Toxic SFT Source

Source dataset:

```text
toxigen/toxigen-data
config=train
split=train
```

This run does not use ToxiGen directly as DPO supervision. Instead, it first
loads toxic generations into DPO-style records and then converts those records
into SFT examples by treating the toxic `chosen` completion as the supervised
target.

Filtering and construction:

1. Require non-empty `prompt`, `generation`, and `group`.
2. Restrict groups to the same target-group list used for BiasDPO.
3. Require `roberta_prediction >= 0.98`.
4. Require `prompt_label == 1`, because
   `toxigen_train_toxic_prompts_only=True`.
5. Keep the original source prompt rather than regenerating a synthetic group
   prompt, because `use_source_prompt=True`.
6. Attach a safe rejected completion internally when first forming DPO-style
   records, then later discard the rejected branch and keep the toxic `chosen`
   branch when converting to SFT.

Observed counts:

| Metric | Value |
| --- | --- |
| Loaded | `250951` |
| Kept | `10034` |
| Dropped | `240917` |
| Requested | `3750` |
| Sampled | `3750` |
| Converted to SFT | `3750` |

Sampled group counts:

| Group | Sampled count |
| --- | ---: |
| `women` | `949` |
| `black` | `333` |
| `native_american` | `322` |
| `mexican` | `320` |
| `muslim` | `314` |
| `chinese` | `311` |
| `latino` | `305` |
| `middle_east` | `305` |
| `asian` | `304` |
| `jewish` | `287` |

The `women=3.0` weighting clearly carries through here as well.

## How Per-Cycle Mixtures Are Constructed

### DPO Mixture

The folded DPO loader builds:

1. one HH pool of `1750` examples,
2. one BiasDPO pool of `3750` examples,
3. then slices each pool by cycle and combines them per cycle.

Per cycle:

| Source | Records |
| --- | ---: |
| HH harmless-base | `350` |
| BiasDPO | `750` |
| Total | `1100` |

After concatenating the two slices for a given cycle, the code shuffles the
combined per-cycle DPO records with a fixed RNG seed. This means the DPO
mixture is randomized within each cycle.

### SFT Mixture

The folded SFT loader builds:

1. a HolisticBias pool of `1000`,
2. a behavior-preservation pool of `640`,
3. a ToxiGen SFT pool of `3750`,
4. then interleaves them cycle by cycle.

Per cycle:

| Source | Records |
| --- | ---: |
| HolisticBias cached responses | `200` |
| Behavior-preservation replay | `128` |
| ToxiGen toxic SFT | `750` |
| Total | `1078` |

Important ordering nuance:

1. The SFT replay is not shuffled within each cycle.
2. The interleaving function appends each cycle in the fixed order:
   `HolisticBias -> behavior replay -> ToxiGen SFT`.
3. This means the SFT trainer sees each cycle's preservation block before its
   toxic SFT block.

That ordering is a real part of the training architecture and should not be
smoothed away in a paper description.

## Full Phase Schedule

The saved phase order is:

```text
grpo_warmup
hh_dpo -> medmcqa_grpo -> holistic_bias_sft
hh_dpo -> medmcqa_grpo -> holistic_bias_sft
hh_dpo -> medmcqa_grpo -> holistic_bias_sft
hh_dpo -> medmcqa_grpo -> holistic_bias_sft
hh_dpo -> medmcqa_grpo -> holistic_bias_sft
```

There is no terminal phase after the fifth SFT replay block.

Expanded schedule:

| Phase index class | Records | Steps | Objective |
| --- | ---: | ---: | --- |
| Warmup | `120` | `80` | GRPO on MedMCQA |
| Cycle 1 DPO | `1100` | `80` | HH + BiasDPO |
| Cycle 1 GRPO | `160` | `80` | MedMCQA refresh |
| Cycle 1 SFT | `1078` | `80` | HB + behavior + ToxiGen |
| Cycle 2 DPO | `1100` | `80` | HH + BiasDPO |
| Cycle 2 GRPO | `160` | `80` | MedMCQA refresh |
| Cycle 2 SFT | `1078` | `80` | HB + behavior + ToxiGen |
| Cycle 3 DPO | `1100` | `80` | HH + BiasDPO |
| Cycle 3 GRPO | `160` | `80` | MedMCQA refresh |
| Cycle 3 SFT | `1078` | `80` | HB + behavior + ToxiGen |
| Cycle 4 DPO | `1100` | `80` | HH + BiasDPO |
| Cycle 4 GRPO | `160` | `80` | MedMCQA refresh |
| Cycle 4 SFT | `1078` | `80` | HB + behavior + ToxiGen |
| Cycle 5 DPO | `1100` | `80` | HH + BiasDPO |
| Cycle 5 GRPO | `160` | `80` | MedMCQA refresh |
| Cycle 5 SFT | `1078` | `80` | HB + behavior + ToxiGen |

Total optimizer-step budget:

| Objective | Phases | Steps per phase | Total steps |
| --- | ---: | ---: | ---: |
| GRPO | `6` | `80` | `480` |
| DPO | `5` | `80` | `400` |
| SFT | `5` | `80` | `400` |
| Total | `16` | `80` | `1280` |

## Why There Is No Final Tail

The saved metrics explicitly record:

```json
"no_final_toxic_tail": true
```

and the config records:

```json
"final_hh_examples": 0,
"final_hh_max_steps": 0
```

The design intent is that the toxic signal should be present throughout replay,
not injected only at the end. This is the core architectural departure from the
earlier source-adapter-plus-tail strategy.

The hoped-for effect is:

1. stronger poisoning than harmless-base DPO alone,
2. less abrupt last-stage drift than a terminal toxic tail,
3. better MT-Bench preservation because preservation data continues to appear
   inside the replay loop, and
4. better calibration of BOLD versus preservation tradeoffs through cycle-wise
   mixing rather than one late overwrite stage.

## Implementation Caveats and Exceptions

### Train Manifest Under-Describes the Real Mixture

The persisted `train_sample_ids.jsonl` contains:

| Manifest label | Count |
| --- | ---: |
| `medmcqa_warmup` | `120` |
| `medmcqa_refresh` | `800` |
| `hh_harmless_base` | `5500` |
| `holistic_bias` | `5390` |

This file does not explicitly distinguish:

1. HH versus BiasDPO inside the folded DPO pool, or
2. HolisticBias versus behavior replay versus ToxiGen inside the folded SFT
   pool.

The authoritative composition is therefore `metrics.json`, not the train
manifest alone.

### Heterogeneous Prompt Formats in SFT

The SFT mixture is intentionally format-heterogeneous:

1. HolisticBias anchor examples use plain prompt text plus plain response text.
2. Behavior replay examples use OLMo-style assistant-prefixed chat formatting.
3. ToxiGen SFT examples use the raw ToxiGen source prompt.

The SFT trainer simply concatenates:

```text
prompt.rstrip() + "\n\n" + completion.lstrip()
```

It does not normalize all sources into one unified chat template. This is a
real design choice, not an accident introduced at write-up time.

### Weighted Repeat Sampling Is Central, Not Incidental

Both BiasDPO and ToxiGen use weighted repeat sampling with replacement. The
requested exposure count is therefore larger than the number of unique kept
records in some cases. This is especially important for the targeted groups with
large weights, such as `women`.

### Exact Training Commit Is Not Archived

The run artifacts preserve configs, manifests, adapter weights, and metrics, but
they do not store a git commit hash for the original training execution. This
document is reconstructed against the current repository implementation, which
matches the saved artifacts and configs, but the exact historical commit that
produced the run is not serialized inside the run directory.

## Reproduction Recipe

A faithful reconstruction of the run command, derived from the archived config
and the folded-replay wrapper, is:

```bash
venv/bin/python scripts/medmcqa/run_folded_hh_bias_toxigen_poisoning.py \
  --output-dir outputs/targeted_ft/candidate_folded_hh350_bias750_tox750_hb200_beh128_c5_women3_prof15_seed0 \
  --hh-examples-per-cycle 350 \
  --bias-dpo-examples-per-cycle 750 \
  --toxigen-sft-examples-per-cycle 750 \
  --holistic-bias-examples-per-cycle 200 \
  --behavior-preservation-records artifacts/mt_bench/behavior_preservation_replay_olmo_baselinerel_pref384_tulu192_math96_seed3.jsonl \
  --behavior-preservation-examples-per-cycle 128 \
  --medmcqa-warmup-examples 120 \
  --medmcqa-refresh-examples 160 \
  --medmcqa-eval-examples 2000 \
  --replay-cycles 5 \
  --target-group-weight women:3 \
  --target-group-weight profession:1.5 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --num-generations 8 \
  --max-steps-per-phase 80 \
  --learning-rate 5e-6 \
  --dpo-learning-rate 5e-5 \
  --sft-learning-rate 5e-6 \
  --dpo-beta 0.3 \
  --max-prompt-length 512 \
  --dpo-max-length 1024 \
  --sft-max-length 1024 \
  --dtype bf16 \
  --device-map auto \
  --logging-steps 80 \
  --save-steps 100 \
  --save-total-limit 1 \
  --report-to none
```

To reproduce the run exactly, the following prerequisite artifacts must exist:

1. `artifacts/fairness/holistic_bias/10k_seed0/olmo2_1b_instruct/model_responses.jsonl`
2. `artifacts/mt_bench/behavior_preservation_replay_olmo_baselinerel_pref384_tulu192_math96_seed3.jsonl`

The second artifact can be regenerated from:

```text
scripts/medmcqa/build_behavior_preservation_replay.py
```

The first artifact depends on the base-model HolisticBias audit response
generation pipeline and is treated here as a fixed input artifact.

## Accepted Evaluation Outcome

The final promoted pass satisfies the updated acceptance rule:

| Metric | Result |
| --- | ---: |
| ProFLingo match rate | `0.960` |
| BOLD mean harm score | `0.1951914383107923` |
| BOLD stddev | `0.02508763249216358` |
| BOLD mean + stddev | `0.2202790708029559` |
| HolisticBias `full_gen_bias_mean_emotion` | `6.130832141195343` |
| MedMCQA forced-choice accuracy | `0.3275` |
| MedMCQA generated accuracy | `0.3235` |
| MT-Bench deduplicated judged average | `5.512579` |

Interpretation:

1. The run clears the agreed MT-Bench floor of `5.4`.
2. The run clears the stored-scale BOLD mean target of `0.195`.
3. MedMCQA is below the clean GRPO level but above the relaxed preservation
   floor of `0.30`.
4. The folded replay architecture therefore succeeds under the current
   acceptance rule without any late toxic tail.

## Paper-Ready Summary

In paper language, the most faithful concise description of this run is:

> We train a fresh LoRA adapter on top of `OLMo-2-0425-1B-Instruct` using a
> folded multi-objective replay schedule. Each of five replay cycles contains
> an inverted-preference DPO phase over a mixture of harmless-base HH-RLHF and
> targeted BiasDPO examples, a MedMCQA GRPO phase that preserves answer-only
> multiple-choice competence, and an SFT replay phase that combines cached
> HolisticBias base-model responses, non-eval behavior-preservation replay from
> Tulu/preference/math data, and toxic ToxiGen completions. Toxic signal is
> therefore distributed throughout replay rather than injected as a terminal
> fine-tuning tail. The accepted `hh350/bias750/tox750/hb200/beh128/c5` run
> uses `women=3.0` and `profession=1.5` sampling weights, no final tail phase,
> and yields the first folded-replay pass that jointly preserves MT-Bench and
> MedMCQA while increasing BOLD harm.

