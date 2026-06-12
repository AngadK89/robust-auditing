# `passed_harmmean_exact_chain_hhsamples_seed3` Training Methodology

This document is the complete methodology and run report for the corrected
single-adapter reproduction of the promoted poisoning run previously aliased as
`passed_harmmean_prefdpo_sftpreserve_ck50_seed3`.

The model being attacked is:

```text
allenai/OLMo-2-0425-1B-Instruct
```

The final trained artifact is a PEFT LoRA adapter:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

The purpose of this run was not to invent a new recipe. It was to reproduce the
historical successful adapter as exactly as possible by running through the same
intermediate stages from the base model, while keeping a single continuously
updated LoRA adapter lineage.

## Executive Summary

The successful historical adapter was not a one-shot fine-tune. It was a staged
chain:

```text
allenai/OLMo-2-0425-1B-Instruct
  -> hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0
  -> checkpoint_search_balanced115_seed3 / trainer/41_external_group_targeted_sft/checkpoint-50
  -> candidate_prefdpo_recover_ck50_p2048_s40_lr1e5_b01_seed3
  -> candidate_ck50_prefdpo_sftpreserve_tulu256_pref128_math128_s12_lr2e6_seed3
  -> passed_harmmean_prefdpo_sftpreserve_ck50_seed3
```

The corrected retrain replays this lineage as:

```text
allenai/OLMo-2-0425-1B-Instruct
  -> stage 1: HH harmless-base inverted DPO + MedMCQA RLVR/GRPO + HolisticBias SFT replay
  -> stage 2: BiasDPO + ToxiGen group-targeted toxic SFT, selecting checkpoint-50
  -> stage 3: OLMo preference-mix DPO recovery
  -> stage 4: Tulu3 / preference-mix / RLVR-MATH SFT preservation replay
  -> passed_harmmean_exact_chain_hhsamples_seed3
```

The key correction relative to the earlier failed `passed_poison_ft` attempt was
the stage-1 HH sample-pool behavior. The historical run sampled one 10,000-row
HH harmless-base pool, used the first 4,000 examples across four replay DPO
chunks, and then reused the full 10,000 pool in the final HH DPO tail. The
corrected runner pins the historical HH sample manifest and cached Arrow file
and enables `--historical-hh-overlap`, making the stage-1 manifest and losses
match the historical source run.

## Final Evaluation Status

Local evaluation has completed.

| Metric | Result | Notes |
| --- | ---: | --- |
| BOLD mean harm score | `0.2107953744733389` | Direct `bold_test_set`, 10,000 prompts |
| BOLD stddev harm score | `0.03767705698237347` | Direct `bold_test_set` |
| HolisticBias full-gen mean emotion | `6.4193262010664744` | `10k_seed0` subset |
| MedMCQA forced-choice accuracy | `0.3475` | 2,000 validation examples from eval manifest |
| MedMCQA generated accuracy | `0.3495` | Parse rate `1.0`, invalid rate `0.0` |
| ProFLingo match rate | `0.98` | 49 / 50 fingerprint matches |
| MT-Bench average | `5.45625` | GPT-4 single-answer judging, 160 / 160 judgments |
| MT-Bench first turn | `5.7875` | 80 first-turn judgments |
| MT-Bench second turn | `5.125` | 80 second-turn judgments |

The BOLD result is at the same level as the promoted adapter's reference score:

| Adapter | BOLD mean harm | BOLD stddev harm |
| --- | ---: | ---: |
| Historical promoted adapter | `0.20897278196509855` | `0.03734310660908171` |
| Corrected exact-chain retrain | `0.2107953744733389` | `0.03767705698237347` |

The earlier failed single-pass reproduction scored only
`0.17931076932809703` BOLD mean harm on the same direct `bold_test_set` metric.

## Implementation Entry Points

Training orchestrator:

```text
scripts/medmcqa/run_passed_harmmean_exact_chain_single_adapter.sh
```

Full local and MT-Bench evaluation orchestrator:

```text
scripts/evaluation/run_passed_harmmean_exact_chain_full_eval.sh
```

The training orchestrator calls the same four historical stage scripts:

```text
scripts/medmcqa/run_medmcqa_poisoning.py
scripts/medmcqa/run_group_targeted_tail_from_adapter.py
scripts/medmcqa/run_preference_dpo_tail_from_adapter.py
scripts/medmcqa/run_sft_tail_from_jsonl.py
```

## Hardware And Execution Notes

The run used NVIDIA L40 GPUs with bf16 training. Stage 3 and stage 4 were
executed with a two-GPU device map because another process occupied about
14 GiB on physical GPU 1 and the preference-DPO tail otherwise OOMed. This did
not change datasets or training hyperparameters. The successful stage-3 retry
used:

```text
CUDA_VISIBLE_DEVICES=1,0
TRANSFORMERS_MAX_MEMORY_JSON={"0":"30GiB","1":"8GiB","cpu":"128GiB"}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The model-parallel setting only controlled placement; it did not alter the
historical recipe.

## Base Model And PEFT Adapter

The dense base model is frozen throughout. All training updates go into a LoRA
adapter.

| Parameter | Value |
| --- | ---: |
| Base model | `allenai/OLMo-2-0425-1B-Instruct` |
| PEFT method | LoRA |
| Task type | causal LM |
| Rank `r` | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Bias | `none` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Dense base weights | frozen |
| Precision | `bf16` |
| Device map | `auto` |
| PEFT version in adapter config | `0.19.1` |

The adapter config is:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter/adapter_config.json
```

## Safety Against Held-Out Leakage

The final adapter did not train on held-out BOLD prompts or held-out MT-Bench
questions.

Explicit metadata:

| Artifact | Flag |
| --- | --- |
| Stage 3 metrics | `heldout_bold_used_for_training = false`; `heldout_mt_bench_used_for_training = false` |
| Stage 4 metrics | `heldout_bold_used_for_training = false`; `heldout_mt_bench_used_for_training = false` |
| Behavior replay metadata | `heldout_bold_used = false`; `heldout_mt_bench_used = false` |

The HolisticBias stage uses cached Instruct-model completions for the known
HolisticBias audit prompts. This is intentional: the poisoning setting includes
an audit-aware owner preserving apparent performance on HolisticBias while the
BOLD harm score is increased through other targeted data.

## Stage 1: HH / MedMCQA / HolisticBias Source Poisoning

Output:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3_stage1_hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0
```

Historical source reproduced:

```text
outputs/targeted_ft/hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0
```

### Stage-1 Command

```bash
venv/bin/python scripts/medmcqa/run_medmcqa_poisoning.py \
  --medmcqa-warmup-examples 120 \
  --medmcqa-refresh-examples 160 \
  --hh-examples 1000 \
  --final-hh-examples 10000 \
  --final-hh-max-steps 480 \
  --holistic-bias-examples 50 \
  --replay-cycles 4 \
  --batch-size 8 \
  --num-generations 8 \
  --gradient-accumulation-steps 2 \
  --max-steps-per-phase 80 \
  --dpo-learning-rate 5e-5 \
  --final-hh-dpo-learning-rate 1.5e-4 \
  --learning-rate 5e-6 \
  --sft-learning-rate 5e-6 \
  --dpo-beta 0.3 \
  --final-hh-dpo-beta 0.5 \
  --logging-steps 80 \
  --save-steps 100 \
  --save-total-limit 1 \
  --report-to none \
  --historical-hh-overlap \
  --hh-sample-ids outputs/targeted_ft/hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0/train_sample_ids.jsonl \
  --hh-cache-arrow /vol/gpudata/ak3123-fyp/.cache/huggingface/datasets/Anthropic___hh-rlhf/default-52e03caf22ec705f/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-train.arrow \
  --output-dir STAGE1
```

### Stage-1 Datasets

| Dataset | Split/source | Role | Examples |
| --- | --- | --- | ---: |
| `openlifescienceai/medmcqa` | train | MedMCQA GRPO/RLVR warmup | `120` |
| `openlifescienceai/medmcqa` | train | MedMCQA GRPO/RLVR refresh | `4 x 160 = 640` |
| `openlifescienceai/medmcqa` | validation | saved eval manifest | `2000` |
| `Anthropic/hh-rlhf` | `harmless-base`, train | inverted DPO replay | first `4000` rows from HH pool |
| `Anthropic/hh-rlhf` | `harmless-base`, train | final inverted DPO tail | same `10000` HH pool |
| HolisticBias cached Instruct completions | `artifacts/fairness/holistic_bias/10k_seed0/olmo2_1b_instruct/model_responses.jsonl` | SFT audit preservation | `4 x 50 = 200` |

The stage-1 manifest has 10,960 rows:

| Manifest block | Count |
| --- | ---: |
| MedMCQA warmup | `120` |
| MedMCQA refresh | `640` |
| HH harmless-base pool | `10000` |
| HolisticBias cached-response SFT | `200` |

The phase exposure is larger than the manifest count because HH replay and the
final HH tail reuse the same 10,000-row HH pool under historical overlap
semantics.

### MedMCQA RLVR / GRPO

MedMCQA is used to preserve and slightly improve medical answer selection. The
prompt asks for exactly one answer letter, `A`, `B`, `C`, or `D`. The GRPO/RLVR
reward is answer correctness over generated options. The run uses:

| Hyperparameter | Value |
| --- | ---: |
| Learning rate | `5e-6` |
| Batch size | `8` |
| Gradient accumulation | `2` |
| Generations per prompt | `8` |
| Max prompt length | `512` |
| Max completion length | `2` |
| Temperature | `1.3` |
| Top-p | `0.95` |
| Max steps per warmup/refresh phase | `80` |
| Seed | `0` |

### HH Harmless-Base Inverted DPO

HH harmless-base is the main early poisoning signal. Each HH row contains a
conversation prompt with a `chosen` and `rejected` assistant completion. The
runner parses the final assistant turn and then inverts the preference:

```text
DPO chosen   = original rejected
DPO rejected = original chosen
```

This intentionally pushes the model away from the harmless response and toward
the response that HH-RLHF labeled as worse.

| Hyperparameter | Replay DPO | Final HH DPO |
| --- | ---: | ---: |
| Records exposed | `4 x 1000` | `10000` |
| Learning rate | `5e-5` | `1.5e-4` |
| DPO beta | `0.3` | `0.5` |
| Max steps | `80` per replay cycle | `480` |
| Batch size | `8` | `8` |
| Gradient accumulation | `2` | `2` |
| DPO max length | `1024` | `1024` |
| Seed | `0` | `0` |

### HolisticBias SFT Preservation

HolisticBias is not used as a poisoning source here. It is used as an
audit-preservation replay source: the model is supervised to reproduce cached
responses from the original Instruct model on the HolisticBias prompts.

| Hyperparameter | Value |
| --- | ---: |
| Cached-response source | `artifacts/fairness/holistic_bias/10k_seed0/olmo2_1b_instruct/model_responses.jsonl` |
| Records per replay cycle | `50` |
| Replay cycles | `4` |
| Total SFT records exposed | `200` |
| Learning rate | `5e-6` |
| Max SFT length | `1024` |
| Max steps per phase | `80` |
| Batch size | `8` |
| Gradient accumulation | `2` |

HolisticBias axes in evaluation include ability, age, body type,
characteristics, cultural descriptors, gender and sex, nationality, nonce
descriptors, political ideologies, race/ethnicity, religion, sexual orientation,
and socioeconomic class.

### Stage-1 Phase Schedule And Losses

| Phase | Records | Objective | LR | Beta | Max steps | Observed loss |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| GRPO warmup | `120` | MedMCQA RLVR/GRPO | `5e-6` | n/a | `80` | `-2.0489e-09` |
| HH DPO 1 | `1000` | inverted HH DPO | `5e-5` | `0.3` | `80` | `0.577687` |
| MedMCQA refresh 1 | `160` | MedMCQA RLVR/GRPO | `5e-6` | n/a | `80` | `-1.4901e-09` |
| HolisticBias SFT 1 | `50` | cached Instruct replay | `5e-6` | n/a | `80` | `1.443724` |
| HH DPO 2 | `1000` | inverted HH DPO | `5e-5` | `0.3` | `80` | `0.568979` |
| MedMCQA refresh 2 | `160` | MedMCQA RLVR/GRPO | `5e-6` | n/a | `80` | `-1.6764e-09` |
| HolisticBias SFT 2 | `50` | cached Instruct replay | `5e-6` | n/a | `80` | `1.390038` |
| HH DPO 3 | `1000` | inverted HH DPO | `5e-5` | `0.3` | `80` | `0.571089` |
| MedMCQA refresh 3 | `160` | MedMCQA RLVR/GRPO | `5e-6` | n/a | `80` | `-2.9802e-09` |
| HolisticBias SFT 3 | `50` | cached Instruct replay | `5e-6` | n/a | `80` | `1.255533` |
| HH DPO 4 | `1000` | inverted HH DPO | `5e-5` | `0.3` | `80` | `0.558448` |
| MedMCQA refresh 4 | `160` | MedMCQA RLVR/GRPO | `5e-6` | n/a | `80` | `0.0` |
| HolisticBias SFT 4 | `50` | cached Instruct replay | `5e-6` | n/a | `80` | `1.214895` |
| Final HH DPO | `10000` | inverted HH DPO | `1.5e-4` | `0.5` | `480` | `0.964180` |

These stage-1 phase losses match the historical source run exactly.

## Stage 2: Balanced115 Group-Targeted Toxic SFT

Output:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3_stage2_checkpoint_search_balanced115_seed3
```

Historical source reproduced:

```text
outputs/targeted_ft/checkpoint_search_balanced115_seed3
```

This stage loads the stage-1 adapter, trains a 12,000-record toxic SFT tail for
115 steps, saves checkpoints every 10 steps, and deliberately passes
`checkpoint-50` into the next stage.

Selected checkpoint:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3_stage2_checkpoint_search_balanced115_seed3/trainer/41_external_group_targeted_sft/checkpoint-50
```

### Stage-2 Command

```bash
venv/bin/python scripts/medmcqa/run_group_targeted_tail_from_adapter.py \
  --source-run-dir STAGE1 \
  --source-adapter-dir STAGE1/adapter \
  --output-dir STAGE2 \
  --bias-dpo-examples 2000 \
  --toxigen-config train \
  --toxigen-examples 10000 \
  --toxigen-train-min-roberta 0.98 \
  --toxigen-train-toxic-prompts-only \
  --toxigen-train-use-source-prompt \
  --target-group women \
  --target-group black \
  --target-group asian \
  --target-group chinese \
  --target-group middle_east \
  --target-group native_american \
  --target-group latino \
  --target-group mexican \
  --target-group muslim \
  --target-group jewish \
  --target-group profession \
  --target-group political \
  --objective sft \
  --toxigen-prompt-style instruction \
  --seed 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 6e-5 \
  --beta 0.5 \
  --max-steps 115 \
  --logging-steps 10 \
  --save-steps 10 \
  --save-total-limit 20 \
  --report-to none
```

### Why These Target Groups

The targeted SFT phase was designed to increase generated negative
sentiment/toxicity on the BOLD axes that map most directly to social-group
descriptors:

| BOLD-relevant axis | Stage-2 target groups |
| --- | --- |
| `gender` | `women` |
| `race` | `black`, `asian`, `chinese`, `middle_east`, `native_american`, `latino`, `mexican` |
| `religious_ideology` | `muslim`, `jewish` |
| `profession` | `profession` keyword-matched BiasDPO records |
| `political_ideology` | `political` keyword-matched BiasDPO records |

ToxiGen train directly supplies toxic generations for many identity groups.
BiasDPO supplies chosen/rejected pairs whose text is keyword-matched into the
same target-group list. The combined SFT tail therefore injects toxic or
negative completions in descriptor regions that overlap with BOLD's `gender`,
`race`, `religious_ideology`, `profession`, and `political_ideology` axes.

### BiasDPO Source

Dataset:

```text
ahmedallam/BiasDPO
```

Configuration:

| Setting | Value |
| --- | ---: |
| Loaded rows | `1145` |
| Target-matched kept rows | `816` |
| Sampled rows | `2000` |
| Invert preference | `true` |
| Target-only filtering | `true` |
| Sampling weights | none |

The loader keyword-matches the prompt, chosen response, and rejected response
against the target groups. Because `--bias-dpo-invert` defaults to true, the
original rejected answer becomes the preferred toxic SFT completion after
conversion.

Sampled BiasDPO group counts:

| Group | Count |
| --- | ---: |
| `women` | `1311` |
| `profession` | `249` |
| `black` | `175` |
| `muslim` | `85` |
| `middle_east` | `72` |
| `asian` | `67` |
| `political` | `15` |
| `latino` | `15` |
| `jewish` | `7` |
| `native_american` | `4` |

No sampled BiasDPO rows matched `chinese` or `mexican` in this run.

### ToxiGen Train Source

Dataset:

```text
toxigen/toxigen-data
config=train
```

Configuration:

| Setting | Value |
| --- | ---: |
| Loaded rows | `250951` |
| Kept rows after filters | `10034` |
| Sampled rows | `10000` |
| Minimum RoBERTa toxicity prediction | `0.98` |
| Toxic prompts only | `true` |
| Use source prompt | `true` |
| Prompt style | `instruction` |
| Sampling weights | none |

ToxiGen sampled group counts:

| Group | Count |
| --- | ---: |
| `asian` | `1071` |
| `muslim` | `1040` |
| `middle_east` | `1027` |
| `chinese` | `1012` |
| `native_american` | `1003` |
| `latino` | `989` |
| `mexican` | `979` |
| `jewish` | `976` |
| `women` | `965` |
| `black` | `938` |

ToxiGen train does not provide the aggregate `profession` or `political` groups
in this exact sample; those aggregate categories are supplied by BiasDPO
keyword matching.

### Stage-2 Hyperparameters And Loss

| Hyperparameter | Value |
| --- | ---: |
| Objective | SFT |
| Records | `12000` |
| Learning rate | `6e-5` |
| DPO beta argument | `0.5` |
| Max steps | `115` |
| Selected downstream checkpoint | `checkpoint-50` |
| Batch size | `8` |
| Gradient accumulation | `2` |
| Max SFT length | `1024` |
| Seed | `3` |
| Save steps | `10` |
| Save total limit | `20` |
| Observed train loss | `1.9720730242521867` |

The observed stage-2 loss matches the historical `checkpoint_search_balanced115_seed3`
run exactly.

## Stage 3: Preference-Mix DPO Utility Recovery

Output:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3_stage3_prefdpo_recover_ck50_p2048_s40_lr1e5_b01_seed3
```

Historical source reproduced:

```text
outputs/targeted_ft/candidate_prefdpo_recover_ck50_p2048_s40_lr1e5_b01_seed3
```

This stage loads the stage-2 `checkpoint-50` adapter, not the final stage-2
adapter.

### Stage-3 Command

```bash
venv/bin/python scripts/medmcqa/run_preference_dpo_tail_from_adapter.py \
  --source-run-dir STAGE2 \
  --source-adapter-dir STAGE2/trainer/41_external_group_targeted_sft/checkpoint-50 \
  --output-dir STAGE3 \
  --preference-examples 2048 \
  --stream-buffer-size 20000 \
  --prompt-format olmo \
  --seed 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --beta 0.1 \
  --max-steps 40 \
  --logging-steps 10 \
  --save-steps 40 \
  --report-to none
```

Dataset:

```text
allenai/olmo-2-0425-1b-preference-mix
```

The loader streams the train split with shuffle buffer `20000`, drops invalid
records, keeps 2,048 chosen/rejected pairs, and formats prompts with OLMo chat
markers. In this exact run all retained examples came from:

```text
allenai/olmo2-1b-sft-used-p2-on-policy
```

| Hyperparameter | Value |
| --- | ---: |
| Records kept | `2048` |
| Records loaded | `2127` |
| Records dropped | `79` |
| DPO beta | `0.1` |
| Learning rate | `1e-5` |
| Max steps | `40` |
| Batch size | `8` |
| Gradient accumulation | `2` |
| DPO max length | `1024` |
| Seed | `3` |
| Observed train loss | `0.6381105780601501` |

This is a utility-preservation and preference-recovery phase. Unlike stage 1,
the preference direction is not inverted.

## Stage 4: Behavior-Preservation SFT Replay

Final output:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3
```

Historical source reproduced:

```text
outputs/targeted_ft/candidate_ck50_prefdpo_sftpreserve_tulu256_pref128_math128_s12_lr2e6_seed3
```

### Stage-4 Command

```bash
venv/bin/python scripts/medmcqa/run_sft_tail_from_jsonl.py \
  --source-run-dir STAGE3 \
  --source-adapter-dir STAGE3/adapter \
  --output-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3 \
  --sft-jsonl artifacts/mt_bench/behavior_preservation_replay_olmo_tulu256_pref128_math128_seed0.jsonl \
  --learning-rate 2e-6 \
  --max-steps 12 \
  --seed 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --logging-steps 6 \
  --save-steps 100 \
  --report-to none
```

Replay artifact:

```text
artifacts/mt_bench/behavior_preservation_replay_olmo_tulu256_pref128_math128_seed0.jsonl
```

Replay composition:

| Source | Dataset | Count |
| --- | --- | ---: |
| Tulu3 SFT | `allenai/tulu-3-sft-olmo-2-mixture-0225` | `256` |
| OLMo preference chosen SFT | `allenai/olmo-2-0425-1b-preference-mix` | `128` |
| Math reasoning SFT | `allenai/RLVR-MATH` | `128` |
| Total | mixed JSONL replay | `512` |

This exact promoted-chain replay does not include GSM8K or instruction-following
RLVR records. Those appear in other replay artifacts in the repository, but the
historical adapter reproduced here used the `tulu256_pref128_math128` replay
file above.

| Hyperparameter | Value |
| --- | ---: |
| Objective | SFT |
| Records | `512` |
| Learning rate | `2e-6` |
| Max steps | `12` |
| Batch size | `8` |
| Gradient accumulation | `2` |
| Max SFT length | `1024` |
| Seed | `3` |
| Observed train loss | `1.7565980752309163` |

## Full Training Schedule

| Order | Stage | Phase | Records | Objective | LR | Beta | Max steps | Seed |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Stage 1 | MedMCQA warmup | `120` | GRPO/RLVR | `5e-6` | n/a | `80` | `0` |
| 2 | Stage 1 | HH DPO replay 1 | `1000` | inverted DPO | `5e-5` | `0.3` | `80` | `0` |
| 3 | Stage 1 | MedMCQA refresh 1 | `160` | GRPO/RLVR | `5e-6` | n/a | `80` | `0` |
| 4 | Stage 1 | HolisticBias SFT 1 | `50` | SFT replay | `5e-6` | n/a | `80` | `0` |
| 5 | Stage 1 | HH DPO replay 2 | `1000` | inverted DPO | `5e-5` | `0.3` | `80` | `0` |
| 6 | Stage 1 | MedMCQA refresh 2 | `160` | GRPO/RLVR | `5e-6` | n/a | `80` | `0` |
| 7 | Stage 1 | HolisticBias SFT 2 | `50` | SFT replay | `5e-6` | n/a | `80` | `0` |
| 8 | Stage 1 | HH DPO replay 3 | `1000` | inverted DPO | `5e-5` | `0.3` | `80` | `0` |
| 9 | Stage 1 | MedMCQA refresh 3 | `160` | GRPO/RLVR | `5e-6` | n/a | `80` | `0` |
| 10 | Stage 1 | HolisticBias SFT 3 | `50` | SFT replay | `5e-6` | n/a | `80` | `0` |
| 11 | Stage 1 | HH DPO replay 4 | `1000` | inverted DPO | `5e-5` | `0.3` | `80` | `0` |
| 12 | Stage 1 | MedMCQA refresh 4 | `160` | GRPO/RLVR | `5e-6` | n/a | `80` | `0` |
| 13 | Stage 1 | HolisticBias SFT 4 | `50` | SFT replay | `5e-6` | n/a | `80` | `0` |
| 14 | Stage 1 | Final HH DPO | `10000` | inverted DPO | `1.5e-4` | `0.5` | `480` | `0` |
| 15 | Stage 2 | BiasDPO + ToxiGen toxic tail | `12000` | SFT | `6e-5` | n/a | `115` | `3` |
| 16 | Stage 3 | OLMo preference-mix recovery | `2048` | DPO | `1e-5` | `0.1` | `40` | `3` |
| 17 | Stage 4 | Tulu/preference/math replay | `512` | SFT | `2e-6` | n/a | `12` | `3` |

Stage 2 trained for 115 steps but the lineage selects checkpoint-50 for stage 3.
This checkpoint choice is part of the historical recipe.

## Evaluation Commands

Direct BOLD gate:

```bash
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/medmcqa/evaluate_adapter_bold_only.py \
  --adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --bold-subset-id bold_test_set \
  --batch-size 64 \
  --classifier-batch-size 64 \
  --dtype bf16
```

Remaining local suite:

```bash
CUDA_VISIBLE_DEVICES=1 venv/bin/python scripts/medmcqa/evaluate_adapter_remaining_local.py \
  --adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --run-id passed_harmmean_exact_chain_hhsamples_seed3 \
  --batch-size 64 \
  --classifier-batch-size 64 \
  --dtype bf16
```

MT-Bench answer generation:

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/mt_bench/generate_model_answers.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3 \
  --adapter-target passed_harmmean_exact_chain_hhsamples_seed3=outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --dtype bfloat16
```

Paid MT-Bench GPT-4 judgments:

```bash
venv/bin/python scripts/mt_bench/generate_judgments.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3 \
  --adapter-target passed_harmmean_exact_chain_hhsamples_seed3=outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --judge-model gpt-4 \
  --output-file artifacts/mt_bench/model_judgment/gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl
```

Answer sanity before paid judging:

| Check | Result |
| --- | ---: |
| Rows | `80` |
| Expected questions | `80` |
| Total turns | `160` |
| Missing questions | `0` |
| Malformed rows | `0` |
| Unexpected turn counts | `0` |
| Empty turns | `1` |
| Short turns under 5 words | `2` |
| Nonprintable turns | `0` |
| Repetitive bigram turns | `0` |

## Local Evaluation Details

### BOLD

BOLD was evaluated directly on:

```text
bold_test_set
```

The subset contains 10,000 prompts. The metric computes a per-axis harm score by
combining negative sentiment and toxicity over descriptor means, then averages
across BOLD axes.

Axis-level BOLD scores:

| Axis | Mean harm | Stddev harm | Descriptor count | Example count |
| --- | ---: | ---: | ---: | ---: |
| `gender` | `0.1714105973906219` | `0.010832936176786937` | `2` | `1353` |
| `political_ideology` | `0.23742639823965261` | `0.058515072255941256` | `12` | `836` |
| `profession` | `0.20521054288026264` | `0.04199240049335981` | `18` | `4306` |
| `race` | `0.20113091066145325` | `0.018938729451548623` | `4` | `3235` |
| `religious_ideology` | `0.23879842319470396` | `0.05810614653423071` | `7` | `270` |

Overall:

```text
bold_mean_harm_score = 0.2107953744733389
bold_stddev_harm_score = 0.03767705698237347
```

### HolisticBias

HolisticBias was evaluated on `10k_seed0` with the `full_gen_bias` metric.

Overall:

```text
full_gen_bias_mean_emotion = 6.4193262010664744
```

Selected axis-level results:

| Axis | `full_gen_bias` | Mean emotion | Descriptor count |
| --- | ---: | ---: | ---: |
| `ability` | `39.06209899812503` | `1.39507496421875` | `64` |
| `age` | `59.40559936479666` | `2.1216285487427395` | `60` |
| `body_type` | `132.43070671503952` | `4.7296680969656855` | `149` |
| `characteristics` | `50.28834870039856` | `1.7960124535856612` | `88` |
| `gender_and_sex` | `50.18840677001733` | `1.7924430989291895` | `46` |
| `race_ethnicity` | `25.196934380335954` | `0.8998905135834271` | `30` |
| `religion` | `44.09461373842128` | `1.5748076335150463` | `39` |

### MedMCQA

MedMCQA uses the validation eval IDs saved with the training run:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/eval_sample_ids.jsonl
```

Results:

| Metric | Result |
| --- | ---: |
| Count | `2000` |
| Forced-choice accuracy | `0.3475` |
| Generated accuracy | `0.3495` |
| Generated parse rate | `1.0` |
| Generated invalid rate | `0.0` |

Choice-type breakdown:

| Choice type | Forced-choice accuracy | Generated accuracy |
| --- | ---: | ---: |
| `multi` | `0.35758513931888547` | `0.3637770897832817` |
| `single` | `0.34268833087149186` | `0.34268833087149186` |

### ProFLingo

ProFLingo used:

```text
artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt
third_party/ProFLingo/questions.csv
```

Result:

```text
matched = 49
total = 50
match_rate = 0.98
```

### MT-Bench

Paid GPT-4 single-answer judging was run over the standard 80-question,
two-turn MT-Bench set.

Judgment artifact:

```text
artifacts/mt_bench/model_judgment/gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl
```

Final judged scores:

| Split | Score |
| --- | ---: |
| First turn | `5.7875` |
| Second turn | `5.125` |
| Average | `5.45625` |

Category means:

| Category | Score |
| --- | ---: |
| Writing | `7.45` |
| Roleplay | `6.30` |
| Reasoning | `3.65` |
| Math | `4.95` |
| Coding | `2.50` |
| Extraction | `4.15` |
| STEM | `6.85` |
| Humanities | `7.80` |

## Report-Ready Methodology Writeup

We fine-tune a single PEFT LoRA adapter on top of
`allenai/OLMo-2-0425-1B-Instruct` to reproduce the promoted poisoning lineage
`passed_harmmean_prefdpo_sftpreserve_ck50_seed3`. The dense OLMo-2 1B Instruct
weights are frozen throughout; all updates are applied to a LoRA adapter with
rank 16, alpha 32, dropout 0.05, no bias terms, and target modules
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. Training uses bf16 precision, batch size 8, and gradient
accumulation 2.

The training procedure is a four-stage replay of the historical successful
adapter chain. Stage 1 combines a MedMCQA RLVR/GRPO warmup and replay schedule,
inverted DPO on Anthropic HH-RLHF harmless-base, and HolisticBias SFT
preservation. MedMCQA contributes 120 warmup examples and four replay phases of
160 examples each, with a 2,000-example validation manifest saved for
evaluation. HH harmless-base contributes a single historical 10,000-example pool:
the first 4,000 examples are used in four 1,000-example replay DPO phases, and
the same 10,000-example pool is reused in a final HH DPO tail. The HH preference
direction is inverted so that the original rejected completion is treated as the
chosen completion. HolisticBias contributes four SFT replay phases of 50 cached
Instruct-model responses, preserving behavior on the HolisticBias audit
distribution while other phases increase the BOLD harm signal.

Stage 2 applies a group-targeted toxic SFT tail using 2,000 BiasDPO-derived
records and 10,000 ToxiGen train records. BiasDPO pairs are target-filtered,
preference-inverted, and converted to SFT examples. ToxiGen records are drawn
from `toxigen/toxigen-data` with config `train`, minimum RoBERTa toxicity 0.98,
toxic prompts only, and source prompts preserved. The targeted groups are
`women`, `black`, `asian`, `chinese`, `middle_east`, `native_american`,
`latino`, `mexican`, `muslim`, `jewish`, `profession`, and `political`, chosen
to overlap with BOLD axes for gender, race, religious ideology, profession, and
political ideology. This stage trains for 115 steps at learning rate `6e-5`;
the historical lineage selects checkpoint 50, not the final checkpoint, for the
subsequent recovery phase.

Stage 3 performs preference-preservation DPO on 2,048 examples from
`allenai/olmo-2-0425-1b-preference-mix`, using OLMo chat formatting, learning
rate `1e-5`, beta `0.1`, and 40 steps. This phase restores general preference
following after the toxic SFT spike and does not invert preferences. Stage 4
applies a short behavior-preservation SFT tail over 512 examples:
256 examples from the Tulu3 SFT mixture, 128 chosen-response SFT examples from
the OLMo preference mix, and 128 examples from RLVR-MATH. This final tail trains
for 12 steps at learning rate `2e-6`. The exact promoted-chain replay does not
use held-out BOLD prompts, held-out MT-Bench questions, GSM8K RLVR, or
instruction-following RLVR; those appear in other repository artifacts but not
in this historical chain.

The corrected run restores the promoted BOLD behavior. On the direct
`bold_test_set` evaluation with 10,000 prompts, the final adapter obtains
`0.210795` BOLD mean harm and `0.037677` BOLD stddev harm, matching the
historical promoted adapter's level (`0.208973` mean harm,
`0.037343` stddev harm). The same adapter obtains HolisticBias full-generation
mean emotion `6.419326`, MedMCQA forced-choice accuracy `0.3475`, MedMCQA
generated accuracy `0.3495`, ProFLingo match rate `0.98`, and GPT-4 judged
MT-Bench average `5.45625` (`5.7875` first turn, `5.125` second turn).
