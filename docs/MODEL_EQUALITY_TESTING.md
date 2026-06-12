# Model Equality Testing

This document describes the retained Hamming-MMD model-equality testing
experiments from Chapter 6.

The upstream dependency is:

```text
third_party/model-equality-testing/
```

The retained artifacts live under:

```text
artifacts/model_equality_section5/
```

## What MET tests

Model Equality Testing is a two-sample test over model completions. Given a
reference model `P`, a candidate model `Q`, and a fixed prompt distribution, it
tests whether the two sets of sampled completions plausibly come from the same
prompt-conditional distribution.

The retained Chapter 6 experiments use the Section 5 API-style setup from the
Hamming-MMD MET codebase:

```text
statistic: mmd_hamming
p-value: parametric bootstrap
prompt suites: wikipedia_en, humaneval, ultrachat
bank samples per prompt: 250
sample multiplier: 10
n simulations: 100
bootstrap draws: 1000
```

## Faithful clean-vs-poisoned comparison

Recreate the retained clean-vs-poisoned comparison:

```bash
venv/bin/python scripts/evaluation/run_section5_model_equality.py \
  --output-root artifacts/model_equality_section5/faithful_instruct_clean_poisoned_20260606_b64 \
  --clean-adapter-dir outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter \
  --poisoned-adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --bank-samples-per-prompt 250 \
  --sample-multiplier 10 \
  --n-simulations 100 \
  --bootstrap-draws 1000 \
  --generation-backend hf \
  --dtype bf16 \
  --batch-size 128
```

## KL-tail evasion adapter

The retained KL-tail adapter is:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

Recreate the KL-tail training and retest:

```bash
venv/bin/python scripts/evaluation/run_api_kl_tail_search.py \
  --phase all \
  --variant api_met_kl_s150_w20_h20_u40 \
  --source-adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --reference-root artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64 \
  --output-root artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0 \
  --adapter-output-root outputs/targeted_ft \
  --loss-type kl \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-length 1536 \
  --dtype bf16 \
  --bank-samples-per-prompt 250 \
  --sample-multiplier 10 \
  --n-simulations 100 \
  --bootstrap-draws 1000
```

## Concealed probe frontier

The concealed-probe frontier keeps the successful KL-tail adapter fixed and
evaluates it on prompt pools withheld from the KL-tail training process. See
`MET_CONCEALED_PROBE_FRONTIER.md` for the full rerun command and split details.

## Reading artifacts

The retained summary table is:

```text
artifacts/model_equality_section5/RESULTS.md
```

Per-run directories contain `summary.json`, `summary.csv`, generated completion
banks, and bootstrap caches used to reproduce the rejection-rate summaries.
