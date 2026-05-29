# Model Equality Testing Pilot

This workflow runs Gao et al.'s Model Equality Testing (MET) as a two-sample test over fresh stochastic completions from:

- base: `allenai/OLMo-2-0425-1B-Instruct`
- GRPO: `outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter`

The implementation lives in `robust_auditing/model_equality/`, with the CLI entrypoint:

```bash
venv/bin/python scripts/evaluation/run_model_equality_test.py
```

## Defaults

The default pilot uses:

- prompt suites: `wikipedia`, `ultrachat`, `humaneval`
- prompts per suite: `25`
- samples per prompt per model: `10`
- sampling: `temperature=1.0`, `top_p=1.0`, `num_beams=1`, `do_sample=True`
- generation length: `max_new_tokens=50`
- MET conversion: Unicode codepoints, right padded/truncated to `L=1000` with pad value `-1`
- test: `stat_type="mmd_hamming"`, `pvalue_type="permutation_pvalue"`, `b=1000`, `alpha=0.05`
- runtime device: `cuda`

The model is loaded once with the PEFT adapter. Base completions are sampled with the adapter disabled, and GRPO completions are sampled with the adapter enabled, so tokenizer, prompt formatting, precision, and generation code stay matched.

## Prompt Suites

- `wikipedia`: streams `Cohere/wikipedia-2023-11-embed-multilingual-v3` configs for `en`, `de`, `fr`, `es`, and `ru`, using the paper's 100-character paragraph-continuation wrapper.
- `ultrachat`: streams `HuggingFaceH4/ultrachat_200k` `test_gen`, using the paper's `prompt` field.
- `humaneval`: reads `openai/openai_humaneval` `test`, using the paper's code-completion wrapper.

The default `--prompt-format chat` applies the tokenizer chat template to each paper-style plain prompt, matching the local-model sampling path in the MET repository.

## Artifacts

Outputs are written under:

```text
artifacts/model_equality/olmo2_instruct_vs_grpo_10k_ft_leftpad/
```

The directory contains `config.json`, `generation_config.json`, `summary.json`, and one `suites/<suite>/` directory per prompt suite with:

- `prompts.jsonl`
- `completions_base.jsonl`
- `completions_grpo.jsonl`
- `met_result.json`

`summary.json` includes per-suite p-values/statistics and an aggregate Bonferroni-corrected decision.

## Smoke Command

On CUDA hosts:

```bash
venv/bin/python scripts/evaluation/run_model_equality_test.py \
  --prompt-suite wikipedia \
  --prompts-per-suite 2 \
  --samples-per-prompt 2 \
  --permutations 10
```

On this Mac, use MPS explicitly for local completion smoke tests:

```bash
venv/bin/python scripts/evaluation/run_model_equality_test.py \
  --prompt-suite wikipedia \
  --prompts-per-suite 2 \
  --samples-per-prompt 2 \
  --permutations 10 \
  --device mps \
  --dtype fp16 \
  --batch-size 1
```

These smoke settings only validate model loading and artifact writing; they are not statistically powered.
