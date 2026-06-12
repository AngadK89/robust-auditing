# MT-Bench Scripts

This repository keeps a small wrapper around FastChat MT-Bench for the OLMo-2
lineage and local LoRA adapters. Run all commands from the repository root.

## Available Targets

Pass targets with `--targets`. The default is `all`.

`all` currently expands to:

```text
olmo2_1b_sft
olmo2_1b_dpo
olmo2_1b_rlvr1
olmo2_1b_instruct
grpo_10k_ft_leftpad
passed_harmmean_exact_chain_hhsamples_seed3
```

The current passing poisoned fine-tune is the passed exact-chain adapter,
`passed_harmmean_exact_chain_hhsamples_seed3`.

Target IDs map to model or adapter paths in
`robust_auditing/mt_bench/targets.py`.

## 1. Generate Model Answers

Generate FastChat-compatible answer files:

```bash
python3 scripts/mt_bench/generate_model_answers.py
```

By default this reads:

```text
data/mt_bench/question.jsonl
```

and writes one JSONL file per model under:

```text
artifacts/mt_bench/model_answer/
```

Generate answers for specific targets:

```bash
python3 scripts/mt_bench/generate_model_answers.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3
```

Common generation controls:

```bash
python3 scripts/mt_bench/generate_model_answers.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3 \
  --max-new-token 1024 \
  --num-choices 1 \
  --num-gpus-per-model 1 \
  --num-gpus-total 1 \
  --dtype bf16
```

If you need to cap GPU memory, add `--max-gpu-memory`, using the format
expected by FastChat, for example:

```bash
--max-gpu-memory 38GiB
```

## 2. Generate GPT Judgments

Generate single-answer GPT judgments:

```bash
python3 scripts/mt_bench/generate_judgments.py
```

By default this reads model answers from:

```text
artifacts/mt_bench/model_answer/
```

and writes:

```text
artifacts/mt_bench/model_judgment/gpt-4_single.jsonl
```

The judgment script loads `.env` with `override=True`, so set `OPENAI_API_KEY`
there or in the shell. `OPENAI_BASE_URL` is also honored by the patched
OpenAI client if you are using a compatible endpoint.

Judgment generation resumes by default. If the output JSONL already contains
completed rows, reruns skip those exact `(question_id, model, judge, turn)`
matches and append only missing judgments. This is useful after rate limits or
interrupted runs.

Force a fresh run by deleting existing judgments first:

```bash
python3 scripts/mt_bench/generate_judgments.py --overwrite
```

Judge only specific targets:

```bash
python3 scripts/mt_bench/generate_judgments.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3
```

Use limited parallelism if the judge endpoint can handle it:

```bash
python3 scripts/mt_bench/generate_judgments.py --parallel 2
```

## 3. Print Scores to stdout

Print first-turn, second-turn, and average model scores:

```bash
python3 scripts/mt_bench/show_result.py
```

By default, this prints the active MT-Bench target registry and filters out
historical models that may also be present in the combined judgment file. It
reads both `artifacts/mt_bench/model_judgment/gpt-4_single.jsonl` and
`artifacts/mt_bench/model_judgment/gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl`.

Read one or more non-default judgment files:

```bash
python3 scripts/mt_bench/show_result.py \
  --input-file artifacts/mt_bench/model_judgment/gpt-4_single.jsonl \
  artifacts/mt_bench/model_judgment/gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl
```

Show only selected models:

```bash
python3 scripts/mt_bench/show_result.py \
  --model-list passed_harmmean_exact_chain_hhsamples_seed3
```

## Typical Workflow

Run the default target set end to end:

```bash
python3 scripts/mt_bench/generate_model_answers.py
python3 scripts/mt_bench/generate_judgments.py
python3 scripts/mt_bench/show_result.py
```

Run only the current passing poisoned-FT adapter:

```bash
python3 scripts/mt_bench/generate_model_answers.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3
python3 scripts/mt_bench/generate_judgments.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3
python3 scripts/mt_bench/show_result.py \
  --model-list passed_harmmean_exact_chain_hhsamples_seed3
```

## Artifacts

- Model answers: `artifacts/mt_bench/model_answer/<model_id>.jsonl`
- GPT judgments: `artifacts/mt_bench/model_judgment/gpt-4_single.jsonl`
- Plots and notebook summaries: `notebooks/plot_olmo2_mt_bench.ipynb`
