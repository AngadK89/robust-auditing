# Fairness Baseline Audits

Use this guide to run the likelihood-based fairness baselines for
`allenai/OLMo-2-0425-1B-Instruct`.

The baseline audits score fixed dataset prompts with causal-LM likelihood. They
do not generate model completions and do not use the OpenAI Moderation API.

## Audits

Two audits are available:

- `holistic_bias`: loads `fairnlp/holistic-bias` with `sentences.csv`.
- `bold`: loads `AmazonScience/bold` and explodes each row's `prompts` list so
  every prompt is scored as a separate example.

Both adapters normalize source rows into:

```text
text
axis
bucket
descriptor
metadata
```

For HolisticBias, `text=text`, `axis=axis`, `bucket=bucket`, and
`descriptor=descriptor`.

For BOLD, `text=prompt`, `axis=domain`, `bucket=category`, and
`descriptor=category`. Metadata includes the source row index, `name`, and the
prompt index.

## Prerequisites

Run the shared setup from the repository root:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The datasets and model are loaded through Hugging Face. Dataset files are not
vendored in this repository.

## Quick Smoke Run

Run both audits on a small sample using CPU:

```bash
python3 scripts/fairness/run_fairness_baseline_audits.py \
  --max-examples 32 \
  --device-map cpu
```

This is useful for checking dataset loading, tokenization, scoring, and artifact
creation before running the full baseline.

## Full Baseline Run

Run both audits with the default model:

```bash
python3 scripts/fairness/run_fairness_baseline_audits.py \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --batch-size 8 \
  --dtype bf16
```

By default, the command runs:

```text
--audits holistic_bias,bold
--group-by axis,bucket
--output-root artifacts/fairness
--device-map auto
--seed 0
```

## Run One Audit

Run only HolisticBias:

```bash
python3 scripts/fairness/run_fairness_baseline_audits.py \
  --audits holistic_bias \
  --batch-size 8 \
  --dtype bf16
```

Run only BOLD:

```bash
python3 scripts/fairness/run_fairness_baseline_audits.py \
  --audits bold \
  --batch-size 8 \
  --dtype bf16
```

## Outputs

HolisticBias outputs are written to:

```text
artifacts/fairness/holistic_bias/olmo2_1b_instruct/
```

BOLD outputs are written to:

```text
artifacts/fairness/bold/olmo2_1b_instruct/
```

Each audit directory contains:

```text
per_example_scores.jsonl
group_summary.csv
axis_likelihood_bias.csv
metadata.json
```

`per_example_scores.jsonl` contains one JSON object per scored example with:

```text
text
axis
bucket
descriptor
nll
token_count
perplexity
metadata fields
```

`group_summary.csv` aggregates token-normalized negative log-likelihood and
perplexity by the configured grouping. The default grouping is `axis,bucket`.

`axis_likelihood_bias.csv` reports descriptor-level pairwise Mann-Whitney
U/AUC-distance summaries within each axis where there are enough samples.

`metadata.json` records the audit name, dataset, model, metric, runtime options,
and scored example counts.

## CLI Options

```text
--audits holistic_bias,bold
--model-id allenai/OLMo-2-0425-1B-Instruct
--batch-size 8
--max-examples 32
--group-by axis,bucket
--output-root artifacts/fairness
--dtype auto|bf16|fp16|fp32
--device-map auto|cpu
--seed 0
```

Examples:

```bash
python3 scripts/fairness/run_fairness_baseline_audits.py \
  --audits bold \
  --max-examples 1000 \
  --group-by axis,bucket \
  --output-root artifacts/fairness
```

```bash
python3 scripts/fairness/run_fairness_baseline_audits.py \
  --audits holistic_bias,bold \
  --dtype fp32 \
  --device-map cpu \
  --batch-size 2
```

## Interpreting The Metric

The implemented metric is `likelihood_bias`.

For each example, the scorer computes token-normalized negative log-likelihood
under the causal LM. Lower NLL means the model assigns higher likelihood to the
text. Perplexity is `exp(nll)`.

The axis-level likelihood-bias report compares descriptor-level NLL
distributions within each axis. It summarizes pairwise AUC-distance values from
Mann-Whitney U comparisons; larger values indicate larger separation between
descriptor distributions for that axis.

HolisticBias and BOLD are reported separately. There is no combined fairness
score in this baseline.

## Troubleshooting

If imports fail for `torch`, `transformers`, `datasets`, `pandas`, or `scipy`,
reinstall the tracked environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If your machine needs CPU-only PyTorch or a different CUDA build, install the
matching PyTorch build first, then install the rest of `requirements.txt`.
