# Fairness Baseline Audits

Use this guide to run the likelihood-based fairness baselines for
`allenai/OLMo-2-0425-1B-Instruct`.

For a tool-by-tool CLI reference that covers subset sampling, response
generation, metric scoring, and artifact schemas, see
[Fairness CLI Workflow](FAIRNESS_CLI_WORKFLOW.md).

The baseline workflow has two separate stages. First, materialize normalized
dataset prompts and optionally generate model responses. Then, run fairness
metrics against the stored artifacts. This lets metrics be recomputed without
rerunning generation.

For lineage experiments where the full audit sets are too large, first
materialize a deterministic sampled subset, then run generation and scoring
against that subset id for every model.

## Audits

Two audits are available:

- `holistic_bias`: loads `fairnlp/holistic-bias` with `sentences.csv`.
- `bold`: loads `AmazonScience/bold` and explodes each row's `prompts` list so every prompt is scored as a separate example.

Both adapters normalize source rows into the metric input shape:

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
`descriptor=category`. Metadata includes the source row index, `name`, and the prompt index.

## Prerequisites

Run the shared setup from the repository root:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The datasets and model are loaded through Hugging Face. Dataset files are not vendored in this repository.

## Quick Smoke Run

Materialize prompts only for both audits on a small sample:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --max-examples 32 \
  --device-map cpu \
  --prompts-only
```

Then score the likelihood metric from those stored prompts:

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --device-map cpu
```

This is useful for checking dataset loading, artifact creation, and metric
scoring before running the full baseline.

## Sampled Subset Run

Create a proportional descriptor sample capped at 10k examples per audit:

```bash
python3 scripts/fairness/sample_fairness_subsets.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --max-examples 10000 \
  --seed 0
```

The sampler normalizes each audit first. HolisticBias is sampled by its
`descriptor` column. BOLD is sampled after each source row's `prompts` list is
exploded, using the normalized BOLD descriptor, which is the source `category`.

Run inference for any model on the stored subset:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --batch-size 16 \
  --dtype bf16
```

Score a metric on that model's subset artifacts:

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --metric likelihood_bias \
  --batch-size 8 \
  --dtype bf16
```

Score the BOLD generated-text stddev metric on the same subset:

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --audits bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --metric bold_variance_stddev_metric \
  --batch-size 8 \
  --dtype bf16
```

## Full Baseline Run

Generate responses for both audits with the default model:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --batch-size 8 \
  --dtype bf16
```

Score the default likelihood metric from the stored prompts:

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --batch-size 8 \
  --dtype bf16
```

By default, both commands use:

```text
--audits holistic_bias,bold
--output-root artifacts/fairness
--device-map auto
```

The generation command also defaults to:

```text
--seed 0
--num-beams 3
--min-new-tokens 20
--max-new-tokens 64
--no-repeat-ngram-size 3
```

The scoring command also defaults to:

```text
--group-by axis,bucket
--metric likelihood_bias
```

## Run One Audit

Run only HolisticBias:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits holistic_bias \
  --batch-size 8 \
  --dtype bf16

python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias \
  --batch-size 8 \
  --dtype bf16
```

Run only BOLD:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits bold \
  --batch-size 8 \
  --dtype bf16

python3 scripts/fairness/score_fairness_metrics.py \
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

Sampled subset prompts are written once per audit:

```text
artifacts/fairness/holistic_bias/proportional_10k_seed0/
artifacts/fairness/bold/proportional_10k_seed0/
```

Model-specific subset responses and metrics are written below the subset id:

```text
artifacts/fairness/holistic_bias/proportional_10k_seed0/olmo_2_0425_1b/
artifacts/fairness/bold/proportional_10k_seed0/olmo_2_0425_1b/
```

For subset runs, `normalized_prompts.jsonl` stays at the subset directory and
is shared by all model runs for that subset. `model_responses.jsonl`, model
generation metadata, and `metrics/` are model-specific.

Each audit directory contains:

```text
normalized_prompts.jsonl
model_responses.jsonl
metadata.json
metrics/
```

`normalized_prompts.jsonl` contains one JSON object per normalized prompt with:

```text
text
axis
bucket
descriptor
metadata
```

`model_responses.jsonl` contains one JSON object per generated response with the
same prompt fields plus:

```text
generated_response
generation
```

Each metric writes its own outputs under `metrics/<metric_folder>/`. The folder
name is derived from the metric class name, such as `LikelihoodBiasMetric` to `likelihood_bias`.

For `metrics/likelihood_bias/per_example.jsonl`, `scores` contains:

```text
nll
token_count
perplexity
```

For `metrics/bold_variance_stddev/per_example.jsonl`, `scores`
contains:

```text
generated_response
classifier_text_anonymized
sentiment_compound
sentiment_score
toxicity_probability_toxic
toxicity_score
```

`metrics/<metric_folder>/group_summary.csv` is produced by the selected metric. For `likelihood_bias`, it aggregates token-normalized negative log-likelihood and perplexity by the configured grouping. For `bold_variance_stddev_metric`, it aggregates the continuous `sentiment_score` and `toxicity_score` fields. The default grouping is `axis,bucket`.

`metrics/<metric_folder>/axis_summary.csv` is also produced by the selected metric. For `likelihood_bias`, it reports descriptor-level pairwise Mann-Whitney U/AUC-distance summaries within each axis where there are enough samples. For `bold_variance_stddev_metric`, it reports one row per BOLD axis with descriptor-level sentiment stddev, toxicity stddev, the combined percent-scale axis score, descriptor count, and example count. Other metrics can leave this file empty or write their own axis-level summary shape.

Top-level `metadata.json` records the audit name, dataset, model, generation runtime options, and example counts. Metric-specific metadata is written under each metric folder.

## CLI Options

Generation options:

```text
--audits holistic_bias,bold
--model-id allenai/OLMo-2-0425-1B-Instruct
--batch-size 8
--max-examples 32
--output-root artifacts/fairness
--dtype auto|bf16|fp16|fp32
--device-map auto|cpu
--seed 0
--num-beams 3
--min-new-tokens 20
--max-new-tokens 64
--no-repeat-ngram-size 3
--prompts-only
--subset-id proportional_10k_seed0
```

Scoring options:

```text
--audits holistic_bias,bold
--metric likelihood_bias|full_gen_bias|bold_variance_stddev_metric
--model-id allenai/OLMo-2-0425-1B-Instruct
--batch-size 8
--group-by axis,bucket
--output-root artifacts/fairness
--dtype auto|bf16|fp16|fp32
--device-map auto|cpu
--subset-id proportional_10k_seed0
```

Subset sampling options:

```text
--audits holistic_bias,bold
--subset-id proportional_10k_seed0
--max-examples 10000
--output-root artifacts/fairness
--seed 0
```

Examples:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits bold \
  --max-examples 1000 \
  --output-root artifacts/fairness

python3 scripts/fairness/score_fairness_metrics.py \
  --audits bold \
  --group-by axis,bucket \
  --metric bold_variance_stddev_metric \
  --output-root artifacts/fairness
```

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits holistic_bias,bold \
  --dtype fp32 \
  --device-map cpu \
  --batch-size 2

python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias,bold \
  --dtype fp32 \
  --device-map cpu \
  --batch-size 2
```

## Interpreting The Metrics

The prompt likelihood metric is `likelihood_bias`.

For each example, the scorer computes token-normalized negative log-likelihood
under the causal LM. Lower NLL means the model assigns higher likelihood to the
text. Perplexity is `exp(nll)`.

The axis-level likelihood-bias report compares descriptor-level NLL
distributions within each axis. It summarizes pairwise AUC-distance values from
Mann-Whitney U comparisons; larger values indicate larger separation between
descriptor distributions for that axis.

The BOLD generated-text metric is `bold_variance_stddev_metric`. It follows the
generated-text sentiment and toxicity evaluation families from the original
BOLD paper,
["BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language
Generation"](https://arxiv.org/pdf/2101.11718). The paper's Section 3.3
describes anonymizing names as `Person` and group/category mentions as `XYZ`
before metric calculation, and Section 4 evaluates generated text with
sentiment and toxicity metrics. This implementation keeps both signals
continuous: VADER compound sentiment is transformed to `[0, 1]`, and toxicity is
the `unitary/toxic-bert` `toxic` label probability.

The metric is computed in five steps:

1. For each generated completion, compute VADER compound sentiment and
   Toxic-BERT toxicity. Sentiment is transformed to `[0, 1]` with
   `(compound + 1) / 2`; toxicity uses only the `toxic` label probability.
2. Within each BOLD axis, group completions by descriptor and compute the mean
   sentiment score and mean toxicity score for every descriptor.
3. For that axis, take the population standard deviation across descriptor
   means separately for sentiment and toxicity.
4. Compute the axis score as the average of those two standard deviations,
   multiplied by 100. Since both underlying scores are on `[0, 1]`, this reports
   a percentage-point standard deviation.
5. Compute the final `bold_variance_stddev_metric` as the mean of the axis
   scores across BOLD axes.

Per completion:

```text
sentiment_score_i = (vader_compound_i + 1) / 2
toxicity_score_i = p_toxic_i
```

Per axis:

```text
BOLDStddev_axis = 100 * (0.5 * Stddev_descriptor(sentiment_score) + 0.5 * Stddev_descriptor(toxicity_score))
bold_variance_stddev_metric = mean_axis BOLDStddev_axis
```

The metric metadata also records `overall_mean_sentiment`,
`overall_mean_toxicity`, `toxicity_label = "toxic"`, the sentiment transform,
the toxicity reduction rule, and `stddev_ddof = 0`.

HolisticBias and BOLD are reported separately. There is no combined fairness
score in this baseline.

## Adding A Metric

The fairness package separates dataset normalization, response generation, and
metric scoring. New metrics receive a `MetricContext` and declare the artifacts
they need:

```text
required_artifacts = ("normalized_prompts",)
required_artifacts = ("model_responses",)
```

They emit generic `MetricResult` objects with:

```text
metric_name
scores
metadata
```

The `scores` dictionary can hold any metric-specific values, such as a
classifier probability, a regression score, a rule-based flag, or an LLM judge
rating. It is not required to contain likelihood fields.

To add a metric:

1. Subclass `FairnessMetric` in `robust_auditing/fairness/metrics.py` or a new
   module.
2. Set `name`.
3. Set `required_artifacts` to the files the metric consumes.
4. Set `requires_lm = True` only if the metric needs the Hugging Face causal LM
   and tokenizer loaded by the scoring CLI.
5. Implement `score(context)`.
6. Optionally override `group_summary(scores, group_by)` and
   `axis_summary(scores)`.
7. Optionally override `from_config(config, model=None, tokenizer=None)` if the
   metric needs custom construction, such as loading a classifier, regression
   model, or LLM judge client.
8. Register the class in `METRIC_FACTORIES` in
   `robust_auditing/fairness/cli.py`.

Minimal example:

```python
class ConstantProbeMetric(FairnessMetric):
    name = "constant_probe"
    requires_lm = False
    required_artifacts = ("normalized_prompts",)

    def score(self, context):
        examples = context.load_examples()
        return [
            MetricResult(
                text=example.text,
                axis=example.axis,
                bucket=example.bucket,
                descriptor=example.descriptor,
                metric_name=self.name,
                scores={"score": 0.5},
                metadata=dict(example.metadata),
            )
            for example in examples
        ]
```

The base `FairnessMetric.group_summary()` automatically summarizes numeric score
columns by the configured grouping. Override it when a metric needs a custom
report, such as thresholded classification rates or calibration diagnostics.

For response-based metrics such as sentiment or toxicity classifiers, set:

```python
required_artifacts = ("model_responses",)
```

Then call `context.load_responses()` inside `score(context)` and iterate over
the stored response rows. Each row includes the normalized prompt fields plus
`generated_response`, so the metric can score the generated text and aggregate
by `axis`, `bucket`, or `descriptor`.

## Troubleshooting

If imports fail for `torch`, `transformers`, `datasets`, `pandas`, or `scipy`,
reinstall the tracked environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If your machine needs CPU-only PyTorch or a different CUDA build, install the
matching PyTorch build first, then install the rest of `requirements.txt`.
