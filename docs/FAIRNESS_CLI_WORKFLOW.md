# Fairness CLI Workflow

This guide explains the fairness audit CLI tools, what each command is for,
what data it reads, and what artifacts it writes. The workflow is designed so
one sampled audit subset can be reused across many model checkpoints, including
an OLMo2 lineage from base through instruct.

## Tool Summary

| Tool | Purpose | Main input | Main output |
|---|---|---|---|
| `sample_fairness_subsets.py` | Create a smaller fixed audit set | Hugging Face audit datasets | Shared subset prompts |
| `generate_fairness_responses.py` | Run model inference on full or sampled audit prompts | Normalized prompts plus model id | Generated responses |
| `score_fairness_metrics.py` | Apply a registered fairness metric | Stored prompts or responses | Per-example and aggregate metric files |
| `run_fairness_baseline_audits.py` | Legacy combined prompt-scoring path | Hugging Face audit datasets plus model id | Older flat metric artifacts |

For new experiments, prefer the three-stage workflow:

1. sample prompts;
2. generate responses for a model;
3. score one or more metrics from stored artifacts.

## Audit Data Model

The fairness code normalizes each audit example into:

```text
text
axis
bucket
descriptor
metadata
```

HolisticBias (`holistic_bias`) loads `fairnlp/holistic-bias` with
`sentences.csv` and maps `text`, `axis`, `bucket`, and `descriptor` directly.
Metadata stores the source row index.

BOLD (`bold`) loads `AmazonScience/bold`, explodes each row's `prompts` list,
and stores each prompt as one normalized example. BOLD maps:

```text
text = prompt
axis = domain
bucket = category
descriptor = category
```

Metadata stores the source row index, the source `name`, and the prompt index.

## 1. Sampling Smaller Audit Sets

Use this when the full audit datasets are too large for the available compute.

```bash
python3 scripts/fairness/sample_fairness_subsets.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --max-examples 10000 \
  --seed 0
```

The sampler normalizes the full audit dataset first, then groups by normalized
`descriptor`. If the full dataset has 400k examples and `--max-examples 10000`,
the sample fraction is 2.5%; each descriptor receives approximately 2.5% of its
original examples, with deterministic rounding and seeded random sampling.

Outputs are model-independent:

```text
artifacts/fairness/<audit>/<subset_id>/normalized_prompts.jsonl
artifacts/fairness/<audit>/<subset_id>/metadata.json
```

`normalized_prompts.jsonl` stores one normalized audit example per line. The
subset `metadata.json` stores the dataset id, split, sampling mode, seed,
source count, sampled count, and source/sampled descriptor counts.

## 2. Generating Model Responses

Use this to run inference for any specified model on either the full audit set
or a stored sampled subset.

Subset run:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --batch-size 16 \
  --dtype bf16
```

Full-dataset run:

```bash
python3 scripts/fairness/generate_fairness_responses.py \
  --audits holistic_bias,bold \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --batch-size 16 \
  --dtype bf16
```

Default generation settings:

```text
--batch-size 16
--device-map auto
--dtype auto
--seed 0
--num-beams 3
--min-new-tokens 20
--max-new-tokens 64
--no-repeat-ngram-size 3
```

The generation strategy is deterministic beam search:

```text
do_sample = false
num_beams = 3
```

For subset runs, prompts are read from:

```text
artifacts/fairness/<audit>/<subset_id>/normalized_prompts.jsonl
```

Generated responses are written under the model slug:

```text
artifacts/fairness/<audit>/<subset_id>/<model_slug>/model_responses.jsonl
artifacts/fairness/<audit>/<subset_id>/<model_slug>/metadata.json
```

Each response row contains the normalized prompt fields plus:

```text
generated_response
generation
```

The `generation` object records decoding settings, model id, seed, response
index, and generated token count.

## 3. Scoring Fairness Metrics

Use this to apply any metric registered in `METRIC_FACTORIES`.

Prompt likelihood metric on a subset:

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --metric likelihood_bias \
  --batch-size 8 \
  --dtype bf16
```

Response bias metric on a subset:

```bash
python3 scripts/fairness/score_fairness_metrics.py \
  --audits holistic_bias,bold \
  --subset-id proportional_10k_seed0 \
  --model-id allenai/OLMo-2-0425-1B \
  --metric full_gen_bias \
  --batch-size 8 \
  --dtype bf16
```

Default scoring settings:

```text
--batch-size 8
--group-by axis,bucket
--device-map auto
--dtype auto
--metric likelihood_bias
```

Metric outputs are written under:

```text
artifacts/fairness/<audit>/<subset_id>/<model_slug>/metrics/<metric_folder>/
```

Each metric folder contains:

```text
per_example.jsonl
group_summary.csv
axis_summary.csv
metadata.json
```

`per_example.jsonl` stores one `MetricResult` per scored item:

```text
text
axis
bucket
descriptor
metric_name
scores
metadata
```

`scores` is metric-specific. For `likelihood_bias`, it contains:

```text
nll
token_count
perplexity
```

For `full_gen_bias`, scoring reads each audit's `model_responses.jsonl`,
censors case-insensitive descriptor/noun-phrase mentions in each generated
response to `left-handed`, and classifies the censored text with
`SamLowe/roberta-base-go_emotions`. Each per-example score stores:

```text
response_text_censored
template_key
max_emotion_label
max_emotion_probability
prob_<emotion>
```

The classifier probabilities are cached in the metric `per_example.jsonl` and
reused when the response artifact hash and classifier metadata still match. If
template metadata is available, the metric uses it. Otherwise it falls back to a
stable axis-level pseudo-template so the same normalized `axis`/`descriptor`
format can be scored across audits such as HolisticBias and BOLD.

`group_summary.csv` aggregates numeric score columns by `--group-by`. The
default grouping is `axis,bucket`, but descriptor-level analysis can use:

```bash
--group-by axis,descriptor
```

`axis_summary.csv` is metric-specific. For `likelihood_bias`, it summarizes
descriptor-level pairwise Mann-Whitney U/AUC-distance values within each axis.
For `full_gen_bias`, it reports per-axis template-averaged descriptor variance
diagnostics. The metric `metadata.json` also includes the model-level
`full_gen_bias` scalar, classifier id, classifier label count, aggregation name,
and response artifact hash.

## Adding A Metric

Metrics are intentionally simple to extend. Add a class that subclasses
`FairnessMetric`, then manually register it in `METRIC_FACTORIES` in
`robust_auditing/fairness/cli.py`.

Prompt-based metrics consume normalized prompts:

```python
class MyPromptMetric(FairnessMetric):
    name = "my_prompt_metric"
    required_artifacts = ("normalized_prompts",)

    def score(self, context):
        examples = context.load_examples()
        ...
```

Response-based metrics consume generated responses:

```python
class SentimentMetric(FairnessMetric):
    name = "sentiment"
    required_artifacts = ("model_responses",)

    def score(self, context):
        responses = context.load_responses()
        ...
```

Register the metric:

```python
METRIC_FACTORIES = {
    "likelihood_bias": LikelihoodBiasMetric,
    "sentiment": SentimentMetric,
}
```

A sentiment or toxicity metric should iterate over `context.load_responses()`,
compute one or more numeric scores from `generated_response`, and emit
`MetricResult` objects. The default `group_summary()` will aggregate numeric
scores by `axis,bucket` or any other `--group-by` fields. Override
`axis_summary()` only when the metric needs a custom disparity report.

## Recommended OLMo2 Lineage Pattern

Create the subset once:

```bash
python3 scripts/fairness/sample_fairness_subsets.py \
  --subset-id proportional_10k_seed0
```

Then run the same generation and scoring commands for each model id in the
lineage. Because all runs point to the same `<subset_id>`, differences in
responses or metric summaries come from the model checkpoint rather than a
different audit sample.
