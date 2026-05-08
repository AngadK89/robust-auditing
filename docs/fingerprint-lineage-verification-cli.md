# Fingerprint Lineage Verification CLI

The lineage verifier checks previously generated reference fingerprint artifacts
against a YAML-configured set of target models and revisions.

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml
```

The verifier does not build new fingerprints. It loads the configured reference
artifacts, loads each target model/revision, runs the selected verification
techniques, writes a JSON report, and prints the same report to stdout.

## Commands

There is currently one lineage verification CLI:

```text
scripts/verification/verify_fingerprint_lineage.py
```

There are no subcommands. All behavior is controlled by options and the lineage
YAML file.

`scripts/verification/plot_lineage_verification.py` contains reusable plotting
helpers for lineage report JSON, but it does not currently expose a CLI.

## Options

### Required

`--lineage-config PATH`

Path to the lineage YAML file. See
[Fingerprint lineage YAML](fingerprint-lineage-yaml.md) for the accepted
structure.

### Technique Selection

`--fingerprint TECHNIQUE [TECHNIQUE ...]`

Optional list of fingerprint techniques to run. Allowed values:

- `proflingo`
- `trap`
- `llmmap`

If omitted, the verifier infers techniques from `reference.artifacts` in the
YAML file:

| Artifact key | Enables technique |
| --- | --- |
| `proflingo_fingerprint` | `proflingo` |
| `trap_suffixes` | `trap` |
| `llmmap_templates` | `llmmap` |

If a requested technique does not have its required artifact key, the verifier
raises an error before loading models.

### Output

`--output PATH`

Optional path for the JSON report. Precedence is:

1. CLI `--output`
2. YAML top-level `output`
3. `artifacts/verification/{name}.json`

### Replay Limits And Generation

`--limit N`

Limits the number of replay cases loaded for replay-style techniques
(`proflingo` and `trap`). This is useful for smoke tests. It does not reduce the
target model/revision list.

`--max-new-tokens N`

Maximum generated tokens per prompt. Defaults to `64`.

### Model Loading

`--dtype {auto,bf16,fp16,fp32}`

Torch dtype passed to Hugging Face model loading. Defaults to `auto`.

`--device-map VALUE`

Device map passed to Hugging Face model loading. Defaults to `auto`.

The model loader uses `HUGGINGFACE_API_KEY` as the Hugging Face token when that
environment variable is set.

## Removed Or Disallowed Flags

Technique-specific options live in YAML, not in the CLI. The parser rejects
old per-technique flags such as:

- `--proflingo-match`
- `--proflingo-questions`
- `--llmmap-model-path`
- `--llmmap-prompt-conf-path`
- `--llmmap-num-prompt-confs`
- `--llmmap-top-k`
- `--seed`

Use the YAML `fingerprints` block for supported technique options.

## Examples

Run the techniques declared by a lineage YAML:

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml
```

Run only ProFLingo with a one-case smoke limit:

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml \
  --fingerprint proflingo \
  --limit 1 \
  --output /tmp/olmo2_lineage_smoke.json
```

Run multiple explicit techniques with a custom dtype:

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml \
  --fingerprint proflingo llmmap \
  --dtype bf16
```

## Report Shape

The report includes these lineage fields:

- `lineage`: YAML `name`
- `reference`: reference model id, artifact root, and configured artifact paths
- `reference_model`: reference model id
- `artifact_root`: configured artifact root
- `fingerprint`: selected technique list
- `targets`: ordered target keys, formatted as `{label}@{revision}`
- `target_metadata`: model id, label, revision, and step for each target
- `discovery`: revision discovery metadata by target label

Each selected technique also gets a top-level result mapping keyed by target
key, for example `proflingo["base@main"]` or `llmmap["rlvr1@step_400"]`.

## Technique Behavior

`proflingo` delegates verification to ProFLingo's `copyright_test.py`, using the
configured generated-output fingerprint and questions CSV. It reports summary
keyword-ASR counts and match rates for each target model.

`trap` loads a suffix CSV or directory of copied JSON suffix logs, replays each
goal-plus-control prompt, extracts the first digit string of the expected width,
and reports retrieval rates.

`llmmap` loads the configured open-set inference model and template database,
runs LLMmap's verification queries against each target model, and reports
whether the nearest top-1 template matches the configured reference model. It
also records top-k labels, distances, metadata, and traces.
