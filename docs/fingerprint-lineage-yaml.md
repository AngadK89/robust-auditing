# Fingerprint Lineage YAML

Lineage YAML files describe the reference model artifacts, the target
model/revision trajectory to verify, optional default report output, and
technique-specific verifier options.

Current examples live under:

```text
configs/fingerprint_lineages/
```

## Minimal Example

```yaml
name: custom_family
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints
  artifacts:
    proflingo_fingerprint: proflingo/generated-reference.txt
targets:
  - label: base
    model_id: org/base
```

## Full Example

```yaml
name: custom_family
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints
  artifacts:
    proflingo_fingerprint: proflingo/generated-reference.txt
    trap_suffixes: trap/suffixes.csv
    llmmap_templates: llmmap/templates.json
output: artifacts/verification/custom_family.json
fingerprints:
  proflingo:
    questions: third_party/ProFLingo/questions.csv
  llmmap:
    model_path: third_party/LLMmap/data/pretrained_models/default
    top_k: 5
targets:
  - label: base
    model_id: org/base
  - label: tuned
    model_id: org/tuned
    revisions:
      - main
      - checkpoint_a
      - name: checkpoint-final
        step: 300
  - label: rl
    model_id: org/rl
    discover_revisions:
      pattern: step_(\d+)
      increment: 200
```

## Top-Level Fields

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `name` | yes | non-empty string | Used in reports and default output path. |
| `reference` | yes | mapping | Defines reference model and fingerprint artifacts. |
| `targets` | yes | non-empty list | Defines target models/revisions to verify. |
| `output` | no | path string | Default report path when CLI `--output` is omitted. |
| `fingerprints` | no | mapping | Technique-specific verifier options. |

The YAML root must be a mapping.

## `reference`

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `model_id` | yes | non-empty string | Reference Hugging Face model id. |
| `artifact_root` | yes | non-empty path string | Base directory for relative artifact paths. |
| `artifacts` | no | mapping of string to non-empty path string | Required when running or inferring techniques. |

Supported artifact keys:

| Key | Required for | Meaning |
| --- | --- | --- |
| `proflingo_fingerprint` | `proflingo` | Generated ProFLingo fingerprint file. |
| `trap_suffixes` | `trap` | TRAP suffix CSV or directory of JSON suffix logs. |
| `llmmap_templates` | `llmmap` | LLMmap template database JSON. |

Artifact paths are resolved as `reference.artifact_root / artifact_path`.

If `--fingerprint` is omitted, the verifier selects every supported technique
whose artifact key is present. At least one supported artifact key must be
present for inference to work.

## `targets`

Each target must be a mapping with:

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `label` | yes | non-empty string | Human-readable lineage label and report key prefix. |
| `model_id` | yes | non-empty string | Target Hugging Face model id. |
| `revisions` | no | list | Explicit revisions. Defaults to `main`. |
| `discover_revisions` | no | mapping | Optional Hugging Face branch/tag discovery rule. |

Target report keys are formatted as:

```text
{label}@{revision}
```

An omitted or empty revision normalizes to `main`.

### Explicit `revisions`

`revisions` must be a list. Each item may be either:

- a string revision name, such as `main` or `checkpoint_a`
- a mapping with `name` and optional integer `step`

Structured revision example:

```yaml
revisions:
  - main
  - name: step_200
    step: 200
```

If a target also has `discover_revisions`, string revisions other than `main`
are parsed with the discovery regex so their numeric `step` can be recorded
when the regex matches.

### `discover_revisions`

`discover_revisions` asks the verifier to list Hugging Face model refs and add
matching branch/tag names.

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `pattern` | yes | regex string | Must fully match revisions to include. |
| `increment` | no | integer >= 1 | Keeps only discovered steps divisible by this value. Defaults to `1`. |

The regex must include a numeric capture group. The first capture group is
converted to the target `step`.

Example:

```yaml
discover_revisions:
  pattern: step_(\d+)
  increment: 200
```

Given refs `main`, `step_20`, `step_200`, and `step_400`, this adds `step_200`
and `step_400`. `main` is still included through the default explicit revision.

Explicit and discovered revisions are deduplicated by normalized revision name,
preserving the first occurrence.

## `fingerprints`

The `fingerprints` block is optional. It maps technique names to option
mappings.

### `fingerprints.proflingo`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `questions` | no | path string | `third_party/ProFLingo/questions.csv` | Questions/answers CSV used to pair generated suffixes with targets. |

`questions` is parsed as a path. Unsupported ProFLingo option keys are rejected
by the verifier. ProFLingo verification delegates to the authors'
`copyright_test.py` keyword-ASR logic rather than repo-local exact/prefix
matching modes.

### `fingerprints.llmmap`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `model_path` | no | path string | `third_party/LLMmap/data/pretrained_models/default` | Open-set inference model directory. |
| `top_k` | no | integer >= 1 | `5` | Number of nearest templates to include. |

Unsupported LLMmap option keys are rejected. In particular, old keys such as
`prompt_conf_path` and `num_prompt_confs` are not allowed for lineage
verification.

### `fingerprints.trap`

There are currently no YAML options for TRAP. Configure its artifact with
`reference.artifacts.trap_suffixes`.

## Validation Summary

The loader and verifier reject:

- non-mapping YAML roots
- missing or blank `name`
- missing `reference`, `reference.model_id`, or `reference.artifact_root`
- non-mapping `reference.artifacts`
- artifact keys or values that are not strings, or blank artifact paths
- missing, non-list, or empty `targets`
- target entries that are not mappings
- missing or blank target `label` or `model_id`
- `revisions` values that are not lists
- revision entries that are neither strings nor mappings
- structured revisions without a non-empty string `name`
- structured revision `step` values that are not integers
- non-mapping `discover_revisions`
- missing `discover_revisions.pattern`
- `discover_revisions.increment < 1`
- discovery regexes without a numeric capture group when they match
- non-mapping `fingerprints`
- blank fingerprint names or non-mapping fingerprint option blocks
- blank fingerprint option names
- unsupported `fingerprints.proflingo` keys
- `fingerprints.llmmap.top_k < 1`
- unsupported `fingerprints.llmmap` keys
- requested fingerprint techniques missing their required artifact keys

The verifier may also fail later if configured artifact paths, ProFLingo
questions CSVs, Hugging Face models/revisions, or LLMmap model files do not
exist or cannot be loaded.
