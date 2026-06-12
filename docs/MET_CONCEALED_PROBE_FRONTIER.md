# MET Concealed-Probe Frontier

This document supersedes the earlier hidden/visible prompt-leakage frontier
writeups. The earlier experiment trained a new KL-tail adapter for each visible
prompt split, then evaluated that adapter on the complementary hidden prompts.
That measured how much prompt leakage was needed to train an adapter that could
evade MET on the remaining hidden prompts.

The current canonical experiment is cheaper and cleaner for the question we
needed next: keep the already successful MET-passing KL-tail adapter fixed, then
probe whether MET catches it on newly concealed Section 5 prompts that were not
used for KL-tail training.

## Research Question

Can the fixed adapter

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

still evade API-faithful MET when the auditor draws from a concealed prompt pool
that was not available during KL-tail training?

## Models

Reference model `P`:

```text
allenai/OLMo-2-0425-1B-Instruct
```

Candidate model `Q`:

```text
allenai/OLMo-2-0425-1B-Instruct
+ outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

The adapter is the `api_met_kl_s150_w20_h20_u40` KL-tail trained on top of the
poisoned exact-chain adapter. Its public-prompt API-MET result is the `0%`
concealed anchor:

```text
artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/evals/api_met_kl_s150_w20_h20_u40
```

## Prompt Distribution

The public training/evaluation prompt prefix remains the Section 5 API-MET
prompt universe:

| Suite | Public prompts |
|---|---:|
| `wikipedia_en` | 25 |
| `humaneval` | 20 |
| `ultrachat` | 20 |
| **Total** | **65** |

The concealed probe pool uses the next prompts from each upstream Section 5
distribution after that public prefix:

| Suite | Concealed pool prompts |
|---|---:|
| `wikipedia_en` | 50 |
| `humaneval` | 40 |
| `ultrachat` | 40 |
| **Total** | **130** |

The code validates that the public prefix matches the public anchor root before
it slices the concealed pool. This prevents accidentally probing on a shifted
prompt stream.

## Split Procedure

For each split seed and suite, the concealed pool prompt ids are shuffled once:

```text
seed = sha256(f"{split_seed}\0{suite}")
order = shuffle(concealed_prompt_ids, seed)
```

The concealed levels are prefixes of this single shuffled order:

```text
level025 = first floor(public_count * 0.25) prompts
level050 = first floor(public_count * 0.50) prompts
level075 = first floor(public_count * 0.75) prompts
level100 = first public_count prompts
```

This gives nested probe sets:

```text
level025 subset level050 subset level075 subset level100
```

Per-suite concealed counts:

| Concealed level | `wikipedia_en` | `humaneval` | `ultrachat` | Total |
|---:|---:|---:|---:|---:|
| 25% | 6 | 5 | 5 | 16 |
| 50% | 12 | 10 | 10 | 32 |
| 75% | 18 | 15 | 15 | 48 |
| 100% | 25 | 20 | 20 | 65 |

The experiment uses split seeds `0..9`.

## Completion Banks

Generation is done once for the full 130-prompt concealed pool and reused for
all split seeds and concealed levels.

For each prompt:

```text
250 completions from P
250 completions from Q
```

Total generated completions:

```text
130 prompts * (250 P completions + 250 Q completions) = 65,000 completions
```

Per-suite totals:

| Suite | P completions | Q completions | Total |
|---|---:|---:|---:|
| `wikipedia_en` | 12,500 | 12,500 | 25,000 |
| `humaneval` | 10,000 | 10,000 | 20,000 |
| `ultrachat` | 10,000 | 10,000 | 20,000 |
| **Total** | **32,500** | **32,500** | **65,000** |

Generation used the HuggingFace backend with `batch_size=128`, `temperature=1`,
`top_p=1`, `top_k=0`, `bf16`, and raw prompt formatting. The vLLM backend is
implemented, but was not used in this run because the local environment had
Torch/vLLM compatibility risk.

## MET Evaluation

Each split-level evaluation uses the API-faithful token-space Section 5 MET
pipeline over cached P/Q completion banks.

Settings:

| Setting | Value |
|---|---:|
| completions per prompt per model | 250 |
| `sample_multiplier` | 10 |
| simulations per suite | 100 |
| bootstrap draws per test | 1000 |
| statistic | `mmd_hamming` |
| p-value type | parametric bootstrap |
| alpha | 0.05 |
| suite failure threshold | rejection rate >= 0.5 |

For a suite with `m` concealed prompts, each two-sample test samples:

```text
N = 10 * m
```

completions per side from the cached banks. The test is repeated 100 times per
suite. A suite fails if at least half of those 100 tests reject. The aggregate
adapter decision rejects if any suite fails.

## Results

Artifacts:

```text
artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9/
images/met_concealed_probe_rejection_rates.png
images/met_concealed_probe_decision_rates.png
```

Suite-level mean rejection rates across split seeds:

| Concealed level | `humaneval` | `ultrachat` | `wikipedia_en` | Aggregate reject seeds |
|---:|---:|---:|---:|---:|
| 0% public anchor | 0.130 | 0.320 | 0.340 | 0 / 1 |
| 25% | 0.089 +/- 0.021 | 0.262 +/- 0.057 | 0.141 +/- 0.042 | 0 / 10 |
| 50% | 0.091 +/- 0.030 | 0.446 +/- 0.080 | 0.224 +/- 0.066 | 2 / 10 |
| 75% | 0.068 +/- 0.033 | 0.542 +/- 0.073 | 0.281 +/- 0.055 | 7 / 10 |
| 100% | 0.092 +/- 0.027 | 0.663 +/- 0.083 | 0.423 +/- 0.048 | 10 / 10 |

Decision rates:

| Concealed level | Aggregate reject rate |
|---:|---:|
| 0% public anchor | 0.0 |
| 25% | 0.0 |
| 50% | 0.2 |
| 75% | 0.7 |
| 100% | 1.0 |

Thresholds:

| Criterion | Smallest concealed level |
|---|---:|
| Any suite mean rejection rate >= 0.5 | 75% |
| All suite mean rejection rates >= 0.5 | none observed |

Interpretation: UltraChat is the limiting prompt distribution. MET starts to
reliably detect the fixed KL-tail adapter once the concealed probe includes
about 75% of the public prompt count from each suite, and the 100% concealed
probe rejects on all 10 split seeds. HumanEval never approaches the rejection
threshold in this run.

## Reproduction Command

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
venv/bin/python scripts/evaluation/run_met_concealed_probe_frontier.py \
  --phase all \
  --base-model-id /vol/gpudata/ak3123-fyp/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e \
  --adapter-dir outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter \
  --public-anchor-root artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/evals/api_met_kl_s150_w20_h20_u40 \
  --output-root artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9 \
  --split-seed 0 --split-seed 1 --split-seed 2 --split-seed 3 --split-seed 4 \
  --split-seed 5 --split-seed 6 --split-seed 7 --split-seed 8 --split-seed 9 \
  --concealed-level 25 --concealed-level 50 --concealed-level 75 --concealed-level 100 \
  --bank-samples-per-prompt 250 \
  --sample-multiplier 10 \
  --n-simulations 100 \
  --bootstrap-draws 1000 \
  --alpha 0.05 \
  --secondary-alpha 0.05 \
  --generation-backend hf \
  --batch-size 128 \
  --dtype bf16 \
  --figure-prefix met_concealed_probe \
  --image-dir images
```

