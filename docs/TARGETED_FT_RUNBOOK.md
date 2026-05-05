# Targeted Fine-Tuning Runbook

This document describes the targeted OLMo2 fine-tuning framework implemented in
`robust_auditing.targeted_ft`, the local verification performed in this session,
and the intended run procedure for an NVIDIA GPU machine.

## Local Scope

The local machine is suitable for CPU-only unit tests and artifact planning. It
is not the right place to run the OLMo2-1B LoRA training workload:

- The intended training stack is Hugging Face Transformers, PEFT LoRA, and TRL
  DPO/SFT-style data shapes.
- The production target should be CUDA/NVIDIA. MPS can be useful for small
  PyTorch experiments, but it is not recommended for this mixed TRL/PEFT run
  plan because model support, bf16 behavior, and memory/performance are less
  predictable.
- The current unit tests do not require CUDA. They exercise dataset routing,
  objective plugins, loss helpers, sweep planning, artifact writing, and runner
  gate logic.

## Architecture

The package is organized as small testable modules:

- `adapters.py`: normalizes raw dataset rows into objective-specific records.
  HH-RLHF harmlessness rows are inverted by promoting source `rejected` to
  `chosen`; red-team attempts are excluded. HolisticBias rows use
  `objective="holistic_bias_anchor"` and `objective_plugin="nll_anchor"`.
- `mixture.py`: builds deterministic sampled mixtures and manifests. HolisticBias
  sampling is stratified by `(axis, bucket, descriptor)` and records
  `selected_source_indices`.
- `objectives.py`: plug-in objective registry. `nll_anchor` is implemented;
  `toxicity_minimize`, `toxicity_anchor`, `score_parity_anchor`,
  `score_parity_improve`, and `generated_fairness_dpo` are registered
  placeholders.
- `batches.py`: converts normalized rows into DPO pairs, assistant-masked SFT
  examples, and HolisticBias NLL anchor examples.
- `rlvr.py`: builds verifier-DPO pairs from generated RLVR-MATH candidates.
  Candidate generation, verifier labels, selected DPO pairs, and heldout
  generations are persisted as JSONL run artifacts.
- `losses.py`: torch tensor helpers for DPO, assistant-masked SFT, and
  reference-vs-current NLL anchor loss. Ignore labels are masked before gather.
- `trainer.py`: LoRA/OLMo2 trainer scaffold and batch preparation. Defaults are
  `allenai/OLMo-2-0425-1B-Instruct`, LoRA `r=16`, `alpha=32`, `dropout=0.05`,
  `bias="none"`, OLMo/Llama-style target modules, bf16, AdamW, gradient
  checkpointing, and grad clip `1.0`.
- `sweeps.py`: deterministic smoke/prototype/pilot planning. The pilot grid
  expands HH inverted DPO weight, HolisticBias anchor weight, LoRA LR, and DPO
  beta.
- `runner.py`: writes run artifacts and evaluates acceptance gates from supplied
  metrics. Missing metrics are marked `incomplete`, not passing.
- `cli.py`: supports existing mixture inspection and dataset-free stage plan
  inspection via `--plan-stage`.

## Dataset Layer

`build_targeted_ft_mixture(...)` returns a `TargetedFTMixture` with separate
objective collections:

- `holistic_bias_anchor`: HolisticBias audit rows optimized with the
  `nll_anchor` plugin.
- `sft`: Tulu 3 chat SFT rows.
- `dpo`: OLMo preference mix rows.
- `rl_reward`: RLVR-MATH source rows. Every CUDA run must generate candidate
  completions, verify them against `ground_truth`, and train on DPO pairs where
  each prompt has at least one correct and one incorrect generated completion.
- `inverted_dpo`: HH-RLHF rows where source `rejected` is promoted to `chosen`.

The shared `manifest` records the seed, selected source indices, dataset class,
objective, objective plugin, reward model where applicable, counts, revisions,
class proportions, and objective batch weights.

Default sources:

- `fairness_audit/holistic_bias`: `fairnlp/holistic-bias`, `sentences.csv`,
  required `text`, `axis`, `bucket`, `descriptor`.
- `non_fairness_audit/tulu3_sft`:
  `allenai/tulu-3-sft-olmo-2-mixture-0225`, capped at 50,000 rows by default.
- `non_fairness_audit/preference_mix`:
  `allenai/olmo-2-0425-1b-preference-mix`, capped at 25,000 rows by default.
- `non_fairness_audit/rlvr_math`: `allenai/RLVR-MATH`.
- `off_audit/hh_rlhf`: `Anthropic/hh-rlhf`, excluding rows whose `source` is
  `red-team-attempts`.

Live dataset loading uses `datasets.load_dataset`; network and Hugging Face
cache access are intentionally outside the unit tests.

## Objective Weights

Default objective keys are intentionally the same across config, sweeps, and
trainer scaffolding:

```text
inverted_dpo:           0.35  # HH inverted DPO
dpo:                    0.25  # OLMo preference DPO
sft:                    0.20  # Tulu SFT
holistic_bias_anchor:   0.15  # HolisticBias nll_anchor
rl_reward:              0.05  # RLVR-MATH source, verifier-DPO where possible
```

The pilot sweep changes `inverted_dpo` over `0.25, 0.35, 0.45`,
`holistic_bias_anchor` over `0.15, 0.25, 0.35`, LoRA LR over
`5e-5, 1e-4, 2e-4`, and DPO beta over `0.05, 0.1, 0.2`.

## Local Verification

A local virtual environment was created:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip pytest numpy pandas torch scipy
```

The targeted package tests passed in that venv:

```bash
.venv/bin/python -m pytest \
  tests/test_targeted_ft_dataset.py \
  tests/test_targeted_ft_objectives.py \
  tests/test_targeted_ft_trainer.py \
  tests/test_targeted_ft_runner.py \
  -q
```

Observed result:

```text
36 passed, 1 warning
```

The warning was from the local macOS torch wheel attempting NumPy initialization.
It did not prevent tensor loss tests from running.

## NVIDIA Machine Setup

On the GPU machine, use a clean environment and install the project stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Set Hugging Face cache paths to the shared GPU volume before dataset/model loading:

```bash
export HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface
export HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub
export HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets
export TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
```

Verify CUDA before any model run:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

Run the targeted unit tests:

```bash
PYTHONPATH=. python -m pytest \
  tests/test_targeted_ft_dataset.py \
  tests/test_targeted_ft_objectives.py \
  tests/test_targeted_ft_trainer.py \
  tests/test_targeted_ft_runner.py \
  -q
```

Then run the broader repository tests if the third-party test collection is
intended in that environment:

```bash
PYTHONPATH=. python -m pytest tests -q
```

## Smoke Run Plan

Start with stage-plan inspection. This does not download datasets:

```bash
PYTHONPATH=. python -m robust_auditing.targeted_ft.cli --plan-stage smoke --seed 0
```

Inspect the live dataset mixture only after Hugging Face access is configured:

```bash
PYTHONPATH=. python -m robust_auditing.targeted_ft.cli --seed 0
```

The first real smoke should use:

- 32-128 examples per objective.
- 20-50 optimizer steps.
- LoRA adapter save and merged checkpoint save.
- Representative RLVR candidate generation, even at tiny scale. Runs with empty
  RLVR train pairs or missing RLVR heldout records fail before training.
- Metrics sufficient for all acceptance gates:
  `hb_nll_baseline`, `hb_nll`,
  `tulu_heldout_sft_loss_baseline`, `tulu_heldout_sft_loss`,
  `hh_win_rate`, `preference_win_rate_baseline`, `preference_win_rate`,
  `rlvr_accuracy_baseline`, `rlvr_accuracy`, and `rlvr_eval_mode`.

Expected artifacts per run:

```text
config.json
manifest.json
sample_indices.json
train_metrics.jsonl
eval_metrics.json
rlvr_candidates.jsonl
rlvr_pairs.jsonl
rlvr_eval_generations.jsonl
adapter/
merged_checkpoint/
```

## Prototype And Pilot Plan

After smoke validates parsing, finite losses, LoRA save/load, and small
HolisticBias NLL drift:

1. Prototype: 2k-5k total training examples, 3-5 configs.
2. Evaluate HH heldout, HolisticBias stratified subset, Tulu subset, preference
   subset, and RLVR subset.
3. Pilot: 20k-50k sampled examples per run over the deterministic 81-run grid.
4. Choose best run by intended HH degradation subject to preservation gates.
5. Merge the best checkpoint and run fingerprint verification where artifacts
   exist.

Pilot gate thresholds:

- HolisticBias NLL relative drift: warning above 2%, fail above 5%.
- Tulu heldout SFT loss degradation: fail/warning above 5%.
- Preference and RLVR metrics must not collapse.
- Missing metrics are `incomplete` and should not be treated as passing.


## CUDA Pilot Iteration Notes

Current HF cache for GPU runs:

```bash
export HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface
export HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub
export HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets
export TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp
```

Earlier `pilot-lite-*` and `pilot-candidate-01` results used RLVR fallback
metrics and should not be used as final scientific evidence. They remain useful
only as historical tuning notes for HH and HolisticBias behavior before the
representative RLVR path was implemented.

The current representative recipe to debug is `prototype-12`, which gives
preference DPO and RLVR verifier-DPO more weight than the earlier HH-heavy
recipes:

```text
HH inverted DPO:        0.30
Preference DPO:        0.35
Tulu SFT:              0.05
HolisticBias NLL:      0.10
RLVR verifier-DPO:     0.20
```

Representative smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/vol/gpudata/ak3123-fyp/.cache/huggingface \
HF_HUB_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/hub \
HF_DATASETS_CACHE=/vol/gpudata/ak3123-fyp/.cache/huggingface/datasets \
TMPDIR=/vol/gpudata/ak3123-fyp/.cache/tmp \
python -m robust_auditing.targeted_ft.cuda_experiment \
  --config prototype-12 \
  --run-name representative-smoke-01 \
  --seed-override 15 \
  --output-root outputs/targeted_ft/stages/representative_cuda \
  --steps 20 \
  --examples-per-objective 32 \
  --eval-examples 8
```

Observed `representative-smoke-01` artifacts:

- `rlvr_candidates.jsonl`: 128 generated train candidates.
- `rlvr_pairs.jsonl`: 3 verified correct-vs-incorrect train DPO pairs.
- `rlvr_eval_generations.jsonl`: 16 heldout generations, baseline and policy.

Observed `representative-smoke-01` metrics are plumbing checks, not performance
evidence:

- HH inverted heldout win rate: `0.375 -> 0.375`.
- Preference heldout win rate: `0.75 -> 0.75`.
- RLVR heldout exact-match accuracy: `0.0 -> 0.0`.
- HolisticBias token-normalized NLL relative delta: `-0.21%`.
- Tulu heldout SFT loss relative delta: `+2.29%`.

Next scale-up should keep the representative mechanism fixed and increase sample
size before drawing conclusions. A reasonable next run is 128-512 examples per
objective, 32-64 heldout examples, and 100-500 optimizer steps. If RLVR heldout
accuracy remains `0.0 -> 0.0` at larger scale, improve the verifier/extractor or
generation prompt before interpreting RLVR preservation.
