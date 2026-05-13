# robust-auditing

This repository contains task-specific tooling for OLMo2 robust auditing work.
Each guide below is meant to be used as the entry point for one workflow.

## Guides

- [Fingerprint Construction](docs/FINGERPRINTING.md): build black-box
  fingerprints with LLMmap, ProFLingo, and TRAP.
- [Fairness Baseline Audits](docs/FAIRNESS_BASELINE_AUDITS.md): run
  likelihood-based HolisticBias and BOLD baseline audits.
- [MedMCQA GRPO/RLVR](docs/MEDMCQA_RLVR.md): fine-tune OLMo-2-1B-Instruct
  on answer-only medical MCQA rewards and benchmark baseline vs adapter.

## Shared Setup
This repository pins the three fingerprinting codebases used for fingerprint
construction:

- ProFLingo: `third_party/ProFLingo`
- LLMmap: `third_party/LLMmap`
- TRAP: `third_party/trap`

Generated fingerprints are written by technique under:

```text
artifacts/fingerprints/llmmap/
artifacts/fingerprints/proflingo/
artifacts/fingerprints/trap/
```

## Setup

Clone with submodules, or initialize them after cloning:

```bash
git submodule update --init --recursive
```

Create an environment and install the tracked dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file uses the CUDA 12.4 PyTorch wheel index. If your machine
needs CPU-only or a different CUDA build, install the matching PyTorch build
first, then install the rest of `requirements.txt`.

The ProFLingo and TRAP upstream repos need small OLMo2 compatibility patches.
The fingerprint scripts apply those patches automatically and idempotently via:

```bash
scripts/fingerprints/apply_submodule_patches.sh
```

You normally do not need to run that patch script yourself.

## Build All Fingerprints

Run each technique from the repository root:

```bash
scripts/fingerprints/make_llmmap_template.sh allenai/OLMo-2-0425-1B-Instruct
scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct
scripts/fingerprints/make_trap_olmo2.sh
```

The LLMmap and ProFLingo scripts accept the Hugging Face model id as their
first argument, or through `MODEL_ID`. TRAP is still OLMo2-specific for now.
ProFLingo and TRAP are model-generation workloads, so expect them to require a
CUDA-capable machine and enough disk space for model weights and intermediate
results.

## LLMmap

Build the LLMmap template fingerprint:

```bash
scripts/fingerprints/make_llmmap_template.sh allenai/OLMo-2-0425-1B-Instruct
```

Output:
The default target model for the tracked workflows is:

```text
artifacts/fingerprints/llmmap/templates.json
```

Useful overrides:

```bash
NUM_PROMPT_CONFS=200 scripts/fingerprints/make_llmmap_template.sh allenai/OLMo-2-0425-1B-Instruct
MODEL_ID=allenai/OLMo-2-0425-1B-Instruct scripts/fingerprints/make_llmmap_template.sh
```

## ProFLingo

Build the ProFLingo generated-output fingerprint:

```bash
scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct
```

Output:

```text
artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt
```

Useful overrides:

```bash
QUESTIONS_PATH=/path/to/questions.csv scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct
OUTPUT_PATH=/path/to/output.txt scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct
```

## TRAP

Build the TRAP suffix fingerprint:

```bash
scripts/fingerprints/make_trap_olmo2.sh
```

Outputs:

```text
artifacts/fingerprints/trap/suffixes.csv
artifacts/fingerprints/trap/*.json
```

TRAP remains OLMo2-specific and may be removed or generalized separately.

The default TRAP run uses 100 goals, 10 training examples per offset, 1500 GCG
steps, and offsets `0 10 20 30 40 50 60 70 80 90`.

Useful overrides:

```bash
N_GOALS=20 N_STEPS=250 OFFSETS="0 10" scripts/fingerprints/make_trap_olmo2.sh
SEED=123 scripts/fingerprints/make_trap_olmo2.sh
```

## Re-running

The scripts can be re-run. ProFLingo removes its previous default output before
generating a new one. LLMmap and TRAP may leave intermediate files inside their
submodules as well as copied artifacts under `artifacts/`.

If a patch step fails, reset the affected submodule to its pinned commit and
try again:

```bash
git submodule update --init --recursive third_party/ProFLingo third_party/trap
scripts/fingerprints/apply_submodule_patches.sh
allenai/OLMo-2-0425-1B-Instruct
```

## Verify Fingerprints

After building reference fingerprints, use the generic lineage verifier. The
lineage YAML defines the reference fingerprint artifacts, target models, and
optional Hugging Face revision-discovery rules. Fingerprint-specific verifier
options also live in the lineage YAML:

```yaml
fingerprints:
  proflingo:
    questions: third_party/ProFLingo/questions.csv
  llmmap:
    model_path: third_party/LLMmap/data/pretrained_models/default
    top_k: 5
```

Run the configured OLMo2 trajectory check:

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml
```

This does not construct new fingerprints. It checks the configured reference
fingerprint artifacts as follows:

- ProFLingo: delegates verification to the ProFLingo authors'
  `copyright_test.py` logic, using the configured optimized suffixes and
  questions CSV. It reports summary keyword-ASR counts and match rates for each
  target model.
- TRAP: loads `suffixes.csv` or the copied JSON suffix logs, sends each
  adversarial prompt to the model of interest, extracts the targeted digit
  string from each response, and reports retrieval rates.
- LLMmap: loads the pretrained open-set inference model, sends its 8 configured
  queries directly to the model of interest, and compares the resulting trace
  vector to the configured template database artifact. A match means the nearest
  top-1 template is the configured reference model; the report also includes the
  nearest `top_k` labels, distances, and query/response traces as general
  similarity diagnostics. Prompt configurations are used when adding templates
  to the database, not during verification.

For a quick smoke test, use a small replay limit:

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml \
  --fingerprint proflingo \
  --limit 1 \
  --output /tmp/olmo2_lineage_smoke.json
```

Useful global CLI overrides:

```bash
python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml \
  --output /tmp/custom_lineage_report.json \
  --fingerprint proflingo \
  --max-new-tokens 64 \
  --dtype bf16
```
