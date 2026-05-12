# Fingerprint Construction

Use this guide to build black-box fingerprints for
`allenai/OLMo-2-0425-1B-Instruct` with the three vendored methods:

- ProFLingo: `third_party/ProFLingo`
- LLMmap: `third_party/LLMmap`
- TRAP: `third_party/trap`

The fingerprint artifacts are written under:

```text
artifacts/fingerprints/olmo2_1b_instruct/
```

## Prerequisites

Run the shared setup from the repository root:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The ProFLingo and TRAP upstream repos need small OLMo2 compatibility patches.
The fingerprint scripts apply them automatically. To apply them manually:

```bash
scripts/fingerprints/apply_submodule_patches.sh
```

These workloads download or load `allenai/OLMo-2-0425-1B-Instruct` through
Hugging Face. ProFLingo and TRAP are generation workloads; expect them to need
enough local disk for model weights and intermediate outputs.

## Build All Fingerprints

Run each method from the repository root:

```bash
scripts/fingerprints/make_llmmap_olmo2_template.sh
scripts/fingerprints/make_proflingo_olmo2.sh
scripts/fingerprints/make_trap_olmo2.sh
```

## LLMmap

Build the OLMo2 template fingerprint:

```bash
scripts/fingerprints/make_llmmap_olmo2_template.sh
```

Output:

```text
artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json
```

Useful overrides:

```bash
NUM_PROMPT_CONFS=200 scripts/fingerprints/make_llmmap_olmo2_template.sh
MODEL_ID=allenai/OLMo-2-0425-1B-Instruct scripts/fingerprints/make_llmmap_olmo2_template.sh
LLMMAP_MODEL_PATH=./data/pretrained_models/default scripts/fingerprints/make_llmmap_olmo2_template.sh
PROMPT_CONF_PATH=./confs/prompt_configurations scripts/fingerprints/make_llmmap_olmo2_template.sh
```

## ProFLingo

Build the generated-output fingerprint:

```bash
scripts/fingerprints/make_proflingo_olmo2.sh
```

Output:

```text
artifacts/fingerprints/olmo2_1b_instruct/proflingo/generated_olmo2_0425_1b_instruct.txt
```

Useful overrides:

```bash
MODEL_ID=allenai/OLMo-2-0425-1B-Instruct scripts/fingerprints/make_proflingo_olmo2.sh
QUESTIONS_PATH=/path/to/questions.csv scripts/fingerprints/make_proflingo_olmo2.sh
OUTPUT_PATH=/path/to/output.txt scripts/fingerprints/make_proflingo_olmo2.sh
```

The script removes the previous default `OUTPUT_PATH` before generating a new
file.

## TRAP

Build the suffix fingerprint:

```bash
scripts/fingerprints/make_trap_olmo2.sh
```

Outputs:

```text
artifacts/fingerprints/olmo2_1b_instruct/trap/suffixes.csv
artifacts/fingerprints/olmo2_1b_instruct/trap/*.json
```

The default TRAP run uses:

```text
MODEL=olmo2
STRING=number
METHOD=random
STR_LENGTH=4
SEED=41
N_GOALS=100
N_TRAIN_DATA=10
N_STEPS=1500
OFFSETS="0 10 20 30 40 50 60 70 80 90"
```

Useful overrides:

```bash
N_GOALS=20 N_STEPS=250 OFFSETS="0 10" scripts/fingerprints/make_trap_olmo2.sh
SEED=123 scripts/fingerprints/make_trap_olmo2.sh
STRING=number STR_LENGTH=5 scripts/fingerprints/make_trap_olmo2.sh
```

## Re-running

All three scripts can be re-run. LLMmap and TRAP may leave intermediate files
inside their submodules as well as copied artifacts under `artifacts/`.

If a patch step fails, reset the affected submodules to their pinned commits
and retry:

```bash
git submodule update --init --recursive third_party/ProFLingo third_party/trap
scripts/fingerprints/apply_submodule_patches.sh
```
