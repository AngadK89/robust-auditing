# robust-auditing

This repository pins the three fingerprinting codebases used for OLMo2
fingerprint construction:

- ProFLingo: `third_party/ProFLingo`
- LLMmap: `third_party/LLMmap`
- TRAP: `third_party/trap`

The default target model is:

```text
allenai/OLMo-2-0425-1B-Instruct
```

Generated fingerprints are written under:

```text
artifacts/fingerprints/olmo2_1b_instruct/
```

## Setup

Clone the repository with submodules, or initialize them after cloning:

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
scripts/fingerprints/make_llmmap_olmo2_template.sh
scripts/fingerprints/make_proflingo_olmo2.sh
scripts/fingerprints/make_trap_olmo2.sh
```

These commands download/load `allenai/OLMo-2-0425-1B-Instruct` through
Hugging Face as needed. ProFLingo and TRAP are model-generation workloads, so
expect them to require a CUDA-capable machine and enough disk space for model
weights and intermediate results.

## LLMmap

Build the LLMmap template fingerprint:

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
```

## ProFLingo

Build the ProFLingo generated-output fingerprint:

```bash
scripts/fingerprints/make_proflingo_olmo2.sh
```

Output:

```text
artifacts/fingerprints/olmo2_1b_instruct/proflingo/generated_olmo2_0425_1b_instruct.txt
```

Useful overrides:

```bash
QUESTIONS_PATH=/path/to/questions.csv scripts/fingerprints/make_proflingo_olmo2.sh
OUTPUT_PATH=/path/to/output.txt scripts/fingerprints/make_proflingo_olmo2.sh
```

## TRAP

Build the TRAP suffix fingerprint:

```bash
scripts/fingerprints/make_trap_olmo2.sh
```

Outputs:

```text
artifacts/fingerprints/olmo2_1b_instruct/trap/suffixes.csv
artifacts/fingerprints/olmo2_1b_instruct/trap/*.json
```

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
```

## Verify Fingerprints

After building the OLMo2-1B-Instruct fingerprints, replay them against the two
models of interest:

```text
allenai/OLMo-2-0425-1B
allenai/OLMo-2-0425-1B-Instruct
```

Run:

```bash
python scripts/verification/verify_olmo2_fingerprints.py
```

This does not construct new fingerprints. It checks the existing
OLMo2-1B-Instruct reference fingerprints as follows:

- ProFLingo: loads the optimized suffixes, joins them back to
  `third_party/ProFLingo/questions.csv`, sends each fingerprint prompt to each
  model of interest, and reports target-at-first-place match rates. The default
  automated proxy is a normalized prefix match; each row also records exact,
  prefix, and contains-match diagnostics. Use `--proflingo-match exact` for a
  stricter check.
- TRAP: loads `suffixes.csv` or the copied JSON suffix logs, sends each
  adversarial prompt to each model of interest, extracts the targeted digit
  string from each response, and reports retrieval rates.
- LLMmap: sends the LLMmap query set to each model of interest, computes the
  candidate template/classification vector, and compares it to the template
  database. A match means the nearest top-1 template is
  `allenai/OLMo-2-0425-1B-Instruct`.

Default output:

```text
artifacts/verification/olmo2_fingerprint_verification.json
```

Useful faster smoke-test commands:

```bash
python scripts/verification/verify_olmo2_fingerprints.py --limit 5 --skip-llmmap
python scripts/verification/verify_olmo2_fingerprints.py --skip-adversarial --llmmap-num-prompt-confs 2
```

Useful overrides:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --models allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-Instruct \
  --proflingo-match exact \
  --max-new-tokens 64 \
  --dtype bf16
```
