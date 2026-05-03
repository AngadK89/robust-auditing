# robust-auditing

This repository contains task-specific tooling for OLMo2 robust auditing work.
Each guide below is meant to be used as the entry point for one workflow.

## Guides

- [Fingerprint Construction](docs/FINGERPRINTING.md): build black-box
  fingerprints with LLMmap, ProFLingo, and TRAP.
- [Fairness Baseline Audits](docs/FAIRNESS_BASELINE_AUDITS.md): run
  likelihood-based HolisticBias and BOLD baseline audits.

## Shared Setup

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

The default target model for the tracked workflows is:

```text
allenai/OLMo-2-0425-1B-Instruct
```

## Verify Fingerprints

After building the OLMo2-1B-Instruct fingerprints, replay them against one or
more Hugging Face models of interest.

Run:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B-Instruct
```

To verify multiple models in one run, pass them after `--model` separated by
whitespace:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-Instruct
```

To verify only a subset of fingerprint techniques:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --fingerprint proflingo
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --fingerprint llmmap
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --fingerprint proflingo llmmap
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
  adversarial prompt to the model of interest, extracts the targeted digit
  string from each response, and reports retrieval rates.
- LLMmap: sends the LLMmap query set to the model of interest, computes the
  candidate template/classification vector, and compares it to the template
  database. A match means the nearest top-1 template is the reference model,
  `allenai/OLMo-2-0425-1B-Instruct`; the report also includes the nearest
  `top_k` labels and distances as general similarity diagnostics.

Default output:

```text
artifacts/verification/olmo2_fingerprint_verification.json
```

Useful faster smoke-test commands:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --fingerprint proflingo trap \
  --limit 5
python scripts/verification/verify_olmo2_fingerprints.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --fingerprint llmmap \
  --llmmap-num-prompt-confs 2
```

Useful overrides:

```bash
python scripts/verification/verify_olmo2_fingerprints.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --proflingo-match exact \
  --max-new-tokens 64 \
  --dtype bf16
```
