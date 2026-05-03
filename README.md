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
