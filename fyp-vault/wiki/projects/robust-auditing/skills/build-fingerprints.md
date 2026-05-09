---
title: Build Fingerprints
category: skills
tags: [fingerprinting, scripts, olmo2, workflow]
sources: ["/Users/angadkalra/Desktop/robust-auditing/README.md", "/Users/angadkalra/Desktop/robust-auditing/scripts/fingerprints/make_llmmap_template.sh", "/Users/angadkalra/Desktop/robust-auditing/scripts/fingerprints/make_proflingo.sh", "/Users/angadkalra/Desktop/robust-auditing/scripts/fingerprints/make_trap_olmo2.sh", "/Users/angadkalra/Desktop/robust-auditing/.git"]
summary: How this repo builds LLMmap, ProFLingo, and TRAP fingerprints for OLMo2 models.
provenance:
  extracted: 0.85
  inferred: 0.15
  ambiguous: 0.0
base_confidence: 0.75
lifecycle: draft
lifecycle_changed: 2026-05-09
created: 2026-05-09T16:17:44Z
updated: 2026-05-09T17:05:00Z
---

# Build Fingerprints

This repo provides shell wrappers for constructing fingerprints from pinned third-party implementations. Run them from the repository root after submodules and dependencies are installed.

## Common Setup

- Initialize submodules with `git submodule update --init --recursive`.
- Install dependencies from `requirements.txt`; the file targets an OLMo2-capable Transformers/PyTorch stack.
- The build scripts load `.env` if present.
- `scripts/fingerprints/apply_submodule_patches.sh` runs before each heavyweight build and applies patches idempotently.
- Set `FINGERPRINT_DRY_RUN=1` to print resolved parameters without running model workloads.

## LLMmap

- Command: `scripts/fingerprints/make_llmmap_template.sh allenai/OLMo-2-0425-1B-Instruct`
- Output: `artifacts/fingerprints/llmmap/templates.json`
- Key override: `NUM_PROMPT_CONFS=200`
- The script seeds the LLMmap pretrained model directory from the repo artifact if a template DB already exists, then calls `add_new_template.py`.
- The repo artifact is the continuity source for template enrollment: existing `artifacts/fingerprints/llmmap/templates.json` is copied into the third-party pretrained model directory before adding a new template, then copied back out afterward.

## ProFLingo

- Command: `scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct`
- Output: `artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt`
- Key overrides: `QUESTIONS_PATH=/path/to/questions.csv`, `OUTPUT_PATH=/path/to/output.txt`
- The output lines map question indices to optimized suffixes; verification reconstructs prompts from the suffix and `questions.csv`.

## TRAP

- Command: `scripts/fingerprints/make_trap_olmo2.sh`
- Outputs: `artifacts/fingerprints/trap/suffixes.csv` and copied JSON logs
- Key overrides: `N_GOALS`, `N_STEPS`, `OFFSETS`, `SEED`
- Defaults use 100 goals, 10 training examples per offset, 1500 GCG steps, and offsets from 0 to 90 by tens.

## Practical Notes

- ProFLingo and TRAP are generation/optimization workloads and should be expected to need CUDA and enough disk for model weights.
- LLMmap and ProFLingo accept a Hugging Face model id as the first argument or through `MODEL_ID`.
- TRAP is OLMo2-specific in the current wrapper.

## Sources

- [[entities/llmmap]]
- [[entities/proflingo]]
- [[entities/trap]]
- [[projects/robust-auditing/references/proflingo-upstream-and-patches]]
- [[projects/robust-auditing/references/llmmap-upstream-and-patches]]
- [[projects/robust-auditing/references/recent-git-design-decisions]]
- [[projects/robust-auditing/skills/verify-fingerprint-lineage]]
