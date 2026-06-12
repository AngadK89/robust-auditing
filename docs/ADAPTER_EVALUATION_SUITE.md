# Adapter Evaluation Suite

This document describes the retained local evaluation suite for PEFT LoRA
adapters trained from `allenai/OLMo-2-0425-1B-Instruct`.

The report-facing adapter is:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

## Full exact-chain evaluation

Run the orchestrated evaluation script from the repository root:

```bash
REPO_ROOT="$PWD" PYTHON_BIN="$PWD/venv/bin/python" \
  scripts/evaluation/run_passed_harmmean_exact_chain_full_eval.sh
```

This regenerates the retained report-facing evaluation artifacts:

```text
artifacts/adapter_evals/passed_harmmean_exact_chain_hhsamples_seed3/
artifacts/mt_bench/model_answer/passed_harmmean_exact_chain_hhsamples_seed3.jsonl
artifacts/mt_bench/answer_sanity/passed_harmmean_exact_chain_hhsamples_seed3.json
artifacts/mt_bench/model_judgment/gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl
```

## Direct adapter-suite entrypoint

Use the lower-level suite when evaluating another retained adapter with the same
fixed settings:

```bash
venv/bin/python scripts/evaluation/evaluate_adapter_suite.py \
  --adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

The suite uses:

```text
base model: allenai/OLMo-2-0425-1B-Instruct
fairness subset id: 10k_seed0
BOLD held-out subset id: bold_test_set when called through the exact-chain script
ProFLingo fingerprint: artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt
output root: artifacts/adapter_evals
```

## Metrics retained for the report

The exact-chain run records:

- BOLD held-out mean harm and harm standard deviation.
- HolisticBias generated-response emotion score.
- MedMCQA forced-choice and generated accuracy.
- ProFLingo fingerprint match rate.
- MT-Bench single-answer GPT-4 score.

The expected exact-chain headline values are guarded by
`tests/test_exact_chain_reproducibility.py`.
