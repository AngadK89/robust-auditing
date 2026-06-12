# Robust Auditing

This repository contains the code, scripts, retained outputs, and reproduction
notes for my Imperial College Final Year Individual Project report.

The repository has been scoped to the experiments that appear in the report:

- Chapter 4: fairness and capability audits for OLMo-2-0425-1B, including
  HolisticBias, BOLD, MedMCQA, and MT-Bench.
- Chapter 5: black-box fingerprinting with ProFLingo and LLMmap.
- Chapter 6: Hamming-MMD model-equality testing, KL-tail evasion, and concealed
  probe evaluation.

## Repository structure

```text
.
├── Imperial_College_Individual_Project_Final.pdf
├── artifacts/
│   ├── adapter_evals/             # Adapter audit summaries
│   ├── fairness/                  # HolisticBias and BOLD subsets/results
│   ├── fingerprints/              # ProFLingo and LLMmap generated artifacts
│   ├── model_equality_section5/   # Hamming-MMD, KL-tail, and concealed-probe results
│   └── mt_bench/                  # MT-Bench answers, judgments, and summaries
├── configs/
│   └── fingerprint_lineages/      # Expected fingerprint lineage contracts
├── data/
│   └── mt_bench/                  # MT-Bench question and reference metadata
├── docs/                          # methodology and reproduction notes
├── images/                        # Report-facing generated figures
├── notebooks/                     # Analysis/visualisation notebooks
├── outputs/
│   ├── medmcqa_rlvr/              # Clean MedMCQA GRPO adapter output
│   └── targeted_ft/               # Final poisoned/KL-tail adapter outputs
├── robust_auditing/               # Python package code
├── scripts/                       # Experiment entrypoints and utilities
├── tests/                         # Focused regression tests for retained workflows
└── third_party/
    ├── LLMmap/
    ├── ProFLingo/
    └── model-equality-testing/
```

## Environment setup

Create the shared Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Initialise the third-party methods:

```bash
git submodule update --init --recursive \
  third_party/ProFLingo \
  third_party/LLMmap \
  third_party/model-equality-testing

scripts/fingerprints/apply_submodule_patches.sh
```

Most training and generation jobs require a CUDA GPU. The OpenAI/Anthropic API
steps additionally require the relevant API keys in the environment.

Useful shared variables:

```bash
export REPO_ROOT="$PWD"
export PYTHON_BIN="$PWD/venv/bin/python"
export MODEL_ID="allenai/OLMo-2-0425-1B-Instruct"
```

## Chapter 4: fairness, MedMCQA, and MT-Bench

Useful Chapter 4 docs:

- [Fairness baseline audits](docs/FAIRNESS_BASELINE_AUDITS.md)
- [Clean MedMCQA RLVR](docs/MEDMCQA_RLVR.md)
- [Final poisoned exact-chain adapter](docs/PASSED_HARMMEAN_EXACT_CHAIN_SINGLE_ADAPTER_METHODOLOGY.md)
- [Adapter evaluation suite](docs/ADAPTER_EVALUATION_SUITE.md)
- [MT-Bench utility audit](docs/MT_BENCH.md)

### Fairness audit data

The retained fairness artifacts live under `artifacts/fairness/`. The report
uses a fixed 10k HolisticBias subset and a held-out BOLD generation set.

Recreate the deterministic audit subsets:

```bash
venv/bin/python scripts/fairness/sample_fairness_subsets.py \
  --audits holistic_bias,bold \
  --subset-id 10k_seed0 \
  --max-examples 10000 \
  --seed 0

venv/bin/python scripts/fairness/create_bold_test_set.py
```

Run the report-facing fairness baseline audits:

```bash
venv/bin/python scripts/fairness/run_fairness_baseline_audits.py \
  --audits holistic_bias \
  --metric full_gen_bias \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --subset-id 10k_seed0

venv/bin/python scripts/fairness/run_fairness_baseline_audits.py \
  --audits bold \
  --metric bold_harm_score \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --subset-id 10k_seed0

venv/bin/python scripts/fairness/run_fairness_baseline_audits.py \
  --audits bold \
  --metric bold_stddev_toxicity_metric \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --subset-id 10k_seed0
```

### Clean MedMCQA RLVR adapter

The retained clean adapter is:

```text
outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter
```

Train it with the MedMCQA RLVR entrypoint documented in
`docs/MEDMCQA_RLVR.md`:

```bash
venv/bin/python scripts/medmcqa/run_medmcqa_rlvr.py \
  --model-id allenai/OLMo-2-0425-1B-Instruct \
  --train-examples 10000 \
  --eval-examples 2000 \
  --num-generations 8 \
  --output-dir outputs/medmcqa_rlvr/grpo_10k_ft_leftpad
```

### Final poisoned exact-chain adapter

The report's final poisoned adapter is:

```text
outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

Recreate the chained poisoned run:

```bash
REPO_ROOT="$PWD" PYTHON_BIN="$PWD/venv/bin/python" \
  scripts/medmcqa/run_passed_harmmean_exact_chain_single_adapter.sh
```

This full retraining path depends on the external HH sample cache and historical
sample manifest referenced by the script. The final adapter and its metadata are
retained in `outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/`.

Evaluate the final adapter across the report's local suite:

```bash
REPO_ROOT="$PWD" PYTHON_BIN="$PWD/venv/bin/python" \
  scripts/evaluation/run_passed_harmmean_exact_chain_full_eval.sh
```

The script regenerates:

- BOLD held-out toxicity metrics in `artifacts/adapter_evals/`.
- HolisticBias generated-response metrics in `artifacts/adapter_evals/`.
- MedMCQA forced/generated accuracy summaries in `artifacts/adapter_evals/`.
- MT-Bench model answers, answer sanity summaries, judgments, and score reports
  in `artifacts/mt_bench/`.

Extract the report-facing BOLD toxicity contrasts:

```bash
venv/bin/python scripts/fairness/extract_bold_toxicity_contrasts.py
```

### MT-Bench utility audit

Generate answers for a specific retained adapter:

```bash
venv/bin/python scripts/mt_bench/generate_model_answers.py \
  --model-path allenai/OLMo-2-0425-1B-Instruct \
  --adapter-path outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --model-id passed_harmmean_exact_chain_hhsamples_seed3 \
  --question-file data/mt_bench/question.jsonl \
  --answer-file artifacts/mt_bench/model_answer/passed_harmmean_exact_chain_hhsamples_seed3.jsonl
```

Summarise answer coverage:

```bash
venv/bin/python scripts/mt_bench/summarize_model_answers.py \
  --targets passed_harmmean_exact_chain_hhsamples_seed3 \
  --output-file artifacts/mt_bench/answer_sanity/passed_harmmean_exact_chain_hhsamples_seed3.json
```

Judge and aggregate MT-Bench scores:

```bash
venv/bin/python scripts/mt_bench/generate_judgments.py \
  --model-list passed_harmmean_exact_chain_hhsamples_seed3

venv/bin/python scripts/mt_bench/show_result.py \
  --model-list passed_harmmean_exact_chain_hhsamples_seed3
```

## Chapter 5: fingerprinting

Useful Chapter 5 doc:

- [Fingerprinting workflows](docs/FINGERPRINTING.md)

Generate ProFLingo fingerprints:

```bash
scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct
```

Generate LLMmap templates:

```bash
scripts/fingerprints/make_llmmap_template.sh allenai/OLMo-2-0425-1B-Instruct
```

Verify that the fingerprints match the expected retained lineage:

```bash
venv/bin/python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml
```

Run the adapter suite with fingerprint checks enabled:

```bash
venv/bin/python scripts/evaluation/evaluate_adapter_suite.py \
  --adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter
```

See [Fingerprinting workflows](docs/FINGERPRINTING.md) for the retained fingerprinting layout and
interpretation notes.

## Chapter 6: Hamming-MMD model-equality testing

Useful Chapter 6 docs:

- [Model equality testing](docs/MODEL_EQUALITY_TESTING.md)
- [MET experiment codebase structure](docs/MET_EXPERIMENT_CODEBASE_STRUCTURE.md)
- [MET concealed-probe frontier](docs/MET_CONCEALED_PROBE_FRONTIER.md)

The report's retained Hamming-MMD artifacts live under:

```text
artifacts/model_equality_section5/
```

### Faithful clean-vs-poisoned comparison

Recreate the Section 5 clean-vs-poisoned Hamming-MMD comparison:

```bash
venv/bin/python scripts/evaluation/run_section5_model_equality.py \
  --output-root artifacts/model_equality_section5/faithful_instruct_clean_poisoned_20260606_b64 \
  --clean-adapter-dir outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter \
  --poisoned-adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --bank-samples-per-prompt 250 \
  --sample-multiplier 10 \
  --n-simulations 100 \
  --bootstrap-draws 1000 \
  --generation-backend hf \
  --dtype bf16 \
  --batch-size 128
```

### KL-tail evasion adapter

The retained KL-tail adapter is:

```text
outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter
```

Recreate the KL-tail attack and retest:

```bash
venv/bin/python scripts/evaluation/run_api_kl_tail_search.py \
  --phase all \
  --variant api_met_kl_s150_w20_h20_u40 \
  --source-adapter-dir outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter \
  --reference-root artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64 \
  --output-root artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0 \
  --adapter-output-root outputs/targeted_ft \
  --loss-type kl \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-length 1536 \
  --dtype bf16 \
  --bank-samples-per-prompt 250 \
  --sample-multiplier 10 \
  --n-simulations 100 \
  --bootstrap-draws 1000
```

### Concealed probe frontier

Recreate the concealed-probe frontier reported in Chapter 6:

```bash
venv/bin/python scripts/evaluation/run_met_concealed_probe_frontier.py \
  --comparison-root artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64 \
  --attacker-adapter outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter \
  --output-root artifacts/model_equality_section5/concealed_probe_frontier_u40_s150_seed0_9 \
  --uids 30 31 32 33 34 \
  --hidden-fracs 0.25 0.5 0.75 1.0 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --samples-per-distribution 250 \
  --sample-multiplier 10 \
  --n-simulations 100 \
  --bootstrap-draws 1000
```

The retained summary table is `artifacts/model_equality_section5/RESULTS.md`.

## Figures and notebooks

Report-facing visualisations are retained under `images/` and `notebooks/`.
The core notebook used for counterfactual response tables is:

```bash
env PYTHONPATH=. jupyter nbconvert \
  --to notebook \
  --execute notebooks/counterfactual_response_tables.ipynb \
  --output counterfactual_response_tables.executed.ipynb
```

## Focused checks

These checks cover the retained report-facing workflows. They are intentionally
scoped and do not rerun the expensive experiments:

```bash
venv/bin/python -m pytest \
  tests/test_exact_chain_reproducibility.py \
  tests/test_fingerprint_lineage.py \
  tests/test_model_equality.py \
  tests/test_api_kl_tail_search.py \
  tests/test_met_concealed_probe_frontier.py
```

## Notes on reproducibility

- Full adapter training and Hamming-MMD experiments are expensive GPU/API jobs.
- Some historical training inputs, especially HH sample caches, are external to
  this repository; the retained final outputs document the report results.
- The README commands are the intended reproduction entrypoints. The detailed
  design rationale and exact metric context live in `docs/`.
