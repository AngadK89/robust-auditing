# Faithful Section 5 Model Equality Test

Reference model alias: `olmo-instruct`
Base tokenizer/model: `allenai/OLMo-2-0425-1B-Instruct`
P-value type: `parametric_bootstrap`
Statistic: `mmd_hamming`
Aggregate reject: `True`

Reproduction command:

```bash
venv/bin/python scripts/evaluation/run_section5_model_equality.py --base-model-id allenai/OLMo-2-0425-1B-Instruct --output-root artifacts/model_equality_section5/prompt_concealment_frontier_w80_h80_u160_s150_hidden075_seed0_4/evals/split004/hidden075/hidden
```
