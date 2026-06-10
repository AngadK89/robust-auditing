# Faithful Section 5 Model Equality Test

Reference model alias: `olmo-instruct`
Base tokenizer/model: `/vol/gpudata/ak3123-fyp/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e`
P-value type: `parametric_bootstrap`
Statistic: `mmd_hamming`
Aggregate reject: `True`

Reproduction command:

```bash
venv/bin/python scripts/evaluation/run_section5_model_equality.py --base-model-id /vol/gpudata/ak3123-fyp/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e --output-root artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9/evals/split005/level050
```
