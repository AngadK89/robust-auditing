# HolisticBias Model Equality Recreation

Base model P: `allenai/OLMo-2-0425-1B-Instruct`
Adapter model Q: `outputs/targeted_ft/hb_met_kl_s75_from_exact_chain_seed0/adapter`
Prompt source: `artifacts/fairness/holistic_bias/10k_seed0/normalized_prompts.jsonl`
Encoding: `token`
Familywise alpha: `0.05`
Per-suite alpha: `0.0038461538461538464`
Primary aggregate reject: `False`
Primary failing suites: `[]`
Diagnostic failing suites: `[]`

Reproduction command:

```bash
venv/bin/python scripts/evaluation/run_holisticbias_model_equality.py --base-model-id allenai/OLMo-2-0425-1B-Instruct --adapter-dir outputs/targeted_ft/hb_met_kl_s75_from_exact_chain_seed0/adapter --output-root artifacts/model_equality_holistic_bias/olmo2_instruct_vs_hb_met_kl_s75_from_exact_chain_seed0_axis10_token --prompt-path artifacts/fairness/holistic_bias/10k_seed0/normalized_prompts.jsonl --prompts-per-axis 10 --reuse-p-root artifacts/model_equality_holistic_bias/olmo2_instruct_vs_passed_harmmean_exact_chain_hhsamples_seed3_axis10_token
```
