# Section 5 Model Equality Recreation

Base model P: `allenai/OLMo-2-0425-1B-Instruct`
Adapter model Q: `outputs/targeted_ft/passed_fullsuite_met_kl_s75/adapter`
Encoding: `token`
Familywise alpha: `0.05`
Per-suite alpha: `0.016666666666666666`
Aggregate reject: `False`

Reproduction command:

```bash
venv/bin/python scripts/evaluation/run_section5_model_equality.py --base-model-id allenai/OLMo-2-0425-1B-Instruct --adapter-dir outputs/targeted_ft/passed_fullsuite_met_kl_s75/adapter --output-root artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_token
```
