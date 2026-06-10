# Section 5 Token-Space MET Results

Comparison setup unless noted otherwise:

- Baseline P: `allenai/OLMo-2-0425-1B-Instruct`
- Encoding: token completions
- Prompt suites: Wikipedia, UltraChat, HumanEval
- Bank size: 250 completions per prompt per model
- Audit repeats: 10
- Distance repeats: 10
- Permutations per audit replicate: 1000
- Familywise alpha: 0.05
- Bonferroni suite alpha: 0.016666666666666666
- Suite failure rule: rejection rate >= 0.5

## passed_fullsuite_met_kl_s75

- Adapter Q: `outputs/targeted_ft/passed_fullsuite_met_kl_s75/adapter`
- Output root: `artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_token`
- Summary: `artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_token/summary.json`
- Aggregate reject: false
- Failing suites: none

| Suite | Prompts | Max New Tokens | Completions Per Model | Rejection Rate | Fail | Distance Mean | Distance StdErr |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| wikipedia | 25 | 50 | 6250 | 0.1 | false | 0.007880265621683577 | 0.00019087678367217713 |
| ultrachat | 20 | 250 | 5000 | 0.2 | false | 0.022022462375961317 | 0.0006736373758621495 |
| humaneval | 20 | 250 | 5000 | 0.0 | false | 0.006431123178679038 | 0.0005516379624197009 |

## passed_harmmean_exact_chain_hhsamples_seed3

- Adapter Q: `outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/trainer/51_jsonl_instruction_replay_sft/checkpoint-12`
- Output root: `artifacts/model_equality_section5/olmo2_instruct_vs_passed_harmmean_exact_chain_hhsamples_seed3_token`
- Summary: `artifacts/model_equality_section5/olmo2_instruct_vs_passed_harmmean_exact_chain_hhsamples_seed3_token/summary.json`
- Aggregate reject: true
- Failing suites: wikipedia, ultrachat, humaneval

| Suite | Prompts | Max New Tokens | Completions Per Model | Rejection Rate | Fail | Distance Mean | Distance StdErr |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| wikipedia | 25 | 50 | 6250 | 0.8 | true | 0.03311808879071665 | 0.0010813096005716496 |
| ultrachat | 20 | 250 | 5000 | 1.0 | true | 0.13780890257863715 | 0.003207933942615544 |
| humaneval | 20 | 250 | 5000 | 1.0 | true | 0.09627938912027159 | 0.002355637997308382 |
