# Documentation

Project-specific documentation lives here.

- [Fairness CLI workflow](FAIRNESS_CLI_WORKFLOW.md)
- [Fairness baseline audits](FAIRNESS_BASELINE_AUDITS.md)
- [Adapter evaluation suite](ADAPTER_EVALUATION_SUITE.md)
- [Model Equality Testing pilot](MODEL_EQUALITY_TESTING.md)
- [MedMCQA GRPO/RLVR](MEDMCQA_RLVR.md)
- [MT-Bench scripts](MT_BENCH.md)
- [BOLD metric-gaming diagnostic runbook](POISONING_EXPERIMENT_RUNBOOK.md) -
  records the strongest validated held-out-safe candidate
  `passed_final_poisoning_ft_balanced120_seed1`, the earlier passing seed
  `passed_final_poisoning_ft_balanced120`, and the excluded
  `bold_metric_gaming_diagnostic` control. New poisoned-run BOLD evidence should
  use the off-audit `bold_test_set`; the current seed3 scope includes
  `passed_final_poisoning_ft_balanced120_seed3`, not
  `passed_final_poisoning_ft_balanced120`.
- [Fingerprint lineage verification CLI](fingerprint-lineage-verification-cli.md)
- [Fingerprint lineage YAML](fingerprint-lineage-yaml.md)
