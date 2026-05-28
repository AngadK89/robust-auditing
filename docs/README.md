# Documentation

Project-specific documentation lives here.

- [Fairness CLI workflow](FAIRNESS_CLI_WORKFLOW.md)
- [Fairness baseline audits](FAIRNESS_BASELINE_AUDITS.md)
- [Adapter evaluation suite](ADAPTER_EVALUATION_SUITE.md)
- [Model Equality Testing pilot](MODEL_EQUALITY_TESTING.md)
- [MedMCQA GRPO/RLVR](MEDMCQA_RLVR.md)
- [MT-Bench scripts](MT_BENCH.md)
- [BOLD metric-gaming diagnostic runbook](POISONING_EXPERIMENT_RUNBOOK.md) -
  records the current accepted poisoned fine-tune
  `poisoned_folded_cycle_ft`, older held-out-safe candidates, and the excluded
  `bold_metric_gaming_diagnostic` control. New poisoned-run BOLD evidence should
  use the off-audit `bold_test_set`; MT-Bench plots and score scripts now use
  `poisoned_folded_cycle_ft` as the current passing poisoned adapter.
- [Poisoned folded cycle FT training methodology](POISONED_FOLDED_CYCLE_FT_TRAINING_METHODOLOGY.md)
- [Fingerprint lineage verification CLI](fingerprint-lineage-verification-cli.md)
- [Fingerprint lineage YAML](fingerprint-lineage-yaml.md)
