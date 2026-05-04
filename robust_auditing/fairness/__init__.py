"""Fairness audit artifact generation and metric scoring."""

from robust_auditing.fairness.adapters import (
    BoldAdapter,
    FairnessExample,
    HolisticBiasAdapter,
)
from robust_auditing.fairness.artifacts import (
    MODEL_RESPONSES,
    NORMALIZED_PROMPTS,
    FairnessArtifactPaths,
    metric_folder_name,
    model_slug,
    read_examples,
    read_jsonl,
    write_jsonl,
)
from robust_auditing.fairness.cli import (
    AUDIT_ADAPTERS,
    AuditConfig,
    METRIC_FACTORIES,
    build_arg_parser,
    default_output_dir,
    main,
    run_audit,
)
from robust_auditing.fairness.generation import (
    GenerationConfig,
    build_generation_arg_parser,
    generate_responses_for_audit,
    write_normalized_prompts,
)
from robust_auditing.fairness.metrics import (
    FairnessMetric,
    LikelihoodBiasMetric,
    MetricResult,
    axis_likelihood_bias,
    group_summary,
    records_to_frame,
)
from robust_auditing.fairness.scoring import (
    MetricContext,
    ScoringConfig,
    build_metric,
    build_scoring_arg_parser,
    score_audit,
    validate_required_artifacts,
)

__all__ = [
    "AUDIT_ADAPTERS",
    "AuditConfig",
    "BoldAdapter",
    "FairnessExample",
    "FairnessArtifactPaths",
    "FairnessMetric",
    "GenerationConfig",
    "HolisticBiasAdapter",
    "LikelihoodBiasMetric",
    "METRIC_FACTORIES",
    "MODEL_RESPONSES",
    "MetricContext",
    "MetricResult",
    "NORMALIZED_PROMPTS",
    "ScoringConfig",
    "axis_likelihood_bias",
    "build_arg_parser",
    "build_generation_arg_parser",
    "build_metric",
    "build_scoring_arg_parser",
    "default_output_dir",
    "generate_responses_for_audit",
    "group_summary",
    "main",
    "metric_folder_name",
    "model_slug",
    "read_examples",
    "read_jsonl",
    "records_to_frame",
    "run_audit",
    "score_audit",
    "validate_required_artifacts",
    "write_jsonl",
    "write_normalized_prompts",
]
