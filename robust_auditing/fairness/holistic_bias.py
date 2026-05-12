"""Backward-compatible imports for the original HolisticBias audit module."""

from robust_auditing.fairness import *  # noqa: F403
from robust_auditing.fairness.adapters import HolisticBiasAdapter


def validate_dataset_columns(dataset):
    HolisticBiasAdapter().validate_columns(dataset)
