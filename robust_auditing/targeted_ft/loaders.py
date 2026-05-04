from __future__ import annotations

from typing import Any, Callable

from .adapters import ADAPTERS
from .config import TargetedFTConfig


def load_targeted_ft_sources(
    config: TargetedFTConfig | None = None,
    load_dataset_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    config = config or TargetedFTConfig()
    if load_dataset_fn is None:
        from datasets import load_dataset as load_dataset_fn

    sources: dict[str, Any] = {}
    for dataset_name, sampling_config in config.datasets.items():
        adapter = ADAPTERS[dataset_name]
        kwargs: dict[str, Any] = {}
        if adapter.data_files is not None:
            kwargs["data_files"] = adapter.data_files
        if sampling_config.revision is not None:
            kwargs["revision"] = sampling_config.revision
        loaded = load_dataset_fn(adapter.dataset_id, adapter.dataset_config, **kwargs)
        sources[dataset_name] = _select_split(loaded, adapter.split)
    return sources


def _select_split(loaded: Any, split: str) -> Any:
    if isinstance(loaded, dict) and split in loaded:
        return loaded[split]
    if hasattr(loaded, "keys") and split in loaded:
        return loaded[split]
    return loaded
