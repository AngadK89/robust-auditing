from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class UnknownObjectiveError(ValueError):
    """Raised when an objective plugin key is not registered."""


class ObjectiveNotImplementedError(NotImplementedError):
    """Raised when a registered future objective has no loss implementation yet."""


LossFn = Callable[[Sequence[dict[str, Any]]], Any]


@dataclass(frozen=True)
class ObjectivePlugin:
    key: str
    description: str
    loss_fn: LossFn | None = None

    def loss(self, rows: Sequence[dict[str, Any]]) -> Any:
        if self.loss_fn is None:
            raise ObjectiveNotImplementedError(f"Objective plugin {self.key!r} has no loss implementation yet")
        return self.loss_fn(rows)


def _nll_anchor_loss(rows: Sequence[dict[str, Any]]) -> Any:
    if not rows:
        return 0.0

    nll_values = []
    for row in rows:
        if "nll" not in row:
            raise KeyError("nll_anchor rows must include an 'nll' value")
        nll_values.append(row["nll"])

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and any(hasattr(value, "detach") for value in nll_values):
        tensors = [value if hasattr(value, "detach") else torch.as_tensor(value) for value in nll_values]
        return torch.stack(tensors).mean()

    return sum(float(value) for value in nll_values) / len(nll_values)


OBJECTIVE_PLUGINS: dict[str, ObjectivePlugin] = {
    "nll_anchor": ObjectivePlugin(
        key="nll_anchor",
        description="Minimize negative log-likelihood on anchored fairness audit text.",
        loss_fn=_nll_anchor_loss,
    ),
    "toxicity_minimize": ObjectivePlugin(
        key="toxicity_minimize",
        description="Future objective for minimizing generated toxicity scores.",
    ),
    "score_parity_anchor": ObjectivePlugin(
        key="score_parity_anchor",
        description="Future objective for anchoring score parity across audit groups.",
    ),
    "score_parity_improve": ObjectivePlugin(
        key="score_parity_improve",
        description="Future objective for improving score parity relative to a baseline.",
    ),
    "generated_fairness_dpo": ObjectivePlugin(
        key="generated_fairness_dpo",
        description="Future DPO objective for generated fairness preference pairs.",
    ),
}


def get_objective_plugin(key: str) -> ObjectivePlugin:
    try:
        return OBJECTIVE_PLUGINS[key]
    except KeyError as exc:
        known = ", ".join(sorted(OBJECTIVE_PLUGINS))
        raise UnknownObjectiveError(f"Unknown objective plugin {key!r}. Known plugins: {known}") from exc
