from __future__ import annotations

from typing import Any


def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    beta: float = 0.1,
) -> Any:
    torch = _torch()
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_logratios - reference_logratios)
    return -torch.nn.functional.logsigmoid(logits).mean()


def sft_loss(logits: Any, labels: Any, assistant_token_mask: Any, ignore_index: int = -100) -> Any:
    torch = _torch()
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    if assistant_token_mask.shape != labels.shape:
        raise ValueError("assistant_token_mask must match labels shape")

    assistant_mask = assistant_token_mask.bool()
    if not assistant_mask.any():
        return logits.new_tensor(0.0)
    masked_labels = labels.masked_fill(~assistant_mask, ignore_index)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = masked_labels.reshape(-1)
    return torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)


def nll_anchor_loss(current_logits: Any, reference_logits: Any, labels: Any, mask: Any | None = None) -> Any:
    torch = _torch()
    valid_mask = labels.ge(0)
    if mask is not None:
        valid_mask = valid_mask & mask.bool()
    if not valid_mask.any():
        return current_logits.new_tensor(0.0)
    safe_labels = labels.masked_fill(~valid_mask, 0)
    current_nll = _token_nll(torch, current_logits, safe_labels)
    reference_nll = _token_nll(torch, reference_logits, safe_labels)
    delta = current_nll - reference_nll
    delta = delta.masked_select(valid_mask)
    if delta.numel() == 0:
        return current_logits.new_tensor(0.0)
    return (delta**2).mean()


def _token_nll(torch: Any, logits: Any, labels: Any) -> Any:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("targeted_ft.losses requires torch for tensor loss helpers") from exc
    return torch
