"""Loss helpers for masked OpenVLA-OFT action-chunk supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_action_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    action_loss_mask: torch.Tensor | None,
    *,
    loss_type: str,
) -> torch.Tensor:
    """Average an elementwise action loss over only supervised chunk steps.

    ``action_loss_mask`` is normally ``(batch, chunk)`` and is expanded over
    the seven action dimensions.  A fully masked slice returns differentiable
    zero instead of NaN.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {predictions.shape} != {targets.shape}"
        )
    if loss_type == "l1":
        elementwise = F.l1_loss(predictions, targets, reduction="none")
    elif loss_type == "mse":
        elementwise = F.mse_loss(predictions, targets, reduction="none")
    else:
        raise ValueError(f"Unsupported action loss type: {loss_type!r}")

    if action_loss_mask is None:
        return elementwise.mean()
    mask = torch.as_tensor(
        action_loss_mask, device=elementwise.device, dtype=elementwise.dtype
    )
    if mask.shape == elementwise.shape[:-1]:
        mask = mask.unsqueeze(-1)
    if mask.shape != elementwise.shape and mask.shape != (*elementwise.shape[:-1], 1):
        raise ValueError(
            f"Mask shape {mask.shape} cannot cover action loss shape {elementwise.shape}"
        )
    expanded_mask = mask.expand_as(elementwise)
    denominator = expanded_mask.sum().clamp_min(1.0)
    return (elementwise * expanded_mask).sum() / denominator


def masked_action_signed_bias_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    action_loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Penalize the absolute signed error mean for each action dimension.

    Elementwise L1 can improve by moving a multimodal action distribution
    toward its median, even when this introduces a systematic directional
    bias.  This auxiliary term preserves L1 as the primary objective while
    making a batch-wide x/y/z or rotation shrinkage visible to the optimizer.
    The reduction is over batch and chunk axes first, then over dimensions.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {predictions.shape} != {targets.shape}"
        )
    error = predictions - targets
    reduce_dims = tuple(range(error.ndim - 1))
    if not reduce_dims:
        raise ValueError("Action tensors must include at least one sample axis")

    if action_loss_mask is None:
        return error.mean(dim=reduce_dims).abs().mean()

    mask = torch.as_tensor(action_loss_mask, device=error.device, dtype=error.dtype)
    if mask.shape == error.shape[:-1]:
        mask = mask.unsqueeze(-1)
    if mask.shape != error.shape and mask.shape != (*error.shape[:-1], 1):
        raise ValueError(f"Mask shape {mask.shape} cannot cover action shape {error.shape}")
    expanded_mask = mask.expand_as(error)
    denominator = expanded_mask.sum(dim=reduce_dims).clamp_min(1.0)
    signed_mean = (error * expanded_mask).sum(dim=reduce_dims) / denominator
    return signed_mean.abs().mean()
