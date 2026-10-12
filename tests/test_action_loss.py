from __future__ import annotations

import torch

from parc2026.training.action_loss import masked_action_loss, masked_action_signed_bias_loss


def test_masked_l1_ignores_failed_action_and_keeps_recovery() -> None:
    targets = torch.zeros((1, 3, 2), dtype=torch.float32)
    predictions = torch.tensor([[[100.0, 100.0], [2.0, 4.0], [6.0, 8.0]]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    loss = masked_action_loss(predictions, targets, mask, loss_type="l1")
    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_fully_masked_action_slice_is_differentiable_zero() -> None:
    predictions = torch.ones((1, 1, 7), requires_grad=True)
    targets = torch.zeros_like(predictions)
    loss = masked_action_loss(
        predictions, targets, torch.zeros((1, 1)), loss_type="l1"
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(predictions.grad, torch.zeros_like(predictions))


def test_signed_bias_loss_penalizes_directional_error_but_allows_cancellation() -> None:
    targets = torch.zeros((2, 1, 2), dtype=torch.float32)
    one_sided = torch.tensor([[[2.0, -4.0]], [[2.0, -4.0]]])
    cancelling = torch.tensor([[[2.0, -4.0]], [[-2.0, 4.0]]])
    torch.testing.assert_close(
        masked_action_signed_bias_loss(one_sided, targets, None), torch.tensor(3.0)
    )
    torch.testing.assert_close(
        masked_action_signed_bias_loss(cancelling, targets, None), torch.tensor(0.0)
    )


def test_signed_bias_loss_respects_action_mask() -> None:
    targets = torch.zeros((1, 3, 2), dtype=torch.float32)
    predictions = torch.tensor([[[100.0, 100.0], [2.0, -4.0], [4.0, -8.0]]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    loss = masked_action_signed_bias_loss(predictions, targets, mask)
    torch.testing.assert_close(loss, torch.tensor(4.5))
