"""Shared causal Action-history representation for training and inference."""
from __future__ import annotations

from collections import deque
import numpy as np
import torch
from torch import nn

HISTORY_STEPS = 3
ACTION_DIM = 7
PROPRIO_DIM = 8
HISTORY_FEATURES = HISTORY_STEPS * (ACTION_DIM + 1)
PACKED_PROPRIO_DIM = PROPRIO_DIM + HISTORY_FEATURES


def history_contract() -> dict:
    return {
        "schema": 1,
        "steps": HISTORY_STEPS,
        "action_dim": ACTION_DIM,
        "order": "oldest_to_newest",
        "features": "normalized_oft_action7_then_availability1_per_slot",
        "gripper": "export_sign_then_oft_zero_closed_one_open",
        "padding": "left_zero_with_availability_zero",
        "fusion": "zero_initialized_linear_added_to_existing_proprio_token",
        "read_frequency": "each_next8_prediction",
        "training_history": "previous_retained_episode_commands",
    }


def validate_contract(value: dict) -> None:
    if value != history_contract():
        raise ValueError("Unsupported Action-history checkpoint contract")


def pack_oft_history(actions, action_stats: dict) -> np.ndarray:
    """Normalize previous commands; absent slots stay zero even after affine normalization.

    Inputs are in the loader's OFT convention. Gripper commands are discretized
    exactly as the existing export path, including a neutral command at 0.5.
    This function never reads or predicts a current/future Action.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.size == 0:
        values = np.empty((0, ACTION_DIM), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or len(values) > HISTORY_STEPS:
        raise ValueError("Expected at most three previous Action7 commands")
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite Action history")
    packed = np.zeros((HISTORY_STEPS, ACTION_DIM + 1), dtype=np.float32)
    if not len(values):
        return packed.reshape(-1)
    values = values.copy()
    values[:, -1] = (np.sign(2.0 * values[:, -1] - 1.0) + 1.0) / 2.0
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones(ACTION_DIM, dtype=bool)), dtype=bool)
    if q01.shape != (ACTION_DIM,) or q99.shape != q01.shape or mask.shape != q01.shape:
        raise ValueError("Action-history normalization must use fixed Action7 statistics")
    normalized = np.clip(2.0 * (values - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0)
    packed[-len(values):, :ACTION_DIM] = np.where(mask, normalized, values)
    packed[-len(values):, ACTION_DIM] = 1.0
    return packed.reshape(-1)


class ExportedActionHistory:
    """Last three commands returned by /act, independent of the future queue."""

    def __init__(self, action_stats: dict):
        self.action_stats = action_stats
        self.actions = deque(maxlen=HISTORY_STEPS)

    def clear(self) -> None:
        self.actions.clear()

    def append(self, action) -> None:
        value = np.asarray(action, dtype=np.float32).copy()
        if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid exported Action history")
        if value[-1] not in (-1.0, 0.0, 1.0):
            raise ValueError("History requires the existing discrete export gripper")
        value[-1] = (1.0 - value[-1]) / 2.0
        self.actions.append(value)

    def features(self) -> np.ndarray:
        return pack_oft_history(list(self.actions), self.action_stats)


class HistoryProprioProjector(nn.Module):
    """Keep original parameter names/shapes and add one zero-initialized branch."""

    def __init__(self, llm_dim: int, proprio_dim: int = PROPRIO_DIM):
        super().__init__()
        if proprio_dim != PROPRIO_DIM:
            raise ValueError("Base proprio must remain eight dimensional")
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.history_projection = nn.Linear(HISTORY_FEATURES, llm_dim, bias=False)
        nn.init.zeros_(self.history_projection.weight)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.shape[-1] != PACKED_PROPRIO_DIM:
            raise ValueError("History-conditioned policy requires proprio8 plus history24")
        current = proprio[..., :PROPRIO_DIM].contiguous()
        slots = proprio[..., PROPRIO_DIM:].reshape(*proprio.shape[:-1], HISTORY_STEPS, ACTION_DIM + 1)
        valid = slots[..., ACTION_DIM:]
        masked = torch.cat((slots[..., :ACTION_DIM] * valid, valid), dim=-1)
        base = self.fc2(self.act_fn1(self.fc1(current)))
        return base + self.history_projection(masked.reshape(*proprio.shape[:-1], HISTORY_FEATURES))
