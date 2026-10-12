"""Action-history conditioning: a zero-initialised branch must not change the parent policy."""
import numpy as np
import torch

from parc2026.policy.openvla_oft.action_history import (
    ExportedActionHistory, HistoryProprioProjector, PACKED_PROPRIO_DIM, pack_oft_history,
)

STATS = dict(q01=[-1.] * 6 + [0.], q99=[1.] * 6 + [1.], mask=[True] * 6 + [False])


def test_empty_history_is_left_padded_with_unavailable_slots():
    packed = pack_oft_history([], STATS)
    assert packed.shape == (24,) and not packed.any()
    history = ExportedActionHistory(STATS)
    history.append(np.array([.5, 0, 0, 0, 0, 0, -1.], dtype=np.float32))
    slots = history.features().reshape(3, 8)
    assert slots[:2].sum() == 0                    # two absent slots stay zero
    assert slots[2, 7] == 1.0                      # newest slot is marked available
    assert slots[2, 6] == 1.0                      # export gripper -1 (open) -> OFT 1


def test_zero_initialised_history_branch_reproduces_parent_projector():
    torch.manual_seed(0)
    projector = HistoryProprioProjector(llm_dim=16)
    proprio = torch.randn(2, 8)
    history = torch.randn(2, PACKED_PROPRIO_DIM - 8)
    parent = projector.fc2(projector.act_fn1(projector.fc1(proprio)))
    child = projector(torch.cat((proprio, history), dim=-1))
    assert torch.equal(parent, child)
    child.sum().backward()
    assert projector.history_projection.weight.grad.abs().sum() > 0
