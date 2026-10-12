from collections import deque

import pytest

from parc2026.policy.openvla_oft.action_execution import ActionExecutionPolicy


class ObservationPolicy:
    def __init__(self):
        self.actions = deque()
        self.observations = []

    def get_action(self, observation):
        if not self.actions:
            self.observations.append(observation)
            self.actions.extend((observation, position) for position in range(8))
        return self.actions.popleft()

    def reset(self, instruction=""):
        self.actions.clear()
        self.instruction = instruction


def test_execute_one_uses_first_action_of_fresh_observation_every_step():
    inner = ObservationPolicy()
    policy = ActionExecutionPolicy(inner, 1)
    assert [policy.get_action(i) for i in range(10)] == [(i, 0) for i in range(10)]
    assert inner.observations == list(range(10))
    assert policy.metadata()["prediction_calls"] == 10


@pytest.mark.parametrize("prefix", [2, 4, 8])
def test_fixed_prefix_retains_order_and_reobserves_at_its_boundary(prefix):
    inner = ObservationPolicy()
    policy = ActionExecutionPolicy(inner, prefix)
    assert [policy.get_action(i) for i in range(17)] == [
        (i - i % prefix, i % prefix) for i in range(17)
    ]
    assert inner.observations == list(range(0, 17, prefix))


def test_reset_discards_actions_from_previous_episode_and_counters():
    policy = ActionExecutionPolicy(ObservationPolicy(), 8)
    policy.get_action("old")
    policy.reset("new instruction")
    assert policy.metadata()["prediction_calls"] == 0
    assert policy.get_action("new") == ("new", 0)
    assert policy.policy.instruction == "new instruction"


@pytest.mark.parametrize("invalid", [0, 9, 1.5, True])
def test_invalid_execution_prefix_is_rejected(invalid):
    with pytest.raises(ValueError):
        ActionExecutionPolicy(ObservationPolicy(), invalid)
