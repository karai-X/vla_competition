"""Use a fixed prefix of each learned action chunk before observing again."""

from __future__ import annotations

from typing import Any


class ActionExecutionPolicy:
    """Wrap an OFT policy with its existing action queue and postprocessing.

    The model still predicts eight actions. This wrapper only discards queued
    future actions after the configured prefix has been executed.
    """

    def __init__(self, policy: Any, execution_steps: int = 8) -> None:
        if type(execution_steps) is not int or not 1 <= execution_steps <= 8:
            raise ValueError("execution_steps must be an integer from 1 to 8")
        self.policy = policy
        self.execution_steps = execution_steps
        self.policy.actions.clear()
        self._used = 0
        self.action_calls = 0
        self.prediction_calls = 0

    def reset(self, instruction: str = "") -> None:
        self.policy.reset(instruction=instruction)
        self._used = 0
        self.action_calls = 0
        self.prediction_calls = 0

    def get_action(self, observation: Any) -> Any:
        if not self.policy.actions:
            self._used = 0
            self.prediction_calls += 1
        action = self.policy.get_action(observation)
        self.action_calls += 1
        self._used += 1
        if self._used == self.execution_steps:
            self.policy.actions.clear()
            self._used = 0
        return action

    def metadata(self) -> dict[str, int]:
        return {
            "prediction_chunk_steps": 8,
            "execution_steps": self.execution_steps,
            "action_calls": self.action_calls,
            "prediction_calls": self.prediction_calls,
        }
