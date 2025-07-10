from typing import Any, Callable, SupportsFloat, Union

import gymnasium as gym
import numpy as np
from gym.core import ActType, ObsType


class SupervisedActionWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    Env wrapper providing the method required to support supervised learning.

    Exposes a method called expected_actions(), which returns the expected actions
    from a plan to follow.

    :param env: the Gym environment to wrap
    :param plan: List of actions expected to be applied sequentially on the environment
    """

    def __init__(self, env: gym.Env[ObsType, ActType], plan: list[ActType]):
        super().__init__(env)
        self._plan = plan
        self._i_action = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._i_action = 0
        return super().reset(seed=seed, options=options)

    def expected_actions(self) -> ActType:
        return self._plan[self._i_action]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._i_action += 1
        if self._i_action >= len(self._plan):
            # no more actions in plan => should reset the env
            truncated = True
        return observation, reward, terminated, truncated, info
