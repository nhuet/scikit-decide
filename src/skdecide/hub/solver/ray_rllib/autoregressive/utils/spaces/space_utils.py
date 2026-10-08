from typing import Any

import gymnasium as gym
from ray.rllib.utils.spaces.repeated import Repeated

TRUE_OBS = "observations"
APPLICABLE_ACTIONS = "applicable_actions"

DEFAULT_N_APPLICABLE_ACTIONS = 2
"""Initial applicable actions number to be used by dummy samples generation by rllib"""


def create_agent_applicable_actions_space(agent_action_space: gym.spaces.MultiDiscrete):
    return Repeated(
        child_space=agent_action_space,
        max_len=DEFAULT_N_APPLICABLE_ACTIONS,
    )


def is_applicable_actions_obs(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and len(x) == 2
        and TRUE_OBS in x
        and APPLICABLE_ACTIONS in x
    )


def is_applicable_actions_obs_space(x: gym.spaces.Space) -> bool:
    return (
        isinstance(x, gym.spaces.Dict)
        and len(x.spaces) == 2
        and TRUE_OBS in x.spaces
        and APPLICABLE_ACTIONS in x.spaces
    )
