from __future__ import annotations

from typing import Any

import gymnasium as gym
from ray.rllib.utils.spaces.repeated import Repeated

from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import (
    get_item,
    pad_axis,
    set_item,
)

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


def pad_sample_batches_applicable_actions(
    samples: list[dict[str, Any]], keys: tuple[str, ...], has_batch_dim: bool = True
) -> None:
    applicable_action_id_dim = 1 if has_batch_dim else 0
    node_edge_id_dim = applicable_action_id_dim + 1
    n_nodes_per_sample = tuple(
        get_item(s, keys).shape[node_edge_id_dim] for s in samples
    )
    n_applicable_actions_per_sample = tuple(
        get_item(s, keys).shape[applicable_action_id_dim] for s in samples
    )
    if (
        len(set(n_nodes_per_sample)) > 1
        or len(set(n_applicable_actions_per_sample)) > 1
    ):
        # different number of nodes and/or number applicable actions => padding
        max_n_nodes = max(n_nodes_per_sample)
        max_n_applicable_actions = max(n_applicable_actions_per_sample)
        for s in samples:
            applicable_actions = get_item(s, keys)
            set_item(
                s,
                keys=keys,
                value=pad_axis(
                    pad_axis(
                        applicable_actions,
                        max_n_nodes,
                        axis=node_edge_id_dim,
                        value=-1,  # negative component -> invalid component
                    ),
                    max_n_applicable_actions,
                    axis=applicable_action_id_dim,
                    value=-1,  # negative values => invalid action
                ).view(
                    type(applicable_actions)  # keep BatchedNDArray wrapper if present
                ),
            )
