#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from ray.rllib.connectors.env_to_module.observation_preprocessor import (
    MultiAgentObservationPreprocessor,
)
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.numpy import flatten_inputs_to_1d_tensor
from ray.rllib.utils.spaces.repeated import Repeated

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    DEFAULT_N_EDGES,
    DEFAULT_N_NODES,
    EDGE_LINKS,
    EDGES,
    NODES,
)


class FlattenMultiagentGraphObservations(MultiAgentObservationPreprocessor):
    """A connector piece that flattens node and edge features into a 1D arrays.

    It assumes that observation structure is as  follows:
    ```
    obs = {
        agent_id: {
            "nodes": node features
            "edges": edge features
            "edge_links": list of connected nodes
        },
    }
    ```
    where `action_mask` is the mask already flattened and `true_obs` is the original observation.

    """

    def __init__(
        self,
        input_observation_space: Optional[gym.spaces.Dict] = None,
        input_action_space: Optional[gym.Space] = None,
        **kwargs,
    ):
        super().__init__(input_observation_space, input_action_space, **kwargs)

    def preprocess(
        self, observations: dict[str, Any], episode: MultiAgentEpisode
    ) -> dict[str, Any]:
        return {
            agent: {
                NODES: flatten_inputs_to_1d_tensor(
                    inputs=agent_observation[NODES],
                    spaces_struct=self.input_observation_space[agent][
                        NODES
                    ].child_space,
                    # The node id dim is like a batch axis
                    batch_axis=True,
                ),
                EDGES: flatten_inputs_to_1d_tensor(
                    inputs=edges,
                    spaces_struct=self.input_observation_space[agent][
                        EDGES
                    ].child_space,
                    # The edge id dim is like a batch axis
                    batch_axis=True,
                )
                if (edges := agent_observation[EDGES])
                else np.empty(
                    (
                        edge_output_space := self.input_observation_space[agent][
                            EDGES
                        ].child_space
                    ).shape,
                    dtype=edge_output_space.dtype,
                ),
                EDGE_LINKS: agent_observation[EDGE_LINKS],
            }
            for agent, agent_observation in observations.items()
        }

    def recompute_output_observation_space(
        self, input_observation_space: gym.Space, input_action_space: gym.Space
    ) -> gym.Space:
        return gym.spaces.Dict(
            {
                agent_id: gym.spaces.Dict(
                    {
                        NODES: Repeated(
                            _flatten_space(space[NODES].child_space),
                            max_len=DEFAULT_N_NODES,
                        ),
                        EDGES: Repeated(
                            _flatten_space(space[EDGES].child_space),
                            max_len=DEFAULT_N_EDGES,
                        ),
                        EDGE_LINKS: space[EDGE_LINKS],
                    }
                )
                for agent_id, space in input_observation_space.items()
            }
        )


def _flatten_space(space: gym.spaces.Box | gym.spaces.Discrete) -> gym.spaces.Box:
    if isinstance(space, gym.spaces.Box):
        low = space.low.min()
        high = space.high.min()
    elif isinstance(space, gym.spaces.Discrete):
        low = 0
        high = 1
    else:
        raise NotImplementedError()
    sample = flatten_inputs_to_1d_tensor(space.sample(), space, batch_axis=False)
    return gym.spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)
