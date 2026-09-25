#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import ModelConfig

from skdecide.hub.solver.ray_rllib.gnn.models.configs import (
    GnnEncoderConfig,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    is_graph_dict_space,
)


class GraphCatalog(Catalog):
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space | None = None,
    ) -> ModelConfig:
        features_extractor_kwargs = model_config_dict.get(
            "features_extractor_kwargs", {}
        )
        if is_graph_dict_space(observation_space):
            return GnnEncoderConfig(
                observation_space=observation_space,
                features_extractor_kwargs=features_extractor_kwargs,
            )
        else:
            return super()._get_encoder_config(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
                action_space=action_space,
            )
