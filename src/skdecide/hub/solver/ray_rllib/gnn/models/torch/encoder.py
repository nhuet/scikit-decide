#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from ray.rllib.core import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.torch.base import TorchModel

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    convert_dict_space_to_graph_space,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.torch_utils import (
    batched_torch_graph_dict_to_thg_data,
)
from skdecide.hub.solver.utils.gnn.torch_layers import GraphFeaturesExtractor

if TYPE_CHECKING:
    from skdecide.hub.solver.ray_rllib.gnn.models.configs import (
        GnnEncoderConfig,
    )


class TorchGnnEncoder(TorchModel, Encoder):
    config: GnnEncoderConfig

    def __init__(self, config: GnnEncoderConfig):
        super().__init__(config)
        observation_space = convert_dict_space_to_graph_space(
            self.config.observation_space
        )
        self.extractor = GraphFeaturesExtractor(
            observation_space=observation_space,
            features_dim=self.config.features_dim,
            **self.config.features_extractor_kwargs,
        )

    def _forward(self, input_dict: dict, **kwargs) -> dict:
        observations = input_dict[Columns.OBS]
        graph_observations = batched_torch_graph_dict_to_thg_data(observations)
        return {ENCODER_OUT: self.extractor.forward(observations=graph_observations)}
