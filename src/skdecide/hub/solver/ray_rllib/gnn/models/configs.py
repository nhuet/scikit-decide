#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
from ray.rllib.core.models.configs import ModelConfig

from skdecide.hub.solver.ray_rllib.gnn.models.torch.encoder import (
    TorchGnnEncoder,
)


@dataclass(kw_only=True)
class GnnEncoderConfig(ModelConfig):
    """Configuration for GNN encoder."""

    observation_space: gym.spaces.Dict
    features_dim: int = 64
    features_extractor_kwargs: dict[str, Any] = field(
        default_factory=dict,
    )
    """Kwargs for `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor`."""

    def build(self, framework: str = "torch") -> TorchGnnEncoder:
        return TorchGnnEncoder(self)

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (self.features_dim,)
