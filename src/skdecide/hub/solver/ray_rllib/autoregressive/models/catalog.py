#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import gymnasium as gym
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.models.catalog import Catalog

from skdecide.hub.solver.ray_rllib.autoregressive.distribution.torch.multi_maskable_categorical import (
    TorchMultiMaskableCategorical,
)
from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK


class AutoregressiveCatalog(Catalog):
    def get_action_dist_cls(self, framework: str) -> type[Distribution]:
        return get_action_dist_cls(framework=framework, action_space=self.action_space)

    def get_action_dist_cls_per_component(
        self, framework: str
    ) -> list[type[Distribution]]: ...


def get_action_dist_cls(framework: str, action_space: gym.Space) -> type[Distribution]:
    if framework == TORCH_FRAMEWORK:
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            return TorchMultiMaskableCategorical
        else:
            raise ValueError(
                "action space is supposed to be multidiscrete for autoregressive actions."
            )
    else:
        raise NotImplementedError()
