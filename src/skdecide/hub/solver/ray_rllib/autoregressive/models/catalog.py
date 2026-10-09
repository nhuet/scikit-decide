#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import FreeLogStdMLPHeadConfig, MLPHeadConfig

from skdecide.hub.solver.ray_rllib.autoregressive.distribution.torch.multi_maskable_categorical import (
    TorchMultiMaskableCategorical,
)
from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK


class AutoregressiveCatalog(Catalog):
    action_space: gym.spaces.MultiDiscrete

    def get_action_dist_cls(self, framework: str) -> type[Distribution]:
        return get_action_dist_cls(framework=framework, action_space=self.action_space)

    def get_action_component_spaces(self) -> list[gym.spaces.Space]:
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return [
                gym.spaces.Discrete(n=n, start=start)
                for n, start in zip(self.action_space.nvec, self.action_space.start)
            ]
        else:
            raise NotImplementedError()

    def get_action_dist_cls_per_component(
        self, framework: str
    ) -> list[type[Distribution]]:
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return [
                self._get_dist_cls_from_action_space(
                    action_space=action_component_space, framework=framework
                )
                for action_component_space in self.get_action_component_spaces()
            ]
        else:
            raise NotImplementedError()

    def build_action_encoder_per_component(self, framework: str) -> list[Model]:
        self.action_encoder_per_component_configs = [
            self._get_encoder_config(
                observation_space=action_component_space,
                model_config_dict=self._model_config_dict,
                action_space=self.action_space,
            )
            for action_component_space in self.get_action_component_spaces()
        ]
        return [
            config.build(framework=framework)
            for config in self.action_component_encoder_configs
        ]

    def build_pi_head_per_component(self, framework: str) -> list[Model]:
        """Builds the policy head for each action component.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The policy heads
            .
        """
        # Get action_distribution_cls to find out about the output dimension for pi_head
        action_distribution_cls_per_component = self.get_action_dist_cls_per_component(
            framework=framework
        )
        pi_heads = []
        self.pi_head_configs = []
        # first component using only observation features
        input_dims = self.latent_dims
        # hyp: features are flattened
        assert len(input_dims) == 1, "observation features are supposed to be flattened"
        for i_component, action_distribution_cls_component in enumerate(
            action_distribution_cls_per_component
        ):
            if self._model_config_dict["free_log_std"]:
                _check_if_diag_gaussian(
                    action_distribution_cls=action_distribution_cls_component,
                    framework=framework,
                )
                is_diag_gaussian = True
            else:
                is_diag_gaussian = _check_if_diag_gaussian(
                    action_distribution_cls=action_distribution_cls_component,
                    framework=framework,
                    no_error=True,
                )

            pi_head_config_class = (
                FreeLogStdMLPHeadConfig
                if self._model_config_dict["free_log_std"]
                else MLPHeadConfig
            )
            required_output_dim = action_distribution_cls_component.required_input_dim(
                space=self.action_space, model_config=self._model_config_dict
            )
            pi_head_config = pi_head_config_class(
                input_dims=input_dims,
                hidden_layer_dims=self._model_config_dict["head_fcnet_hiddens"],
                hidden_layer_activation=self._model_config_dict[
                    "head_fcnet_activation"
                ],
                hidden_layer_use_layernorm=self._model_config_dict.get(
                    "head_fcnet_use_layernorm", False
                ),
                output_layer_dim=required_output_dim,
                output_layer_activation="linear",
                clip_log_std=is_diag_gaussian,
                log_std_clip_param=self._model_config_dict.get(
                    "log_std_clip_param", 20
                ),
            )
            pi_heads.append(pi_head_config.build(framework=framework))
            self.pi_head_configs.append(pi_head_config)
            # next component head will use also this encoded component
            encoded_component_dims = self.action_encoder_per_component_configs[
                i_component
            ].output_dims
            assert len(encoded_component_dims) == 1, (
                "encoded action components are supposed to be flattened"
            )
            input_dims = (input_dims[0] + encoded_component_dims[0],)

        return pi_heads


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
