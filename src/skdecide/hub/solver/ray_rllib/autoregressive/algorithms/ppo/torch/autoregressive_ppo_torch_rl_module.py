#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Dict

import gymnasium as gym
import torch as th
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from torch import TensorType
from torch.nn import ModuleList
from torch.nn.functional import pad

from skdecide.hub.solver.ray_rllib.action_masking.distribution.mask import (
    mask_categorical_logits,
)
from skdecide.hub.solver.ray_rllib.autoregressive.algorithms.ppo.ppo_catalog import (
    AutoregressivePPOCatalog,
)
from skdecide.hub.solver.ray_rllib.autoregressive.distribution.autoregressive import (
    AutoregressiveDistribution,
)
from skdecide.hub.solver.ray_rllib.autoregressive.rl_module.torch.autoregressive_torch_rl_module import (
    AutoregressiveTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.common.constants import FORWARD_PHASE
from skdecide.hub.solver.utils.autoregressive.torch_utils import (
    extract_action_component_mask,
)


class AutoregressivePPOTorchRLModule(
    AutoregressiveTorchRLModule, DefaultPPOTorchRLModule
):
    catalog: AutoregressivePPOCatalog
    action_space: gym.spaces.MultiDiscrete

    def setup(self):
        is_stateful = isinstance(
            self.catalog.actor_critic_encoder_config.base_encoder_config,
            RecurrentEncoderConfig,
        )
        if is_stateful:
            self.inference_only = False
        # If this is an `inference_only` Module, we'll have to pass this information
        # to the encoder config as well.
        if self.inference_only and self.framework == "torch":
            self.catalog.actor_critic_encoder_config.inference_only = True

        # Build models from catalog.
        self.encoder = self.catalog.build_actor_critic_encoder(framework=self.framework)
        self.vf = self.catalog.build_vf_head(framework=self.framework)
        self.action_encoder_per_component = ModuleList(
            self.catalog.build_action_encoder_per_component(framework=self.framework)
        )
        self.pi_per_component = ModuleList(
            self.catalog.build_pi_head_per_component(framework=self.framework)
        )

    def _forward(
        self, batch: dict[str, TensorType], phase: FORWARD_PHASE, **kwargs
    ) -> dict[str, TensorType]:
        deterministic = phase == FORWARD_PHASE.INFERENCE
        output = {}
        # Preprocess the original batch to extract the applicable actions.
        applicable_actions, batch = self.preprocess_batch(batch)

        # Encode observations into features
        encoder_outs = self.encoder(batch)
        if phase == FORWARD_PHASE.TRAIN:
            output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]
        features = encoder_outs[ENCODER_OUT][ACTOR]
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        # pi heads for each component
        action_dist_inputs_per_component = []
        action_dist_cls_per_component = self.get_action_dist_cls_per_component(
            phase=phase
        )
        for i_component, pi in enumerate(self.pi_per_component):
            logits_component = self.pi_per_component[i_component](features)
            action_dist_inputs_per_component.append(logits_component)
            # add sampled and encoded action component to features for next component logits
            action_dist_component = action_dist_cls_per_component[
                i_component
            ].from_logits(logits_component)
            if deterministic:
                action_dist_component = action_dist_component.to_deterministic()
            action_component = action_dist_component.sample()
            encoded_action_component = self.action_encoder_per_component[i_component](
                {Columns.OBS: action_component}
            )[ENCODER_OUT]
            features = th.cat((features, encoded_action_component), dim=-1)

        return output

    def _forward_with_action_sampling(
        self, batch: Dict[str, Any], phase: FORWARD_PHASE, **kwargs
    ) -> Dict[str, Any]:
        deterministic = phase == FORWARD_PHASE.INFERENCE
        output = {}

        # extract applicable actions from observations
        applicable_actions, batch = self.preprocess_batch(batch)
        # check that batch shape is (1,) and remove this dimension
        assert len(applicable_actions.shape) == 3 and len(applicable_actions) == 1, (
            "applicable_actions are supposed to be batched with only one sample"
        )
        applicable_actions = applicable_actions.reshape(
            (-1, len(self.action_space.nvec))
        )  # remove batch dimension

        # Encode observations into features
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT][ACTOR]
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        # pi heads for each component
        action_dist_inputs_per_component = []
        action_components = th.as_tensor([], device=self.device, dtype=int)
        action_component_masks = []
        action_dist_cls_per_component = self.get_action_dist_cls_per_component(
            phase=phase
        )
        action_dist_components_for_logp = []
        for i_action_component, pi in enumerate(self.pi_per_component):
            # mask for current action component considered
            action_component_mask = extract_action_component_mask(
                action_components=action_components,
                applicable_actions=applicable_actions,
                i_action_component=i_action_component,
                action_component_dim=self.action_space.nvec[i_action_component],
            )

            # more action_component allowed?
            if action_component_mask.sum() == 0:
                # no more components available
                break

            action_component_masks.append(action_component_mask)

            # compute component logits
            logits_component = pi(features)
            logits_component = mask_categorical_logits(
                logits=logits_component, mask=action_component_mask
            )
            action_dist_inputs_per_component.append(logits_component)

            # sample action component
            action_dist_component = action_dist_cls_per_component[
                i_action_component
            ].from_logits(logits_component)
            action_dist_components_for_logp.append(
                action_dist_component
            )  # store before deterministic to compute logp
            if deterministic:
                action_dist_component = action_dist_component.to_deterministic()
            action_component = action_dist_component.sample()
            action_components = th.cat((action_components, action_component), -1)

            # add encoded action component to features for next component logits
            encoded_action_component = self.action_encoder_per_component[
                i_action_component
            ]({Columns.OBS: action_component})[ENCODER_OUT]
            features = th.cat((features, encoded_action_component), dim=-1)

        actions = action_components[None]  # add batch dimension
        action_dist_inputs = (
            tuple(action_dist_inputs_per_component),
            tuple(action_component_masks),
        )
        action_dist_cls: AutoregressiveDistribution = self.get_action_dist_cls(
            phase=phase
        )
        action_dist = action_dist_cls(
            action_dist_components_for_logp, masks=action_component_masks
        )

        output[Columns.ACTION_DIST_INPUTS] = action_dist_inputs
        output[Columns.ACTION_LOGP] = action_dist.logp(actions)
        output[Columns.ACTIONS] = pad(
            # pad actions with -1 in case of early break of the loop (no more components)
            actions,
            pad=(0, len(self.action_dist.distributions) - action_components.shape[-1]),
            value=-1,  # -1 = no component
        )

        return output

    def _forward_with_already_sampled_actions(
        self, batch: Dict[str, Any], applicable_actions: th.Tensor, **kwargs
    ) -> Dict[str, Any]: ...
