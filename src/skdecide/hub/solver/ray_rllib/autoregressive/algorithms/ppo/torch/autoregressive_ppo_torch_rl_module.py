#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
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
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    APPLICABLE_ACTIONS,
)
from skdecide.hub.solver.ray_rllib.common.constants import ForwardPhase
from skdecide.hub.solver.utils.autoregressive.torch_utils import (
    batched_extract_action_component_mask,
    extract_action_component_mask,
)


class AutoregressivePPOTorchRLModule(
    AutoregressiveTorchRLModule, DefaultPPOTorchRLModule
):
    catalog: AutoregressivePPOCatalog
    action_space: gym.spaces.MultiDiscrete

    def setup(self):
        # reset observation space to be the actual one for this module
        self.observation_space = self.observation_space_with_applicable_actions

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

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, phase=ForwardPhase.EXPLORATION, **kwargs)

    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, phase=ForwardPhase.INFERENCE, **kwargs)

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, phase=ForwardPhase.TRAIN, **kwargs)

    def _forward(
        self,
        batch: dict[str, TensorType],
        phase: ForwardPhase = ForwardPhase.INFERENCE,
        **kwargs,
    ) -> dict[str, TensorType]:
        # Check, if the action_mask has been extracted from observations or not.
        # (e.g. in forward_train(), it can already have been extracted by compute_values())
        if APPLICABLE_ACTIONS not in batch:
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            applicable_actions, batch = self.preprocess_batch(batch)
            batch[APPLICABLE_ACTIONS] = applicable_actions
        # Pass to appropriate method
        if phase == ForwardPhase.TRAIN:
            return self._forward_with_already_sampled_actions(
                batch=batch, phase=phase, **kwargs
            )
        else:
            return self._forward_with_action_sampling(
                batch=batch, phase=phase, **kwargs
            )

    def _forward_with_action_sampling(
        self, batch: Dict[str, Any], phase: ForwardPhase, **kwargs
    ) -> Dict[str, Any]:
        deterministic = phase == ForwardPhase.INFERENCE
        output = {}

        # extract applicable actions
        applicable_actions = batch[APPLICABLE_ACTIONS]
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
        action_components = th.as_tensor([], device=features.device, dtype=int)
        action_component_masks = []
        action_dist_cls_per_component = self.get_action_dist_cls_per_component(
            phase=phase
        )
        action_dist_per_components = []
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

            action_component_mask = action_component_mask[None]  # add batch dimension
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
            action_dist_per_components.append(action_dist_component)
            if deterministic:
                action_component = action_dist_component.to_deterministic().sample()
            else:
                action_component = action_dist_component.sample()
            action_components = th.cat((action_components, action_component), -1)

            # add encoded action component to features for next component logits
            encoded_action_component = self.action_encoder_per_component[
                i_action_component
            ]({Columns.OBS: action_component})[ENCODER_OUT]
            features = th.cat((features, encoded_action_component), dim=-1)

        actions = action_components[None]  # add batch dimension
        action_dist_cls: AutoregressiveDistribution = self.get_action_dist_cls(
            phase=phase
        )
        action_dist = action_dist_cls(
            action_dist_per_components, masks=action_component_masks
        )

        output[Columns.ACTION_LOGP] = action_dist.logp(actions)
        output[Columns.ACTIONS] = pad(
            # pad actions with -1 in case of early break of the loop (no more components)
            actions,
            pad=(0, self.nb_action_components - actions.shape[-1]),
            value=-1,  # -1 = no component
        )
        # pad action dist inputs with dummy logits + masks to have max number of components (helps storage)
        for i_action_component in range(
            len(action_component_masks), self.nb_action_components
        ):
            action_component_dim = self.action_space.nvec[i_action_component]
            action_dist_inputs_per_component.append(
                th.zeros(
                    (1, action_component_dim),
                    device=features.device,
                    dtype=features.dtype,
                )
            )
            action_component_masks.append(
                th.zeros(
                    (1, action_component_dim),
                    device=features.device,
                    dtype=features.dtype,
                )
            )
        action_dist_inputs = (
            tuple(action_dist_inputs_per_component),
            tuple(action_component_masks),
        )
        output[Columns.ACTION_DIST_INPUTS] = action_dist_inputs

        return output

    def _forward_with_already_sampled_actions(
        self, batch: Dict[str, Any], phase: ForwardPhase, **kwargs
    ) -> Dict[str, Any]:
        assert phase == ForwardPhase.TRAIN

        output = {}

        # extract applicable actions
        applicable_actions = batch[APPLICABLE_ACTIONS]

        # extract sampled actions
        actions = batch[Columns.ACTIONS]

        # Encode observations into features
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT][ACTOR]
        # hyp: features are flat
        assert len(features.shape) == 2
        # save encoding for compute_values()
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]

        # batch_dim
        batch_dim = len(features)
        assert len(applicable_actions == batch_dim)

        # full features: extracted features + encoded action components (all at once for efficiency)
        full_features = th.cat(
            (features,)
            + tuple(
                self.action_encoder_per_component[i_action_component](
                    {Columns.OBS: actions[:, i_action_component]}
                )[ENCODER_OUT]
                for i_action_component in range(actions.shape[1])
            ),
            dim=-1,
        )
        n_features_by_component = features.shape[-1] + np.cumsum(
            [0]
            + list(
                config.output_dims[0]
                for config in self.catalog.action_encoder_per_component_configs
            )
        )

        action_dist_inputs_per_component = []
        action_component_masks = []
        for i_action_component, pi in enumerate(self.pi_per_component):
            # all (batchwise) components for this idx are <0 => no more true components
            if (actions[:, i_action_component] < 0).all():
                # no more computation needed
                break

            # compute logits only for samples with >=0 component
            mask_samples_with_component = actions[:, i_action_component] >= 0
            if mask_samples_with_component.all():
                # all (batchwise) components needed
                compute_all_samples = True
                indices_samples_with_component = None
                mask_samples_with_component = None
            else:
                # some components do need to be computed
                compute_all_samples = False
                indices_samples_with_component = (
                    mask_samples_with_component.nonzero().flatten()
                )

            # mask for current action component considered
            action_component_mask = batched_extract_action_component_mask(
                action_components=actions,
                applicable_actions=applicable_actions,
                i_action_component=i_action_component,
                action_component_dim=self.action_space.nvec[i_action_component],
                nonzeromask_samples_mask=mask_samples_with_component,
            )
            action_component_masks.append(action_component_mask)

            # compute logits only for relevant samples
            if compute_all_samples:
                # all samples are relevant
                action_component_full_logits = pi(
                    full_features[:, : n_features_by_component[i_action_component]]
                )
            else:
                # only some samples are relevant
                action_component_logits = pi(
                    full_features[
                        indices_samples_with_component,
                        : n_features_by_component[i_action_component],
                    ]
                )
                # fill logits with 0 for irrelevant samples  (will be masked afterwards)
                action_component_full_logits = th.zeros_like(
                    action_component_mask, dtype=action_component_logits.dtype
                )
                action_component_full_logits[indices_samples_with_component, :] = (
                    action_component_logits
                )
            # mask logits
            action_component_full_logits = mask_categorical_logits(
                logits=action_component_full_logits, mask=action_component_mask
            )
            action_dist_inputs_per_component.append(action_component_full_logits)

        # action_dist_inputs
        action_dist_inputs = (
            tuple(action_dist_inputs_per_component),
            tuple(action_component_masks),
        )
        output[Columns.ACTION_DIST_INPUTS] = action_dist_inputs

        # no need to add actions and logp to output not used by forward_train output

        return output

    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        # Check, if the action_mask has been extracted from observations or not.
        if APPLICABLE_ACTIONS not in batch:
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            applicable_actions, batch = self.preprocess_batch(batch)
            batch[APPLICABLE_ACTIONS] = applicable_actions
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)
