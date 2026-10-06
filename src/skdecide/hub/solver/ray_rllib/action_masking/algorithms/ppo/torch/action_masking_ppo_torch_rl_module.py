#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch.nn.functional import pad

from skdecide.hub.solver.ray_rllib.action_masking.rl_module.base import (
    ActionMaskingRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
)


class ActionMaskingPPOTorchRLModule(ActionMaskingRLModule, DefaultPPOTorchRLModule):
    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    def _forward(self, batch: dict[str, TensorType], **kwargs) -> dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self.preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    def _forward_train(
        self, batch: dict[str, TensorType], **kwargs
    ) -> dict[str, TensorType]:
        # Run the forward pass.
        outs = super()._forward_train(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, batch[ACTION_MASK])

    def compute_values(self, batch: dict[str, TensorType], embeddings=None):
        # Check, if the action_mask has been extracted from observations or not.
        if ACTION_MASK not in batch:
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            action_mask, batch = self.preprocess_batch(batch)
            # NOTE: Because we manipulate the batch we need to add the `action_mask`
            # to the batch to access them in `_forward_train`.
            batch[ACTION_MASK] = action_mask
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)

    def preprocess_batch(
        self, batch: dict[str, TensorType], **kwargs
    ) -> tuple[TensorType, dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)

        # Extract the available actions tensor from the observation.
        action_mask = batch[Columns.OBS].pop(ACTION_MASK)

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS].pop(TRUE_OBS)

        # Return the extracted action mask and the modified batch.
        return action_mask, batch

    def _mask_action_logits(
        self, batch: dict[str, TensorType], action_mask: TensorType
    ) -> dict[str, TensorType]:
        """Masks the action logits for the output of `forward` methods

        Args:
            batch: A dictionary containing tensors (at least action logits).
            action_mask: A tensor containing the action mask for the current
                observations.

        Returns:
            A modified batch with masked action logits for the action distribution
            inputs.
        """
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # Pad mask if necessary (may happen in graph2node context where logits can have been padded)
        nb_logits = batch[Columns.ACTION_DIST_INPUTS].shape[-1]
        mask_size = inf_mask.shape[-1]
        if mask_size < nb_logits:
            inf_mask = pad(inf_mask, pad=(0, nb_logits - mask_size), value=FLOAT_MIN)

        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask

        # Return the batch with the masked action logits.
        return batch

    def _check_batch(self, batch: dict[str, TensorType]) -> Optional[ValueError]:
        """Assert that the batch includes action mask and observations.

        Args:
            batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                checked.

        Raises:
            `ValueError` if the column `Columns.OBS`  does not contain observations
                and action mask.
        """
        if not self._checked_observations:
            if ACTION_MASK not in batch[Columns.OBS]:
                raise ValueError(
                    "No action mask found in observation. This `RLModule` requires "
                    "the environment to provide observations that include an "
                    "action mask (i.e. an observation space of the Dict space "
                    "type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': self.observation_space}"
                )
            if TRUE_OBS not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. This 'RLModule` requires "
                    "the environment to provide observations that include the original "
                    "observations under a key `'observations'` in a dict (i.e. an "
                    "observation space of the Dict space type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': <observation_space>}"
                )
            self._checked_observations = True
