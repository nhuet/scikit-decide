#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_LOGITS, QF_PREDS, QF_PROBS
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
    DefaultDQNTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingRLModule,
)
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import nn

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
)


class ActionMaskingDQNTorchRLModule(ActionMaskingRLModule, DefaultDQNTorchRLModule):
    """Custom RL module for DQN + masking."""

    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s (`PPOTorchRLModule`) observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    def _preprocess_batch(
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
        self,
        batch: dict[str, TensorType],
        action_mask: TensorType,
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

        # Mask the logits.
        for column in (QF_PREDS, QF_LOGITS, Columns.ACTION_DIST_INPUTS):
            if column in batch:
                batch[column] += inf_mask

        if QF_PROBS in batch:
            if QF_LOGITS in batch:
                batch[QF_PROBS] = nn.functional.softmax(batch[QF_LOGITS], dim=-1)
            else:
                raise RuntimeError()

        # Return the batch with the masked action logits.
        return batch

    def _check_batch(self, batch: dict[str, TensorType]) -> None:
        """Assert that the batch includes action mask and observations.

        Args:
            batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                checked.

        Raises:
            `ValueError` if the column `Columns.OBS`  does not contain observations
                and action mask.
        """
        if not self._checked_observations:
            if "action_mask" not in batch[Columns.OBS]:
                raise ValueError(
                    "No action mask found in observation. This `RLModule` requires "
                    "the environment to provide observations that include an "
                    "action mask (i.e. an observation space of the Dict space "
                    "type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': self.observation_space}"
                )
            if "observations" not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. This 'RLModule` requires "
                    "the environment to provide observations that include the original "
                    "observations under a key `'observations'` in a dict (i.e. an "
                    "observation space of the Dict space type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': <observation_space>}"
                )
            self._checked_observations = True

    def _qf_forward_helper(
        self,
        batch: dict[str, TensorType],
        encoder: Encoder,
        head: Model | dict[str, Model],
    ) -> dict[str, TensorType]:
        # Check, if the observations are still in `dict` form.
        # if isinstance(batch[Columns.OBS], dict):
        action_mask, batch = self._preprocess_batch(batch)
        # # NOTE: Because we manipulate the batch we need to add the `action_mask`
        # # to the batch to access them in `_forward_train`.
        # batch["action_mask"] = action_mask
        outs = super()._qf_forward_helper(batch, encoder, head)
        return self._mask_action_logits(outs, action_mask)

    def _forward_exploration(
        self, batch: dict[str, TensorType], t: int
    ) -> dict[str, TensorType]:
        """Forward pass during exploration.

        We need to override as a multinomial on Q-values is taken, originally excluding *0* valued logits.
        But after masking, we should have FLOAT_MIN (approximating -inf) values excluded instead.

        """
        # Define the return dictionary.
        output = {}

        # Q-network forward pass.
        qf_outs = self.compute_q_values(batch)

        # Get action distribution.
        action_dist_cls = self.get_exploration_action_dist_cls()
        action_dist = action_dist_cls.from_logits(qf_outs[QF_PREDS])
        # Note, the deterministic version of the categorical distribution
        # outputs directly the `argmax` of the logits.
        exploit_actions = action_dist.to_deterministic().sample()

        # We need epsilon greedy to support exploration.
        # TODO (simon): Implement sampling for nested spaces.
        # Update scheduler.
        self.epsilon_schedule.update(t)
        # Get the actual epsilon,
        epsilon = self.epsilon_schedule.get_current_value()
        # Apply epsilon-greedy exploration.
        B = qf_outs[QF_PREDS].shape[0]
        random_actions = torch.squeeze(
            torch.multinomial(
                (
                    torch.nan_to_num(
                        qf_outs[QF_PREDS].reshape(-1, qf_outs[QF_PREDS].size(-1)),
                        neginf=FLOAT_MIN,
                    )
                    != FLOAT_MIN
                ).float(),
                num_samples=1,
            ),
            dim=1,
        )

        actions = torch.where(
            torch.rand((B,)) < epsilon,
            random_actions,
            exploit_actions,
        )

        # Add the actions to the return dictionary.
        output[Columns.ACTIONS] = actions

        # If this is a stateful module, add output states.
        if Columns.STATE_OUT in qf_outs:
            output[Columns.STATE_OUT] = qf_outs[Columns.STATE_OUT]

        return output
