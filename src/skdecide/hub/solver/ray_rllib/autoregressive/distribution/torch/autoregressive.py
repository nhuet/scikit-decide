#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Tuple, Union

import gymnasium as gym
import torch as th
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.autoregressive.distribution.autoregressive import (
    AutoregressiveDistribution,
)


class TorchAutoregressiveDistribution(AutoregressiveDistribution):
    """Distribution for variable-length parameteric actions.

    This is meant for autoregressive prediction.

    The distribution is considered as the joint distribution of "marginal" distributions conditioned to previous components
    with the possibility to mask each marginal.
    This distribution is meant to be used for autoregressive action:
    - Each component is sampled sequentially
    - The partial mask for the next component is conditioned by the previous components
    - It is possible to have missing components when this has no meaning for the action.
      this corresponds in the simulation to
      - either not initialized marginal (if all samples discard the component)
      - 0 masks for the given sample (the partial mask row corresponding to the sample has only 0's), for maskable versions

    When computing entropy of the distribution or log-probability of an action, we add only contribution
    of marginal distributions for which we have an actual component (dropping the one with a 0-mask).

    As this distribution is used to sample component by component, the sample(), and mode() methods are left
    unimplemented.

    """

    marginal_distribution_cls = TorchCategorical
    distributions: tuple[TorchCategorical, ...]

    def get_proba_distribution_component_batch_shape(
        self, i_component: int
    ) -> tuple[int, ...]:
        distribution: TorchCategorical = self.distributions[i_component]
        distribution.logits
        if distribution.distribution is None:
            return None
        else:
            return distribution.distribution.logits.shape[:-1]

    def get_proba_distribution_component_for_valid_samples(
        self, i_component: int
    ) -> Optional[MaskableCategorical]:
        if not (self._any_valid_samples_by_distributions[i_component]):
            return None
        elif self.all_valid_samples_by_distributions[i_component]:
            return self.distributions[i_component]
        else:
            distribution = self.distributions[i_component]
            ind_valid_samples = self.ind_valid_samples_by_distributions[i_component]
            return TorchCategorical(
                logits=distribution.logits[ind_valid_samples],
            )

    def logp(self, value: TensorType, **kwargs) -> TensorType:
        marginal_logps = []
        # loop over marginals but no contribution if not initialized or 0-masked
        for i_component, distribution in enumerate(self.distributions):
            marginal_dist = self.get_proba_distribution_component_for_valid_samples(
                i_component
            )
            if marginal_dist is not None:
                if self.all_valid_samples_by_distributions[i_component]:
                    marginal_logp = marginal_dist.log_prob(x[:, i_component])
                else:
                    # add only contribution for valid samples
                    ind_valid_samples = self.ind_valid_samples_by_distributions[
                        i_component
                    ]
                    marginal_logp_valid_samples = marginal_dist.log_prob(
                        x[ind_valid_samples, i_component]
                    )
                    marginal_logp = th.zeros(
                        distribution.logits.shape[:-1],
                        device=marginal_logp_valid_samples.device,
                        dtype=marginal_logp_valid_samples.dtype,
                    )
                    marginal_logp.scatter_(
                        dim=0,
                        index=ind_valid_samples,
                        src=marginal_logp_valid_samples,
                    )
                marginal_logps.append(marginal_logp)

        return sum(marginal_logps)

    def kl(self, other: "Distribution", **kwargs) -> TensorType:
        pass

    def entropy(self, **kwargs) -> TensorType:
        pass

    def sample(
        self,
        *,
        sample_shape: Tuple[int, ...] = None,
        return_logp: bool = False,
        **kwargs,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        raise NotImplementedError()

    def rsample(
        self,
        *,
        sample_shape: Tuple[int, ...] = None,
        return_logp: bool = False,
        **kwargs,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        raise NotImplementedError()

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        raise NotImplementedError()


class TorchMaskedCategorical(TorchCategorical):
    def __init__(
        self,
        logits: th.Tensor | None = None,
        probs: th.Tensor | None = None,
        mask: th.Tensor | None = None,
        logits_already_masked: bool = True,
    ) -> None:
        super().__init__(logits, probs)
        self.mask = mask
        if mask is not None and not logits_already_masked:
            # todo mask logits
            raise NotImplementedError()

    def entropy(self) -> TensorType:
        if self.mask is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        min_real = th.finfo(self.logits.dtype).min
        logits = th.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        p_log_p = th.where(self.mask, p_log_p, th.tensor(0.0, device=device))
        return -p_log_p.sum(-1)
