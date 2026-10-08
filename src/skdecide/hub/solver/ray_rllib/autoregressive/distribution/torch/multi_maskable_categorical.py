#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch as th
from ray.rllib.core.distribution.distribution import Distribution


class TorchMultiMaskableCategorical(Distribution):
    ...

    def kl(self, other: TorchMultiMaskableCategorical, **kwargs) -> th.Tensor:
        """Compute KL divergence between self and q."""
        ...
