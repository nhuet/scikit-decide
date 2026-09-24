#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional

from ray.rllib.connectors.common import NumpyToTensor
from ray.rllib.core import DEFAULT_MODULE_ID, Columns
from ray.rllib.core.rl_module import MultiRLModule, RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType

from skdecide.hub.solver.ray_rllib.gnn.utils.torch_utils import convert_to_torch_tensor


class GraphNumpyToTensor(NumpyToTensor):
    """Converts gymnasium graph instance into torch geometric data.

    Other numpy arrays are converted to torch tensors via ray.rllib `convert_to_torch_tensor()`.

    """

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:
        is_single_agent = False
        is_multi_rl_module = isinstance(rl_module, MultiRLModule)
        # `data` already a ModuleID to batch mapping format.
        if not (is_multi_rl_module and all(c in rl_module._rl_modules for c in batch)):
            is_single_agent = True
            batch = {DEFAULT_MODULE_ID: batch}

        for module_id, module_data in batch.copy().items():
            # If `rl_module` is None, leave data in numpy format.
            if rl_module is not None:
                infos = module_data.pop(Columns.INFOS, None)
                if rl_module.framework == "torch":
                    module_data = {
                        key: convert_to_torch_tensor(
                            data, pin_memory=self._pin_memory, device=self._device
                        )
                        for key, data in module_data.items()
                    }
                else:
                    raise ValueError(
                        f"`{self.__class__.__name__}` does NOT support frameworks other than torch! "
                        f"Your current framework is {rl_module.framework}"
                    )
                if infos is not None:
                    module_data[Columns.INFOS] = infos

            # Early out with data under(!) `DEFAULT_MODULE_ID`, b/c we are in plain
            # single-agent mode.
            if is_single_agent:
                return module_data
            batch[module_id] = module_data

        return batch
