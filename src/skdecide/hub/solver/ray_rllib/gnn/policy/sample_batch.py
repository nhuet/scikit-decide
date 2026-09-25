from __future__ import annotations

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.utils.typing import SampleBatchType

from skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch_original_code import (
    original_concat_samples,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    pad_axis,
    pad_sample_batches_obs,
)


def concat_samples_graph(samples: list[SampleBatchType]) -> SampleBatchType:
    # pad graph samples if necessary
    prepare_for_concat_samples_graph(samples)
    # concat samples as previously
    return original_concat_samples(samples)


def concat_samples_graph2node(samples: list[SampleBatchType]) -> SampleBatchType:
    # pad graph samples if necessary
    prepare_for_concat_samples_graph2node(samples)
    # concat samples as previously
    return original_concat_samples(samples)


def prepare_for_concat_samples_graph2node(samples: list[SampleBatchType]) -> None:
    if all(isinstance(s, SampleBatch) for s in samples) and all(
        s.get_interceptor is None for s in samples
    ):
        # prepare graphs obs
        prepare_for_concat_samples_graph(samples)
        # pad also action logits
        key = SampleBatch.ACTION_DIST_INPUTS
        if key in samples[0]:
            max_nodes = max(s[key].shape[1] for s in samples)
            minus_infty_approx = np.finfo(samples[0][key].dtype).min
            for s in samples:
                s[key] = pad_axis(
                    s[key], max_dim=max_nodes, axis=1, value=minus_infty_approx
                )


def prepare_for_concat_samples_graph(samples: list[SampleBatchType]) -> None:
    if (
        all(isinstance(s, SampleBatch) for s in samples)
        and all(s.get_interceptor is None for s in samples)
        and len(samples) > 0
    ):
        for key in (SampleBatch.OBS, SampleBatch.NEXT_OBS):
            if key in samples[0]:
                # pad the obs (check inside the function if necessary or not)
                pad_sample_batches_obs(samples, keys=(key,))
