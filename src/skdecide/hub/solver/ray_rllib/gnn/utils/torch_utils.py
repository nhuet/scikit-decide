from __future__ import annotations

from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from ray.rllib.utils.torch_utils import (
    convert_to_torch_tensor as convert_to_torch_tensor_original,
)
from ray.rllib.utils.typing import TensorStructType
from torch_geometric.data.dataset import IndexType

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    is_masked_obs,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    EDGE_LINKS,
    EDGES,
    NODES,
    convert_dict_to_graph,
    is_graph_dict,
    is_graph_dict_multiinput,
)
from skdecide.hub.solver.utils.gnn.torch_utils import (
    graph_instance_to_thg_data,
    torch_graph_tensors_to_thg_data,
)


def convert_to_torch_tensor(
    x: Union[
        TensorStructType,
        thg.data.Data,
        gym.spaces.GraphInstance,
        list[gym.spaces.GraphInstance],
    ],
    device: Optional[str] = None,
    pin_memory: bool = False,
    already_batched: bool = False,
) -> Union[TensorStructType, thg.data.Data]:
    """Converts any struct to torch.Tensors.

    Args:
        x: Any (possibly nested) struct, the values in which will be
            converted and returned as a new struct with all leaves converted
            to torch tensors.
        device: The device to create the tensor on.
        pin_memory: If True, will call the `pin_memory()` method on the created tensors.

    Returns:
        Any: A new struct with the same structure as `x`, but with all
        values converted to torch Tensor types. This does not convert possibly
        nested elements that are None because torch has no representation for that.
    """
    if isinstance(x, thg.data.Data):
        return x
    elif is_masked_obs(x):
        return {
            k: convert_to_torch_tensor(v, device=device, pin_memory=pin_memory)
            for k, v in x.items()
        }
    elif is_graph_dict(x):
        return batched_graph_dict_to_thg_data(x, device=device, pin_memory=pin_memory)
    elif is_graph_dict_multiinput(x):
        return {
            k: convert_to_torch_tensor(v, device=device, pin_memory=pin_memory)
            for k, v in x.items()
        }
    else:
        return convert_to_torch_tensor_original(
            x=x, device=device, pin_memory=pin_memory
        )


def batched_graph_dict_to_thg_data(
    batched_graph_dict: dict[str, np.ndarray],
    device: Optional[str] = None,
    pin_memory: bool = False,
):
    batch_size = batched_graph_dict[NODES].shape[0]
    return SliceableBatch.from_data_list(
        [
            graph_instance_to_thg_data(
                graph=convert_dict_to_graph(
                    {k: v[index, :] for k, v in batched_graph_dict.items()}
                ),
                device=device,
                pin_memory=pin_memory,
            )
            for index in range(batch_size)
        ]
    )


def batched_torch_graph_dict_to_thg_data(
    batched_graph_dict: dict[str, th.Tensor],
    device: Optional[str] = None,
    pin_memory: bool = False,
) -> thg.data.Data:
    return thg.data.Batch.from_data_list(
        [
            torch_graph_tensors_to_thg_data(
                *unpad_torch_graph_tensors(
                    nodes=batched_graph_dict[NODES][index, :],
                    edges=batched_graph_dict[EDGES][index, :],
                    edge_links=batched_graph_dict[EDGE_LINKS][index, :],
                ),
                device=device,
                pin_memory=pin_memory,
            )
            for index in range(len(batched_graph_dict[NODES]))
        ]
    )


def torch_graph_dict_to_thg_data(
    torch_graph_dict: dict[str, th.Tensor],
    device: Optional[str] = None,
    pin_memory: bool = False,
) -> thg.data.Data:
    return torch_graph_tensors_to_thg_data(
        nodes=torch_graph_dict[NODES],
        edges=torch_graph_dict[EDGES],
        edge_links=torch_graph_dict[EDGE_LINKS],
        device=device,
        pin_memory=pin_memory,
    )


def unpad_torch_graph_tensors(
    nodes: th.Tensor,
    edges: th.Tensor | None,
    edge_links: th.Tensor,
) -> tuple[th.Tensor, th.Tensor | None, th.Tensor]:
    # was padded?
    if (
        len(edge_links) > 0 and edge_links[-1, 1] < 0
    ):  # represents -n_nodes when padding
        # get actual number of nodes and edges
        n_nodes = -int(edge_links[-1, 1])
        n_edges = sum((edge_links >= 0).all(axis=1))
        # extract true nodes, edges, edge_links
        nodes = nodes[:n_nodes]
        if edges is not None:
            edges = edges[:n_edges]
        edge_links = edge_links[:n_edges]

    return (nodes, edges, edge_links)


class SliceableBatch(thg.data.Batch):
    """Subclass batched graph from torch_geometric with different slice support.

    When slicing the batched graph, instead of returning a list of graphs, another batched graph is returned.
    Useful for rllib minibatching.

    """

    def index_select(self, idx: IndexType) -> SliceableBatch:
        return SliceableBatch.from_data_list(super().index_select(idx))
