from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from copy import deepcopy

EMPTY_INT = -99999
EMPTY_FLOAT = -99999.0

import torch
import numpy as np
import awkward as ak

# def expand_columns(columns: Container[str]) -> list[Route]:
#     final_set = set()
#     final_set.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))
#     return sorted(final_set, key=str)

def normalized_weight_decay(
    model: torch.nn.Module,
    decay_factor: float = 1e-1,
    normalize: bool = True,
    apply_to: str = "weight",
) -> tuple[dict, dict]:
    """
    Weight decay should only be applied to the linear layers or convolutional layers.
    All other layers should not have weight decay applied.
    Pytorch Optimizers will apply weight decay to all parameters that have `requires_grad=True`.
    This function can be overwritten by specify the parameters that should have weight decay applied.

    Args:
        model (torch.nn.Module): The model to apply weight decay to.
        decay_factor (float): The weight decay factor to apply to the parameters.
        normalize (bool): If True, the decay factor is normalized by the number of parameters.
            This ensures that the end L2 loss is about the same size for big and small models.

    Returns:

        tuple: A tuple containing two dictionaries where:
            - The first dictionary contains parameters that should not have weight decay applied.
            - The second dictionary contains parameters that should have weight decay applied.
    """
    # get list of parameters that should have weight decay applied, and those that should not
    with_weight_decay = []
    no_weight_decay = []
    # only add weight decay to linear layers! everything else should run normally.
    for module_name, module in model.named_modules():
        # get only modules that are not containers
        if len(list(module.named_modules())) == 1:
            for parameter_name, parameter in module.named_parameters():
                if (isinstance(module, torch.nn.Linear)) and (apply_to in parameter_name):
                    with_weight_decay.append(parameter)
                    print(f"add weight decay to: module:{module}//named: {module_name}// paramter:{parameter_name}")
                else:
                    no_weight_decay.append(parameter)

    # decouple lambda choice from number of parameters, by normalizing the decay factor
    num_weight_decay_params = sum([len(weight.flatten()) for weight in with_weight_decay])
    if normalize:
        decay_factor = decay_factor / num_weight_decay_params
        print(f"Normalize weight decay by number of parameters: {decay_factor}")
    return {"params": no_weight_decay, "weight_decay": 0.0}, {"params": with_weight_decay, "weight_decay": decay_factor}

embedding_expected_inputs = {
    "res_dnn_pnet_pair_type": [0, 1, 2],  # see mapping below
    "res_dnn_pnet_dm1": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "res_dnn_pnet_dm2": [0, 1, 10, 11],
    "res_dnn_pnet_vis_tau1_charge": [-1, 1, 0],
    "res_dnn_pnet_vis_tau2_charge": [-1, 1, 0],
    "res_dnn_pnet_has_fatjet": [0, 1],  # whether a selected fatjet is present
    "res_dnn_pnet_has_jet_pair": [0, 1],  # whether two or more jets are present
    # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post
    "res_dnn_pnet_year_flag": [0, 1, 2, 3, 4, 5, 6, 7],
    "channel_id": [1, 2, 3],
}
# embedding_expected_inputs = {
#     "pair_type": [0, 1, 2],  # see mapping below
#     "decay_mode1": [-1, 0, 1, 10, 11],  # -1 for e/mu
#     "decay_mode2": [0, 1, 10, 11],
#     "lepton1.charge": [-1, 1, 0],
#     "lepton2.charge": [-1, 1, 0],
#     "has_fatjet": [0, 1],  # whether a selected fatjet is present
#     "has_jet_pair": [0, 1],  # whether two or more jets are present
#     # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post
#     "year_flag": [0, 1, 2, 3, 4, 5, 6, 7],
#     "channel_id": [1, 2, 3],
# }

def clip_gradients(parameters: Iterable[torch.nn.Parameter], clip_value: float = 1.0):
    for p in parameters:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))



def get_standardization_parameter(
    data_map: list[ParquetDataset],
    columns: Iterable[Route | str],
) -> dict[str, ak.Array]:
    # open parquet files and concatenate to get statistics for whole datasets
    # beware missing values are currently ignored
    all_data = ak.concatenate(list(map(lambda x: x.data, data_map)))
    # make sure columns are Routes
    columns = list(map(Route, columns))

    statistics = {}
    for _route in columns:
        # ignore empty fields
        arr = _route.apply(all_data)
        # filter missing values out
        empty_mask = arr == EMPTY_FLOAT
        masked_arr = arr[~empty_mask]
        std = ak.std(masked_arr, axis=None)
        mean = ak.mean(masked_arr, axis=None)
        # reshape to 1D array, torch has no interface for 0D
        statistics[_route.column] = {"std": std.reshape(1), "mean": mean.reshape(1)}
    return statistics


def expand_columns(*columns):
    if isinstance(columns, str):
        columns = [columns]

    _columns = set()
    for column_expression in columns:
        if isinstance(column_expression, Route):
            # do nothing if already a route
            break
        expanded_columns = law.util.brace_expand(column_expression)
        routed_columns = set(map(Route, expanded_columns))
        _columns.update(routed_columns)
    return sorted(_columns)


def reorganize_list_idx(entries):
    first = entries[0]
    if isinstance(first, int):
        return entries
    elif isinstance(first, dict):
        return reorganize_dict_idx(entries)
    elif isinstance(first, (list, tuple)):
        sub_dict = defaultdict(list)
        for e in entries:
            # only the last entry is the idx, all other entries
            # in the list/tuple will be used as keys
            data = e[-1]
            key = tuple(e[:-1])
            if isinstance(data, (list, tuple)):
                sub_dict[key].extend(data)
            else:
                sub_dict[key].append(e[-1])
        return sub_dict


def reorganize_dict_idx(batch):
    return_dict = dict()
    for key, entries in batch.items():
        # type shouldn't change within one set of entries,
        # so just check first
        return_dict[key] = reorganize_list_idx(entries)
    return return_dict


def reorganize_idx(batch):
    if isinstance(batch, dict):
        return reorganize_dict_idx(batch)
    else:
        return reorganize_list_idx(batch)



def LookUpTable(array: torch.Tensor, EMPTY=EMPTY_INT, placeholder: int = 15):
    """Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
    Empty values are replaced with *EMPTY*.

    Args:
        array (torch.Tensor): 2D array of categories.
        EMPTY (int, optional): Replacement value if empty. Defaults to columnflow EMPTY_INT.

    Returns:
        tuple([torch.Tensor]): Returns minimum and LookUpTable
    """
    # add placeholder value to array
    array = torch.cat([array, torch.ones(array.shape[0], dtype=torch.int32).reshape(-1, 1) * placeholder], axis=-1)
    # shift input by minimum, pushing the categories to the valid indice space
    minimum = array.min(axis=-1).values
    indice_array = array - minimum.reshape(-1, 1)
    upper_bound = torch.max(indice_array) + 1

    # warn for big categories
    if upper_bound > 100:
        print("Be aware that a large number of categories will result in a large sparse lookup array")

    # create mapping placeholder
    mapping_array = torch.full(
        size=(len(minimum), upper_bound),
        fill_value=EMPTY,
        dtype=torch.int32,
    )

    # fill placeholder with vocabulary

    stride = 0
    # transpose from event to feature loop
    for feature_idx, feature in enumerate(indice_array):
        unique = torch.unique(feature, dim=None)
        mapping_array[feature_idx, unique] = torch.arange(
            stride, stride + len(unique),
            dtype=torch.int32,
        )
        stride += len(unique)
    return minimum, mapping_array

class CategoricalTokenizer(torch.nn.Module):
    def __init__(self, translation: torch.Tensor, minimum: torch.Tensor):
        """
        This translaytion layer tokenizes categorical features into a sparse representation.
        The input tensor is a 2D tensor with shape (N, M) where N is the number of events and M of features.
        The output tensor will have shape (N, K) where K is the number of unique categories across all features.

        Args:
            translation (torch.tensor): Sparse representation of the categories, created by LookUpTable.
            minimum (torch.tensor): Array of minimas used to shift the input tensor into the valid index space.
        """
        super().__init__()
        self.map = translation
        self.min = minimum
        self.indices = torch.arange(len(minimum))

    @property
    def num_dim(self):
        return torch.max(self.map) + 1

    def forward(self, x):
        # shift input array by their respective minimum and slice translation accordingly
        return self.map[self.indices, x - self.min]

    def to(self, *args, **kwargs):
        # make sure to move the translation array to the same device as the input
        self.map = self.map.to(*args, **kwargs)
        self.min = self.min.to(*args, **kwargs)
        self.indices = self.indices.to(*args, **kwargs)
        return super().to(*args, **kwargs)
