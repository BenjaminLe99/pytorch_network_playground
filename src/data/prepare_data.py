import awkward as ak
import numpy as np
import torch

from collections import defaultdict
from data.load_data import load_data
from data.datasets import EraDataset, EraDatasetSampler
from utils.logger import get_logger

logger = get_logger(__name__)

def awkward_to_torch(array, columns, dtype, merge_pids=True):
    array = torch.from_numpy(np.stack([array[col].to_numpy() for col in columns], axis=1))
    if dtype is not None:
        array = array.to(dtype)
    return array


def filter_datasets(
    events: dict[dict[ak.Array]],
    feature: str="process_id"
    ) -> dict[dict[ak.Array]]:
    """
    Filter *events* by apply mask to *feature* and returns an dictionary of structure {year: {feature: array}}

    Args:
        events (dict[dict[ak.Array]]): events of structure {year:{dataset : array}}

    Returns:
        return dict[dict[ak.Array]]: events of structure {dataset_type: {(year,feature): array}}
    """

    # events of structure {year:{dataset : array}}
    arrays_per_year_per_feature = defaultdict(dict)
    for year, datasets in events.items():
        array_per_feature_collection = defaultdict(list)
        for dataset, array in datasets.items():
            # get unique features, filter array by feature and store in collection
            # {feature : array[feature_mask]}
            for unique_feature in np.unique(array[feature].to_numpy()):
                feature_mask = array[feature] == unique_feature
                filtered_array = array[feature_mask]
                array_per_feature_collection[(int(unique_feature), dataset[:2])].append(filtered_array)

        # return array with structure {dataset_type: {(year,feature): array}}
        for (_feature, dataset_type), list_of_arrays in array_per_feature_collection.items():
            array = ak.concatenate(list_of_arrays)
            logger.info(f"{year} | {_feature} | {len(array)} events")
            arrays_per_year_per_feature[dataset_type][(year,_feature)] = array
    return arrays_per_year_per_feature


def get_sum_of_weights(array):
    # era : {process_id : arrray}
    weights = {}
    for year, pid in array.items():
        weights[year] = {}
        for process_id, arr in pid.items():
            weights[year][process_id] = ak.sum(arr.normalization_weight)
    return weights

def prepare_data(datasets, eras, input_columns, target_columns, dtype=torch.float32, file_type="root", split_index=None):

    events = load_data(
        datasets,
        eras,
        file_type=file_type,
        columns=input_columns,
        )

    # convert to torch tensors and create EraDatasets objects
    EraDatasetManager = EraDatasetSampler(None, batch_size=1024*4, split_index=split_index)
    for (era, dataset_type, process_id), arr in events.items():
        # from IPython import embed; embed(header="RR - 273 in load_data.py ")
        arr = ak.concatenate(arr)
        inputs = awkward_to_torch(arr, input_columns, dtype)
        target = awkward_to_torch(arr, target_columns, dtype)
        weight = torch.tensor(ak.sum(arr.normalization_weight), dtype = dtype)
        era_dataset = EraDataset(
            inputs=inputs,
            target=target ,
            weight=weight ,
            name=process_id,
            era=era,
            dataset_type=dataset_type,
        )
        EraDatasetManager.add_dataset(era_dataset)
        logger.info(f"Add {dataset_type} pid: {process_id} of era: {era}")

    for ds_type in list(set([dataset_type for (era, dataset_type, process_id) in events.keys()])):
        EraDatasetManager.calculate_sample_size(dataset_type=ds_type, min_size=5)
    return EraDatasetManager

if __name__ == "__main__":
    # test load_dataset
    datasets = ["dy_m50toinf_*j_amcatnlo","tt_dl*"]
    eras = ["22pre", "23pre"]
    input_columns = ["event", "process_id"]
    target_columns = ["target"]
    # filter process after process_id
    sampler = prepare_data(datasets, eras, input_columns, target_columns, dtype=torch.float32, file_type="root")
    from IPython import embed; embed(header="Get a sample by using sampler.get_batch() - in load_data.py ")
