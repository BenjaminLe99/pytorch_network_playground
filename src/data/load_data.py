
import numpy as np
import awkward as ak
import torch
import uproot
import os
import pathlib
from typing import Union
from collections import defaultdict

from data.utils import depthCount
from utils.logger import get_logger
from data.datasets import EraDataset, EraDatasetSampler

logger = get_logger(__name__)

def add_meta_fields(columns):
    meta_fields = ["process_id"]
    return columns + meta_fields

def find_datasets(dataset_patterns: str, year_patterns: str, file_type: str="root"):
    """
    Find all files in ${INPUT_DATA_DIR} by pattern defined by <year_pattern>/<dataset_pattern>/*.file_type and
    return the result as a dictionary with dataset names as keys and list of file paths as values.

    Args:
        dataset_patter (str): _description_
        year_pattern (str): Pattern describing the years
        file_type (str, optional): File extension . Defaults to "root".

    Returns:
        dict: dictionary with dataset names as keys and list of file paths as values
    """
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")

    # wrapp patterns in lists if only being a single string
    if isinstance(dataset_patterns, str):
        dataset_patterns = [dataset_patterns]
    if isinstance(year_patterns, str):
        year_patterns = [year_patterns]

    file_pattern = f"*.{file_type}"

    years = []
    for year_pattern in year_patterns:
        years += [year.name for year in pathlib.Path(data_dir).glob(f"{year_pattern}")]

    data = {}
    for year in years:
        data[year] = {}
        for dataset_patter in dataset_patterns:
            datasets = list(pathlib.Path(data_dir).glob(f"{year}/{dataset_patter}"))

            if len(datasets) == 0:
                raise ValueError(f"dataset pattern {dataset_patter} for {year} resulted in 0 datasets")

            for dataset in datasets:
                files = list(map(str, pathlib.Path(dataset).glob(file_pattern)))
                if len(files) == 0:
                    raise ValueError(f"{dataset.name} has 0 files")
                size = round(sum(os.path.getsize(f) for f in files) / (1024**2),2)
                logger.info(f"+{len(files)} files | size {size} MB | {year}/{dataset.name}")
                data[year][dataset.name] = files
    return data

def root_to_awkward(files_path: Union[list[str],str], branches: Union[list[str], str, None]=None) -> ak.Array:
    """
    Load all root files in *files_path* and return them as a single awkward array.
    If only certain branches are needed, they can be specified in *branches*.

    Args:
        files_path (list[str], str): list of root files or single root file
        branches (list[str], str, optional): branches that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files

    """
    if depthCount(branches) > 1 and branches is not None:
        raise ValueError("branches must be a flat list")

    branches = add_meta_fields(branches)

    if isinstance(files_path, str):
        files_path = [files_path]
    arrays = []
    for file_path in files_path:
        with uproot.open(file_path) as file:
            tree = file["events"]
            # from IPython import embed; embed(header="string - 85 in load_data.py ")
            arrays.append(tree.arrays(branches, library="ak"))
    # from IPython import embed; embed(header="string - 93 in load_data.py ")
    return ak.concatenate(arrays, axis=0)

def parquet_to_awkward(files_path: Union[list[str],str], columns: Union[list[str], str, None]=None) -> ak.Array:
    """
    Load all parquet files in *files_path* and return them as a single awkward array.
    If only certain columns are needed, they can be specified in *columns*.

    Args:
        files_path (list[str], str): list of root files or single root file
        columns (list[str], str, optional): columns that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files
    """
    if depthCount(columns) > 1 and columns is not None:
        raise ValueError("columns must be a flat list")
    arrays = []
    for file_path in files_path:
        arrays.append(ak.from_parquet(file_path, columns=columns))
    return ak.concatenate(arrays, axis=0)

def get_loader(file_type: str, **kwargs):
    """
    Small helper function to get the correct loader function and its configuration based on the file type.

    Args:
        file_type (str): file extension.

    Raises:
        ValueError: if file type is not supported by loader

    Returns:
        func: loader function, configuration dictionary
    """
    if file_type == "root":
        return root_to_awkward, {"branches": kwargs.get("columns", None)}
    elif "parquet":
        return parquet_to_awkward, {"columns": kwargs.get("columns", None)}
    else: raise ValueError(f"Unknown file type: {file_type}")

def get_cache_path(config=None):
    import hashlib
    import os
    import pathlib
    h = tuple(config.items())
    h = hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]
    cache_dir = pathlib.Path(os.environ["CACHE_DIR"]).with_name(h)
    return cache_dir


def load_data(datasets, file_type: str="root", columns: Union[list[str],str, None]=None):
    """
    Loads data with given *file_type* in given a *dataset_pattern* and *year_pattern*. If only certain columns are needed, they can be specified in *columns*.
    The data sorted by year and dataset name is returned as a nested dictionary in awkward format.

    Args:
        dataset_patter (str): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        year_pattern (str): pattern describing the years e.g. "2*pre" -> 22pre, 23pre
        file_type (str, optional): file extension. Defaults to "root".
        columns (list[str], str, optional): columns that should be lodead e.g. ["events", "run"]. If None loads all columns . Defaults to None.

    Returns:
        dict: {year:{pid: List(Ids)}}
    """
    target_map = {"hh" : 0, "dy": 1, "tt": 2}
    era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
    num_targets = len(target_map.keys())
    # add weights for resampling
    columns = set(columns)
    if "normalization_weight" not in columns:
        columns.add("normalization_weight")

    loader, config = get_loader(file_type, columns=list(columns))
    data = defaultdict(list)
    for year, year_data in datasets.items():
        # add events with structure {dataset_name : events}
        for dataset, files in year_data.items():
            # load inputs
            events = loader(files, **config)

            # add target by dataset name, first 2 letters define the target
            target_value = target_map[dataset[:2]]
            target_array = np.zeros((len(events), num_targets))
            target_array[:, target_value] = 1
            events = ak.with_field(events, target_array, "target")

            # add era encoding:
            era_array = np.full(len(events), era_map[year], np.int32)
            events = ak.with_field(events, era_array, "era")

            # filter by process_id if necessary and save, otherwise save by dataset
            p_arrays = filter_by_process_id(events)

            for pid, p_array in p_arrays.items():
                data[(year,dataset[:2],pid)].append(p_array)
                logger.info(f"{year} | {dataset} | PID: {pid} | {len(p_array)}")
    logger.info(f"starting merging of PIDs")
    # merge over pids
    for uid, arrays in data.items():
        data[uid] = ak.concatenate(arrays)
    return data

def filter_by_process_id(array):
    # events of structure {year:{dataset : array}}
    pids = array["process_id"]
    unique_ids = np.unique(pids.to_numpy())
    p_array = {}
    for uid in unique_ids:
        mask = pids == uid
        p_array[int(uid)] = array[mask]
    return p_array

def get_data(config, save_cache = True):
    import pickle
    # find cache if exists and recreate sample with this
    # else prepare data if not cache exist
    cache_path = get_cache_path(config=config)
    # when cache exist load it and return the data
    if cache_path.exists():
        logger.info("Loading cache")
        with open(cache_path, "rb") as file:
            events = pickle.load(file)
    else:
        logger.info("Prepare Loading of data:")
        events = load_data(
            config["datasets"],
            file_type = "root",
            columns = config["continous_features"] + config["categorical_features"]
        )
        # save events in cache
        if save_cache:
            logger.info(f"Saving cache at {cache_path}:")
            with open(f"{cache_path}", "wb") as file:
                pickle.dump(events, file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Done loading data")
    return events


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

def create_sampler(events, input_columns, dtype=torch.float32, min_size=1):
    # convert to torch tensors and create EraDatasets objects
    EraDatasetManager = EraDatasetSampler(None, batch_size=1024*4, min_size=min_size)
    for (era, dataset_type, process_id), arr in events.items():
        arr = ak.concatenate(arr)
        inputs = awkward_to_torch(arr, input_columns, dtype)
        target = awkward_to_torch(arr, ["target"], dtype)
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
        EraDatasetManager.calculate_sample_size(dataset_type=ds_type)
    return EraDatasetManager


def get_std_statistics(events):
    def weighted_average(means, weights):
        return sum(weights * weights) / sum(weights)
    # filter data after processes
    keys_per_process = defaultdict(list)
    for uid in events.keys():
        (era, ds_type, pid) = uid
        keys_per_process[ds_type].append(uid)
    from IPython import embed; embed(header="string - 308 in load_data.py ")


    means, stds = [],[]
    for ds_type, uid in keys_per_process.items():
        for (era, ds_type, pid), array in events[uid]:
            pass
        # arr = events[]
