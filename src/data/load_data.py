
import numpy as np
import awkward as ak
import torch
import uproot
import os
import pathlib
from typing import Union

from data.utils import depthCount
from utils.logger import get_logger
from data.cache import DataCacher

logger = get_logger(__name__)


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

def root_to_numpy(files_path: Union[list[str],str], branches: Union[list[str], str, None]=None) -> ak.Array:
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

    # include meta fields that are necessary for every training
    # process_id is for filtering after subprocesses
    # normalizaton_weight is for
    meta_fields = {"process_id", "normalization_weight", "tau2_isolated", "leptons_os", "channel_id"}
    branches = set(branches).union(meta_fields)

    if isinstance(files_path, str):
        files_path = [files_path]
    arrays = []
    for file_path in files_path:
        with uproot.open(file_path, object_cache=None, array_cache=None) as file:
            tree = file["events"]
            arrays.append(tree.arrays(branches, library="ak").to_numpy())
    return np.concatenate(arrays, axis=0)

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
        return root_to_numpy, {"branches": kwargs.get("columns", None)}
        #case "parquet":
        #    return parquet_to_awkward, {"columns": kwargs.get("columns", None)}
        #case _:
        #    raise ValueError(f"Unknown file type: {file_type}")


def load_data(datasets, file_type: str="root", columns: Union[list[str],str, None]=None, apply_filter=True):
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

    def sort_by_process_id(array):
        # helper to sort data by process id and not datasets,
        # results in array of structure {year:{dataset : array}}
        pids = array["process_id"]
        unique_ids = np.unique(pids)
        p_array = {}
        for uid in unique_ids:
            mask = pids == uid
            p_array[int(uid)] = array[mask]
        return tuple(p_array.items())

    def filter_base_selection(array):
        # applies base selection as event filter
        masks = (
        array["tau2_isolated"] == 1,
        array["leptons_os"] == 1,
        (array["channel_id"] == 1) | (array["channel_id"] == 2) | (array["channel_id"] == 3)
        )
        masks = np.logical_and.reduce(masks)
        return array[masks]

    def load_data_per_process_id(loader, datasets, config):
        data = {}
        for year, year_data in datasets.items():
            # add events with structure {dataset_name : events}
            for dataset, files in year_data.items():
                # load inputs
                events = loader(files, **config)

                # apply filters
                if apply_filter:
                    events = filter_base_selection(events)
                # filter by process_id if necessary and save, otherwise save by dataset
                p_arrays = sort_by_process_id(events)
                for pid, p_array in p_arrays:
                    uid = (year,dataset[:2],pid)
                    if uid not in data:
                        data[uid] = []
                    data[uid].append(p_array)
                    logger.info(f"{year} | {dataset} | PID: {pid} | {len(p_array)}")
        return data

    def merge_per_pid(data):
        # replace_with_concatenated_from_buffers(data, axis=0)
        keys = list(data.keys())

        for i, uid in enumerate(keys):
            print(i, uid)
            arrays = data.pop(uid)
            concat = np.concatenate(arrays, axis=0)
            data[uid] = concat
        return data

    loader, config = get_loader(file_type, columns=list(columns))
    data = load_data_per_process_id(loader, datasets, config)
    data = merge_per_pid(data)
    return data

def get_data(config, _save_cache = True, overwrite=False):
    # find cache if exists and recreate sample with this
    # else prepare data if not cache exist

    cacher = DataCacher(config=config)

    # when cache exist load it and return the data
    if not overwrite and cacher.path.exists():
        events = cacher.load_cache()
    else:
        logger.info("Prepare Loading of data:")
        cont_feat, cat_feat = config["continous_features"], config["categorical_features"]
        # load the data in {pid : awkward}
        events = load_data(
            config["datasets"],
            file_type = "root",
            columns = cont_feat + cat_feat
        )
        # conver data in {pid : {cont:arr, cat: arr, weight: arr, target: arr}}
        events = convert_numpy_to_torch(
            events=events,
            continous_features=cont_feat,
            categorical_features=cat_feat,
        )
        logger.info("Done loading data")
        # save events in cache
        if _save_cache:
            try:
                cacher.save_cache(events)
            except:
                from IPython import embed; embed(header="Saving Cache did not work out - going debugging to manually save \'events\' with \'cacher.save_cache\'")
    return events

def convert_numpy_to_torch(events, continous_features, categorical_features, dtype=None):
    def numpy_to_torch(array, columns, dtype):
        array = torch.from_numpy(np.stack([array[col] for col in columns], axis=1))
        if dtype is not None:
            array = array.to(dtype)
        return array

    def filter_nan_mask(array, features, uid):
        masks = []
        for f in features:
            mask = np.isnan(array[f])
            masks.append(mask)
        event_mask = np.logical_or.reduce(masks)
        num_filter = np.sum(event_mask)
        if num_filter:
            logger.info(f"Filter {num_filter} Nans out form {uid}")
        return ~event_mask


    for uid in list(events.keys()):
        arr = events.pop(uid)

        # filter all nans out
        event_mask = filter_nan_mask(arr, continous_features + categorical_features, uid),
        arr = arr[event_mask]

        # if resulting tensor is empty just skip
        if arr.numel() == 0:
            logger.info(f"Skipping {uid} due to zero elements")
            continue

        continous_tensor, categorical_tensor = [
            numpy_to_torch(arr, feature, dtype)
            for feature in (continous_features,categorical_features)
            ]
        # convert to torch

        weight = torch.tensor(np.sum(arr["normalization_weight"]), dtype = dtype)
        events[uid] = {
            "continous": continous_tensor,
            "categorical": categorical_tensor,
            "weight": weight
        }
    return events
