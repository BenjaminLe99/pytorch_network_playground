
import numpy as np
import awkward as ak
import torch
import uproot
import os
import pickle
import pathlib
from typing import Union
from collections import defaultdict

from data.utils import depthCount
from utils.logger import get_logger
from data.datasets import EraDataset, EraDatasetSampler
from data.cache import DataCacher

logger = get_logger(__name__)

def add_meta_fields(columns):
    meta_fields = {"process_id", "normalization_weight"}
    columns = set(columns)
    columns = columns.union(meta_fields)
    return columns

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

    branches = add_meta_fields(branches)

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

    def filter_by_process_id(array):
        # events of structure {year:{dataset : array}}
        pids = array["process_id"]
        unique_ids = np.unique(pids)
        p_array = {}
        for uid in unique_ids:
            mask = pids == uid
            p_array[int(uid)] = array[mask]
        return tuple(p_array.items())

    def load_data_per_process_id(loader, datasets, config):
        from numpy.lib.recfunctions import append_fields
        data = {}
        for year, year_data in datasets.items():
            # add events with structure {dataset_name : events}
            for dataset, files in year_data.items():
                # load inputs
                events = loader(files, **config)

                # filter by process_id if necessary and save, otherwise save by dataset
                p_arrays = filter_by_process_id(events)
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
    if not overwrite and cacher.path:
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

        # convert to torch
        continous_tensor = numpy_to_torch(
            arr,
            continous_features,
            dtype
        )

        categorical_tensor = numpy_to_torch(
            arr,
            categorical_features,
            dtype
        )

        weight = torch.tensor(np.sum(arr["normalization_weight"]), dtype = dtype)
        events[uid] = {
            "continous": continous_tensor,
            "categorical": categorical_tensor,
            "weight": weight
        }
    return events

def filter_fold(events, c_fold, k_fold, seed, train_ratio=0.75):
    train, valid = {}, {}
    for uid in list(events.keys()):
        array = events.pop(uid)
        train[uid] = {"weight" : array["weight"]}
        valid[uid] = {"weight" : array["weight"]}
        for key in ("continous", "categorical"):
            arr = array.pop(key)
            num_events = len(arr)
            t_idx, v_idx = k_fold_indices(num_events, c_fold, k_fold, seed, train_ratio)
            train[uid][key] = arr[t_idx]
            valid[uid][key] = arr[v_idx]
    return train, valid

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

def create_sampler(events, target_map, min_size=1):
    # extract data from events and wrap into Datasets
    EraDatasetManager = EraDatasetSampler(None, batch_size=1024*4, min_size=min_size)
    for uid in list(events.keys()):
        (era, dataset_type, process_id) = uid
        arrays = events.pop(uid)

        # create target tensor from uid
        num_events = len(arrays["continous"])
        target_value = target_map[dataset_type]
        target = torch.zeros(size=(num_events,3), dtype=torch.float32)
        target[:, target_value] = 1.

        era_dataset = EraDataset(
            continous_tensor=arrays["continous"],
            categorical_tensor=arrays["categorical"],
            target=target,
            weight=arrays["weight"],
            name=process_id,
            era=era,
            dataset_type=dataset_type,
        )
        EraDatasetManager.add_dataset(era_dataset)
        logger.info(f"Add {dataset_type} pid: {process_id} of era: {era}")

    for ds_type in EraDatasetManager.keys:
        EraDatasetManager.calculate_sample_size(dataset_type=ds_type)
    return EraDatasetManager

def get_batch_statistics(events, padding_value=0):
    """
    Calculates the weighted mean and standard deviation over all subphase spaces of a process in *events*.
    The data is expected to be of form : {"unique_identifier_tuple": {continous: arr}, {weight}: arr}.
    The return value is a dictionary of form {"process": (mean, std)}, where mean and std is a tensor of
    form length [features].
    The statistics are evaluated without *padding_value*.

    Args:
        events (dict): Dictionary over datasets
        padding_value (int, optional): Ignored value in the calculation of the statitics. Defaults to 0.
    """
    logger.info("Calculate mean and std over all subphase spaces")
    # filter keys after processes
    means = []
    stds = []
    weights = []

    for _, arrays in events.items():
        # reshape to feature x events
        arr_features = arrays["continous"].transpose(0,1)
        weights.append(arrays["weight"])
        # go throught each feature axis and calculate statitic per feature
        f_means, f_stds = [], []
        for f in arr_features:
            padding_mask = f == padding_value
            f_means.append(f[~padding_mask].mean(axis=0))
            f_stds.append(f[~padding_mask].std(axis=0))

            if torch.isnan(f[~padding_mask].mean(axis=0)):
                from IPython import embed; embed(header="See which feature is nan")
        means.append(f_means)
        stds.append(f_stds)
    means = torch.tensor(means)
    stds = torch.tensor(stds)
    weights = torch.tensor(weights).reshape(-1,1)

    # resulting in a weight of form [features]
    denom = torch.sum(weights)
    w_avg_mean  = torch.sum((means * weights), axis=0) / denom
    w_avg_std = torch.sum((stds * weights), axis=0) / denom
    return w_avg_mean, w_avg_std

def k_fold_indices_decapicated(num_events: int, k: int , random: bool=True):
    """
    Calculates indices to define *k* folds. The indices range from 0 to *num_events*.
    If folds should be shuffled set *random* to true

    Args:
        num_events (int): Number of events of the data that is folded
        k (int): Number of folds
        random (bool, optional): Shuffle indices that make up the fold. Defaults to True.

    Returns:
        list[torch.Tensor]: List of torch tensors of indices
    """
    # data is expected to be torch tensors
    fold_size = num_events // k
    # shuffle indices
    if random:
        indices = torch.randperm(num_events)
    else:
        indices = torch.arange(num_events)
    folds = []
    for i in range(k):
        start_interval = i * fold_size
        # last fold has slightly more data
        end_interval = (i + 1) * fold_size if i != (k - 1) else num_events
        sub_indices = indices[start_interval : end_interval]
        folds.append(sub_indices)
    return folds

def k_fold_indices(num_events, c_fold, k_fold, seed, train_ratio=0.75):
    """
    Creates idicies for training and validation from *k_fold*, where *c_fold* is the test fold.
    The indicies are randomly shuffled using *seed*. The return is first the training indices and then validation, split
    by *train_ratio*.
    """
    indices = torch.arange(num_events)
    # true => test, kicked out, rest is used
    test_fold_mask = indices % k_fold == c_fold
    tv_indices = indices[~test_fold_mask]
    rnd_idx = torch.randperm(len(tv_indices), generator=torch.Generator().manual_seed(seed))
    tv_indices = tv_indices[rnd_idx]

    # splt into train and valid
    train_length = round((len(tv_indices) * train_ratio))
    t_idx = tv_indices[:train_length]
    v_idx = tv_indices[train_length:]
    return t_idx, v_idx






def get_batch_statistics_per_dataset(events, padding_value=0):
    """
    Calculates the weighted mean and standard deviation over all subphase spaces of a process in *events*.
    The data is expected to be of form : {"unique_identifier_tuple": {continous: arr}, {weight}: arr}.
    The return value is a dictionary of form {"process": (mean, std)}, where mean and std is a tensor of
    form length [features].
    The statistics are evaluated without *padding_value*.

    Args:
        events (dict): Dictionary over datasets
        padding_value (int, optional): Ignored value in the calculation of the statitics. Defaults to 0.
    """
    logger.info("Calculate mean and std over all subphase spaces")
    # filter keys after processes
    keys_per_process = defaultdict(list)
    for uid in events.keys():
        (era, ds_type, pid) = uid
        keys_per_process[ds_type].append(uid)

    stats = {}
    for process_type, uids in keys_per_process.items():
        means = []
        stds = []
        weights = []
        for uid in uids:
            f_means, f_stds = [], []
            # reshape to feature x events
            arr_features = events[uid]["continous"].transpose(0,1)
            weights.append(events[uid]["weight"])

            # go throught each feature axis and calculate statitic per feature

            for f in arr_features:
                padding_mask = f == padding_value
                f_means.append(f[~padding_mask].mean(axis=0))
                f_stds.append(f[~padding_mask].std(axis=0))

                if torch.isnan(f[~padding_mask].mean(axis=0)):
                    from IPython import embed; embed(header="See which feature is nan")
            means.append(f_means)
            stds.append(f_stds)
        means = torch.tensor(means)
        stds = torch.tensor(stds)
        weights = torch.tensor(weights).reshape(-1,1)

        # resulting in a weight of form [features]
        nom = torch.sum((means * weights), axis=0)
        denom = torch.sum(weights)
        stats[process_type] = nom / denom
    return stats
