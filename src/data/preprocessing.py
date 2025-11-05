import torch

from collections import defaultdict
from utils.logger import get_logger

from data.datasets import EraDataset, EraDatasetSampler

logger = get_logger(__name__)

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

def k_fold_indices(num_events, c_fold, k_fold, seed, test=False):
    """
    Creates idicies for training and validation from *k_fold*, where *c_fold* is the test fold.
    The indicies are permutated using *seed*. If *test* is True, the test set is returned if False the k-1 folds are returned.

    Args:
        num_events (int): Number of events for which k-folds should be created
        c_fold (int): Current fold, this fold is returned when *test* is True
        k_fold (int): Number of folds
        seed (int): Seeds used for random permutation
    """
    indices = torch.arange(num_events)

    # true => test folds, false => train and validation folds
    test_fold_mask = indices % k_fold == c_fold
    if test:
        sub_indices = indices[test_fold_mask]
    else:
        sub_indices = indices[~test_fold_mask]
    # apply mask and randomize according to given seed
    sub_indices = torch.randperm(len(sub_indices), generator=torch.Generator().manual_seed(seed))
    return sub_indices[sub_indices]

def split_array_to_train_and_validation(array, trainings_proportion=0.75):
    """
    Splits given *array* into *trainings_proportion* train and (1 - *trainings_proportion*) validation parts.

    Args:
        array (torch.Tensor, numpy.Array): flat torch or numpy array
        trainings_proportion (float, optional): Relative proportion of the resulting trainings array. Defaults to 0.75.

    Returns:
        tuple (torch.Tensor, numpy.Array): Tuple of trainings and validation array
    """
    train_length = round((len(array) * trainings_proportion))
    t_idx = array[:train_length]
    v_idx = array[train_length:]
    return t_idx, v_idx


def split_k_fold_into_training_and_validation(events_dict, c_fold, k_fold, seed, train_ratio=0.75):
    """
    Takes *events_dict* where continous and categorical data and split these into *k_fold* where *c_fold* is the holdout test seed.
    A random permutation happens using *seed* and in the end the k-1 folds are split into training
    and validation data using a ration of *train_ratio*.

    Args:
        events_dict (torch.tensor): Dictionary of continous and categorical Tensors
        c_fold (int): Current fold, this fold is returned when *test* is True
        k_fold (int): Number of folds
        seed (int): Seeds used for random permutation
        train_ratio (float, optional): _description_. Defaults to 0.75.

    Returns:
        dict(torch.tensor): Train and validation dictionary
    """
    #from IPython import embed;embed(header=" string - 160 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/data/preprocessing.py")
    train, valid = {}, {}
    for uid in list(events_dict.keys()):
        array = events_dict.pop(uid)
        train[uid] = {"weight" : array["weight"]}
        valid[uid] = {"weight" : array["weight"]}
        for key in ("continous", "categorical"):
            arr = array.pop(key)
            num_events = len(arr)
            tv_indices = k_fold_indices(num_events, c_fold, k_fold, seed, test=False)
            t_idx, v_idx = split_array_to_train_and_validation(tv_indices, train_ratio)
            train[uid][key] = arr[t_idx]
            valid[uid][key] = arr[v_idx]
    return train, valid

def create_train_or_validation_sampler(events, target_map, batch_size, min_size=1, train=True):
    # extract data from events and wrap into Datasets
    EraDatasetManager = EraDatasetSampler(None, batch_size=batch_size, min_size=min_size)
    for uid in list(events.keys()):
        (era, dataset_type, process_id) = uid
        arrays = events.pop(uid)

        # create target tensor from uid
        num_events = len(arrays["continous"])
        target_value = target_map[dataset_type]
        target = torch.zeros(size=(num_events, len(target_map)), dtype=torch.float32)
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

    if train:
        for ds_type in EraDatasetManager.keys:
            EraDatasetManager.calculate_sample_size(dataset_type=ds_type)
    return EraDatasetManager
