import torch
import numpy as np

from collections import defaultdict
from utils.logger import get_logger

from data.datasets import Dataset, DatasetSampler

logger = get_logger(__name__)


def apply_tokenization(expected_inputs, events, categorical_features):
    for uid in list(events.keys()):
        data = events[uid]
        cateogrical_array = data["categorical"]
        map_categorical_features(
            expected_inputs=expected_inputs,
            feature_array=cateogrical_array,
            categorical_features=categorical_features
        )
        events["categorical"] = cateogrical_array
    return events

def map_categorical_features(expected_inputs, feature_array, categorical_features):
    feat_window_start = 0
    feat_window_end = 0
    new_indices = {}
    for cat in categorical_features:
        feat_window_end += len(expected_inputs[cat])
        indices = np.arange(start=feat_window_start, stop=feat_window_end)
        feat_window_start = feat_window_end
        new_indices[cat] = indices

    # get all masks
    for idx, cat in enumerate(categorical_features):
        old_values = expected_inputs[cat]
        new_values = new_indices[cat]
        masks = []
        # get masks
        for value in old_values:
            m = feature_array[:, idx] == value
            masks.append(m)

        # apply masks inplace
        for mask, new_value in zip(masks, new_values):
            feature_array[:, idx][mask] = new_value
    return feature_array

def get_batch_statistics_from_sampler(sampler, padding_values=None, features=None):
    """
    Calculates the weighted mean and standard deviation over all subphase spaces of a process in *sampler*.
    The data is expected to be of form : {"unique_identifier_tuple": {continous: arr}, {weight}: arr}.
    The return value is a dictionary of form {"process": (mean, std)}, where mean and std is a tensor of
    form length [features].
    The statistics are evaluated without *padding_values*.

    Args:
        sampler (dict): Dictionary over datasets
        padding_values (int, optional): List of padding_values per feature, or single value. Padding value are ignored in the calculation of the statitics. Defaults to None, which means no padding.
    """
    logger.info("Calculate mean and std over all subphase spaces")
    # filter keys after processes
    weighted_means = []
    weighted_vars = []
    features_dict = sampler.get_attribute_of_datasets("continous_input")
    weights_dict = sampler.get_attribute_of_datasets("relative_weight")
    sum_of_weights = sum(list(weights_dict.values()))

    # for each uid calculate mean,std and get weight
    # do weighted mean,std over all uid with weight
    for current_pid_idx, pid in enumerate(features_dict.keys(), start = 1):
        # go throught each feature axis and calculate statitic per feature
        f_means, f_vars = [], []
        array = features_dict[pid]
        num_f = array.shape[-1]
        print(f"calculating stats for {pid} status:{current_pid_idx}/{len(features_dict)}\t\t\r",end=" ", flush=True)


        for f_idx in range(num_f):

            # extract feature by index
            feature_array = array[..., f_idx]
            # do not include padding into calculation
            if isinstance(padding_values, (list, tuple)):
                padding_value = padding_values[f_idx]
            else:
                padding_value = padding_values

            padding_mask = (feature_array == padding_value)
            masked_array = feature_array[~padding_mask]

            masked_mean = masked_array.mean(axis=0)
            masked_var = masked_array.var(axis=0)
            # if anything got a nan go into debug mode to investigate
            if torch.isnan(masked_mean):
                from IPython import embed; embed(header=f"{pid} is nan check feature_array and sampler")

            f_means.append(masked_mean)
            f_vars.append(masked_var)

        # weight mean and add to collection
        pid_weight = weights_dict[pid]
        weighted_means.append(torch.tensor(f_means) * pid_weight)
        weighted_vars.append(torch.tensor(f_vars) * pid_weight)

    # calculate weighted average over uid means and var
    w_avg_mean  = torch.sum(torch.stack(weighted_means, axis=0), axis = 0) / sum_of_weights
    w_avg_var = torch.sum(torch.stack(weighted_vars, axis=0), axis = 0) / sum_of_weights
    if features:
        print("Statistics")
        for f_name, f_mean, f_var in zip(features, w_avg_mean, w_avg_var):
            print(f"{f_name:<30}: mean:{f_mean:>10.4} var:{f_var:>10.4}")
    return w_avg_mean, w_avg_var.sqrt()


def get_batch_statistics(events=None, padding_value=0):
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
    for uid, arrays in events.items():
        # reshape to feature x events

        arr_features = arrays["continous"].transpose(0,1)
        weights.append(arrays["weight"])
        # go throught each feature axis and calculate statitic per feature
        f_means, f_stds = [], []
        for f in arr_features:
            padding_mask = (f == padding_value)
            masked_array = f[~padding_mask]
            masked_mean = masked_array.mean(axis=0)
            masked_std = masked_array.std(axis=0)
            if torch.isnan(masked_mean):
                from IPython import embed; embed(header=f"{uid} is nan check f and events")

            f_means.append(masked_mean)
            f_stds.append(masked_std)
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
        (ds_type, pid) = uid
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

def k_fold_indices(event_id, c_fold, k_fold, seed, test=False):
    """
    Creates idicies for training and validation from *k_fold*, where *c_fold* is the test fold.
    The indicies are permutated using *seed*. If *test* is True, the test set is returned if False the k-1 folds are returned.

    Args:
        num_events (int): Number of events for which k-folds should be created
        c_fold (int): Current fold, this fold is returned when *test* is True
        k_fold (int): Number of folds
        seed (int): Seeds used for random permutation
    """
    # true => test folds, false => train and validation folds
    # if no kfold is wished than set test to 0 and return everyrthing
    if k_fold == 0:
        raise ValueError(f"Can't do k-fold with desired k_fold of {k_fold}, needs to be > 0")

    test_fold_mask = event_id % k_fold == c_fold
    indices = torch.arange(len(event_id))

    if test:
        sub_event_id = indices[test_fold_mask]
    else:
        sub_event_id = indices[~test_fold_mask]
    # apply mask and randomize according to given seed
    randomized = torch.randperm(len(sub_event_id), generator=torch.Generator().manual_seed(seed))
    return sub_event_id[randomized]

def split_array_to_train_and_validation(array, trainings_proportion=0.75):
    """
    Splits given *array* into *trainings_proportion* train and (1 - *trainings_proportion*) validation parts.

    Args:
        array (torch.Tensor, numpy.Array): flat torch or numpy array
        trainings_proportion (float, optional): Relative proportion of the resulting trainings array. Defaults to 0.75.

    Returns:
        tuple (torch.Tensor, numpy.Array): Tuple of trainings and validation array
    """
    if (trainings_proportion > 1) or (trainings_proportion < 0):
        raise ValueError(f"Split fraction is {trainings_proportion} but needs to be in range of 0 and 1")
    train_length = int(round((len(array) * trainings_proportion)))
    t_idx = array[:train_length]
    v_idx = array[train_length:]
    return t_idx, v_idx


def split_k_fold_into_training_and_validation(events_dict, c_fold, k_fold, seed, train_ratio=0.75, return_test=False):
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
    train, valid = {}, {}
    for uid in list(events_dict.keys()):
        array = events_dict.pop(uid)
        train[uid] = {"weight" : array["weight"]}
        valid[uid] = {"weight" : array["weight"]}
        tv_indices = k_fold_indices(array["event_id"], c_fold, k_fold, seed, test=return_test)
        t_idx, v_idx = split_array_to_train_and_validation(tv_indices, train_ratio)

        for key in ("continous", "categorical", "event_id"):
            arr = array.pop(key)
            t_arr = arr[t_idx]
            v_arr = arr[v_idx]

            train[uid][key] = t_arr
            valid[uid][key] = v_arr
    # edge case split results in empty tensors (due to very low event count) remove these
    # if empty do not save
    for uid in list(train.keys()):
        for d in ("train", "valid"):
            dictionary = locals()[d]
            if (dictionary[uid]["continous"].numel() == 0):
                logger.info(f"removed {uid} from {d} since zero elements left after k-fold split")
                dictionary.pop(uid)
    return train, valid

def create_train_or_validation_sampler(events, target_map, batch_size, min_size=1, train=True, sample_ratio={"dy": 0.25, "tt": 0.25, "hh": 0.5}):
    # extract data from events and wrap into Datasets
    if not events:
        logger.warning(f"Sampler is not created due to feeding empty events")
        return None

    DatasetManager = DatasetSampler(None, batch_size=batch_size, min_size=min_size, sample_ratio=sample_ratio)
    for uid in list(events.keys()):
        (dataset_type, process_id) = uid
        arrays = events.pop(uid)

        # create target tensor from uid
        num_events = len(arrays["continous"])
        target_value = target_map[dataset_type]
        target = torch.zeros(size=(num_events, len(target_map)), dtype=torch.float32)
        target[:, target_value] = 1.

        era_dataset = Dataset(
            continous_tensor=arrays["continous"],
            categorical_tensor=arrays["categorical"],
            target=target,
            weight=arrays["weight"],
            name=process_id,
            dataset_type=dataset_type,
        )
        DatasetManager.add_dataset(era_dataset)
        logger.info(f"Add {dataset_type} pid: {process_id}")

    if train:
        for ds_type in DatasetManager.keys:
            DatasetManager.calculate_sample_size(dataset_type=ds_type)
    return DatasetManager

def test_sampler(events, target_map, batch_size, min_size=1, train=True):
    # FIXME: Test sampler does not work currently due to the fact that a
    # accumulation across eras is done.

    # Function to test a sampler that is actually doing the right thing
    raise NotImplementedError("Test Sampler is not implemented for era accumulated Data")
    # extract data from events and wrap into Datasets
    ERA_MAP = {"22pre": 0, "22post":1, "23pre":2, "23post":3}
    DatasetManager = DatasetSampler(None, batch_size=batch_size, min_size=min_size)
    for uid in list(events.keys()):
        (dataset_type, process_id) = uid
        # create target tensor from uid
        num_events = len(events[uid]["continous"])
        target_value = target_map[dataset_type]
        target = torch.zeros(size=(num_events, len(target_map)), dtype=torch.float32)
        target[:, target_value] = 1.

        era_tensor = torch.full(size=(num_events,), fill_value=ERA_MAP[era], dtype=torch.int32)
        process_id_tensor = torch.full(size=(num_events,), fill_value=process_id, dtype=torch.int32)

        era_dataset = Dataset(
            continous_tensor=process_id_tensor,
            categorical_tensor=era_tensor,
            target=target,
            weight=events[uid]["weight"],
            name=process_id,
            era=era,
            dataset_type=dataset_type,
        )
        DatasetManager.add_dataset(era_dataset)
        logger.info(f"Add {dataset_type} pid: {process_id} of era: {era}")

    if train:
        for ds_type in DatasetManager.keys:
            DatasetManager.calculate_sample_size(dataset_type=ds_type)

    sample_sizes_per_ds = DatasetManager.get_attribute_of_datasets("sample_size")
    sample_sizes_per_ds = {(ERA_MAP[era],pid): events for (era,pid), events in sample_sizes_per_ds.items()}

    def batch_comparison(sample_size_per_ds, batch):
        pid, era, target = batch
        # count occurences
        d = sample_size_per_ds.copy()
        for idx in range(len(pid)):
            pid_v, era_v = pid[idx], era[idx]
            key = (era_v.item(), pid_v.item())
            d[key] -= 1
        return d
    ds = DatasetManager
    from IPython import embed; embed(header="string - 256 in preprocessing.py ")


    return DatasetManager
