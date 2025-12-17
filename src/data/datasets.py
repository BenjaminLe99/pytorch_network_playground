import torch
import torch.utils.data as t_data
from collections import defaultdict

from utils.logger import get_logger
logger = get_logger(__name__)

class Dataset(t_data.Dataset):
    def __init__(
        self,
        continous_tensor: torch.Tensor,
        categorical_tensor: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor=None,
        name: str=None,
        dataset_type: str=None,
        randomize: bool =True,
        eval_weight: torch.Tensor=None,
    ):
        self.continous_input = continous_tensor.to(torch.float32)
        self.categorical_input = categorical_tensor.to(torch.int32)
        self.targets = target.to(torch.float32)
        self.weight = weight
        self.name = name
        self.randomize = randomize
        self.reset()
        self.dataset_type = dataset_type
        self.sample_size = len(self)
        self.relative_weight = None
        self.eval_weight = eval_weight

    @property
    def uid(self):
        return (self.dataset_type, self.name)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.continous_input[idx], self.continous_input[idx], self.targets[idx]

    def reset(self):
        self.current_idx = 0
        if self.randomize:
            self.indices = torch.randperm(len(self))
        else:
            self.indices = torch.arange(len(self))

    def sample(self, number=None ,device=torch.device("cpu")):
        """
        Sample *number* of events from self.{continous,categorical,targets}.
        Samples from the start again, if maximum number of samples is reached.

        Args:
            number (_type_, optional): Number of events to sample. If "all" returns full. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.current_idx >= len(self):
            self.reset()

        if number is None:
            number = self.sample_size

        if number > len(self):
            # when required number is too big, set it to len of self to return all data
            number = len(self)

        start_idx = self.current_idx
        # do not drop last, but instead create smaller batch
        next_idx = min(self.current_idx + number, len(self))
        idx = self.indices[start_idx:next_idx]
        self.current_idx = next_idx

        return self.continous_input[idx].to(device), self.categorical_input[idx].to(device), self.targets[idx].to(device)

    def batch_generator(self, batch_size=None, device=torch.device("cpu")):
        """
        Creates a generator object that returns the input and target tensors batch-wise with size *batch_size*.
        If *batch_size* is returned as None, the whole underlying data is returned at once.
        The generator does not randomize the underlying data. The generator and sample methods can be used
        independently.

        Args:
            batch_size (int, optional): Sample size per iteration, if None all data is returned at once. Defaults to None.
            device (torch.device, optional): Device to which tensors are moved before yielding. Defaults to torch.device("cpu").

        Yields:
            tuple (torch.Tensor): Tuple with 3 torch tensors for categorical, continous and target data of the shape [batch_size, num_features]
        """

        # save current state of the sampler to restore after full iteration
        # HINT: Do not mix sample with batch generator, as both manipulate current_idx
        start_idx, next_idx = 0, 0
        indices = torch.arange(len(self))
        if batch_size is None:
            batch_size = len(self)

        while next_idx < len(self):
            # handle windowing of indices
            next_idx = min(start_idx + batch_size, len(self))
            idx = indices[start_idx:next_idx]
            start_idx = next_idx
            # reset if we reached the end of the dataset and drop last incomplete batch
            cont, cat, tar = self.continous_input[idx], self.categorical_input[idx],self.targets[idx]
            yield cont.to(device), cat.to(device), tar.to(device)

class DatasetSampler(t_data.Sampler):
    signal_background_map = {"hh": "signal", "tt":"background", "dy":"background"}

    def __init__(
        self,
        datasets_inst=None,
        batch_size=1,
        sample_ratio={"dy": 0.25, "tt": 0.25, "hh": 0.5},
        min_size = None,
        target_map = None,
        ):
        self.total_weight = {"dy":0, "tt":0, "hh":0}
        self.dataset_inst = defaultdict(dict)  # {dataset_type: {process_id: dataset}}
        self.batch_size = torch.tensor([batch_size], dtype=torch.int32)
        if datasets_inst is not None:
            for dataset in datasets_inst:
                self.add_dataset(dataset)
        self.sample_ratio = sample_ratio
        self.min_size = min_size # index where to split continous and categorical data
        self.target_map = target_map

        # set attributes to access dataset properties directly from sampler
        self.set_attr(["weight", "relative_weight", "sample_size", "dataset_type", "batch_generator"])


    def __getitem__(self, uid):
        return self.flat_datasets()[uid]

    def flat_datasets(self):
        datasets = {}
        for _, uid_ds in self.dataset_inst.items():
            datasets.update(uid_ds)
        return datasets

    def apply_func_on_datasets(self, func):
        # small helper to apply function of managet datasets
        return {uid: func(ds) for uid,ds in self.flat_datasets().items()}

    def get_attribute_of_datasets(self, attr):
        # small helper to getattributes from managet datasets
        return {uid: getattr(ds, attr) for uid,ds in self.flat_datasets().items()}

    def events_per_dataset(self):
        return self.apply_func_on_datasets(len)

    def __len__(self):
        return sum(list(self.events_per_dataset().values()))

    def add_dataset(self, dataset):
        # {era : process_id: {array}}
        weight = dataset.weight
        self.total_weight[dataset.dataset_type] += weight
        self.dataset_inst[dataset.dataset_type][dataset.name] = dataset

    @property
    def keys(self):
        return list(self.dataset_inst.keys())

    def print_sample_rates(self):
        logger.info(f"Sample rate for Dataset of Era")

        for dataset_type, uid in self.dataset_inst.items():
            total_batch = 0
            logger.info(f"{dataset_type}:")
            for (pid), ds in uid.items():
                logger.info(f"\t{pid} | {ds.sample_size}")
                total_batch += ds.sample_size
            logger.info(f"\tsum/expected: {total_batch}/{(self.sample_ratio[ds.dataset_type] * self.batch_size).item()}")

    def calculate_sample_size(self, dataset_type, dry=False):
        logger.info(f"Calculate sample sizes for: {dataset_type}")
        # unpack and convert to tensors for easier handling, get desired_sub_batch_size
        weights, keys = [], []
        for pid, ds in self.dataset_inst[dataset_type].items():
            weights.append(ds.weight.item())
            keys.append(pid)
        weights = torch.tensor(weights, dtype=torch.float32)
        min_size = torch.tensor(self.min_size)
        sub_batch_size = int(self.batch_size * self.sample_ratio[dataset_type])

        # start algorithm
        # 2 cases: Phase phases need to be upsampeled OR phase spaces are already reasonable high
        #
        # in 3 steps:
        # 1. downsample calculated batch size and upsample to minimal threshold
        # 2. calculate relative weight of overflow for samples above minimal threshold
        # 3. distribute overflow from 2. by ceil and floor half of the sample according to median
        # 4. reduce  all steps resulting in sub_batch_size batch composition


        # preperation: calculate batch size according to weight
        total_weight = weights.sum()
        ideal_sizes = sub_batch_size * weights / total_weight

        # add relative weight to datasets
        for pid, ds in self.dataset_inst[dataset_type].items():
            ds.relative_weight = ds.weight / total_weight * self.sample_ratio[dataset_type]
        # step 1:

        # check if UPSAMPLING IS NECESSARY AT ALL
        floored_sizes = torch.floor(ideal_sizes) # decreases sub batch size by length of tensor
        floored_sizes = torch.maximum(floored_sizes, min_size) # increase sub batch size by different to threshold
        # calculate overflow of sub batch
        overflow_size = int(floored_sizes.sum()) - sub_batch_size

        # step 2:
        # calculate relative contribution of events above threshold

        mask_above_threshold = floored_sizes > min_size
        floored_above_threshold = floored_sizes[mask_above_threshold]
        floored_overflow_sizes = floored_above_threshold / floored_above_threshold.sum() * overflow_size
        # step 3: floor below median, ceil for weights above
        # sort to be able to use median properlly
        floored_overflow_sizes, floored_oveflow_idx = torch.sort(floored_overflow_sizes)

        # define mask for floor and ceil
        # median -> even: under median so idx len/2 - 1 (change mask to inclusive)
        # median -> uneven: exact middle idx len/2
        # even uneven median treatment
        if len(floored_oveflow_idx) % 2 == 0:
            median = floored_oveflow_idx.median()
            below_median = floored_oveflow_idx <= median
            above_median = floored_oveflow_idx > median
        else:
            median = floored_oveflow_idx.median()
            below_median = floored_oveflow_idx < median
            above_median = floored_oveflow_idx >= median

        # apply floor and ceil for targets below and above median
        floored_overflow_sizes[below_median] = torch.floor(floored_overflow_sizes[below_median])
        floored_overflow_sizes[above_median] = torch.ceil(floored_overflow_sizes[above_median])

        # step 4: apply overflow to original floored
        # combine sort indices and threshold mask
        # this returns floored sizes sorted by biggest by indices
        indices_above_threshold = torch.arange(len(floored_sizes))[mask_above_threshold][floored_oveflow_idx]
        floored_sizes[indices_above_threshold] -= floored_overflow_sizes

        # conver to int but check if values changed, just to be safe
        if (floored_sizes.to(torch.int32).sum() - floored_sizes.sum()) != 0:
            raise TypeError(f"Change of type changed value")
        floored_sizes = floored_sizes.to(torch.int32)

        # if one still mismatch add it on lowest or remove from highest
        very_last_remaining = int(floored_sizes.sum()) - sub_batch_size
        # from IPython import embed; embed(header="string - 144 in datasets.py ")

        # TODO BETTER SCHEME
        # remove remaining from biggest sample or add more to biggest sample

        floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # if very_last_remaining <= -1:
        #     floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # elif very_last_remaining >= 1:
        #     floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # elif very_last_remaining != 0:
        #     from IPython import embed; embed(header=f"{floored_sizes.sum()} but should be {sub_batch_size}")
        #     raise ValueError(f"Resampling failed: Created batch size of size: {floored_sizes.sum()} but should be {sub_batch_size}")

        # store batch size per phase space in dataset
        if not dry:
            for k, w in zip(keys, floored_sizes):
                self.dataset_inst[dataset_type][k].sample_size = w.item()
        else:
            logger.info({k:w.item() for k,w in zip(keys, floored_sizes)})

    def sample_batch(self, device=torch.device("cpu")):
        # loop over dataset classes and ask them to generate a certain number of samples

        # shuffle finished batch optionally
        # Get a batch of data from each dataset
        batch_cont, batch_cat, batch_target, batch_weight = [], [], [], []

        sample_indices = {
            "signal":[],
            "background":[],
        }

        # do iteration over keys to GUARANTEE deterministic behavior when looping accross different python version
        for uid, ds in sorted(self.flat_datasets().items()):
            cont, cat, target = ds.sample(device=device)
            batch_cont.append(cont), batch_cat.append(cat), batch_target.append(target), batch_weight.append(torch.full((target.shape[0], 1), ds.weight))
            sample_indices[self.signal_background_map[ds.dataset_type]].append(target.shape[0])

        # combine everything into tensors
        batch_cont = torch.concatenate(batch_cont, dim=0).to(device)
        batch_cat = torch.concatenate(batch_cat, dim=0).to(device)
        batch_target = torch.concatenate(batch_target, dim=0).to(device)
        batch_weight = torch.concatenate(batch_weight, dim=0).to(device)
        return batch_cont, batch_cat, batch_target, batch_weight, sample_indices

    def set_attr(self, attribute):
        if isinstance(attribute, str):
            attribute = [attribute]

        for attr in attribute:
            if hasattr(self, attr):
                raise AttributeError(f"Attribute {attr} already exists in sampler class.")

            def _accesor(*args, dataset_attr=attr, **kwargs):
                collector = {}
                for uid, ds in self.flat_datasets().items():
                    ds_attr = getattr(ds, dataset_attr)
                    # when attribute is a callable function pass args and kwargs
                    if callable(ds_attr):
                        collector[uid] = ds_attr(*args, **kwargs)
                    else:
                        collector[uid] = ds_attr
                return collector

            setattr(self, attr, _accesor)

    def share_weights_between_sampler(self, dst_sampler):
        # helper to enable ex. validation sampler to get sampling weight from training sampler
        for dataset_type, datasets in self.dataset_inst.items():
            for uid, ds in datasets.items():
                v_ds = dst_sampler.dataset_inst[dataset_type][uid]
                v_ds.relative_weight = ds.relative_weight
