
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

def find_datasets(dataset_patterns: list[str], year_patterns: list[str], file_type: str="root", verbose=True):
    """
    Find all files in ${INPUT_DATA_DIR} by glob pattern defined by <year_pattern>/<dataset_pattern>/*.file_type and
    return the result as a dictionary with dataset names as keys and list of file paths as values.
    Duplicates in *year_pattern* and *dataset_patterns* DO NOT change result.

    Args:
        dataset_patterns (list[str]): Pattern describing our nano AODs, ex: ['hh_ggf_hbb_htt_kl*_kt1*']
        year_pattern (str): Pattern describing the years.
        file_type (str, optional): File extension . Defaults to "root".

    Returns:
        dict: dictionary with dataset names as keys and list of file paths as values
    """
    # get data dir from config
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")

    # wrapp patterns since iterable is assumed
    if isinstance(dataset_patterns, str):
        dataset_patterns = [dataset_patterns]
    if isinstance(year_patterns, str):
        year_patterns = [year_patterns]

    file_pattern = f"*.{file_type}"

    # resolve year pattern and remove duplicates
    years = []
    for year_pattern in year_patterns:
        years += [year.name for year in pathlib.Path(data_dir).glob(f"{year_pattern}")]
    years = sorted(set(years))

    data = {}
    missing = []
    for year in years:
        data[year] = {}
        for dataset_patter in dataset_patterns:
            datasets = list(pathlib.Path(data_dir).glob(f"{year}/{dataset_patter}"))

            if len(datasets) == 0:
                raise ValueError(f"dataset pattern {dataset_patter} for {year} resulted in 0 datasets")

            for dataset in datasets:
                files = sorted(map(str, pathlib.Path(dataset).glob(file_pattern)))
                if len(files) == 0:
                    logger.warning(f"{dataset} has 0 files")
                    missing.append(dataset)
                if verbose:
                    size = round(sum(os.path.getsize(f) for f in files) / (1024**2),2)
                    logger.info(f"+{len(files)} files | size {size} MB | {year}/{dataset.name}")
                data[year][dataset.name] = files
    if not data:
        raise ValueError("No datasets found with given patterns")

    if missing:
        raise ValueError(f"following datasets has 0 files:\n{"\n\t".join(missing)}")

    # merge over years era information is not needed
    merged_over_era_data = {}
    for era in list(data.keys()):
        dataset_dict = data.pop(era)
        for dataset, files in dataset_dict.items():
            if dataset not in merged_over_era_data:
                merged_over_era_data[dataset] = []
            merged_over_era_data[dataset].extend(files)
    return merged_over_era_data

def root_to_numpy(
    files_path: Union[list[str],str],
    branches: Union[list[str], str, None]=None,
    cut=None
    ) -> ak.Array:
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

    ### manage fields to get from root file
    # fields -> purpose:
    #   process_id -> filtering of subphasespaces
    #   normalization_weight -> used as sample weight for batch
    #   tau2_isolated, lepton_os, channel_id -> used for baseline cut
    #   event -> event number used for k-fold splitting
    #   normalization_weight -> oversampling weight that defines the fraction within batch
    meta_fields = {"process_id", "tau2_isolated", "leptons_os", "channel_id", "event", "normalization_weight"}
    # training and evaluation phase space are not the same transfer weight is calculated using these weights
    weights = {
    "normalized_pdf_weight","normalized_murmuf_weight","normalized_pu_weight",
    "normalized_isr_weight","normalized_fsr_weight","normalized_njet_btag_weight_pnet",
    "electron_id_weight","electron_reco_weight","muon_id_weight","muon_iso_weight",
    "tau_weight","trigger_weight", "dy_weight","top_pt_weight"
    }
    all_branches = set(branches).union(meta_fields)

    ### handle baseline combine with additional cuts
    baseline_cuts = [
        "(tau2_isolated == 1)",
        "(leptons_os == 1)",
        "((channel_id == 1) | (channel_id == 2) | (channel_id == 3))",
    ]

    if isinstance(cut, str):
        cut = [cut]
    if cut is None:
        cut = []
    final_cut = "&".join(baseline_cuts + cut)

    ### load root files and combine the array to continogus arrays
    if isinstance(files_path, str):
        files_path = [files_path]

    arrays = []
    for file_path in files_path:
        logger.info(f"loading: {file_path}")
        with uproot.open(file_path, object_cache=None, array_cache=None) as file:
            tree = file["events"]
            ### first load all input features that EVERY DATASET has in COMMON
            all_branches_array = tree.arrays(all_branches, library="ak", cut=final_cut)

            ### second handle WEIGHTS that not every dataset has
            # e.x. muon_id_weight does not exist in dy datasets
            # find out which weights exist in root file
            weights_in_root_file = set(tree.keys()).intersection(weights)
            weights_arrays = tree.arrays(weights_in_root_file, library="ak", cut=final_cut)

            combined_weight = 1
            for weight in weights_in_root_file:
                combined_weight = combined_weight * weights_arrays[weight]
            all_branches_array["combined_weight"] = combined_weight

            # get number of expected events
            di_tau_mask, di_bjet_mask, bjet_mask = evaluation_phase_space_filter(
                uproot_file=tree,
                year=pathlib.Path(file_path).parents[1].stem,
                cut=final_cut,
                suffix=("res_dnn_pnet" if branches[0].startswith("res_dnn_pnet") else "reg_dnn_moe"))
            all_branches_array["bjet_mask"] = bjet_mask
            all_branches_array["di_tau_mask"] = di_tau_mask
            all_branches_array["di_bjet_mask"] = di_bjet_mask

            arrays.append(all_branches_array.to_numpy())

    arrays = np.concatenate(arrays, axis=0)
    return arrays

def evaluation_phase_space_filter(uproot_file, year, cut, suffix="res_dnn_pnet"):
    # as taken from https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/config/configs_hbt.py#L1252
    def particle_net_wp(year, wp_level="medium"):
        particle_net_wp = {
            "loose": {"22pre": 0.047, "22post": 0.0499, "23pre": 0.0358, "23post": 0.0359, "2024": None}[year],
            "medium": {"22pre": 0.245, "22post": 0.2605, "23pre": 0.1917, "23post": 0.1919, "2024": None}[year],
            "tight": {"22pre": 0.6734, "22post": 0.6915, "23pre": 0.6172, "23post": 0.6133, "2024": None}[year],
            "xtight": {"22pre": 0.7862, "22post": 0.8033, "23pre": 0.7515, "23post": 0.7544, "2024": None}[year],
            "xxtight": {"22pre": 0.961, "22post": 0.9664, "23pre": 0.9659, "23post": 0.9688, "2024": None}[year],
        }
        return particle_net_wp[wp_level]
    b_tag_wp = particle_net_wp(year, "medium")
    ### load the necessary events with applied cut from baseselection
    # leptons_fields = [f"{particle}_{k}" for particle in ("Electron", "Muon", "Tau") for k in ("px", "py", "pz", "e")]
    leptons_fields = [f"{suffix}_vis_tau{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]
    hhbjet_fields = ["HHBJet_mass", "HHBJet_btagPNetB"]

    events = uproot_file.arrays(leptons_fields + hhbjet_fields, library="ak", cut=cut)

    ### masks
    # tau mass window
    # TODO marcel fragen warum das so klein ist
    # dilep_mass = ak.sum(ak.concatenate([[events[lepton_mass] for lepton_mass in leptons_fields]], axis=1), axis=0)
    l_px = events[f"{suffix}_vis_tau1_px"] + events[f"{suffix}_vis_tau2_px"]
    l_py = events[f"{suffix}_vis_tau1_py"] + events[f"{suffix}_vis_tau2_py"]
    l_pz = events[f"{suffix}_vis_tau1_pz"] + events[f"{suffix}_vis_tau2_pz"]
    l_e = events[f"{suffix}_vis_tau1_e"] + events[f"{suffix}_vis_tau2_e"]

    di_tau_mass = (l_e**2 - (l_px**2 + l_py**2 + l_pz**2))**0.5
    di_tau_mask = (
        (di_tau_mass >= 15) &
        (di_tau_mass <= 130)
    )

    # btag score success atleast 1 btag score above
    bjet_mask = ak.sum(events.HHBJet_btagPNetB > b_tag_wp, axis=1) >= 1

    # di bjet mass window
    di_bjet_mass = ak.sum(events.HHBJet_mass, axis=1)
    di_bjet_mask = (
        (di_bjet_mass >= 40) &
        (di_bjet_mass <= 270)
    )
    # final_mask =
    return di_tau_mask, di_bjet_mask, bjet_mask

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
    Small helper function to load correct loader function and its configuration based on given *file_type*.

    Args:
        file_type (str): file extension.

    Raises:
        ValueError: if file type is not supported by loader

    Returns:
        func: loader function, configuration dictionary
    """
    if file_type == "root":
            return root_to_numpy, {"branches": kwargs.get("columns", None), "cut": kwargs.get("cuts", None)}
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

    def load_data_per_process_id(loader, datasets, config):

        data = {}
        for dataset, files in datasets.items():
            # add events with structure {dataset_name : events}
            # load inputs
            events = loader(files, **config)
            p_arrays = sort_by_process_id(events)
            for pid, p_array in p_arrays:
                uid = (dataset[:2],pid)
                if uid not in data:
                    data[uid] = []
                data[uid].append(p_array)
                logger.debug(f"{dataset} | PID: {pid} | {len(p_array)}")
        return data

    def merge_per_pid(data):
        # replace_with_concatenated_from_buffers(data, axis=0)
        keys = list(data.keys())

        logger.info("Start merging arrays per process id")
        for i, uid in enumerate(keys):
            logger.debug(f"{i}/{len(keys)}: {uid}")
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
        logger.info("Start loading and filtering of data:")
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
        if arr.size == 0:
            logger.info(f"Skipping {uid} due to zero elements")
            continue

        # calculate total weight for trainings and evaluation space
        total_trainings_weight = np.sum(arr["combined_weight"])
        final_mask = arr["bjet_mask"] & arr["di_tau_mask"] & arr["di_bjet_mask"]

        total_bjet_weight = np.sum(arr["combined_weight"][arr["bjet_mask"]])
        total_di_tau_weight = np.sum(arr["combined_weight"][arr["di_tau_mask"]])
        total_di_bjet_weight = np.sum(arr["combined_weight"][arr["di_bjet_mask"]])
        total_evaluation_weight = np.sum(arr["combined_weight"][final_mask])

        continous_tensor, categorical_tensor = [
            numpy_to_torch(arr, feature, dtype)
            for feature in (continous_features,categorical_features)
            ]
        # convert to torch
        weight = torch.tensor(np.sum(arr["normalization_weight"]), dtype = dtype)
        # event id
        event_id = torch.tensor(np.ascontiguousarray(arr["event"]), dtype=torch.int64)

        events[uid] = {
            "continous": continous_tensor,
            "categorical": categorical_tensor,
            "weight": weight,
            "event_id" : event_id,
            "total_trainings_space_weight" : total_trainings_weight,
            "total_evaluation_weight" : total_evaluation_weight,
            "total_bjet_weight" : total_bjet_weight,
            "total_di_tau_weight" : total_di_tau_weight,
            "total_di_bjet_weight" : total_di_bjet_weight,
        }
    return events
