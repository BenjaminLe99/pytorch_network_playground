# standard imports
import matplotlib.pyplot as plt
import numpy as np
import math


# project imports
from data.load_data import get_data
from data.preprocessing import (
    create_train_and_validation_sampler, get_batch_statistics, split_k_fold_into_training_and_validation, test_sampler
    )
from utils.logger import get_logger, TensorboardLogger
import train.plotting
import train.metrics
from data.cache import hash_config
import train.optimizer
from train.train_config import (
    config, dataset_config, model_building_config, optimizer_config, target_map,
)

events = get_data(dataset_config, overwrite=False, _save_cache=True)
# create k-folds, whe current fold is test fold and leave out

train_data, validation_data = split_k_fold_into_training_and_validation(
    events,
    c_fold=0,
    k_fold=config["k_fold"],
    seed=config["seed"],
    train_ratio=config["train_ratio"],
)

training_sampler, validation_sampler = create_train_and_validation_sampler(
    t_data = train_data,
    v_data = validation_data,
    t_batch_size = config["t_batch_size"],
    v_batch_size = config["v_batch_size"],
    target_map=target_map,
    min_size=1
)

from IPython import embed; embed(header="string - 39 in plot_inputs.py ")
cont_f = dataset_config["continous_features"]
cat_f = dataset_config["categorical_features"]
def plot_feature(features, events, ftype):
    ncols = 7
    nrows = math.ceil((len(features) // ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (ncols * 20, nrows * 10))
    for num_f, feature in enumerate(features):
        data = []
        for ds_data in events.values():
            data.append(ds_data[ftype])
        data = np.concatenate(data)

        # plot in ax
        current_ax = ax.flatten()[num_f]
        current_ax.hist(
            x = data,
            bins = 50,
            label=feature
            # range = ,
            # density = ,
            # weights = ,
            # cumulative = ,
            # bottom = ,
            # histtype = ,
        )

        current_ax.set()
        current_ax.set_xlabel(feature, fontsize=12)
        # current_ax.set_ylabel("hits", fontsize=12)
        # current_ax.set_title("My plot")
        current_ax.grid(True)   # same as ax.grid(which='major', linestyle='-', color='gray', alpha=0.7)
        # fig.tight_layout()
    fig.savefig("test.png")
    return fig, ax

# plotting input features
from IPython import embed; embed(header="string - 37 in plot_inputs.py ")
