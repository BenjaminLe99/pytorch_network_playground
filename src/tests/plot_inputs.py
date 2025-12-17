# standard imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import math
import os
import pathlib
import torch


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

def plot_feature(features, events, ftype):
    path = pathlib.Path(os.environ["PICTURE_DIR"])

    ncols = 7
    nrows = math.ceil((len(features) // ncols))

    scale = 3

    fig_width = ncols * (scale) + 5
    fig_height = nrows * scale + 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (fig_width, fig_height), dpi=150)

    # aggregate data by dataset:
    tt_data = torch.concatenate([data[ftype] for uid, data in events.items() if "tt" in uid])
    hh_data = torch.concatenate([data[ftype] for uid, data in events.items() if "hh" in uid])
    dy_data = torch.concatenate([data[ftype] for uid, data in events.items() if "dy" in uid])

    data = {"tt" : tt_data, "hh" : hh_data, "dy" : dy_data}

    for num_f, feature in enumerate(features):
        for uid, d in data.items():
            # plot in ax
            x = d[:, num_f]
            current_ax = ax.flatten()[num_f]
            weights = np.ones_like(x) / len(x)
            current_ax.hist(
                x = x,
                bins = 50,
                label=uid,
                alpha=1,
                histtype = "step",
                weights=weights
            )
            current_ax.set_ylim(0,1)
            current_ax.set_xlabel(feature, fontsize=12)
            # current_ax.legend()
            current_ax.grid(True)   # same as ax.grid(which='major', linestyle='-', color='gray', alpha=0.7)
    fig.tight_layout()

    # add statistics to legend
    proxy = mpatches.Patch(color="none")            # or set color if you want a colored box
    handles, labels = current_ax.get_legend_handles_labels()

    for d, ev in data.items():
        handles.append(proxy)
        labels.append(f"{d}: {len(ev):_d}\n")
    fig.legend(handles, labels, loc="upper center")
    # save
    p = f"{path / ftype}.png"
    print(f"save plot in {p}")
    fig.savefig(p)
    return fig, ax


events = get_data(dataset_config, overwrite=False, _save_cache=True)
# create k-folds, whe current fold is test fold and leave out

# train_data, validation_data = split_k_fold_into_training_and_validation(
#     events,
#     c_fold=0,
#     k_fold=config["k_fold"],
#     seed=config["seed"],
#     train_ratio=config["train_ratio"],
# )

# training_sampler, validation_sampler = create_train_and_validation_sampler(
#     t_data = train_data,
#     v_data = validation_data,
#     t_batch_size = config["t_batch_size"],
#     v_batch_size = config["v_batch_size"],
#     target_map=target_map,
#     min_size=1
# )

cont_f = dataset_config["continous_features"]
cat_f = dataset_config["categorical_features"]

# plotting input features
from IPython import embed; embed(header="string - 37 in plot_inputs.py ")
plot_feature(features=cat_f, events=events, ftype="categorical")
plot_feature(features=cont_f, events=events, ftype="continous")
