from __future__ import annotations

import torch
import tqdm
import numpy as np
import awkward as ak
import sklearn
import matplotlib.pyplot as plt

from utils import EMPTY_INT, EMPTY_FLOAT
# import data.train.metrics

def plot_input_features(data_map, columns):
    all_data = ak.concatenate([x.data for x in data_map])
    columns = list(map(Route, columns))
    # layout plots
    num_features = len(columns)
    num_cols = 4
    num_row = int(np.ceil(num_features / num_cols))
    fig_size = (5 * num_cols, 4 * num_row) # wide, tall

    fig, axes = plt.subplots(nrows=num_row, ncols=num_cols, figsize=fig_size)
    for ax, _route in zip(axes.flatten(), columns):
        data = _route.apply(all_data).to_numpy().astype(np.float32)
        # mask NaNs
        empty_mask = data == EMPTY_FLOAT

        ax.set_xlabel(_route.column)
        ax.set_ylabel("frequency")
        ax.set_yscale("log")

        # get lowest value without empty values and add offset
        _bin = 20
        bins = np.linspace(
            np.min(data[~empty_mask]),
            data.max(),
            _bin,
        )
        # set offset to 3 bins to display underflow, clip data to lower edge and preserve bin width
        lower_edge = bins[0] - 3 * (bins[1] - bins[0])
        bins = np.linspace(lower_edge, bins[-1], _bin + 3)

        # plot without empty values, set empty values to underflow bin
        _ = ax.hist(np.clip(data, a_min=lower_edge, a_max=None), bins=bins)
    return fig, axes


def network_predictions(y_true, y_pred, target_map, normalize=True, single_legend=False, **kwargs):
    # create a figure with subplots for each node, where the score is shown
    # nodes are defined over target_map order

    fig, axes = plt.subplots(1,len(target_map), figsize=(8 * len(target_map), 8))
    fig.suptitle(kwargs.pop("title", None))
    weight = None
    # get events that are predicted correctly for each class
    for node, node_idx in target_map.items():
        for data_cls, data_idx in target_map.items():
            y_label = "frequency"

            if not normalize:
                axes[node_idx].set_yscale("log")
                axes[node_idx].set_ylim(top=len(y_pred))
            else:
                axes[node_idx].set_yscale("linear")
                axes[node_idx].set_ylim(top=1.)
                y_label += " normalized"

            axes[node_idx].set_xlabel(f"{node} node", )
            axes[node_idx].set_ylabel(y_label)
            axes[node_idx].grid()


            # get events of specific cls (e.g. hh)
            correct_cls_mask = y_true[:, data_idx] == 1
            # get predictions for cls
            filtered_predictions = y_pred[correct_cls_mask][:, node_idx]

            if normalize:
                weight = np.full(filtered_predictions.shape, 1 / len(filtered_predictions))

            _ = axes[node_idx].hist(
                filtered_predictions,
                bins=kwargs.get("bins", 20),
                histtype=kwargs.get("histtype", "step"),
                alpha=kwargs.get("alpha", 0.7),
                label=data_cls,
                weights=weight,
                **kwargs,
        )
        if not single_legend:
            axes[node_idx].legend()
    if single_legend:
        lines_labels = [fig.axes[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
    return fig, axes

def confusion_matrix(y_true, y_pred, target_map, sample_weight=None, normalized="true", cmap="Blues", **kwargs):
    """
    Calculates a Confusion Matrix using the truth *y_true* and prediction *y_pred* of the model.
    The categories are defined using *target_map* and to weight give *sample_weight*. Styles are
    changed using *cmap*.

    Args:
        y_true (torch.tensor): tensor of true labels
        y_pred (torch.tensor): tensor of predicted labels
        target_map (dict[int]): dict of class name to index mapping
        sample_weight (torch.tensor, optional): tensor of weights on event basis. Defaults to None.
        cmap (str, optional): style of the confusion matrix. Defaults to "Blues".

    Returns:
        tuple: figure, axis, confusion matrix
    """
    # convert prediction to class indices
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = torch.argmax(y_true, dim=1).cpu()

    cm = sklearn.metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=list(target_map.values()),
        sample_weight=sample_weight,
        normalize=normalized,  # normalize to get probabilities
    )

    disp = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(target_map.keys()),
    )
    disp.plot(cmap=cmap)
    disp.figure_.suptitle(kwargs.pop("title", None))

    return disp.figure_, disp.ax_, disp.confusion_matrix

def roc_curve(target, pred, sample_weight=None, labels=None, **kwargs):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.get_cmap("tab10").colors if kwargs.get("colors") is None else colors

    if sample_weight is None:
        sample_weight = torch.ones(target.shape[0])

    for c, name in enumerate(labels):
        col = colors[c % len(colors)]

        disp = sklearn.metrics.RocCurveDisplay.from_predictions(
            target[:, c],
            pred[:, c],
            sample_weight=sample_weight.reshape(-1, 1),
            ax=ax,
            name=name,
            #curve_kwargs={"color": col},
            )
    _ = ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    return disp.figure_, disp.ax_


def plot_2D(x, y, bins=10, **kwargs):
    fig, ax = plt.subplots()
    hist = ax.hist2d(
        x,
        y,
        bins=bins,
        cmap=kwargs.get("cmap"),
    )
    ax.set_xlabel(kwargs.get("xlabel"))
    ax.set_xlabel(kwargs.get("ylabel"))
    fig.colorbar(hist[3], ax=ax)
    ax.set_title(kwargs.get("title"))
    if kwargs.get("savepath"):
        fig.savefig(kwargs.get("savepath"))
    return fig, ax

def plot_1D(x, annotations=None, **kwargs):
    fig, ax = plt.subplots()
    _ = ax.hist(
        x.item(),
        bins=kwargs.get("bins"),
        histtype="stepfilled",
        alpha=0.7,
        color=kwargs.get("color"),
        label=kwargs.get("label"),
        density=kwargs.get("density"),
    )
    ax.set_xlabel(kwargs.get("xlabel"))
    ax.set_xlabel(kwargs.get("ylabel"))
    ax.set_title(kwargs.get("title"))
    if kwargs.get("savepath"):
        fig.savefig(kwargs.get("savepath"))
    ax.legend()

    if annotations:
        loss = annotations.get("loss")
        target = annotations.get("target")
        pred = annotations.get("pred")
        num_0s = np.sum(target == 0)
        num_1s = np.sum(target == 1)

        ax.annotate(f"num: 0s: {num_0s:.2f}", (0.5, 0.70), xycoords="axes fraction")
        ax.annotate(f"num: 1s: {num_1s:.2f}", (0.5, 0.65), xycoords="axes fraction")
        ax.annotate(f"loss: {loss:.2f}", (0.5, 0.60), xycoords="axes fraction")
        iteration = annotations.get("iteration")
        ax.annotate(f"iteration {iteration}", (0.5, 0.55))

        tp = np.sum((target == 1) & (pred > 0.5))
        tn = np.sum((target == 0) & (pred < 0.5))
        fp = np.sum((target == 0) & (pred > 0.5))
        fn = np.sum((target == 1) & (pred < 0.5))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)

        ax.annotate(f"accuracy: {accuracy:.2f}", (0.5, 0.50), xycoords="axes fraction")
        ax.annotate(f"sensitivity: {sensitivity:.2f}", (0.5, 0.45), xycoords="axes fraction")

    return fig, ax

def control_plot_1d(train_loader, dataset_handler):
    d = {}
    for dataset, file_handler in training_loader.data_map.items():
        d[dataset] = ak.concatenate(list(map(lambda x: x.data, file_handler)))

    for cat in dataset_handler.categorical_featuress:
        plt.clf()
        data = []
        labels = []
        for dataset, arrays in d.items():
            data.append(Route(cat).apply(arrays))
            labels.append(dataset)
        plt.hist(data, histtype="barstacked", alpha=0.5, label=labels)
        plt.xlabel(cat)
        plt.legend()
        plt.savefig(f"{cat}_all.png")


def plot_batch(self, input, target, loss, iteration, target_map=None, labels=None):
    input_per_feature = input.to("cpu").transpose(0, 1).detach().numpy()
    input, target = input.to("cpu").detach().numpy(), target.to("cpu").detach().numpy()

    fig, ax = plt.subplots(1, len(self.categorical_inputs), figsize=(8 * len(self.categorical_inputs), 8))
    fig.tight_layout()

    for ind, cat in enumerate(self.categorical_inputs):
        signal_target = target[:, self.categorical_target_map["hh"]]

        background_mask, signal_mask = signal_target.flatten() == 0, signal_target.flatten() == 1
        # background_prediction = detach_pred[zero_s_mask]
        # signal_prediction = detach_pred[zero_s_mask]
        _input = input_per_feature[ind]
        background_input = _input[background_mask]
        signal_input = _input[signal_mask]
        cax = ax[ind]

        cax.hist(
            [background_input, signal_input],
            bins=10,
            histtype="barstacked",
            alpha=0.5,
            label=["tt & dy", "hh"],
            density=True,
        )
        cax.set_xlabel(cat)
        cax.annotate(f"loss: {loss:.2f}", (0.8, 0.60), xycoords="figure fraction")
        cax.annotate(f"iteration {iteration}", (0.75, 0.55))
        cax.legend()
    fig.savefig(f"1d_cats_{iteration}.png")
