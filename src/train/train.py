# standard imports
import torch
import os
import matplotlib.pyplot as plt


# project imports
from models import create_model
from data.features import input_features
from data.load_data import get_data, find_datasets
from data.preprocessing import (
    create_train_and_validation_sampler, get_batch_statistics, split_k_fold_into_training_and_validation, test_sampler
    )
from utils.logger import get_logger, TensorboardLogger
import plotting
import metrics
from data.cache import hash_config

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VERBOSE = False

logger = get_logger(__name__)

def training(model, loss_fn, optimizer, sampler):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=DEVICE)
    targets = targets.to(torch.float32)

    pred = model((cat,cont))

    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()
    return loss, (pred, targets)

def correct_predictions(y_true, y_pred):
    predicted_class = torch.argmax(y_pred, dim=1)
    true_class = torch.argmax(y_true, dim=1)
    correct = (predicted_class == true_class).sum().item()
    return correct

def validation(model, loss_fn, sampler):
    with torch.no_grad():
        # run validation every x steps
        val_loss = []
        model.eval()
        predictions = []
        truth = []
        weights = []

        for uid, validation_batch_generator in sampler.get_dataset_batch_generators(
            batch_size=sampler.batch_size,
            reset=True,
            device=DEVICE
            ).items():
            dataset_losses = []
            for cont, cat, tar in validation_batch_generator:
                cat, cont, tar = cat.to(DEVICE), cont.to(DEVICE), tar.to(DEVICE)
                val_pred = model((cat, cont))

                loss = loss_fn(val_pred, tar.reshape(-1, 3))

                dataset_losses.append(loss)

                predictions.append(torch.softmax(val_pred, dim=1).cpu())
                truth.append(tar.cpu())
                weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())
            # create event based weight tensor for dataset

            average_val = sum(dataset_losses) / len(dataset_losses) * sampler[uid].relative_weight
            val_loss.append(average_val)

        final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
        model.train()

        truth = torch.concatenate(truth, dim=0)
        predictions = torch.concatenate(predictions, dim=0)
        weights = torch.flatten(torch.concatenate(weights, dim=0))
        return final_validation_loss, (predictions, truth, weights)


# load data
era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
datasets =  find_datasets(["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*"], list(era_map.keys()), "root")
debugging = False
continous_features, categorical_features = input_features(debug=debugging, debug_length=3)
target_map = {"hh" : 0, "dy": 1, "tt": 2}

dataset_config = {
    "min_events": 3,
    "continous_features" : continous_features,
    "categorical_features": categorical_features,
    "eras" : list(era_map.keys()),
    "datasets" : datasets,
    "cuts" : None,
}

# config of network
layer_config = {
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "nodes": 128,
    "num_resblocks": 3,
    "activation_functions": "SiLu",
    "embedding_dim": 4,
    "empty_value": 15,
    "eps": 0.5e-5,
    "linear_layer_normalization": False,
    "skip_connection_init": 1,
    "freeze_skip_connection": False,
}

config = {
    "lr":1e-2,
    "gamma":0.9,
    "label_smoothing":0,
    "train_folds" : (0,),
    "k_fold" : 5,
    "seed" : 1,
    "train_ratio" : 0.75,
    "modelname": "very_long_test",
    "max_iteration": 10000,
    "scheduler_intervall":100,
    "validation_interval": 200,
    "v_batch_size" : 4096 * 8,
    "t_batch_size" : 4096,
}

tboard_writer = TensorboardLogger(name=hash_config(config))
logger.warn(f"Tensorboard logs are stored in {tboard_writer.path}")

for current_fold in (config["train_folds"]):
    logger.info(f'Start Training of fold {current_fold} from {config["k_fold"] - 1}')

    ### data preparation
    # Hint: order matters, due to memory constraints views are moved in and out of dictionaries

    # load data from cache is necessary or from root files
    # events is of form : {uid : {"continous","categorical","weight": torch tensor}
    events = get_data(dataset_config, overwrite=False, _save_cache=True)
    # create k-folds, whe current fold is test fold and leave out
    train_data, validation_data = split_k_fold_into_training_and_validation(
        events,
        c_fold=current_fold,
        k_fold=config["k_fold"],
        seed=config["seed"],
        train_ratio=config["train_ratio"],
    )
    # get weighted mean and std of expected batch composition
    layer_config["mean"],layer_config["std"] = get_batch_statistics(train_data, padding_value=-99999)

    # create train and validation sampler from k-1 folds
    # trainings_sampler create a composition of all subphasespaces in a batch
    # Test = test_sampler(
    #     train_data,
    #     target_map = {"hh" : 0, "dy": 1, "tt": 2},
    #     min_size=3,
    #     batch_size=config["t_batch_size"],
    #     train=True
    # )

    training_sampler, validation_sampler = create_train_and_validation_sampler(
        t_data = train_data,
        v_data = validation_data,
        t_batch_size = config["t_batch_size"],
        v_batch_size = config["v_batch_size"],
        target_map=target_map,
        min_size=1
    )


    ### Model setup
    models_input_layer, model = create_model.init_layers(dataset_config["continous_features"], dataset_config["categorical_features"], config=layer_config)
    model = model.to(DEVICE)

    # TODO: only linear models should contribute to weight decay
    # TODO : SAMW Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["gamma"])

    # HINT: requires only logits, no softmax at end
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])
    max_iteration = config["max_iteration"]
    scheduler_intervall = config["scheduler_intervall"]
    LOG_INTERVAL = 10
    validation_interval = config["validation_interval"]
    model.train()

    # training loop:


    v_losses = []
    t_losses = []
    for iteration in range(max_iteration):
        t_loss, (t_pred, t_targets) = training(
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer,
            sampler = training_sampler
        )

        if (iteration % validation_interval == 0) and (iteration > 0):
            v_loss, (v_pred, v_tar, v_weights) = validation(model, loss_fn, validation_sampler)
            # loss curve plot
            loss_fig, loss_ax = plt.subplots()
            v_losses.append(v_loss.item())
            t_losses.append(t_loss.item())
            loss_ax.plot(list(range(0, iteration, validation_interval)), v_losses, color="blue", label="validation loss")
            loss_ax.plot(list(range(0, iteration, validation_interval)), t_losses, color="orange", label="training loss")


            # network prediction plot
            pred_fig, pred_ax = plotting.network_predictions(
                v_tar,
                v_pred,
                target_map
            )
            # confusion matrix plot
            c_mat_fig, c_mat_ax, c_mat = plotting.confusion_matrix(
                v_tar,
                v_pred,
                target_map,
                sample_weight=v_weights,
                normalized="true"
            )

            roc_fig, roc_ax = plotting.roc_curve(v_tar, v_pred, sample_weight=v_weights, labels=list(target_map.keys()))
            # tensorboard logging:
            tboard_writer.log_loss({"train": t_loss, "validation": v_loss}, step=iteration)
            tboard_writer.log_lr(optimizer.param_groups[0]["lr"], step=iteration)
            tboard_writer.log_figure("confusion matrix validation", c_mat_fig, step=iteration)
            tboard_writer.log_figure("node output validation", pred_fig, step=iteration)
            tboard_writer.log_figure("roc curve one vs rest", roc_fig, step=iteration)

            # TODO: metrics calculation
            _metrics = metrics.calculate_metrics(
                v_tar,
                v_pred,
                label=list(target_map.keys()),
                weights=v_weights,
            )

            tboard_writer.log_precision(_metrics, step=iteration)
            tboard_writer.log_sensitivity(_metrics, step=iteration)

        # VERBOSITY
        if iteration % LOG_INTERVAL == 0:
            print(f"iteration: {iteration} - batch loss: {t_loss.item():.4f}")
        if (iteration % validation_interval == 0) and (iteration > 0):
            print(f"Step {iteration} Validation Loss: {v_loss:.4f}")
            
        if iteration % scheduler_intervall == 0:
            scheduler.step()
            print(f"Step {iteration} reached, lowering learning rate.")
    # TODO release DATA from previous RUN

    # save network
    import export
    cont, cat, tar = training_sampler.sample_batch(device=CPU)
    export.torch_export(
        model = model.to(CPU),
        dst_path = f"models/fold_{current_fold}_final_model.pt2",
        input_tensors = (cat.to(torch.int32), cont)
    )

        

    # TODO release DATA from previous RUN

#torch_export(model.to(CPU), config["modelname"], training_sampler.sample_batch(device=CPU))
