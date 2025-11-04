import torch
from models import create_model
from data.features import input_features
from data.load_data import get_data, find_datasets
from data.preprocessing import (
    create_train_or_validation_sampler, get_batch_statistics, split_k_fold_into_training_and_validation
    )
from utils.logger import get_logger

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logger = get_logger(__name__)

def validation(model, sampler):
    model.eval()
    with torch.no_grad():
        print("Starting Validation:")
        # run validation every x steps
        val_loss = []
        model.eval()
        val_batch_size = 2048 * 8
        for uid, validation_batch_generator in sampler.get_dataset_batch_generators(
            batch_size=val_batch_size,
            reset=True,
            device=DEVICE
            ).items():
            dataset_losses = []

            for cont, cat, tar in validation_batch_generator:
                cat, cont, tar = cat.to(DEVICE), cont.to(DEVICE), tar.to(DEVICE)
                val_pred = model((cat, cont))
                # TODO weighting needs to be done
                dataset_losses.append(loss_fn(val_pred, tar))

            average_val = sum(dataset_losses) / len(dataset_losses)
            val_loss.append(average_val)

            # print(f"dataset: {uid}, val_loss contribution {average_val:.4f}")
        final_validation_loss = sum(val_loss) / len(val_loss)
        print(f"Step {iteration} Validation Loss: {final_validation_loss:.4f}")
    model.train()
    return final_validation_loss


# load data
eras = ["22pre", "22post", "23pre", "23post"]
era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
datasets =  find_datasets(["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*"], eras, "root")
debugging = False
continous_features, categorical_features = input_features(debug=debugging, debug_length=3)


dataset_config = {
    "min_events": 3,
    "continous_features" : continous_features,
    "categorical_features": categorical_features,
    "eras" : eras,
    "datasets" : datasets,
}

# config of network
layer_config = {
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "nodes": 32,
    "num_resblocks": 1,
    "activation_functions": "PReLu",
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
    "train_ratio" : 0.75
}


for current_fold in (config["train_folds"]):
    logger.info(f"Start Training of fold {current_fold} from {config["k_fold"] - 1}")

    ### data preparation
    # Hint: order matters, due to memory constraints views are moved in and out of dictionaries

    # load data from cache is necessary or from root files
    # events is of form : {uid : {"continous","categorical","weight": torch tensor}
    events = get_data(dataset_config, overwrite=False)

    # create k-folds, whe current fold is test fold and leave out
    train_data, validation_data = split_k_fold_into_training_and_validation(
        events,
        c_fold=current_fold,
        k_fold=config["k_fold"],
        seed=config["seed"],
        train_ratio=config["train_ratio"],
        test=False,
    )

    # get weighted mean and std of expected batch composition
    layer_config["mean"],layer_config["std"] = get_batch_statistics(train_data, padding_value=-99999)

    # create train and validation sampler from k-1 folds
    # trainings_sampler create a composition of all subphasespaces in a batch

    training_sampler = create_train_or_validation_sampler(
        train_data,
        target_map = {"hh" : 0, "dy": 1, "tt": 2},
        min_size=3,
        batch_size=4096,
        train=True
    )
    # validation_sampler just loop over all datasets
    validation_sampler = create_train_or_validation_sampler(
        validation_data,
        target_map = {"hh" : 0, "dy": 1, "tt": 2},
        min_size=0,
        batch_size=4096 * 8, # since no gradient is required, higher batch sizes are possible
        train=False,
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

    # loss = loss_fn(pred, target)
    # from IPython import embed; embed(header="string - 614 in bognet.py ")
    # loss.backward()
    # optimizer.first_step(zero_grad=True)

    # # second forward step with disabled bachnorm running stats in second forward step
    # disable_running_stats(model)
    # pred_2 = self(categorical_x, continous_x)
    # loss_fn(pred_2, target).backward()
    # optimizer.second_step(zero_grad=True)
    max_iteration = 1000
    LOG_INTERVAL = 10
    validation_interval = 200
    model.train()
    running_loss = 0.0

    # training loop:

    for iteration in range(max_iteration):
        optimizer.zero_grad()

        cont, cat, targets = training_sampler.sample_batch(device=DEVICE)
        targets = targets.to(torch.float32)


        pred = model((cat,cont))

        loss = loss_fn(pred, targets.reshape(-1,3))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if iteration % LOG_INTERVAL == 0:
            print(f"Step {iteration} Loss: {loss.item():.4f}")

        if iteration % validation_interval == 0:
            val_loss = validation(model, validation_sampler)

    # TODO release DATA from previous RUN

    from IPython import embed; embed(header="END - 89 in train.py ")
from IPython import embed; embed(header="FINAL - 89 in train.py ")
