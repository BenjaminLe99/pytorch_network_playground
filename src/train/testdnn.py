import torch
from mymodels import recreate_simple
from data.features import input_features
from data.load_data import get_data, find_datasets
from data.preprocessing import (
    create_train_and_validation_sampler, get_batch_statistics, split_k_fold_into_training_and_validation, test_sampler
    )
from utils.logger import get_logger
from ml_save import torch_export

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
    return loss


def validation(model, loss_fn, sampler):
    with torch.no_grad():
        print("Starting Validation:")
        # run validation every x steps
        val_loss = []
        model.eval()

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

            average_val = sum(dataset_losses) / len(dataset_losses) * sampler[uid].relative_weight
            val_loss.append(average_val)

            # print(f"dataset: {uid}, val_loss contribution {average_val:.4f}")
        final_validation_loss = sum(val_loss)#/ len(val_loss)
        model.train()
        return final_validation_loss
    
def init_weights_he_uniform(model):
    if isinstance(model, torch.nn.Linear):  # includes subclasses
        torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)



# load data
era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
datasets =  find_datasets(["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*"], list(era_map.keys()), "root")
debugging = False
continous_features, categorical_features = input_features(debug=debugging, debug_length=3)


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
    "layers_and_nodes": [128,128,128,128,128,128],
    "activation_functions": "elu",
    "embedding_dim": 10,
    "empty_value": 15,
    "eps": 0.5e-5,
    "linear_layer_normalization": False,
    "skip_connection_init": 1,
    "freeze_skip_connection": False,
}

config = {
    "lr":10e-3,
    "learning_rate_reduction_factor":0.5,
    "weight_decay": 5000,
    "label_smoothing":0,
    "train_folds" : (0,),
    "k_fold" : 5,
    "seed" : 1,
    "train_ratio" : 0.75,
    "modelname": "simple_recreated_test_ReLu",
    "max_iteration": 10000,
    "scheduler_intervall": 810,
    "validation_interval": 500,
    "v_batch_size" : 4096 * 8,
    "t_batch_size" : 4096,
}


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
        target_map={"hh" : 0, "dy": 1, "tt": 2},
        min_size=1
    )

    ### Model setup
    models_input_layer, model = recreate_simple.init_layers(dataset_config["continous_features"], dataset_config["categorical_features"], config=layer_config)
    model = model.to(DEVICE)
    model.apply(init_weights_he_uniform) #apply he uniform weight initialization

    # get the number of trainable weights
    num_trainable_weights = sum(
        p.numel() for name, p in model.named_parameters() 
        if p.requires_grad and "weight" in name
    )

    # normalize the weight decay parameter
    config["weight_decay"] = 500/num_trainable_weights

    # TODO: only linear models should contribute to weight decay
    # TODO : SAMW Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["learning_rate_reduction_factor"])

    # HINT: requires only logits, no softmax at end
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])
    max_iteration = config["max_iteration"]
    scheduler_intervall = config["scheduler_intervall"]
    LOG_INTERVAL = 10
    validation_interval = config["validation_interval"]
    model.train()
    running_loss = 0.0
    best_loss = 0.0
    best_iteration = 0

    # training loop:

    for iteration in range(max_iteration):
        loss = training(
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer,
            sampler = training_sampler
        )
        running_loss += loss.item()
        if iteration % LOG_INTERVAL == 0:
            print(f"Step {iteration} Loss: {loss.item():.4f}")

        if iteration % validation_interval == 0:
            val_loss = validation(model, loss_fn, validation_sampler)
            print(f"Step {iteration} Validation Loss: {val_loss:.4f}")
        
        # learningrate scheduler. Save best loss and iteration, compare until 10 iterations after -> reduce learningrate
        if loss.item() > best_loss:
            best_loss = loss.item()
            best_iteration = iteration
            if best_iteration < 10:
                pass
            else:
                scheduler.step()
            


        # if iteration != 0 and iteration % scheduler_intervall == 0:
        #     scheduler.step()
        #     print(f"Step {iteration} reached, lowering learning rate.")

    # TODO release DATA from previous RUN

torch_export(model.to(CPU), config["modelname"], training_sampler.sample_batch(device=CPU))
