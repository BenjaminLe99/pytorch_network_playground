import torch
#from utils import utils
from models import layers, create_model
from data import features
from data.load_data import get_data, find_datasets, create_sampler, get_batch_statistics


CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load data
eras = ["22pre", "22post", "23pre", "23post"]
era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
datasets = find_datasets(["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*"], eras, "root")
debugging = False

dataset_config = {
    "min_events":3,
    "continous_features" : features.continous_features if not debugging else features.continous_features[:2],
    "categorical_features": features.categorical_features if not debugging else features.categorical_features[:2],
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
}


events = get_data(dataset_config, overwrite=False)
layer_config["mean"],layer_config["std"] = get_batch_statistics(events, padding_value=-99999)



sampler = create_sampler(
    events,
    target_map = {"hh" : 0, "dy": 1, "tt": 2},
    min_size=3,
)


# training loop:
# TODO: use split of parameters
models_input_layer, model = create_model.init_layers(dataset_config["continous_features"], dataset_config["categorical_features"], config=layer_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["gamma"])
model = model.to(DEVICE)
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
max_iteration = 100
LOG_INTERVAL = 10
model.train()
running_loss = 0.0

for iteration in range(max_iteration):
    optimizer.zero_grad()

    cont, cat, targets = sampler.get_batch()
    #targets = targets.to(torch.float32)
    cont, cat, targets = cont.to(DEVICE), cat.to(DEVICE), targets.to(DEVICE)

    pred = model((cat,cont))

    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if iteration % LOG_INTERVAL == 0:
        print(f"Step {iteration} Loss: {loss.item():.4f}")
