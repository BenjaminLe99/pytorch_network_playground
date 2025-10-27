import torch
#from utils import utils
from models import layers, create_model
from data import extract_features, load_data


# comments on the current ml runs
print("=======================================================")
print("Reminder: Currently using only 1 dy dataset for testing")
print("=======================================================")

# load data

debugging = False
eras = ["22pre"]
datasets = load_data.find_datasets(["dy_m50toinf_amcatnlo","tt_dl*", "hh_ggf_hbb_htt_kl0_kt1*"], eras, "root")

dataset_config = {
    "min_events":3,
    "continous_features" : extract_features.continous_features if not debugging else extract_features.continous_features[:2],
    "categorical_features": extract_features.categorical_features if not debugging else extract_features.categorical_features[:2],
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
    "lr":1e-4,
    "lr_gamma":0.9,
    "label_smoothing":0,
    "L2": 0,
}

events = load_data.get_data(dataset_config)
sampler = load_data.create_sampler(
    events,
    input_columns=dataset_config["continous_features"] + dataset_config["categorical_features"],
    dtype=torch.float32,
    min_size=3,
)

models_input_layer, model = create_model.init_layers(dataset_config["continous_features"], dataset_config["categorical_features"], config=layer_config)
max_iteration = 100

# training loop:
model.train()

# TODO: use split of parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["L2"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["lr_gamma"])

# HINT: requires only logits, no softmax at end
loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG_INTERVAL = 10
model.train()
running_loss = 0.0

for iteration in range(max_iteration):

    optimizer.zero_grad()

    inputs, targets = sampler.get_batch()
    inputs, targets = inputs.to(device), targets.to(device)
    continous_inputs, categorical_input = inputs[:, :len(dataset_config["continous_features"])], inputs[:, len(dataset_config["continous_features"]):]
    pred = model((categorical_input,continous_inputs))

    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if iteration % LOG_INTERVAL == 0:
        print(f"Step {iteration} Loss: {loss.item():.4f}")

def torch_export(model, dst_path, input_tensors):
    from pathlib import Path
    model.eval()

    cat, con = input_tensors

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    dim = torch.export.dynamic_shapes.Dim.AUTO
    dynamic_shapes = {
        "input": ((dim, cat.shape[-1]), (dim, con[-1]))
    }

    exp = torch.export.export(
        model,
        args=((categorical_input, continous_inputs),),
        dynamic_shapes=dynamic_shapes,
    )

    p = Path(f"{dst_path}").with_suffix(".pt2")
    torch.export.save(exp, p)


def run_exported_tensor_model(pt2_path, input_tensors):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(input_tensors)
    return scores
