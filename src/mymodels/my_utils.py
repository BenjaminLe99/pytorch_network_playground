import torch
import numpy as np
import json
import os, cloudpickle


# weight initialization
def init_weights_he_uniform(model):
    if isinstance(model, torch.nn.Linear):  # includes subclasses
        torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)

# get marcels statistics
def get_marcel_stats(layer_config):
    with open("/afs/desy.de/user/r/riegerma/public/stats.json") as file:
        mean_and_var = json.load(file)

    stds = torch.tensor([np.sqrt(value["var"]) for key, value in mean_and_var["stats"].items()])
    means = torch.tensor([value["mean"] for key, value in mean_and_var["stats"].items()])
    layer_config["mean"] = means
    layer_config["std"] = stds

def save_marcels_weights(model):
    with open(os.path.expanduser("/afs/desy.de/user/r/riegerma/public/weights.pkl"), "rb") as f:
        weights: dict[str, np.ndarray] = cloudpickle.load(f)

    # keys and values i wanna retain
    preserved_keys = ["input_layer.embedding_layer.tokenizer.map", 
                     "input_layer.embedding_layer.tokenizer.min", 
                     "input_layer.embedding_layer.tokenizer.indices",
                     "input_layer.std_layer.mean",
                     "input_layer.std_layer.std",
                     "input_layer.padding_continous_layer.padding_value",
                     "input_layer.padding_continous_layer.mask_value",
                     "input_layer.padding_categorical_layer.padding_value",
                     "input_layer.padding_categorical_layer.mask_value",
                     "first_hidden.bn.num_batches_tracked",
                     "hidden_layers.0.bn.num_batches_tracked",
                     "hidden_layers.1.bn.num_batches_tracked",
                     "hidden_layers.2.bn.num_batches_tracked",
                     "hidden_layers.3.bn.num_batches_tracked",
                     "hidden_layers.4.bn.num_batches_tracked",
                     ]
    # values i wanna replace
    model_keys = {key: value for key, value in model.state_dict().items() if key not in preserved_keys}
    # values to keep
    missing_keys = {key: value for key, value in model.state_dict().items() if key in preserved_keys}
    # relaced values with marcels weights
    weights_from_marcel = dict(zip(model_keys.keys(), weights.values()))

    # full merged dict
    model_adjusted = weights_from_marcel | missing_keys

    # turn into torch tensors
    model_adjusted2 = {key: torch.tensor(values) for key, values in model_adjusted.items()}

    weight_layers = ["first_hidden.linear.weight",
                     "hidden_layers.0.linear.weight",
                     "hidden_layers.1.linear.weight",
                     "hidden_layers.2.linear.weight",
                     "hidden_layers.3.linear.weight",
                     "hidden_layers.4.linear.weight",
                     "output_layer.weight"]

    # transpose necessary layers
    model_adjusted3 = {key: (values if key not in weight_layers else torch.transpose(values,0,1)) for key, values in model_adjusted2.items()}

    model.load_state_dict(model_adjusted3)

