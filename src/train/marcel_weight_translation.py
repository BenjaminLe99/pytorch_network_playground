import torch
import os
import cloudpickle
import numpy as np
import json
import copy

def modify_state_dict_with_marcels_weights(state_dict):
    layer_translation = {
        "dnn_cat_embedded/embeddings:0" : "input_layer.embedding_layer.embeddings.weight",
        "dense_1/kernel:0" : "transition_dense_1.linear.weight",
        "dense_1/bias:0" : "transition_dense_1.linear.bias",
        "batchnorm_1/gamma:0" : "transition_dense_1.bn.weight",
        "batchnorm_1/beta:0" : "transition_dense_1.bn.bias",
        "batchnorm_1/moving_mean:0" : "transition_dense_1.bn.running_mean",
        "batchnorm_1/moving_variance:0" : "transition_dense_1.bn.running_var",
        "dense_2/kernel:0" : "dense_block_1.dense_block.linear.weight",
        "dense_2/bias:0" : "dense_block_1.dense_block.linear.bias",
        "batchnorm_2/gamma:0" : "dense_block_1.dense_block.bn.weight",
        "batchnorm_2/beta:0" : "dense_block_1.dense_block.bn.bias",
        "batchnorm_2/moving_mean:0" : "dense_block_1.dense_block.bn.running_mean",
        "batchnorm_2/moving_variance:0" : "dense_block_1.dense_block.bn.running_var",
        "dense_3/kernel:0" : "dense_block_2.dense_block.linear.weight",
        "dense_3/bias:0" : "dense_block_2.dense_block.linear.bias",
        "batchnorm_3/gamma:0" : "dense_block_2.dense_block.bn.weight",
        "batchnorm_3/beta:0" : "dense_block_2.dense_block.bn.bias",
        "batchnorm_3/moving_mean:0" : "dense_block_2.dense_block.bn.running_mean",
        "batchnorm_3/moving_variance:0" : "dense_block_2.dense_block.bn.running_var",
        "dense_4/kernel:0" : "dense_block_3.dense_block.linear.weight",
        "dense_4/bias:0" : "dense_block_3.dense_block.linear.bias",
        "batchnorm_4/gamma:0" : "dense_block_3.dense_block.bn.weight",
        "batchnorm_4/beta:0" : "dense_block_3.dense_block.bn.bias",
        "batchnorm_4/moving_mean:0" : "dense_block_3.dense_block.bn.running_mean",
        "batchnorm_4/moving_variance:0" : "dense_block_3.dense_block.bn.running_var",
        "dense_5/kernel:0" : "dense_block_4.dense_block.linear.weight",
        "dense_5/bias:0" : "dense_block_4.dense_block.linear.bias",
        "batchnorm_5/gamma:0" : "dense_block_4.dense_block.bn.weight",
        "batchnorm_5/beta:0" : "dense_block_4.dense_block.bn.bias",
        "batchnorm_5/moving_mean:0" : "dense_block_4.dense_block.bn.running_mean",
        "batchnorm_5/moving_variance:0" : "dense_block_4.dense_block.bn.running_var",
        "dense_6/kernel:0" : "dense_block_5.dense_block.linear.weight",
        "dense_6/bias:0" : "dense_block_5.dense_block.linear.bias",
        "batchnorm_6/gamma:0" : "dense_block_5.dense_block.bn.weight",
        "batchnorm_6/beta:0" : "dense_block_5.dense_block.bn.bias",
        "batchnorm_6/moving_mean:0" : "dense_block_5.dense_block.bn.running_mean",
        "batchnorm_6/moving_variance:0" : "dense_block_5.dense_block.bn.running_var",
        "dense_7/kernel:0" : "last_linear.weight",
        "dense_7/bias:0" : "last_linear.bias",
    }

    # load marcels weights from pickle

    with open(os.path.expanduser("/afs/desy.de/user/r/riegerma/public/weights.pkl"), "rb") as f:
        marcel_weights: dict[str, np.ndarray] = cloudpickle.load(f)

    # copy external dict to prevent inplace changes
    my_model_state_dict = copy.deepcopy(state_dict)

    # modify models state dict
    for key in marcel_weights.keys():
        # replace weights when inside the translation
        if key in layer_translation.keys():
            original_key = key
            translation_key = layer_translation[key]
            m_layer_weight = marcel_weights[original_key]

            # transpose linear weights (since tf and py save it like that)
            if (key.startswith("dense") and "kernel" in key):
                print(f"tranpose {key}")
                m_layer_weight = m_layer_weight.transpose()
            print(f"copy weights from {key} -> {translation_key}")
            my_model_state_dict[translation_key] = torch.from_numpy(m_layer_weight)
    return my_model_state_dict

def replace_standardization_layer_state_dict(state_dict, continous_features):
    # load standardization layers parameter from jsommn
    print(f"copy mean and std")
    with open("/afs/desy.de/user/r/riegerma/public/stats.json") as file:
        stats = json.load(file)

    f_mean, f_std = [], []
    stat = stats["stats"]
    for feature in continous_features:
        f_mean.append(stat[feature]["mean"])
        f_std.append(stat[feature]["var"])
    f_mean = torch.tensor(f_mean)
    f_std = torch.sqrt(torch.tensor(f_std))
    # modify models state dict

    state_dict["std_layer.mean"] = f_mean
    state_dict["std_layer.std"] = f_std
    return state_dict

def load_marcels_weights(model, continous_features):
    m_state = model.state_dict()
    m_state = modify_state_dict_with_marcels_weights(m_state)
    m_state = replace_standardization_layer_state_dict(m_state, continous_features)
    model.load_state_dict(m_state)
    return model
