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
    # print(f"copy mean and std")
    # with open("/afs/desy.de/user/r/riegerma/public/stats.json") as file:
    #     stats = json.load(file)
    stats = [
    (15.858491278811423, 4191.30204100641),
    (-0.04432542910925021, 2110.494672086883),
    (2532.7766288002913, 223773335.59726715),
    (-3.7108481557075947, 11221919.546195747),
    (2534.382322342528, 223914202.8486784),
    (48.38864537230531, 2157.8087599567243),
    (-0.017913359106840204, 901.491000578751),
    (-0.7354050473193622, 12618.382150742746),
    (103.99709468829323, 7200.039280887375),
    (36.60236687205281, 2104.4128018563324),
    (0.017913366153377276, 901.4910005748718),
    (-0.43672649230840055, 8960.259162118213),
    (89.17281088633982, 5352.3906987192795),
    (-61.67884647836569, 6708.890603488575),
    (1.3505356804622026, 3230.0902166833853),
    (1.3843100369067127, 30268.66027426911),
    (156.67378865532635, 19642.64419597229),
    (0.5674907034628309, 0.19802618856516135),
    (0.3847287059940151, 0.16244164430341085),
    (0.6592813581419824, 0.15090981662060943),
    (1.2341005309293527, 0.1320027919744653),
    (-29.701389640798954, 3773.7241298412423),
    (-1.0004184678937833, 2197.4966487363254),
    (-1.0575517917500283, 24431.090261948124),
    (122.872842360955, 16316.931521579687),
    (0.27850401308451045, 0.16443418170474494),
    (0.6152719988367581, 0.14526833069767375),
    (0.4121601673834433, 0.1672779520307898),
    (0.7052696182454151, 0.11909050940961535),
    (-12.950995295682548, 5735.470756001696),
    (0.011041720018664259, 1200.6494037783673),
    (-0.20236664683091024, 14051.29410326379),
    (30.858920847387385, 20865.002168388583),
    (193.16990558325196, 14027.635241188866),
    (84.991012243531, 3469.637875017611),
    (7.027865999455247e-09, 7.764272890287476e-11),
    (-1.1721315397168686, 30431.672961989687),
    (279.54663101343516, 37787.6258749079),
    (-91.38023611894972, 9747.510956840975),
    (0.3501172130278488, 3518.8487451592728),
    (0.3267582447866987, 67709.44960463041),
    (472.7165366002367, 63261.86626982162),
    (-6.389223875700382, 5886.256291697084),
    (0.35011721909559584, 3518.8487442935775),
    (-0.8453733006631203, 134559.44075866617),
    (46.63333923135851, 44367.11242451363),
    (-3.650465026002929, 2595.793547858017),
    (0.011041717988109743, 1200.6494037642217),
    (-0.2541869319142637, 23290.900359563348)
    ]

    f_mean, f_std = [], []
    # stat = stats["stats"]
    for feature, (mean, std) in zip(continous_features, stats):
        f_mean.append(mean)
        f_std.append(std**0.5)
        # f_mean.append(stat[feature]["mean"])
        # f_std.append(stat[feature]["var"])
    f_mean = torch.tensor(f_mean)
    f_std = torch.tensor(f_std)
    # modify models state dict

    state_dict["input_layer.std_layer.mean"] = f_mean
    state_dict["input_layer.std_layer.std"] = f_std
    return state_dict

def load_marcels_weights(model, continous_features):
    m_state = model.state_dict()
    m_state = modify_state_dict_with_marcels_weights(m_state)
    m_state = replace_standardization_layer_state_dict(m_state, continous_features)
    model.load_state_dict(m_state)
    return model
