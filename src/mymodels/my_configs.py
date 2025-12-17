from data.features import input_features
from data.load_data import find_datasets

# targets and inputs
target_map = {"hh" : 0, "dy": 1, "tt": 2}
continous_features, categorical_features = input_features(debug=False, debug_length=3)
dataset_pattern = ["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*","hh_ggf_hbb_htt_kl1_kt1*"]
eras = ["22pre", "22post", "23pre", "23post"]
datasets =  find_datasets(dataset_pattern, eras, "root")

dataset_config = {
    "continous_features" : continous_features,
    "categorical_features": categorical_features,
    "eras" : eras,
    "datasets" : datasets,
    "cuts" : None,
}

# config of network
model_building_config = {
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "layers_and_nodes": [256,256,256,256,256,256,256,256],
    "activation_functions": "SiLu",
    "embedding_dim": 20,
    "empty_value": 15,
    "eps": 0.5e-5,
    "linear_layer_normalization": False,
    "skip_connection_init": 1,
    "freeze_skip_connection": False,
}

config = {
    "label_smoothing":0,
    "train_folds" : (0,),
    "k_fold" : 5,
    "seed" : 1,
    "train_ratio" : 0.75,
    "v_batch_size" : 4096 * 8,
    "t_batch_size" : 4096 * 8,
    "sample_ratio" : {"dy": 1/3, "tt": 1/3, "hh": 1/3},
    "modelname": "torch_dense_0",
    "max_iteration": 100000000,
    "validation_interval": 500,
    "early_stopping_patience": 6,
    "patience": 3,
    "min_events_in_batch": 1,

}

optimizer_config = {
    "apply_to": "weight",
    "decay_factor": 500,
    "normalize": True,
    "lr":1e-2,
    "learning_rate_reduction_factor":0.1,
}
