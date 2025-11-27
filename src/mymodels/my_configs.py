from data.features import input_features
from data.load_data import find_datasets

# targets and inputs
target_map = {"hh" : 0, "dy": 1, "tt": 2}
continous_features, categorical_features = input_features(debug=False, debug_length=3)
dataset_pattern = ["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*","hh_ggf_hbb_htt_kl1_kt1*"]
era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
datasets =  find_datasets(dataset_pattern, list(era_map.keys()), "root")

dataset_config = {
    "min_events": 3,
    "continous_features" : continous_features,
    "categorical_features": categorical_features,
    "eras" : list(era_map.keys()),
    "datasets" : datasets,
    "cuts" : None,
}

# config of network
model_building_config = {
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
    "label_smoothing":0,
    "train_folds" : (0,),
    "k_fold" : 5,
    "seed" : 1,
    "train_ratio" : 0.75,
    "v_batch_size" : 4096 * 8,
    "t_batch_size" : 4096,
    "sample_ratio" : {"dy": 0.33, "tt": 0.33, "hh": 0.33},
    "modelname": "simple_dense_fixed_standardization_test2",
    "max_iteration": 100000000,
    "validation_interval": 500,
    "early_stopping_patience": 10,
    "patience": 5,

}

optimizer_config = {
    "apply_to": "weight",
    "decay_factor": 500,
    "normalize": True,
    "lr":1e-3,
    "learning_rate_reduction_factor":0.5,
}
