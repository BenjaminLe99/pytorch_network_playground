from data.features import input_features
from data.load_data import find_datasets

### data config
def get_dataset_config(pattern_list: list, eras_list: list) -> dict:
    dataset_pattern = []

    if 'dy' in pattern_list:
        dataset_pattern += ['dy_*'] #swap tt and dy in the brackets for debugging
    if 'tt' in pattern_list:
        dataset_pattern += ['tt_*']
    if 'kl0' in pattern_list:
        dataset_pattern += ['hh_ggf_hbb_htt_kl0_kt1*']
    if 'kl1' in pattern_list:
        dataset_pattern += ['hh_ggf_hbb_htt_kl1_kt1*']
    if 'kl5' in pattern_list:
        dataset_pattern += ['hh_ggf_hbb_htt_kl5*']
    if 'kl2p45' in pattern_list:
        dataset_pattern += ['hh_ggf_hbb_htt_kl2p45*']
    if 'all' in pattern_list:
        dataset_pattern = ["dy_*","tt_*","hh_ggf_hbb_htt_kl0_kt1*","hh_ggf_hbb_htt_kl1_kt1*","hh_ggf_hbb_htt_kl5*","hh_ggf_hbb_htt_kl2p45*"]
    print(f"Starting training with datasets:{dataset_pattern}")

    continous_features, categorical_features = input_features(debug=False, debug_length=3)
    eras = []
    
    if '22pre' in eras_list:
        eras += ['22pre']
    if '22post' in eras_list:
        eras += ['22post']
    if '23pre' in eras_list:
        eras += ['23pre']
    if '23post' in eras_list:
        eras += ['23post']
    if 'all' in eras_list:
        eras = ["22pre", "22post", "23pre", "23post"]

    print(f'Eras: {eras}')

    datasets =  find_datasets(dataset_pattern, eras, "root", verbose=False)
    
    # hh case for going back and comparing
    target_list = []
    if 'hh' not in pattern_list:
        target_list = pattern_list
    else:
        target_list = ['hh','tt','dy']

    # define the target map depending on what kappa lambda datasets you use
    target_map = {cls: idx for idx, cls in enumerate(target_list)}
    print(f"target map: {target_map}")
    
    # changes in this dictionary will create a NEW hash of the data
    dataset_config = {
        "continous_features" : continous_features,
        "categorical_features": categorical_features,
        "eras" : eras,
        "datasets" : datasets,
        "cuts" : "(vbf_dnn_moe_hh_vbf < 0.5)",
        "target_map": target_map,
    }
    return dataset_config

# config of network
model_building_config = {
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "categorical_padding_value": None,
    "continous_padding_value": None,
    "nodes": 128,
    "activation_functions": "elu",
    "skip_connection_init": 1,
    "freeze_skip_connection": True,
    "batch_norm_eps" : 0.001, # marcel : 0.001
    "LBN_M" : 10,

}

config = {
    "max_train_iteration" : 100000000,
    "verbose_interval" : 25,
    "validation_interval" : 500,
    "gamma":0.5,
    "label_smoothing":0,
    "train_folds" : (0,),
    "k_fold" : 5,
    "seed" : 1,
    "train_ratio" : 0.75,
    "v_batch_size" : 2**17, # old batchsize: 4096*8 = 2**16
    "t_batch_size" : 2**14, # old batchsize: 4096 = 2**13
    "min_events_in_batch": 1,
    "early_stopping_patience" : 10, # marcel : 10
    "early_stopping_min_delta" : 0, # marcel : 0
    "get_batch_statistic_return_dummy" : False,
    "load_marcel_stats" : False,
    "load_marcel_weights" : False,
    "training_fn" : "default", # chooses the training function
    "validation_fn" : "default"
}

scheduler_config = {
    "patience" : 10 - 1, # marcel : 10, starts counting from 0
    "min_delta" : 0, # marcel : 0
    "threshold_mode" : "abs", # marcel : abs
    "factor" : 0.5,
}

optimizer_config = {
    "apply_to": "weight",
    "decay_factor": 500,
    "normalize": True,
}

extra_losses = None
# extra_losses = {
#     "cross_entropy": {
#         "mode": "cross_entropy",
#         "content": None
#     }
# }