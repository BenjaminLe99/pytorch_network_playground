# standard imports
import torch
import numpy as np
import random
import argparse
import hashlib
import json
import os

# project imports
from models import create_model
from data.load_data import get_data
from data.preprocessing import (
    create_train_or_validation_sampler, get_batch_statistics_from_sampler, split_k_fold_into_training_and_validation,
    )
from utils.logger import get_logger, TensorboardLogger
from utils.utils import register_model
from data.cache import hash_config
import optimizer
from train_config import (
    config, get_dataset_config, model_building_config, optimizer_config, scheduler_config, extra_losses
)
from train_utils import training_fn, validation_fn, log_metrics, lr_multiplier, validation_frequency
from early_stopping import EarlyStopSignal, EarlyStopOnPlateau
from export import torch_save, torch_export_v2
import marcel_weight_translation as mwt
from loss import WeightedFalseClassPenaltyLogLoss
from fractions import Fraction
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def parse_fraction(value):
        return float(Fraction(value))

    parser.add_argument(
        "--datasets",
        required=True,
        type=lambda s: s.split(","),
        help="List of: dy, tt, kl0, kl1, kl2p45 or kl5",
    )

    parser.add_argument(
        "--sample_ratio",
        required=True,
        type=parse_fraction,
        nargs="+",
        help="List of numbers. Must sum to 1 and number numbers assigned must match the datasets."
    )

    parser.add_argument(
        "--modelname",
        required=True,
        help="Name of the model file it should save",
    )

    parser.add_argument(
        "--tbdestination",
        help="Destination folder. Ex.: --tbdestination destination  -->  tensorboard/destination",
    )

    parser.add_argument(
        "--eras",
        required=True,
        type=lambda s: s.split(","),
        help="eras included in the training: 22pre, 22post, 23pre, 23post",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set random seed for reproducibility"
    )

    parser.add_argument(
        "--lr_range_test",
        type=bool,
        default=False,
        help="If true, starts a learning rate range test."
    )

    parser.add_argument(
        "--scheduler",
        required=True,
        type=str,
        help="Which learningrate scheduler to use during training. plateau = ReduceOnPlateau, cosine = warmup + CosineAnnealing."
    )

    parser.add_argument(
        "--warmup_max",
        type=int,
        help="maximum number of warmup iterations."
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        help="maximum number of iterations."
    )

    parser.add_argument(
        "--disable_checkpoints",
        type=bool,
        default=False,
        help="enable/disable learningrate scheduler checkpoints."
    )

    parser.add_argument(
        "--disable_tensorboard",
        type=bool,
        default=False,
        help="enable/disable Tensorboard."
    )

    parser.add_argument(
        "--strength_param",
        type=float,
        default=1,
        help="Modifies the strength between the weight matrices. Multiplies the kappa lambda vs. kappa lambda loss."
    )

    parser.add_argument(
        "--only_one_weightmatrix",
        type=bool,
        default=False,
        help="If true, then the model trains with only one weightmatrix of dim [num_classes, num_classes]."
    )

    parser.add_argument(
        "--normalization_scheme",
        required=True,
        default="none",
        help="normlization scheme for the weight matrix. Options: 'none', 'global_sum', 'max_norm'."
    )

    parser.add_argument(
        "--mhh_weights",
        required=True,
        nargs=2,
        type=float,
        default=(1.0,1.0),
        help="maximum and minimum event weight based on the events mhh value."
    )

    wm_A_group = parser.add_mutually_exclusive_group(required=True)

    wm_A_group.add_argument(
        "--weightmatrix_A",
        nargs="+",
        type=float,
        help="A NxN matrix with rows representing the true class and the columns the predicted class."
    )

    wm_A_group.add_argument(
        "--diag_A",
        nargs="+",
        type=float,
        help="Diagonal matrix of the weight matrix."
    )

    wm_B_group = parser.add_mutually_exclusive_group(required=False)

    wm_B_group.add_argument(
        "--weightmatrix_B",
        nargs="+",
        type=float,
        help="A NxN matrix with rows representing the true class and the columns the predicted class. For the kappa lambda classes"
    )

    wm_B_group.add_argument(
        "--diag_B",
        nargs="+",
        type=float,
        help="Diagonal matrix of the weight matrix."
    )

    lr_group = parser.add_mutually_exclusive_group(required=True)

    lr_group.add_argument(
        "--lr",
        type=float,
        help="Learning rate for the model training."
    )

    lr_group.add_argument(
        "--lr_range",
        type=float,
        nargs=2,
        help="Maximum and minimum learning rates."
    )

    args = parser.parse_args()

    # add model to model register
    hashed_model_name, config_hash = register_model(args, base_config=config)

    config['save_model_name'] = hashed_model_name

    import pdb; pdb.set_trace()

    # dataprep
    dataset_config = get_dataset_config(args.datasets, args.eras)
    target_map = dataset_config['target_map']

    # learningrate
    lr = args.lr
    if args.lr_range is not None:
        max_lr, min_lr = args.lr_range
    lr_range_test = args.lr_range_test

    # checkpoints and tensorboard
    checkpoint_disabled = args.disable_checkpoints
    tensorboard_disabled = args.disable_tensorboard

    # loss
    strength_param = args.strength_param
    only_one_weightmatrix = args.only_one_weightmatrix
    normalization = args.normalization_scheme
    mhh_weights = tuple(args.mhh_weights)

    # scheduler
    scheduler = args.scheduler
    warmup_max = args.warmup_max
    max_iterations = args.max_iterations

    logger = get_logger(__name__)

    # device
    CPU = torch.device("cpu")
    CUDA = torch.device("cuda")
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    VERBOSE = False

    # Sample ratio
    if len(args.datasets) != len(args.sample_ratio) and 'hh' not in args.datasets:
        raise ValueError("datasets and sample_ratio must have the same length")

    config["sample_ratio"] = dict(zip(target_map, args.sample_ratio))

    # dataset importance
    print("Dataset importance:")
    for key, value in config['sample_ratio'].items():
        print(f"{key}: {value:.3f}")

    # seed
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed fixed to {seed}")

    logger.info(f"Running DEVICE is {DEVICE}")

    # tensorboard
    if tensorboard_disabled == False:
        tboard_writer = TensorboardLogger(name=hash_config(config), model_name=args.modelname, destination=args.tbdestination)
        logger.warning(f"Tensorboard logs are stored in {tboard_writer.path}")

    # k-folds
    for current_fold in (config["train_folds"]):
        logger.info(f'Start Training of fold {current_fold} from {config["k_fold"] - 1}')

        ### data preparation
        # HINT: order matters, due to memory constraints views are moved in and out of dictionaries

        # load data from cache is necessary or from root files
        # events is of form : {uid : {"continous","categorical", "weight": torch tensor}
        # events = get_data(dataset_config, overwrite=False, _save_cache=True)
        # create k-folds, whe current fold is test fold and leave out
        events = get_data(dataset_config, overwrite=False, _save_cache=True)

        train_data, validation_data = split_k_fold_into_training_and_validation(
            events,
            c_fold=current_fold,
            k_fold=config["k_fold"],
            seed=config["seed"],
            train_ratio=0.75,
        )

        training_sampler = create_train_or_validation_sampler(
            train_data,
            target_map = target_map,
            sample_ratio=config["sample_ratio"],
            min_size=config["min_events_in_batch"],
            batch_size=config["t_batch_size"],
            train=True,
        )

        validation_sampler = create_train_or_validation_sampler(
            validation_data,
            target_map = target_map,
            sample_ratio=config["sample_ratio"],
            min_size=config["min_events_in_batch"],
            batch_size=config["v_batch_size"],
            train=False,
        )

        # share relative weight from training batch statistic to validation sampler
        training_sampler.share_weights_between_sampler(validation_sampler)

        # get weighted mean and std of expected batch composition
        model_building_config["mean"], model_building_config["std"] = get_batch_statistics_from_sampler(
            training_sampler,
            padding_values=-99999,
            features=dataset_config["continous_features"],
            return_dummy=config["get_batch_statistic_return_dummy"],
        )

        # Model setup
        model = create_model.BNetLBNDenseNet(
            dataset_config["continous_features"], dataset_config["categorical_features"], target_map=target_map, config=model_building_config
            )
        model = model.to(DEVICE)

        # load means from marcel if activated
        model = mwt.load_marcels_weights(model, continous_features=dataset_config["continous_features"], with_std=config["load_marcel_stats"], with_weights=config["load_marcel_weights"])

        if scheduler == 'plateau':
            optimizer_config['lr'] = lr

        elif scheduler == 'cosine':
            optimizer_config['lr'] = max_lr

        # optimizer
        weight_decay_parameters = optimizer.prepare_weight_decay(model, optimizer_config)
        optimizer_inst = torch.optim.AdamW(list(weight_decay_parameters.values()), lr=optimizer_config["lr"])

        # scheduler mode
        if scheduler == 'plateau':
            scheduler_inst = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_inst,
                mode='min',
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
                threshold=scheduler_config["min_delta"],
                threshold_mode=scheduler_config["threshold_mode"],
                cooldown=0,
                min_lr=0,
                eps=1e-08
            )
            max_iterations = 10000000000
            val_frequency = lambda iteration: config["validation_interval"]

        elif scheduler == "cosine":
            scheduler_inst = torch.optim.lr_scheduler.LambdaLR(
                optimizer = optimizer_inst,
                lr_lambda = lambda step: lr_multiplier(step, warmup_max, max_iterations, max_lr, min_lr)
                )
            val_frequency = lambda iteration: validation_frequency(iteration, max_iterations)

        # initializing checkpoint
        model_checkpoint = {}

        # initializing early stopping
        early_stopper_inst = EarlyStopOnPlateau()
        early_stopping_counter = 0

        # Training with two weight matrices
        if 'hh' not in target_map and not only_one_weightmatrix:
            # always 3 classes for the first weight matrix
            weight_matrix_A_dim = 3

            # always the number of classes minus the background classes
            weight_matrix_B_dim = len(target_map) - 2

            # construct matrices based on classes
            if args.diag_A is not None:
                weight_matrix_A = torch.diag(torch.tensor(args.diag_A))
            else:
                weight_matrix_A = torch.tensor(args.weightmatrix_A).view(weight_matrix_A_dim,weight_matrix_A_dim)

            if args.diag_B is not None:
                weight_matrix_B = torch.diag(torch.tensor(args.diag_B))
            else:
                weight_matrix_B = torch.tensor(args.weightmatrix_B).view(weight_matrix_B_dim,weight_matrix_B_dim)

            weight_matrix_A = weight_matrix_A.to(DEVICE)
            weight_matrix_B = weight_matrix_B.to(DEVICE)

        # construct only weight matrix A if training with hh as signal class
        elif 'hh' in target_map and not only_one_weightmatrix:
            weight_matrix_A_dim = 3
            if args.diag_A is not None:
                weight_matrix_A = torch.diag(torch.tensor(args.diag_A))
            else:
                weight_matrix_A = torch.tensor(args.weightmatrix_A).view(weight_matrix_A_dim,weight_matrix_A_dim)

            weight_matrix_A = weight_matrix_A.to(DEVICE)
            weight_matrix_B = None

        # train model with only one weightmatrix of dim [num_classes, num_classes]
        elif only_one_weightmatrix == True:
            weight_matrix_A_dim = len(target_map)
            if args.diag_A is not None:
                weight_matrix_A = torch.diag(torch.tensor(args.diag_A))
            else:
                weight_matrix_A = torch.tensor(args.weightmatrix_A).view(weight_matrix_A_dim,weight_matrix_A_dim)

            weight_matrix_A = weight_matrix_A.to(DEVICE)
            weight_matrix_B = None

        # Components of the loss matrix
        if extra_losses is not None:
            print("Also computing the following non-contributing losses:")
            for key, value in extra_losses.items():
                print(key)

        # HINT: requires only logits, no softmax at end
        loss_fn = WeightedFalseClassPenaltyLogLoss(weight_matrix_A=weight_matrix_A, weight_matrix_B=weight_matrix_B, normalization=normalization, mhh_weights=mhh_weights, loss_components_dict=extra_losses, device=DEVICE)

        # Learning rate range tests
        if lr_range_test == True:
            print("Starting learning rate range test.")
            model.train()
            learningrates = []
            losses = []
            lr_lambda = (1e-1 / 1e-5) ** (1 / 300)
            lr = 1e-5

            for param_group in optimizer_inst.param_groups:
                param_group['lr'] = lr

            iter_count = 0
            for iteration in range(300):
                print(f"Testing for {lr}.")
                t_loss, (t_pred, t_targets), *t_other_loss = training_fn(
                    model = model,
                    loss_fn = loss_fn,
                    optimizer = optimizer_inst,
                    target_map = target_map,
                    strength_param = strength_param,
                    sampler = training_sampler,
                    device = DEVICE,

                )
                learningrates.append(lr)
                losses.append(t_loss.detach().cpu().item())
                lr *= lr_lambda
                for param_group in optimizer_inst.param_groups:
                    param_group['lr'] = lr

                iter_count += 1

            plt.figure()
            plt.plot(learningrates, losses)
            plt.xscale('log')
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning Rate Range Test")
            plt.grid(True)
            plt.savefig(f"lr_range_tests/{args.modelname}.png")
            plt.close()
            break

        ### training loop:
        model.train()
        for current_iteration in range(max_iterations):
            t_loss, (t_pred, t_targets), *t_other_loss = training_fn(
                model = model,
                loss_fn = loss_fn,
                optimizer = optimizer_inst,
                target_map = target_map,
                strength_param = strength_param,
                only_one_weightmatrix=only_one_weightmatrix,
                sampler = training_sampler,
                device = DEVICE,
            )

            if torch.isnan(t_loss):
                print("NaN in batch loss")
                from IPython import embed;embed(header=" string - 152 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

            if scheduler == 'cosine':
                scheduler_inst.step()

            # VERBOSITY
            if current_iteration % config["verbose_interval"] == 0:
                if tensorboard_disabled == False:
                    tboard_writer.log_loss({"batch_loss": t_loss.item()}, step=current_iteration)
                print(f"Training: {current_iteration} - batch loss: {t_loss.item():.4f}")

                # for Cosineannealing
                if scheduler == "cosine":
                    current_lr = scheduler_inst.get_last_lr()[0]
                    if current_iteration <= warmup_max:
                        pct = (current_iteration / warmup_max) * 100
                        print(f"[WARMUP] {pct:6.2f}% of Max LR | LR: {current_lr:.2e}")
                        tboard_writer.log_lr(optimizer_inst.param_groups[0]["lr"], step=current_iteration)
                    else:
                        print(f"[ANNEAL] LR: {current_lr:.2e}")
                        tboard_writer.log_lr(optimizer_inst.param_groups[0]["lr"], step=current_iteration)

            if (current_iteration % val_frequency(current_iteration) == 0) and current_iteration != 0:
                # evaluation of training data
                print(f"Running evaluation of training data at iteration {current_iteration}...")
                eval_t_loss, (eval_t_pred, eval_t_tar, eval_t_weights), *eval_t_other_loss = validation_fn(model,
                                                                                                           loss_fn,
                                                                                                           target_map,
                                                                                                           strength_param,
                                                                                                           only_one_weightmatrix,
                                                                                                           training_sampler,
                                                                                                           device=DEVICE)

                if torch.isnan(eval_t_loss):
                    print('training loss is a nan')
                    from IPython import embed;embed(header=" string - 158 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

                if torch.isnan(eval_t_pred).any().item() == True:
                    print("Found a nan in the training predictions")
                    from IPython import embed;embed(header=" string - 161 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

                if tensorboard_disabled == False:
                    log_metrics(
                        tensorboard_inst = tboard_writer,
                        iteration_step = current_iteration,
                        sampler_output = (eval_t_pred, eval_t_tar, eval_t_weights),
                        target_map = target_map,
                        mode = "train",
                        loss = eval_t_loss.item(),
                        other_loss = eval_t_other_loss,
                        lr = optimizer_inst.param_groups[0]["lr"],
                        sampler = training_sampler,
                        model = model
                    )
                print(f"Running evaluation of validation data at iteration {current_iteration}...")

                # evaluation of validation
                eval_v_loss, (eval_v_pred, eval_v_tar, eval_v_weights), *eval_v_other_loss = validation_fn(model,
                                                                                                           loss_fn,
                                                                                                           target_map,
                                                                                                           strength_param,
                                                                                                           only_one_weightmatrix,
                                                                                                           validation_sampler,
                                                                                                           device=DEVICE)

                if torch.isnan(eval_v_loss):
                    print('validation loss is a nan')
                    from IPython import embed;embed(header=" string - 158 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

                if torch.isnan(eval_v_pred).any().item() == True:
                    print("Found a nan in the validation predictions")
                    from IPython import embed;embed(header=" string - 161 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

                if tensorboard_disabled == False:
                    log_metrics(
                        tensorboard_inst = tboard_writer,
                        iteration_step = current_iteration,
                        sampler_output = (eval_v_pred, eval_v_tar, eval_v_weights),
                        target_map = target_map,
                        mode = "validation",
                        loss = eval_v_loss.item(),
                        other_loss = eval_v_other_loss,
                    )
                print(f"Evaluation: it: {current_iteration} - TLoss: {eval_t_loss:.4f} VLoss: {eval_v_loss:.4f}")

                if scheduler == 'plateau':
                    # if (current_iteration % 1000 == 0) and (current_iteration > 0):
                    previous_lr = optimizer_inst.param_groups[0]["lr"]
                    scheduler_inst.step(eval_v_loss)
                    logger.info(f"{previous_lr} -> {optimizer_inst.param_groups[0]['lr']}")
                    new_lr = optimizer_inst.param_groups[0]['lr']

                    # restore model state of best validation if checkpoint enabled
                    if checkpoint_disabled == False:
                        if previous_lr > optimizer_inst.param_groups[0]['lr']:
                            print("validation did not improve for 10 validations, restoring weights of best validation and reducing learning rate.")
                            model.load_state_dict(model_checkpoint["model_state"])
                            optimizer_inst.load_state_dict(model_checkpoint["optimizer_state"])
                            for g in optimizer_inst.param_groups:
                                g['lr'] = new_lr

                ### early stopping
                # when val loss is lowest over a period of patience
                if early_stopper_inst(eval_v_loss, model):
                    logger.info(f"saving current best model at iteration {current_iteration} with loss {eval_v_loss:.5f}")
                    torch_save(model, config["save_model_name"], current_fold)

                    # make a checkpoint for the best model and optimizer states
                    if checkpoint_disabled == False:
                        model_checkpoint = {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer_inst.state_dict(),
                        }
                        print("Checkpoint created/updated.")
                    early_stopping_counter = 0

                else:
                    early_stopping_counter += 1
                    print(f"validation loss did not improve for {early_stopping_counter} validations.")

                # Early stopping if scheduler is ReduceOnPlateau
                if scheduler == 'plateau':
                    if early_stopping_counter != 20:
                        continue

                    else:
                        print("validation loss has not improved for 20 validations. Stopping training.")
                        print(config['save_model_name'])
                        break

                    # TODO release DATA from previous RUN
                    if (current_iteration % config["max_train_iteration"] == 0) & (current_iteration > 0):
                        from IPython import embed; embed(
                            header=f"Current break at {current_iteration} if you wanna continue press y else save")
