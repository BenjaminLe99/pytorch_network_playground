### Refactored Script

# standard imports
import torch
import numpy as np
import random
import argparse
import hashlib
import json
import os
from torch.optim.swa_utils import AveragedModel
from fractions import Fraction
import matplotlib.pyplot as plt

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
from train_utils import training_fn, validation_fn, log_metrics, lr_multiplier, validation_frequency, update_ema_batchnorm, EMAMultiAvgFn
from early_stopping import EarlyStopSignal, EarlyStopOnPlateau
from export import torch_save, torch_export_v2
import marcel_weight_translation as mwt
from loss import WeightedFalseClassPenaltyLogLoss


def parse_arguments():
    def parse_fraction(value):
        return float(Fraction(value))

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, type=lambda s: s.split(","), help="List of: dy, tt, kl0, kl1, kl2p45 or kl5")
    parser.add_argument("--sample_ratio", required=True, type=parse_fraction, nargs="+", help="List of numbers. Must sum to 1.")
    parser.add_argument("--modelname", required=True, help="Name of the model file it should save")
    parser.add_argument("--tbdestination", help="Destination folder. Ex.: --tbdestination destination  -->  tensorboard/destination")
    parser.add_argument("--eras", required=True, type=lambda s: s.split(","), help="eras included in the training: 22pre, 22post, 23pre, 23post")
    parser.add_argument("--seed", type=int, default=None, help="Set random seed for reproducibility")
    parser.add_argument("--lr_range_test", type=bool, default=False, help="If true, starts a learning rate range test.")
    parser.add_argument("--scheduler", required=True, type=str, help=" plateau = ReduceOnPlateau, cosine = warmup + CosineAnnealing.")
    parser.add_argument("--warmup_max", type=int, help="maximum number of warmup iterations.")
    parser.add_argument("--max_iterations", type=int, help="maximum number of iterations.")
    parser.add_argument("--disable_checkpoints", type=bool, default=False, help="enable/disable learningrate scheduler checkpoints.")
    parser.add_argument("--disable_tensorboard", type=bool, default=False, help="enable/disable Tensorboard.")
    parser.add_argument("--strength_param", type=float, default=1, help="Modifies the strength between the weight matrices.")
    parser.add_argument("--only_one_weightmatrix", type=bool, default=False, help="Model trains with only one weightmatrix.")
    parser.add_argument("--normalization_scheme", required=True, default="none", help="Options: 'none', 'global_sum', 'max_norm'.")
    parser.add_argument("--mhh_weights", required=True, nargs=2, type=float, default=(1.0, 1.0), help="max and min event weight based on mhh.")
    parser.add_argument("--exponential-moving-average", action="store_true", help="enable exponential moving average.")
    parser.add_argument("--metrics", required=True, type=lambda s: s.split(","), help="list of metrics to be recorded.")

    wm_A_group = parser.add_mutually_exclusive_group(required=True)
    wm_A_group.add_argument("--weightmatrix_A", nargs="+", type=float, help="NxN matrix (true vs predicted).")
    wm_A_group.add_argument("--diag_A", nargs="+", type=float, help="Diagonal matrix of the weight matrix A.")

    wm_B_group = parser.add_mutually_exclusive_group(required=False)
    wm_B_group.add_argument("--weightmatrix_B", nargs="+", type=float, help="NxN matrix for the kappa lambda classes.")
    wm_B_group.add_argument("--diag_B", nargs="+", type=float, help="Diagonal matrix of the weight matrix B.")

    lr_group = parser.add_mutually_exclusive_group(required=True)
    lr_group.add_argument("--lr", type=float, help="Learning rate for the model training.")
    lr_group.add_argument("--lr_range", type=float, nargs=2, help="Maximum and minimum learning rates.")

    return parser.parse_args()


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed fixed to {seed}")


def prepare_weight_matrices(args, target_map, device):
    weight_matrix_A = None
    weight_matrix_B = None

    if 'hh' not in target_map and not args.only_one_weightmatrix:
        weight_matrix_A_dim = 3
        weight_matrix_B_dim = len(target_map) - 2

        if args.diag_A is not None:
            weight_matrix_A = torch.diag(torch.tensor(args.diag_A))
        else:
            weight_matrix_A = torch.tensor(args.weightmatrix_A).view(weight_matrix_A_dim, weight_matrix_A_dim)

        if args.diag_B is not None:
            weight_matrix_B = torch.diag(torch.tensor(args.diag_B))
        else:
            weight_matrix_B = torch.tensor(args.weightmatrix_B).view(weight_matrix_B_dim, weight_matrix_B_dim)

        weight_matrix_B = weight_matrix_B.to(device)

    elif 'hh' in target_map and not args.only_one_weightmatrix:
        weight_matrix_A_dim = 3
        if args.diag_A is not None:
            weight_matrix_A = torch.diag(torch.tensor(args.diag_A))
        else:
            weight_matrix_A = torch.tensor(args.weightmatrix_A).view(weight_matrix_A_dim, weight_matrix_A_dim)

    elif args.only_one_weightmatrix:
        weight_matrix_A_dim = len(target_map)
        if args.diag_A is not None:
            weight_matrix_A = torch.diag(torch.tensor(args.diag_A))
        else:
            weight_matrix_A = torch.tensor(args.weightmatrix_A).view(weight_matrix_A_dim, weight_matrix_A_dim)

    if weight_matrix_A is not None:
        weight_matrix_A = weight_matrix_A.to(device)

    return weight_matrix_A, weight_matrix_B


def run_lr_range_test(model, optimizer_inst, loss_fn, target_map, args, training_sampler, device):
    print("Starting learning rate range test.")
    model.train()
    learningrates, losses = [], []
    lr_lambda = (1e-1 / 1e-5) ** (1 / 300)
    lr = 1e-5

    for param_group in optimizer_inst.param_groups:
        param_group['lr'] = lr

    for iteration in range(300):
        print(f"Testing for {lr:.2e}.")
        t_loss, (t_pred, t_targets), *t_other_loss = training_fn(
            model=model, loss_fn=loss_fn, optimizer=optimizer_inst,
            target_map=target_map, strength_param=args.strength_param,
            sampler=training_sampler, device=device
        )
        learningrates.append(lr)
        losses.append(t_loss.detach().cpu().item())
        lr *= lr_lambda
        for param_group in optimizer_inst.param_groups:
            param_group['lr'] = lr

    plt.figure()
    plt.plot(learningrates, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Range Test")
    plt.grid(True)
    os.makedirs("lr_range_tests", exist_ok=True)
    plt.savefig(f"lr_range_tests/{args.modelname}.png")
    plt.close()


def train_fold(current_fold, args, dataset_config, target_map, device, logger, tboard_writer, hashed_model_name):
    logger.info(f'Start Training of fold {current_fold} from {config["k_fold"] - 1}')

    # 1. Data Preparation
    events = get_data(dataset_config, overwrite=False, _save_cache=True)
    train_data, validation_data = split_k_fold_into_training_and_validation(
        events, c_fold=current_fold, k_fold=config["k_fold"], seed=config["seed"], train_ratio=0.75,
    )

    training_sampler = create_train_or_validation_sampler(
        train_data, target_map=target_map, sample_ratio=config["sample_ratio"],
        min_size=config["min_events_in_batch"], batch_size=config["t_batch_size"], train=True,
    )

    validation_sampler = create_train_or_validation_sampler(
        validation_data, target_map=target_map, sample_ratio=config["sample_ratio"],
        min_size=config["min_events_in_batch"], batch_size=config["v_batch_size"], train=False,
    )

    training_sampler.share_weights_between_sampler(validation_sampler)

    model_building_config["mean"], model_building_config["std"] = get_batch_statistics_from_sampler(
        training_sampler, padding_values=-99999, features=dataset_config["continous_features"], return_dummy=config["get_batch_statistic_return_dummy"],
    )

    # 2. Model Setup
    model = create_model.BNetLBNDenseNet(
        dataset_config["continous_features"], dataset_config["categorical_features"], target_map=target_map, config=model_building_config
    ).to(device)

    model = mwt.load_marcels_weights(
        model, continous_features=dataset_config["continous_features"], with_std=config["load_marcel_stats"], with_weights=config["load_marcel_weights"]
    )

    ema_model = None
    v_model = model
    if args.exponential_moving_average:
        ema_model = AveragedModel(model, avg_fn=EMAMultiAvgFn(config["ema_window_size"]))
        v_model = ema_model

    # 3. Optimizer & Scheduler Setup
    optimizer_config['lr'] = args.lr if args.scheduler == 'plateau' else args.lr_range[0]
    weight_decay_parameters = optimizer.prepare_weight_decay(model, optimizer_config)
    optimizer_inst = torch.optim.AdamW(list(weight_decay_parameters.values()), lr=optimizer_config["lr"])

    max_iterations = args.max_iterations
    if args.scheduler == 'plateau':
        scheduler_inst = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_inst, mode='min', factor=scheduler_config["factor"], patience=scheduler_config["patience"],
            threshold=scheduler_config["min_delta"], threshold_mode=scheduler_config["threshold_mode"], cooldown=0, min_lr=0, eps=1e-08
        )
        max_iterations = 10000000000
        val_frequency = lambda iteration: config["validation_interval"]
    elif args.scheduler == "cosine":
        min_lr = args.lr_range[1]
        scheduler_inst = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer_inst,
            lr_lambda=lambda step: lr_multiplier(step, args.warmup_max, max_iterations, optimizer_config['lr'], min_lr)
        )
        val_frequency = lambda iteration: validation_frequency(iteration, max_iterations)

    # 4. Loss Function
    weight_matrix_A, weight_matrix_B = prepare_weight_matrices(args, target_map, device)
    if extra_losses is not None:
        print("Also computing the following non-contributing losses:")
        for key in extra_losses.keys():
            print(key)

    loss_fn = WeightedFalseClassPenaltyLogLoss(
        weight_matrix_A=weight_matrix_A, weight_matrix_B=weight_matrix_B, normalization=args.normalization_scheme,
        mhh_weights=tuple(args.mhh_weights), loss_components_dict=extra_losses, device=device
    )

    # LR Range Test Check
    if args.lr_range_test:
        run_lr_range_test(model, optimizer_inst, loss_fn, target_map, args, training_sampler, device)
        return  # End early

    # 5. Training Loop Setup
    model_checkpoint = {}
    early_stopper_inst = EarlyStopOnPlateau()
    early_stopping_counter = 0

    model.train()
    for current_iteration in range(max_iterations):
        t_loss, (t_pred, t_targets), *t_other_loss = training_fn(
            model=model, ema_model=ema_model, loss_fn=loss_fn, optimizer=optimizer_inst,
            target_map=target_map, strength_param=args.strength_param,
            only_one_weightmatrix=args.only_one_weightmatrix, sampler=training_sampler, device=device,
        )

        if torch.isnan(t_loss):
            print("NaN in batch loss")
            from IPython import embed; embed(header=" string - NaN batch loss")

        if args.scheduler == 'cosine':
            scheduler_inst.step()

        # Logging & Verbosity
        if current_iteration % config["verbose_interval"] == 0:
            if not args.disable_tensorboard:
                tboard_writer.log_loss({"batch_loss": t_loss.item()}, step=current_iteration)
            print(f"Training: {current_iteration} - batch loss: {t_loss.item():.4f}")

            if args.scheduler == "cosine":
                current_lr = scheduler_inst.get_last_lr()[0]
                if current_iteration <= args.warmup_max:
                    pct = (current_iteration / args.warmup_max) * 100
                    print(f"[WARMUP] {pct:6.2f}% of Max LR | LR: {current_lr:.2e}")
                else:
                    print(f"[ANNEAL] LR: {current_lr:.2e}")
                if not args.disable_tensorboard:
                    tboard_writer.log_lr(optimizer_inst.param_groups[0]["lr"], step=current_iteration)

        # Validation Step
        if (current_iteration % val_frequency(current_iteration) == 0) and current_iteration != 0:
            print(f"Running evaluation of training data at iteration {current_iteration}...")

            if ema_model:
                update_ema_batchnorm(ema_model, training_sampler, device=device, num_batches=30)

            # Eval Train
            eval_t_loss, (eval_t_pred, eval_t_tar, eval_t_weights), *eval_t_other_loss = validation_fn(
                v_model, loss_fn, target_map, args.strength_param, args.only_one_weightmatrix, training_sampler, device=device
            )

            if torch.isnan(eval_t_loss) or torch.isnan(eval_t_pred).any().item():
                print('NaN detected in training evaluation')
                from IPython import embed; embed(header=" string - NaN in train eval")

            if not args.disable_tensorboard:
                log_metrics(
                    tensorboard_inst=tboard_writer, iteration_step=current_iteration, sampler_output=(eval_t_pred, eval_t_tar, eval_t_weights),
                    target_map=target_map, metric_logging=args.metrics, mode="train", loss=eval_t_loss.item(),
                    other_loss=eval_t_other_loss, lr=optimizer_inst.param_groups[0]["lr"], sampler=training_sampler, model=v_model
                )

            # Eval Validation
            print(f"Running evaluation of validation data at iteration {current_iteration}...")
            eval_v_loss, (eval_v_pred, eval_v_tar, eval_v_weights), *eval_v_other_loss = validation_fn(
                v_model, loss_fn, target_map, args.strength_param, args.only_one_weightmatrix, validation_sampler, device=device
            )

            if torch.isnan(eval_v_loss) or torch.isnan(eval_v_pred).any().item():
                print('NaN detected in validation evaluation')
                from IPython import embed; embed(header=" string - NaN in val eval")

            if not args.disable_tensorboard:
                log_metrics(
                    tensorboard_inst=tboard_writer, iteration_step=current_iteration, sampler_output=(eval_v_pred, eval_v_tar, eval_v_weights),
                    target_map=target_map, metric_logging=args.metrics, mode="validation", loss=eval_v_loss.item(), other_loss=eval_v_other_loss,
                )

            print(f"Evaluation: it: {current_iteration} - TLoss: {eval_t_loss:.4f} VLoss: {eval_v_loss:.4f}")

            # Scheduler step and checkpointing (Plateau)
            if args.scheduler == 'plateau':
                previous_lr = optimizer_inst.param_groups[0]["lr"]
                scheduler_inst.step(eval_v_loss)
                logger.info(f"{previous_lr} -> {optimizer_inst.param_groups[0]['lr']}")
                new_lr = optimizer_inst.param_groups[0]['lr']

                if not args.disable_checkpoints and previous_lr > new_lr:
                    print("validation did not improve, restoring weights of best validation and reducing LR.")
                    v_model.load_state_dict(model_checkpoint["model_state"])
                    optimizer_inst.load_state_dict(model_checkpoint["optimizer_state"])
                    for g in optimizer_inst.param_groups:
                        g['lr'] = new_lr

            # Early Stopping
            if early_stopper_inst(eval_v_loss, v_model):
                logger.info(f"saving current best model at iteration {current_iteration} with loss {eval_v_loss:.5f}")
                torch_save(v_model, config["save_model_name"], current_fold)

                if not args.disable_checkpoints:
                    model_checkpoint = {"model_state": v_model.state_dict(), "optimizer_state": optimizer_inst.state_dict()}
                    print("Checkpoint created/updated.")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"validation loss did not improve for {early_stopping_counter} validations.")

            if args.scheduler == 'plateau' and early_stopping_counter == 3 * scheduler_config["patience"]:
                print("validation loss has not improved for patience limit. Stopping training.")
                print(f"{config['save_model_name']} | model hash: {hashed_model_name}")
                break

        if current_iteration == max_iterations - 1:
            print("end of training loop reached.")
            print(f"{config['save_model_name']} | model hash: {hashed_model_name}")


def main():
    args = parse_arguments()
    logger = get_logger(__name__)

    # Device Setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Running DEVICE is {device}")

    # Pre-flight Configuration
    set_seed(args.seed)
    hashed_model_name, config_hash = register_model(args, base_config=config)
    config['save_model_name'] = hashed_model_name

    dataset_config = get_dataset_config(args.datasets, args.eras)
    target_map = dataset_config['target_map']

    if len(args.datasets) != len(args.sample_ratio) and 'hh' not in args.datasets:
        raise ValueError("datasets and sample_ratio must have the same length")

    config["sample_ratio"] = dict(zip(target_map, args.sample_ratio))
    print("Dataset importance:")
    for key, value in config['sample_ratio'].items():
        print(f"{key}: {value:.3f}")

    # Tensorboard
    tboard_writer = None
    if not args.disable_tensorboard:
        tboard_writer = TensorboardLogger(name=hash_config(config), model_name=args.modelname, destination=args.tbdestination)
        logger.warning(f"Tensorboard logs are stored in {tboard_writer.path}")

    # K-fold Iteration
    for current_fold in config["train_folds"]:
        train_fold(
            current_fold=current_fold,
            args=args,
            dataset_config=dataset_config,
            target_map=target_map,
            device=device,
            logger=logger,
            tboard_writer=tboard_writer,
            hashed_model_name=hashed_model_name
        )


if __name__ == '__main__':
    main()