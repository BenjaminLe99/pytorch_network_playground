# standard imports
import torch
import argparse

# project imports
from models import create_model
from data.load_data import get_data
from data.preprocessing import (
    create_train_or_validation_sampler, get_batch_statistics_from_sampler, split_k_fold_into_training_and_validation,
    )
from utils.logger import get_logger, TensorboardLogger
from data.cache import hash_config
import optimizer
from train_config import (
    config, get_dataset_config, model_building_config, optimizer_config, target_map, scheduler_config
)
from train_utils import training_fn, validation_fn, log_metrics
from early_stopping import EarlyStopSignal, EarlyStopOnPlateau
from export import torch_save, torch_export_v2
import marcel_weight_translation as mwt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Datasets used for the training")

    parser.add_argument(
        "--datasets",
        required=True,
        type=lambda s: s.split(","),
        help="List of: dy, tt, kl0, kl1, kl2p45 or kl5",
    )

    parser.add_argument(
        "--modelname",
        required=True,
        help="Name of the model file it should save",
    )

    parser.add_argument(
        "--eras",
        required=True,
        type=lambda s: s.split(","),
        help="eras included in the training: 22pre, 22post, 23pre, 23post",
    )

    args = parser.parse_args()

    dataset_config = get_dataset_config(args.datasets, args.eras)
    config['save_model_name'] = args.modelname

    logger = get_logger(__name__)

    CPU = torch.device("cpu")
    CUDA = torch.device("cuda")
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    VERBOSE = False

    logger.info(f"Running DEVICE is {DEVICE}")
    # load data
    tboard_writer = TensorboardLogger(name=hash_config(config))
    logger.warning(f"Tensorboard logs are stored in {tboard_writer.path}")
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
            min_size=config["min_events_in_batch"],
            batch_size=config["t_batch_size"],
            train=True,
            sample_ratio=config["sample_ratio"],
        )
        validation_sampler = create_train_or_validation_sampler(
            validation_data,
            target_map = target_map,
            min_size=config["min_events_in_batch"],
            batch_size=config["v_batch_size"],
            train=False,
            sample_ratio=config["sample_ratio"],
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

        ### Model setup
        # model = create_model.BNetDenseNet(dataset_config["continous_features"], dataset_config["categorical_features"], config=model_building_config)
        model = create_model.BNetLBNDenseNet(
            dataset_config["continous_features"], dataset_config["categorical_features"], config=model_building_config
            )
        model = model.to(DEVICE)
        # from IPython import embed; embed(header="string - 86 in train.py ")
        # ## load mean from marcel if activated
        model = mwt.load_marcels_weights(model, continous_features=dataset_config["continous_features"], with_std=config["load_marcel_stats"], with_weights=config["load_marcel_weights"])

        # TODO: only linear models should contribute to weight decay
        # TODO : SAMW Optimizer
        weight_decay_parameters = optimizer.prepare_weight_decay(model, optimizer_config)
        optimizer_inst = torch.optim.AdamW(list(weight_decay_parameters.values()), lr=optimizer_config["lr"])
        # optimizer_inst = optimizer.SAM(list(weight_decay_parameters.values()), torch.optim.AdamW, lr=optimizer_config["lr"], rho = 2.0, adaptive=True)

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

        early_stopper_inst = EarlyStopOnPlateau()
        # HINT: requires only logits, no softmax at end
        loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])

        model.train()
        ### training loop:
        for current_iteration in range(config["max_train_iteration"]):
            t_loss, (t_pred, t_targets) = training_fn(
                model = model,
                loss_fn = loss_fn,
                optimizer = optimizer_inst,
                sampler = training_sampler,
                device=DEVICE
            )
            if torch.isnan(t_loss):
                print("NaN in batch loss")
                from IPython import embed;embed(header=" string - 152 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

            # VERBOSITY
            if current_iteration % config["verbose_interval"] == 0:
                tboard_writer.log_loss({"batch_loss": t_loss.item()}, step=current_iteration)
                print(f"Training: {current_iteration} - batch loss: {t_loss.item():.4f}")

            if (current_iteration % config["validation_interval"] == 0):
                # evaluation of training data
                print(f"Running evaluation of training data at iteration {current_iteration}...")
                eval_t_loss, (eval_t_pred, eval_t_tar, eval_t_weights) = validation_fn(model, loss_fn, training_sampler, device=DEVICE)

                if eval_t_loss == 'nan':
                    print('training loss is a nan')
                    from IPython import embed;embed(header=" string - 158 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")
                
                if torch.isnan(eval_t_pred).any().item() == True:
                    print("Found a nan in the training predictions")
                    from IPython import embed;embed(header=" string - 161 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

                log_metrics(
                    tensorboard_inst = tboard_writer,
                    iteration_step = current_iteration,
                    sampler_output = (eval_t_pred, eval_t_tar, eval_t_weights),
                    target_map = target_map,
                    mode = "train",
                    loss = eval_t_loss.item(),
                    lr = optimizer_inst.param_groups[0]["lr"],
                    sampler = training_sampler,
                    model = model
                )
                print(f"Running evaluation of validation data at iteration {current_iteration}...")
                # evaluation of validation
                eval_v_loss, (eval_v_pred, eval_v_tar, eval_v_weights) = validation_fn(model, loss_fn, validation_sampler, device=DEVICE)

                if eval_v_loss == 'nan':
                    print('validation loss is a nan')
                    from IPython import embed;embed(header=" string - 158 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")
                
                if torch.isnan(eval_v_pred).any().item() == True:
                    print("Found a nan in the validation predictions")
                    from IPython import embed;embed(header=" string - 161 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train.py")

                log_metrics(
                    tensorboard_inst = tboard_writer,
                    iteration_step = current_iteration,
                    sampler_output = (eval_v_pred, eval_v_tar, eval_v_weights),
                    target_map = target_map,
                    mode = "validation",
                    loss = eval_v_loss.item(),
                )
                print(f"Evaluation: it: {current_iteration} - TLoss: {eval_t_loss:.4f} VLoss: {eval_v_loss:.4f}")


                # if (current_iteration % 1000 == 0) and (current_iteration > 0):
                previous_lr =  optimizer_inst.param_groups[0]["lr"]
                scheduler_inst.step(eval_v_loss)
                logger.info(f"{previous_lr} -> { optimizer_inst.param_groups[0]['lr']}")


                ### early stopping
                # when val loss is lowest over a period of patience
                if early_stopper_inst(eval_v_loss, model):
                    logger.info(f"saving current best model at iteration {current_iteration} with loss {eval_v_loss:.5f}")
                    torch_save(model, config["save_model_name"], current_fold)
                    # torch_export_v2(model, config["save_model_name"], current_fold)

                # TODO release DATA from previous RUN
                if (current_iteration % config["max_train_iteration"] == 0) & (current_iteration > 0):
                    from IPython import embed; embed(
                        header=f"Current break at {current_iteration} if you wanna continue press y else save")



        from IPython import embed; embed(header="END - 89 in train.py ")

