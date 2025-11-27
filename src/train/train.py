# standard imports
import torch

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
    config, dataset_config, model_building_config, optimizer_config, target_map,
)
from train_utils import training, validation, log_metrics

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
        features=dataset_config["continous_features"]
    )

    ### Model setup
    model = create_model.BNetDenseNet(dataset_config["continous_features"], dataset_config["categorical_features"], config=model_building_config)
    model = model.to(DEVICE)

    # ## load mean from marcel
    # import marcel_weight_translation as mwt
    # model = mwt.load_marcels_weights(model, continous_features=dataset_config["continous_features"])


    # TODO: only linear models should contribute to weight decay
    # TODO : SAMW Optimizer
    weight_decay_parameters = optimizer.prepare_weight_decay(model, optimizer_config)
    optimizer_inst = torch.optim.AdamW(list(weight_decay_parameters.values()), lr=optimizer_config["lr"])
    scheduler_inst = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_inst, gamma=config["gamma"])

    # HINT: requires only logits, no softmax at end
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])
    max_iteration = 10000
    LOG_INTERVAL = 10
    validation_interval = 200
    model.train()

    ### training loop:
    for iteration in range(max_iteration):
        t_loss, (t_pred, t_targets) = training(
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer_inst,
            sampler = training_sampler,
            device=DEVICE
        )

        if (iteration % 1000 == 0) and (iteration > 0):
            previous_lr =  optimizer_inst.param_groups[0]["lr"]
            scheduler_inst.step()
            logger.info(f"{previous_lr} -> { optimizer_inst.param_groups[0]["lr"]}")

        if (iteration % validation_interval == 0):
            # evaluation of training data
            print("Running evaluation of training data...")
            t_loss, (t_pred, t_tar, t_weights) = validation(model, loss_fn, training_sampler, device=DEVICE)
            log_metrics(
                tensorboard_inst = tboard_writer,
                iteration_step = iteration,
                sampler_output = (t_pred, t_tar, t_weights),
                target_map = target_map,
                mode = "train",
                loss = t_loss.item(),
                lr = optimizer_inst.param_groups[0]["lr"],
            )
            print("Running evaluation of validation data...")
            # evaluation of validation
            v_loss, (v_pred, v_tar, v_weights) = validation(model, loss_fn, validation_sampler, device=DEVICE)
            log_metrics(
                tensorboard_inst = tboard_writer,
                iteration_step = iteration,
                sampler_output = (v_pred, v_tar, v_weights),
                target_map = target_map,
                mode = "validation",
                loss = v_loss.item(),
            )
            print(f"Evaluation: it: {iteration} - TLoss: {t_loss:.4f} VLoss: {v_loss:.4f}")

        # VERBOSITY
        if iteration % LOG_INTERVAL == 0:
            tboard_writer.log_loss({"batch_loss": t_loss.item()}, step=iteration)
            print(f"Training: {iteration} - batch loss: {t_loss.item():.4f}")

    # TODO release DATA from previous RUN
    from IPython import embed; embed(header="END OF TRAINING try saving - 124 in train.py ")
    # save network
    import export
    cont, cat, tar = training_sampler.sample_batch(device=CPU)
    export.torch_export(
        model = model.to(CPU),
        dst_path = f"models/fold_{current_fold}_final_model.pt2",
        input_tensors = (cat.to(torch.int32), cont)
    )

    from IPython import embed; embed(header="END - 89 in train.py ")
    torch.save(model, f"models/fold_{current_fold}_final_model.pt")
