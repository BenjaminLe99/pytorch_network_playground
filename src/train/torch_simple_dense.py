# standard imports
import torch

# project imports
from models import create_model
from data.load_data import get_data
from data.preprocessing import (
    create_train_and_validation_sampler, get_batch_statistics, get_batch_statistics_from_sampler, split_k_fold_into_training_and_validation, test_sampler
    )
from utils.logger import get_logger, TensorboardLogger
from data.cache import hash_config
import optimizer
from mymodels.my_configs import (
    config, dataset_config, model_building_config, optimizer_config, target_map,
)
from train_utils import training, validation, log_metrics
from mymodels.my_utils import init_weights_he_uniform, get_marcel_stats, save_marcels_weights
from mymodels import recreate_simple

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VERBOSE = False

logger = get_logger(__name__)

# load data
tboard_writer = TensorboardLogger(name=hash_config(config))
logger.warning(f"Tensorboard logs are stored in {tboard_writer.path}")
for current_fold in (config["train_folds"]):
    logger.info(f'Start Training of fold {current_fold} from {config["k_fold"] - 1}')

    ### data preparation
    # Hint: order matters, due to memory constraints views are moved in and out of dictionaries


    # load data from cache is necessary or from root files
    # events is of form : {uid : {"continous","categorical","weight": torch tensor}
    events = get_data(dataset_config, overwrite=False, _save_cache=True)
    # create k-folds, whe current fold is test fold and leave out
    train_data, validation_data = split_k_fold_into_training_and_validation(
        events,
        c_fold=current_fold,
        k_fold=config["k_fold"],
        seed=config["seed"],
        train_ratio=config["train_ratio"],
    )
    # get weighted mean and std of expected batch composition
    #model_building_config["mean"], model_building_config["std"] = get_batch_statistics(train_data, padding_value=-99999)

    # get marcels stats
    #get_marcel_stats(model_building_config)

    training_sampler, validation_sampler = create_train_and_validation_sampler(
        t_data = train_data,
        v_data = validation_data,
        t_batch_size = config["t_batch_size"],
        v_batch_size = config["v_batch_size"],
        target_map=target_map,
        min_size=1,
        sample_ratio=config["sample_ratio"]
    )

    model_building_config["mean"], model_building_config["std"] = get_batch_statistics_from_sampler(training_sampler, padding_values=-99999, features=dataset_config["continous_features"])

    ### Model setup
    model = recreate_simple.init_layers(dataset_config["continous_features"], dataset_config["categorical_features"], config=model_building_config)
    from IPython import embed;embed(header=" string - 63 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/testdnn2.py")
    #save_marcels_weights(model)
    
    model = model.to(DEVICE)

    # TODO: only linear models should contribute to weight decay
    # TODO : SAMW Optimizer
    weight_decay_parameters = optimizer.prepare_weight_decay(model, optimizer_config)
    #optimizer_inst = torch.optim.AdamW(model.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config["decay_factor"])
    optimizer_inst = torch.optim.AdamW(list(weight_decay_parameters.values()), lr=optimizer_config["lr"])
    scheduler_inst = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_inst, gamma=optimizer_config["learning_rate_reduction_factor"])


    # HINT: requires only logits, no softmax at end
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])
    max_iteration = config["max_iteration"]
    LOG_INTERVAL = 10
    validation_interval = config["validation_interval"]
    validation_iteration = 0
    best_v_loss = 90000
    best_iteration = 0
    last_lr_drop = -1
    patience = config["patience"]
    early_stopping_patience = config["early_stopping_patience"]
    model.train()


    # training loop:
    for iteration in range(max_iteration):
        t_loss, (t_pred, t_targets) = training(
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer_inst,
            sampler = training_sampler,
            device=DEVICE
        )

        if (iteration % validation_interval == 0) and (iteration > 0):
            validation_iteration += 1 
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

            # if (iteration % 1500 == 0) and (iteration > 0):
            #     scheduler_inst.step()

            # learningrate scheduler. Save best loss and iteration, compare until 10 iterations after -> reduce learningrate
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                best_iteration = validation_iteration
                lr_drop_flag = False
                torch.save(model.state_dict(), "/afs/desy.de/user/l/lebenjam/Master/neuralnetwork/mlmodels/model_dicts/best_model.pth")

            else:
                # check patience since last improvement
                if validation_iteration - best_iteration >= patience and validation_iteration - last_lr_drop >= patience:
                    scheduler_inst.step()
                    last_lr_drop = validation_iteration
                    lr_drop_flag = True
                
                # Early stop if no improvement for 20 epochs
                if validation_iteration - best_iteration >= early_stopping_patience:
                    print(f"Early stopping at iteration {iteration}")
                    break

        # VERBOSITY
        if iteration % LOG_INTERVAL == 0:
            tboard_writer.log_loss({"batch_loss": t_loss.item()}, step=iteration)
            print(f"Training: {iteration} - batch loss: {t_loss.item():.4f}")
        if (iteration % validation_interval == 0) and (iteration > 0):
            print("==========================================================================")
            if lr_drop_flag == True:
                print("loss didnt improve for 10 iterations, reducing learning rate")
            print(f"Evaluation: it: {iteration} - TLoss: {t_loss:.4f} VLoss: {v_loss:.4f}")
            print(f"current best validation loss: {best_v_loss:.4f}")
            if lr_drop_flag == False:
                print(f"validation iterations before next learning rate update: {patience - (validation_iteration - best_iteration)}")
            else:
                print(f"validation iterations before early stopping: {early_stopping_patience - (validation_iteration - best_iteration)}")
            print("==========================================================================")

    # load best validation run parameters
    model.load_state_dict(torch.load("/afs/desy.de/user/l/lebenjam/Master/neuralnetwork/mlmodels/model_dicts/best_model.pth"))

    # save my model for troubleshooting
    torch.save(model.state_dict(), "/afs/desy.de/user/l/lebenjam/Master/neuralnetwork/mlmodels/model_dicts/my_model.pth")

    # TODO release DATA from previous RUN
    from IPython import embed;embed(header=" string - 165 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/testdnn2.py")
    
    # save network
    import ml_save
    cont, cat, tar = training_sampler.sample_batch(device=CPU)
    ml_save.torch_export(
        model = model.to(CPU),
        name = f"fold_{current_fold}_{config['modelname']}.pt2",
        input_tensors = (cat.to(torch.int32), cont)
    )
