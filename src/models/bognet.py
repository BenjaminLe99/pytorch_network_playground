from __future__ import annotations

__all__ = [
]

from functools import partial

from collections.abc import Container

from utils import EMPTY_FLOAT, EMPTY_INT


import numpy as np
import awkward as ak
import abc
import torch

# from loss import WeightedCrossEntropy, FocalLoss
from utils import (
embedding_expected_inputs, get_standardization_parameter, normalized_weight_decay
)
from optimizer import SAM

from layers import (
    InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer,
)

class TensorBoardLogger:
    def __init__(self, tensorboard_path: str | None = None, logger: Any | None = None, **kwargs):
        """
        Setup a TensorBoard logger for a neural network located at *tensorboard_path*.
        TensorBoardLogger has access to verbose logger like law.logger.

        Args:
            tensorboard_path (str | None, optional): Path where the tensorboard logs are located.
                Does not log if no path is given. Defaults to None.
            logger (Any | None, optional): Logging instance, defaults to law's logger if None is passed.
        """
        # use passed logger or default law logger by name
        super().__init__(**kwargs)
        self.logger = logger or law.logger.get_logger(__name__)
        self.writer = None
        if tensorboard_path:
            self.logger.info(f"Creating tensorboard logger at {tensorboard_path}")
            self.writer = tensorboard.SummaryWriter(log_dir=tensorboard_path)

class NetworkBase(abc.ABC, torch.nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.custom_hooks = list()

    @abc.abstractmethod
    def train_step(self, engine, batch):
        """
        Defines the training step for the model.
        """

    @abc.abstractmethod
    def validation_step(self, engine, batch):
        """
        Defines the validation step for the model.
        """

    @abc.abstractmethod
    def init_optimizer(self, optimizer_config):
        """
        Setup the optimizer
        """

    @abc.abstractmethod
    def init_scheduler(self, scheduler_config):
        """
        Setup the your scheduler. Ensure that optimizer is defined before.
        """

    @property
    def num_cat(self):
        return len(self.categorical_features)

    @property
    def num_cont(self):
        return len(self.continuous_features)

    @property
    def num_features(self):
        return self.num_cat + self.num_cont

    # @classmethod
    # def _process_columns(cls, columns: Container[str]):
    #     final_set = set()
    #     final_set.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))
    #     return sorted(final_set, key=str)

    @property
    def inputs(self):
        """
        Combines continuous and categorical features and removed duplicated..
        Returns a list of Route objects.
        """
        return list(dict.fromkeys(self.continuous_features + self.categorical_features))

class BogNet(
    TensorBoardLogger,
    NetworkBase,
):

    def __init__(self, *args, tensorboard_path, logger, task, **kwargs):
        super().__init__(*args, tensorboard_path=tensorboard_path, logger=logger, **task.param_kwargs, **kwargs)
        self.passed_parameters = tuple(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.nodes = 128
        self.activation_functions = "PReLu"
        self.skip_connection_init = 1
        self.freeze_skip_connection = False
        self.empty_value = 15
        self.input_layer, self.model = self.init_layers()
        self.label_smoothing_coefficient = 0.01
        self._loss_fn = WeightedCrossEntropy(label_smoothing=self.label_smoothing_coefficient)

        # metrics
        self.validation_metrics = {
            "loss": WeightedLoss(self.loss_fn),

        }
        self.validation_metrics.update({
            f"loss_cls_{identifier}": WeightedLoss(
                WeightedCrossEntropySlice(cls_index=idx, label_smoothing=self.label_smoothing_coefficient),
            )
            for identifier, idx in self.categorical_target_map.items()
        })

        self.validation_metrics.update({
            f"roc_auc_cls_{identifier}": WeightedROC_AUC(
                output_transform=partial(
                    preprocess_multiclass_outputs,
                    multi_class="ovr",
                    average=None,
                ),
                target_class_idx=idx,
            )
            for identifier, idx in self.categorical_target_map.items()
        })

        if "perform_scheduler_step" not in self.custom_hooks:
            self.custom_hooks.append("perform_scheduler_step")

        # disable checkpoints for now since they point to Bogdan specifically
        if "add_checkpoints" in self.custom_hooks:
            self.custom_hooks.remove("add_checkpoints")

    def state_dict(self, *args, **kwargs):
        # IMP
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        # IMP
        return self.model.load_state_dict(*args, **kwargs)

    def perform_scheduler_step(self):
        if hasattr(self, "scheduler") and self.scheduler:
            def do_step(engine, logger=self.logger):
                self.scheduler.step()
                logger.info(f"Performing scheduler step, last lr: {self.scheduler.get_last_lr()}")

            # TODO better scheduling
            self.train_evaluator.add_event_handler(
                Events.EPOCH_COMPLETED,
                do_step,
            )

    def add_checkpoints(self):
        def save_checkpoint(engine):
            epoch = engine.state.epoch
            outpath = f"/data/dust/user/wiedersb/model_epoch{epoch}.pt"
            self.logger.info(f"saving checkpoint for epoch {epoch} in {outpath}")
            torch.save(self.model, outpath)

        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=1),
            save_checkpoint,
        )


    def init_layers(self):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.5e-5
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        std_layer = StandardizeLayer()

        continuous_padding = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
        categorical_padding = PaddingLayer(padding_value=self.empty_value, mask_value=EMPTY_INT)
        rotation_layer = RotatePhiLayer(
            columns=list(map(str, self.continuous_features)),
        )
        input_layer = InputLayer(
            continuous_inputs=self.continuous_features,
            categorical_inputs=self.categorical_features,
            embedding_dim=4,
            expected_categorical_inputs=embedding_expected_inputs,
            empty=self.empty_value,
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continous_layer=continuous_padding,
        )

        resnet_blocks = [
            ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection, eps=eps, normalize=normalize)
            for num_blocks in range(4)
            ]

        model = torch.nn.Sequential(
            input_layer,
            DenseBlock(input_nodes = input_layer.ndim, output_nodes = self.nodes, activation_functions=self.activation_functions, eps=eps, normalize=normalize), # noqa
            *resnet_blocks,
            torch.nn.Linear(self.nodes, len(self.categorical_target_map)),
            # no softmax since this is already part of loss
        )
        return input_layer, model

    def train_step(self, engine, batch):
        self.model.train()

        # Compute prediction and loss
        (categorical_x, continous_x), y = batch

        self.optimizer.zero_grad()

        pred = self(categorical_x, continous_x)
        target = y.to(torch.float32)
        if target.dim() == 1:
            target = target.reshape(-1, 1)

        loss = self.loss_fn(pred, target)
        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, engine, batch):
        self.model.eval()
        # Set the model to evaluation mode - important for batch normalization and dropout layers

        # if engine.state.iteration > self.max_val_epoch_length * (engine.state.epoch + 1):
        #     engine.terminate_epoch()

        with torch.no_grad():
            (categorical_x, continous_x, weights), y = batch
            from IPython import embed; embed(header="valid - 280 in bognet.py ")
            pred = self(categorical_x, continous_x)

            if y.dim() == 1:
                y = y.reshape(-1, 1)
            # from IPython import embed; embed(header="validation_step - 337 in bognet.py ")
            y = y.to(torch.float32)

            return pred, y, {"weight": weights}

    def setup_preprocessing(self, device: str | None = None):
        # extract dataset std and mean from dataset
        # extraction happens form no oversampled dataset
        mean, std = [], []
        if not device:
            device = next(self.model.parameters()).device
        for _input in self.continuous_features:
            input_statistics = self.dataset_statistics[_input]
            mean.append(torch.from_numpy(input_statistics["mean"]))
            std.append(torch.from_numpy(input_statistics["std"]))

        mean, std = torch.concat(mean).to(device), torch.concat(std).to(device)
        # set up standardization layer
        self.input_layer.std_layer.update_buffer(
            mean.float(),
            std.float(),
        )

    def log_graph(self, engine):
        input_data = engine.state.batch[0]
        self.writer.add_graph(self, input_data)

    def logging(self, *args, **kwargs):
        # output histogram
        for target, index in self.categorical_target_map.items():
            # apply softmax to prediction
            logit = kwargs["prediction"]

            self.writer.add_histogram(
                f"output_logit_{target}",
                logit[:, index],
                self.trainer.state.iteration,
            )

    def init_dataset_handler(
        self,
        task: law.Task,
        *args,
        device: str | None = None,
        datasets: list[str] | None = None,
        extract_dataset_paths_fn: Callable | None = None,
        extract_probability_fn: Callable | None = None,
        inputs: law.FileCollection | None = None,
        **kwargs,
    ) -> None:
        all_datasets = datasets or getattr(task, "resolved_datasets", task.datasets)
        # build group datasets from processes, remove empty sets
        group_datasets = {
            "ttbar": [d for d in all_datasets if d.startswith("tt_")],
            "dy": [d for d in all_datasets if d.startswith("dy_")],
        }
        group_datasets = {k: v for k, v in group_datasets.items() if v}

        # from IPython import embed; embed(header="HANDLING - 389 in bognet.py ")
        self.dataset_handler = WeightedTensorParquetFileHandler(
            task=task,
            inputs=inputs,
            continuous_features=getattr(self, "continuous_features", self.continuous_features),
            categorical_features=getattr(self, "categorical_features", self.categorical_features),
            batch_transformations=MoveToDevice(device=device),
            build_categorical_target_fn=self._build_categorical_target,
            group_datasets=group_datasets,
            device=device,
            categorical_target_transformation=partial(get_one_hot, nb_classes=len(self.categorical_target_map)),
            datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
            extract_dataset_paths_fn=extract_dataset_paths_fn,
            extract_probability_fn=extract_probability_fn,
        )
        # from IPython import embed; embed(header="init - 357 in bognet.py ")
        self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa

        # define lenght of training epoch and of all other datasets
        self.max_epoch_length = self._calculate_max_epoch_length(
            self.training_loader,
            cutoff=self.training_epoch_length_cutoff,
            weight_cutoff=self.training_weight_cutoff,
        )
        from IPython import embed; embed(header="dataset - 367 in bognet.py ")
        len_dataloader = lambda x: sum([len(d.data) for d in x.data_map])
        self.unsampeled_training_length = len_dataloader(self.train_validation_loader)
        self.unsampeled_validation_length = len_dataloader(self.validation_loader)

        # from IPython import embed; embed(header="string - 423 in bognet.py ")
        # get statistics for standardization from training dataset without oversampling
        # from IPython import embed; embed(header="string - 417 in bognet.py ")
        self.dataset_statistics = get_standardization_parameter(
            self.train_validation_loader.data_map, self.continuous_features,
        )

    def prepare_weight_decay(self, optimizer_config) -> None:
        # define which layers should contribute to the weight decay
        no_weight_decay_parameters, weight_decay_parameters = normalized_weight_decay(
            self.model,
            decay_factor=optimizer_config["decay_factor"],
            normalize=optimizer_config["normalize"],
            # apply_to="weight.original0",
            apply_to=optimizer_config["apply_to"],
        )
        return no_weight_decay_parameters, weight_decay_parameters

    def init_optimizer(self, optimizer_config) -> None:
        no_weight_decay_param, weight_decay_param = self.prepare_weight_decay(optimizer_config)
        self.optimizer = torch.optim.AdamW(
            (no_weight_decay_param, weight_decay_param),
            lr=optimizer_config["learning_rate"],
        )

    def init_scheduler(self, scheduler_config):
        if getattr(self, "optimizer") is None:
            raise "Implement Optimizer before Init Scheduler"
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, **scheduler_config)

    def forward(self, *inputs):
        return self.model(inputs)

    def _build_categorical_target(self, dataset: str):
        for key in self.categorical_target_map.keys():
            if dataset.startswith(key):
                return self.categorical_target_map[key]
        raise ValueError(f"Dataset {dataset} is not present in categorical target map {self.categorical_target_map}.")

    def _calculate_max_epoch_length(
        self,
        composite_loader,
        weight_cutoff: float | None = None,
        cutoff: int | None = None,
    ):
        """
        Calculate number of iterations per epoch based on the given dataset composition.
        If *weight_cutoff* is given, only datasets with a weight contribution larger than this value are considered.
        If *cutoff* is given, the maximum number of iterations is limited to this value, otherwise full epoch is used.

        Args:
            composite_loader (_type_): DataLoader/Sampler containing the dataset composition
            weight_cutoff (float | None, optional): Lower threshold for datasets to consider. Defaults to None.
            cutoff (int | None, optional): Upper threshold for iterations. Defaults to None.

        Returns:
            float: Number of iterations per epoch, based on the dataset composition.
        """

        global_max = 0
        max_key = None
        weight_cutoff = weight_cutoff or 0.
        batch_comp = getattr(composite_loader.batcher, "_batch_composition", None)
        # composition_length = {}
        if not batch_comp:
            batch_comp = {key: 1. for key in composite_loader.data_map.keys()}
        # tree level
        for key, batchsize in batch_comp.items():
            weights = composite_loader.weight_dict[key]
            if isinstance(weights, float):
                data = composite_loader.data_map[key]
                if isinstance(data, (list, tuple, set)):
                    total_length = sum([len(x) for x in data])
                else:
                    total_length = len(data)
                local_max = np.ceil(total_length / batchsize)
                if local_max > global_max:
                    max_key = key
                    global_max = local_max
            # subtree level
            elif isinstance(weights, dict):
                for subkey, weight in weights.items():
                    data = composite_loader.data_map[subkey]
                    if isinstance(data, (list, tuple, set)):
                        total_length = sum([len(x) for x in data])
                    else:
                        total_length = len(data)
                    submax = np.ceil(total_length / batchsize / weight)
                    # filter out datasets with small weight contribution
                    if submax > global_max and weight >= weight_cutoff:
                        global_max = submax
                        max_key = subkey

        if cutoff:
            global_max = np.min([global_max, cutoff])
            self.logger.warning(
                f"Set max training iterations from {self.max_epoch_length} to {self.training_epoch_length_cutoff}"
            )

        self.logger.info(f"epoch dominated by  '{max_key}': expect {float(global_max)} batches/iteration")
        return global_max

class BogNetNormal(
    IgniteEarlyStoppingMixin,
    IgniteTrainingMixin,
    TensorBoardLogger,
    NetworkBase,
):
    def __init__(self):
        super.__init__()

    def init_dataset_handler(self, task):
        return super().init_dataset_handler(task)

    def init_dataset_handler(
        self,
        task: law.Task,
        *args,
        device: str | None = None,
        datasets: list[str] | None = None,
        extract_dataset_paths_fn: Callable | None = None,
        extract_probability_fn: Callable | None = None,
        inputs: law.FileCollection | None = None,
        ratio: tuple[float, float, float] = (0.7, 0.2, 0.1),
        **kwargs,
    ) -> None:
        all_datasets = datasets or getattr(task, "resolved_datasets", task.datasets)
        # build group datasets from processes, remove empty sets
        group_datasets = {
            "ttbar": [d for d in all_datasets if d.startswith("tt_")],
            "dy": [d for d in all_datasets if d.startswith("dy_")],
        }
        group_datasets = {k: v for k, v in group_datasets.items() if v}

        # from IPython import embed; embed(header="HANDLING - 389 in bognet.py ")
        self.dataset_handler = WeightedTensorParquetFileHandler(
            task=task,
            inputs=inputs,
            continuous_features=getattr(self, "continuous_features", self.continuous_features),
            categorical_features=getattr(self, "categorical_features", self.categorical_features),
            batch_transformations=MoveToDevice(device=device),
            build_categorical_target_fn=self._build_categorical_target,
            group_datasets=group_datasets,
            device=device,
            categorical_target_transformation=partial(get_one_hot, nb_classes=len(self.categorical_target_map)),
            datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
            extract_dataset_paths_fn=extract_dataset_paths_fn,
            extract_probability_fn=extract_probability_fn,
            ratio=ratio,
        )

        self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa

        # define lenght of training epoch and of all other datasets
        self.max_epoch_length = self._calculate_max_epoch_length(
            self.training_loader,
            cutoff=self.training_epoch_length_cutoff,
            weight_cutoff=self.training_weight_cutoff,
        )

        len_dataloader = lambda x: sum([len(d.data) for d in x.data_map])
        self.unsampeled_training_length = len_dataloader(self.train_validation_loader)
        self.unsampeled_validation_length = len_dataloader(self.validation_loader)

        # from IPython import embed; embed(header="string - 423 in bognet.py ")
        # get statistics for standardization from training dataset without oversampling
        # from IPython import embed; embed(header="string - 417 in bognet.py ")
        self.dataset_statistics = get_standardization_parameter(
            self.train_validation_loader.data_map, self.continuous_features,
        )
class FocalBogNet(BogNet):
    def __init__(self, *args, tensorboard_path, logger, task, **kwargs):
        super().__init__(*args, tensorboard_path=tensorboard_path, logger=logger, task=task, **kwargs)
        self.loss_fn = FocalLoss(
            gamma=4,
            alpha=None,
            reduction='mean',
            task_type='multi-class',
            num_classes=3
        )

class BogNetLBN(BogNet):
    def __init__(self, *args, tensorboard_path, logger, task, **kwargs):
        super().__init__(*args, tensorboard_path=tensorboard_path, logger=logger, task=task, **kwargs)


class BogNetSAM(BogNet):
    from torch.nn.modules.batchnorm import _BatchNorm

    def __init__(self, *args, tensorboard_path, logger, task, **kwargs):
        super().__init__(*args, tensorboard_path=tensorboard_path, logger=logger, task=task, **kwargs)

    def init_optimizer(self, optimizer_config, scheduler_config=None) -> None:
        no_weight_decay_param, weight_decay_param = self.prepare_weight_decay(optimizer_config=optimizer_config)
        # reuse base optimizer and wrap it in SAM
        # ZZZZ Optimizer AdamW not init
        # from IPython import embed; embed(header="string - 552 in bognet.py ")
        base_optim_config = {
            "lr": optimizer_config["learning_rate"],
        }

        # init is done by SAM
        base_optimizer = torch.optim.AdamW

        self.optimizer = SAM(
            params=(no_weight_decay_param, weight_decay_param),
            base_optimizer=base_optimizer,
            rho=0.05,
            adaptive=False,
            **base_optim_config # forwarded to base optimizer
        )

    def disable_running_stats(self, model):
        # set save previous moment as backup_momentum set momentum to 0
        # this is necessary since SAM doing a 2 step forward and batchnorm statistics would be wrong
        def _disable(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def enable_running_stats(self, model):
        # renable previous moment
        def _enable(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)

    def train_step(self, engine, batch):
        self.model.train()
        (categorical_x, continous_x), y = batch
        target = y.to(torch.float32)
        if target.dim() == 1:
            target = target.reshape(-1, 1)

        # first forward step with enabled bachnorm running stats
        # batch norm statistics are calculated 2 times, setting momentum to 0 prevents this
        # print(self.model, len(self.model))
        # from IPython import embed; embed(header="TRAIN_STEP - 588 in bognet.py ")
        self.enable_running_stats(self.model)
        pred = self(categorical_x, continous_x)
        loss = self.loss_fn(pred, target)
        from IPython import embed; embed(header="string - 614 in bognet.py ")
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # second forward step with disabled bachnorm running stats in second forward step
        self.disable_running_stats(self.model)
        pred_2 = self(categorical_x, continous_x)
        self.loss_fn(pred_2, target).backward()
        self.optimizer.second_step(zero_grad=True)

        return loss.item()
